import os
import json
import regex as re
import multiprocessing
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List, Optional
from collections import defaultdict

# internal imports
from ..utils.logger import get_logger

# globals
VOCAB_SIZE = 12260 # GPT-2 vocab size for 10B tokens of trainig was 50000
NUM_MERGES = VOCAB_SIZE - 256
GPT4_SPLIT_PAT = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
SPECIAL_TOKENS = {
    "<|endoftext|>": 12256, # token marking the End of a File (similar to OpenAI)
    "<|fim_prefix|>": 12257,
    "<|fim_suffix|>": 12258,
    "<|fim_middle|>": 12259,
}
_logger = get_logger(__name__, level="DEBUG")

# --- WORKER FUNCTION ---
def worker_loop(rank: int, chunk_ids: List[List[int]], cmd_queue, res_queue):
    try:
        while True:
            cmd, payload = cmd_queue.get()
            
            if cmd == 'stats':
                stats = defaultdict(int)
                for chunk in chunk_ids:
                    # OPTIMIZATION: Check len once
                    if len(chunk) < 2: continue
                    # Fast iteration
                    for pair in zip(chunk, chunk[1:]):
                        stats[pair] += 1
                res_queue.put(dict(stats))
                
            elif cmd == 'merge':
                pair, idx = payload
                p0, p1 = pair
                
                # Iterate and merge
                for i, chunk in enumerate(chunk_ids):
                    if len(chunk) < 2: continue
                    
                    new_chunk = []
                    j = 0
                    n = len(chunk)
                    while j < n:
                        # Safety check for bounds
                        if j < n - 1 and chunk[j] == p0 and chunk[j+1] == p1:
                            new_chunk.append(idx)
                            j += 2
                        else:
                            new_chunk.append(chunk[j])
                            j += 1
                    chunk_ids[i] = new_chunk
                
                res_queue.put(True)
                
            elif cmd == 'stop':
                break
    except Exception as e:
        # Send error and properly exit
        try:
            res_queue.put(e)
        except:
            pass # Queue might be closed

# --- MAIN CLASS ---
class Tokenizer:
    # ... (Your __init__ remains the same) ...
    def __init__(self, 
                 special_tokens: Optional[Dict[str, int]] = None, 
                 pattern: Optional[str] = None, 
                 vocab_size: Optional[int] = None, 
                 num_merges: Optional[int] = None,
                 vocab_output_path: Optional[str] = None):
        
        if not special_tokens:
            special_tokens = SPECIAL_TOKENS
        self.special_tokens = special_tokens
        self.inv_special_tokens = self._invert_special_tokens(special_tokens)
        pattern = GPT4_SPLIT_PAT if pattern is None else pattern
        self.pattern = re.compile(pattern)
        self.vocab_size = VOCAB_SIZE if vocab_size is None else vocab_size
        self.num_merges = NUM_MERGES if num_merges is None else num_merges

        if not vocab_output_path:
            vocab_path = "data/vocab"
            self.vocab_output_path = Path(vocab_path)
        else:
            vocab_path = Path(vocab_output_path)
            if os.path.isdir(vocab_path):
                self.vocab_output_path = vocab_path
            elif os.path.exists(vocab_path):
                raise Exception(f"'{vocab_path}' exists but is not a directory.")
            else:
                os.mkdir(vocab_path)
                self.vocab_output_path = vocab_path

    def train(self, text, verbose: bool=False):
        """Train the tokenizer using persistent workers with RAM protection"""
        
        _logger.info("Splitting text...")
        text_chunks = re.findall(self.pattern, text)
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]
        
        # --- CRITICAL FIX 1: Limit Workers to save RAM ---
        # Even if you have 100 CPUs, using them all will explode RAM due to Copy-On-Write
        available_cpus = os.cpu_count()
        # Cap at 8 workers to prevent RAM explosion. Increase only if you have 500GB+ RAM.
        max_workers = min(8, available_cpus - 1) 
        max_workers = max(1, max_workers) # Ensure at least 1
        
        _logger.info(f"Distributing data across {max_workers} processes (Capped to save RAM)...")

        chunk_size = (len(ids) + max_workers - 1) // max_workers
        id_batches = [ids[i:i + chunk_size] for i in range(0, len(ids), chunk_size)]
        
        cmd_queues = [multiprocessing.Queue() for _ in range(max_workers)]
        res_queue = multiprocessing.Queue()
        
        processes = []
        for i in range(max_workers):
            if i < len(id_batches):
                p = multiprocessing.Process(
                    target=worker_loop, 
                    args=(i, id_batches[i], cmd_queues[i], res_queue)
                )
                p.start()
                processes.append(p)
            else:
                cmd_queues[i].close()

        active_workers = len(processes)
        _logger.info(f"Workers ready. Starting BPE loop for {self.num_merges} merges...")

        vocab = {idx: bytes([idx]) for idx in range(256)}
        merges = {}

        try:
            for i in range(self.num_merges):
                # 1. BROADCAST STATS
                for q in cmd_queues[:active_workers]:
                    q.put(('stats', None))
                
                # 2. AGGREGATE STATS (With Timeouts)
                global_stats = defaultdict(int)
                for _ in range(active_workers):
                    try:
                        # --- CRITICAL FIX 2: Timeout ---
                        # If a worker dies, this will raise Empty after 60s instead of freezing forever
                        res = res_queue.get(timeout=600) 
                    except multiprocessing.queues.Empty:
                        raise RuntimeError("Worker process timed out. Likely OOM Kill.")
                    
                    if isinstance(res, Exception):
                        raise res
                    for pair, count in res.items():
                        global_stats[pair] += count
                
                if not global_stats:
                    break

                top_pair = max(global_stats, key=global_stats.get)
                idx = 256 + i
                
                # 3. BROADCAST MERGE
                for q in cmd_queues[:active_workers]:
                    q.put(('merge', (top_pair, idx)))
                
                # 4. WAIT FOR CONFIRMATION
                for _ in range(active_workers):
                    try:
                        res = res_queue.get(timeout=600)
                    except multiprocessing.queues.Empty:
                        raise RuntimeError("Worker stuck during merge. Check RAM usage.")
                    
                    if isinstance(res, Exception):
                        raise res
                
                merges[top_pair] = idx
                vocab[idx] = vocab[top_pair[0]] + vocab[top_pair[1]]
                
                if verbose or i % 10 == 0:
                    _logger.info(f"merge {i+1}/{self.num_merges}: {top_pair} -> {idx} ({vocab[idx]}) had {global_stats[top_pair]} occurrences")

        except Exception as e:
            _logger.error(f"Training crashed: {e}")
            raise e # Re-raise so you see the error
        finally:
            _logger.info("Stopping workers...")
            for q in cmd_queues[:active_workers]:
                q.put(('stop', None))
            for p in processes:
                # Force kill if they don't exit quickly
                p.join(timeout=5)
                if p.is_alive():
                    p.terminate()

        _logger.info(f"Tokenization completed!")
        self.merges = merges
        self.vocab = vocab
        self._save_config()

    def _save_config(self):
        serializable_merges = {f"{pair[0]}, {pair[1]}": idx for pair, idx in self.merges.items()}
        tokenizer_config = {
            "name": "Custom_BPE_Tokenizer",
            "vocab_size": len(self.vocab),
            "pattern": self.pattern.pattern,
            "special_tokens": self.special_tokens,
            "merges": serializable_merges
        }
        config_file_name = f"tokenizer_config_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
        full_path = self.vocab_output_path / config_file_name
        try:
            with open(full_path, "w", encoding="utf-8") as f:
                json.dump(tokenizer_config, f, indent=4)
            _logger.info(f"Configuration saved to {full_path}")
        except Exception as e:
            _logger.error(f"Failed to save config: {e}")

    def decode(self, ids: List) -> str:
        """Decodes tokens to natural language text
        
        This already handle any special tokens that could exist in the token list.

        Args:
            ids (List): List of tokens
        
        Returns:
            The decoded text string
        """
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inv_special_tokens:
                part_bytes.append(self.inv_special_tokens[idx].encode("utf-8"))
            else:
                _logger.error(f"Invalid token id: {idx}")
                raise ValueError(f"Invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        
        return text

    def encode(self, text: str) -> List:
        """Encodes text to tokens w.r.t spedial tokens
        
        Args:
            text (str): Text to encode
        
        Returns:
            List of encoded ids
        """
        # we handle special tokens by splitting the text whenever we find the 
        # exact occurence of any of the special tokens. We use re.split()
        # to extract every occurence of special tokens and change it for the corresponding token
        # mapped on the self.pattern dict.
        # Reference for the special_pattern: https://github.com/karpathy/minbpe/blob/master/minbpe/regex.py
        # This works because special characters are defined like OAI's, e.g.: <|specialcharacter|>
        special_pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"
        special_chunks = re.split(special_pattern, text)
        ids = []
        for part in special_chunks:
            if part in self.special_tokens:
                _logger.debug(f"Special token detected at part: {part}")
                ids.append(self.special_tokens[part]) # handle special tokens
            else:
                _logger.debug(f"Non special token at part: {part}")
                ids.extend(self.encode_no_special(part)) # simply encode the no-special part
        return ids

    def encode_no_special(self, text: str) -> List:
        """Encodes text ignoring the special tokens completely.
        
        In this case, special token s will be trated and encoded like
        any other regular token. This follows the tiktoken implementation.

        Args:
            text (str): Text to encode
        
        Returns:
            List of encoded ids
        """
        text_chunks = re.findall(self.pattern, text)
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)

        return ids
    
    @classmethod
    def from_file(cls, config_path: str):
        """Loads a tokenizer instance from a saved config file.
        
        Args:
            config_path (str): For for the tokenizer generated config file
        
        Returns:
            Tokenizer: trained tokenizer
        """
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        instance = cls(
            vocab_size=config["vocab_size"],
            pattern=config["pattern"],
            special_tokens=config["special_tokens"]
        )
        instance.merges = {
            tuple(map(int, k.split(", "))): v 
            for k, v in config["merges"].items()
        }
        instance.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in instance.merges.items():
            instance.vocab[idx] = instance.vocab[p0] + instance.vocab[p1] 
        return instance

    def _invert_special_tokens(self, special_tokens_dict: Dict[str, int]) -> Dict[int, str]:
        return {v : k for k, v in special_tokens_dict.items()}
    
    def _encode_chunk(self, text_bytes: bytes) -> List:
        ids = list(text_bytes)
        if self.merges is None:
            raise ValueError(f"Tokenizer is not trained.")
        while len(ids) >= 2: 
            stats = {}
            # Quick local stats for inference (no need for multiprocessing here)
            for bigram in zip(ids, ids[1:]):
                stats[bigram] = stats.get(bigram, 0) + 1
            
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break 
            idx = self.merges[pair]
            
            # Simple merge for inference
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                    new_ids.append(idx)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids
        return ids