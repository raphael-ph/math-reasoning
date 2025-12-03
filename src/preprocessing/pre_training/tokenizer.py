# --- Tokenizer class ---
# This implementation has many inspirations and it took me sometime to build this rather simple
# Acknowledgements to:
# 
# 1. Karpathy (as always): https://www.youtube.com/watch?v=zduSFxRajkE
# 2. This blog: https://sebastianraschka.com/blog/2025/bpe-from-scratch.html
# 3. GPT-2 paper: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

import os
import json
import regex as re # A more advanced, feature-rich alternative for complex and performance-critical regular expression operations
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List, Optional

# internal imports
from ...utils.logger import get_logger

# globals
VOCAB_SIZE = 12257 # GPT-2 vocab size for 10B tokens of trainig was 50000
NUM_MERGES = VOCAB_SIZE - 256
GPT4_SPLIT_PAT = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
SPECIAL_TOKENS = {
    "<|endoftext|>": 12256, # token marking the End of a File (similar to OpenAI)
}
# set up logging
_logger = get_logger(__name__, level="DEBUG")
class Tokenizer:
    def __init__(self, 
                 special_tokens: Dict[str, int], 
                 pattern: Optional[str] = None, 
                 vocab_size: Optional[int] = None, 
                 num_merges: Optional[int] = None,
                 vocab_output_path: Optional[str] = None):
        self.special_tokens = special_tokens
        self.inv_special_tokens = self._invert_special_tokens(special_tokens)
        # split pattern
        pattern = GPT4_SPLIT_PAT if pattern is None else pattern
        self.pattern = re.compile(pattern)
        self.vocab_size = VOCAB_SIZE if vocab_size is None else vocab_size
        self.num_merges = NUM_MERGES if num_merges is None else num_merges

        # defining the output vocab path
        if not vocab_output_path:
            vocab_path = "data/vocab"
            _logger.info(f"No output path provided. Fallback vocab to default path: {vocab_path}")
            self.vocab_output_path = Path(vocab_path)
        else:
            vocab_path = Path(vocab_output_path)
            if os.path.isdir(vocab_path):
                self.vocab_output_path = vocab_path
            elif os.path.exists(vocab_path):
                print(f"'{vocab_path}' exists but is not a directory.")
                raise Exception(f"'{vocab_path}' exists but is not a directory.")
            else:
                _logger.info(f"'{vocab_path}' does not exist. Creating...")
                os.mkdir(vocab_path)
                self.vocab_output_path = vocab_path
    
    # --- MAIN FUNCTIONS ---
    def train(self, text, verbose: bool=False):
        """Train the tokenizer on the list of tokens"""
        merges = {} # have the mapping (int, int) -> int of the pair to the new token

        # break the chunks
        text_chunks = re.findall(self.pattern, text)

        if verbose: 
            _logger.info(f"Text chunks: {text_chunks[:10]}")
            _logger.info(f"Converting input text to utf-8 ids...")
            _logger.info(f"--- Example of convertion: ---")
            for ch in text_chunks[:10]:
                _logger.info(f"{ch} -> {list(ch.encode("utf-8"))}")
        
        # input for text processing        
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)} # initialize the vocab
        for i in range(self.num_merges):
            stats = {}
            for chunk_ids in ids:
                self._get_stats(chunk_ids, stats)
            if not stats:
                break
            top_pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = [self._merge(chunk_ids, top_pair, idx) for chunk_ids in ids]
            merges[top_pair] = idx
            vocab[idx] = vocab[top_pair[0]] + vocab[top_pair[1]]
            if verbose:
                _logger.info(f"merge {i+1}/{self.num_merges}: {top_pair} -> {idx} ({vocab[idx]}) had {stats[top_pair]} occurrences")
        
        # Tokenization completion
        _logger.info(f"Tokenization completed!")
        _logger.info(f"Vocab size : {len(vocab)}")
        _logger.info(f"New tokens: {len(vocab) - 256}")
        
        # Saving in memory
        self.merges = merges # used at encode()
        self.vocab = vocab   # used at decode()

        # Formatting the tokenization config file
        serializable_merges = {
            f"{pair[0]}, {pair[1]}": idx
            for pair, idx in self.merges.items()
        }

        tokenizer_config = {
            "name": "Custom_BPE_Tokenizer",
            "vocab_size": len(self.vocab),
            "pattern": self.pattern.pattern, # Save the regex pattern string
            "special_tokens": self.special_tokens,
            "merges": serializable_merges
        }
        
        # saving vocab
        _logger.info(f"Saving vocab to {self.vocab_output_path}")
        config_file_name = f"tokenizer_config_{datetime.now().strftime("%Y%m%d%H%M%S")}.txt"
        full_path = self.vocab_output_path / config_file_name

        _logger.info(f"Saving tokenizer configuration to {full_path}")

        try:
            with open(full_path, "w", encoding="utf-8") as f:
                # json.dump serializes the dictionary to the file, using indent for readability
                json.dump(tokenizer_config, f, indent=4)
            
            _logger.info(f"Configuration saved successfully.")
            
        except Exception as e:
            _logger.error(f"Failed to save tokenizer configuration to {full_path}: {e}")

    def decode(self):
        pass

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

    # --- HELPER FUNCTIONS ---
    def _merge(self, ids: List, bigram: Tuple, idx: int) -> List[int]:
        """Helper function to substitute pairs
        
        This is equivalent to Karpathy's `merge()` function. It finds the 
        byte pairs in ids and changes it to the byte-pair encoded token it to the tokenizer's vocab.

        Args:
            ids (List): list of tokens from the corpus
            bigram (Tuple): These are the byte pairs
            idx (int): The correspondent idx for the pair
        
        Returns:
            List[int]: the new ids, byte-encoded
        """
        bpe = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == bigram[0] and ids[i+1] == bigram[1]:
                bpe.append(idx)
                i += 2
            else:
                bpe.append(ids[i])
                i += 1
        return bpe

    def _get_stats(self, ids: List, counts: Dict[Tuple, int] = None) -> Dict[Tuple, int]:
        """Helper function to get information of bigrams:
        
        Args:
            ids (list): list of the encoded tokens
        
        Returns:
            Dict[Tuple, int]: dictionary conatining the bigram and it's frequency
            of appearance on the corpus
        """
        counts = {} if counts is None else counts
        for bigram in zip(ids, ids[1:]):  # getting together n and n+1 bytes
            # Get value if key does not exist, with a specified default value
            counts[bigram] = counts.get(bigram, 0) + 1
        return counts
    
    def _invert_special_tokens(self, special_tokens_dict: Dict[str, int]) -> Dict[int, str]:
        """Since my implementation follows Karpathy's, this functions is needed to invert the dict.
        
        It is more intuitive to write the special tokens dict with "str" as a key. The implementation, however,
        requires the tokens to have an int as a key.

        Args:
            special_tokens_dict (Dict[str, int]): The written special tokens dict
        
        Returns: Dict[int, str]: inverted dict
        """
        return {v : k for k, v in special_tokens_dict.items()}
    
    def _encode_chunk(self, text_bytes: bytes) -> List:
        """Encodes chunks into tokens
        
        This function is a direct reference to Karpathy's implementation. It
        is designed as a helper function that will encode chunks exclusively. Only change
        is that it reeceives the byte text.

        Args:
            text_bytes (bytes): Byte text representation

        Returns:
            List of ids
        """
        ids = list(text_bytes)
        # assert merges exist, which means the tokenizer is trained
        if self.merges is None:
            _logger.error(f"Tokenizer is not trained. Train the tokenizer before encoding")
            raise ValueError(f"Tokenizer is not trained. Train the tokenizer before encoding")
        while len(ids) >= 2: # at least two tokens, else return the simple tokenization
            # we need to start substituting the pairs with the first index first
            stats = self._get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break # no more pairs
            idx = self.merges[pair]
            ids = self._merge(ids, pair, idx)
        return ids
    
if __name__ == "__main__":
    text = "The bigger they are, the harder they fall."
    tok = Tokenizer(special_tokens=SPECIAL_TOKENS)
    tok.train(text)
    ids = tok.encode("The bigger they are the harder the fall <|endoftext|>")

    print(ids)
