"""This objective of this is to prepare the dataset for training. Since the dataset is enourmous
the plan is to tokenize and save it as a binary file in the disk, so we can load it to memory only during 
actual training. This inspiration comes from Karpathy's nanoGPT implementation.

link: https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py
"""
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pyarrow.parquet as pq
from tokenizers import Tokenizer

# internal imports
from ..utils.logger import get_logger

# --- Configuration ---
VOCAB_PATH = "./data/vocab/tokenizer_vocab.json"
PRETRAINING_DIR = Path("./data/pretraining")
FILE_EXT = ".parquet"
CORPUS_METADATA_PATH = "./data/corpus/metadata.json"

# special tokens
tokenizer = Tokenizer.from_file(VOCAB_PATH)
BOS_ID = tokenizer.token_to_id("<|bos|>")
EOT_ID = tokenizer.token_to_id("<|endoftext|>")
CONTEXT_SIZE = 1024

# logging
_logger = get_logger("memmap")

# --- Helper Functions ---
def _iter_all_documents(directory: Path):
    for item in directory.iterdir():
        if item.is_dir():
            for shard in item.glob(f"*{FILE_EXT}"):
                pf = pq.ParquetFile(shard)
                for batch in pf.iter_batches(columns=["text"]):
                    yield from batch.column("text").to_pylist()

def process(text: str):
    ids = [BOS_ID] + tokenizer.encode(text).ids + [EOT_ID]
    out = {"ids": ids, "len": len(ids)}
    return out

# --- Main Function ---
def main():
    Path("./data/corpus").mkdir(parents=True, exist_ok=True)
    _logger.info("** Initiating MemMap Builder 1st Pass - Counting Total Tokens **")
    total_tok = 0

    for text in _iter_all_documents(PRETRAINING_DIR):
        out = process(text)
        total_tok += out["len"]

    _logger.info("="*60)
    _logger.info("First pass completed!")
    _logger.info(f"Total tokens present in Corpus: {total_tok}")
    _logger.info("="*60)
    _logger.info("** Initiating MemMap Builder 2nd Pass - Creating MemMap File **")

    arr_len = total_tok
    filename = "./data/corpus/corpus.bin"
    dtype = np.uint16
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    
    offset = 0
    for text in tqdm(_iter_all_documents(PRETRAINING_DIR), desc=f'writing {filename}'):
        out = process(text)
        # Write into mmap
        arr[offset : offset + len(out["ids"])] = out["ids"]
        offset += len(out["ids"])
    arr.flush()

    # We'll create pointers for both the train and validation set
    _logger.info("Defining Train and Validation Splits")
    total_chunks = arr_len // CONTEXT_SIZE
    val_size = int((arr_len * 0.0005) // CONTEXT_SIZE) # WITH 9B tokens, we get approx 4M tokens for validation


    shuffle_index = np.arange(total_chunks)
    shuffle_index = np.random.permutation(shuffle_index)
    val_set = shuffle_index[-val_size:]
    train_set = shuffle_index[:-val_size]
    _logger.info("Saving Train and Validation indices")
    np.save("./data/corpus/train_indices.npy", train_set)
    np.save("./data/corpus/val_indices.npy", val_set)

    metadata = {
        "total_tokens": total_tok,
        "vocab_size": tokenizer.get_vocab_size(),
        "context_size": CONTEXT_SIZE,
        "total_chunks": int(total_chunks),
        "train_chunks": int(len(train_set)),
        "val_chunks": int(val_size),
    }
    _logger.info(f"Saving corpus metadata to: {CORPUS_METADATA_PATH}")
    with open(CORPUS_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    _logger.info("**CORPUS PREPARATION DONE!**")
    _logger.info("="*60)
    _logger.info(f"Total Chunks: {arr_len // CONTEXT_SIZE}")
    _logger.info(f"Total Corpus Tokens: {arr_len / 1e9:.2f}B")
    _logger.info(f"Train split tokens: {(arr_len - (arr_len * 0.0005)) / 1e9:.2f}B")
    _logger.info(f"Val split tokens: {(arr_len * 0.0005) / 1e9:.2f}B")
    _logger.info("="*60)

if __name__ == "__main__":
    main()