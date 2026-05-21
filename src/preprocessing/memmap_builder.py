"""This objective of this is to prepare the dataset for training. Since the dataset is enourmous
the plan is to tokenize and save it as a binary file in the disk, so we can load it to memory only during 
actual training. This inspiration comes from Karpathy's nanoGPT implementation.

link: https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py
"""
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

# special tokens
tokenizer = Tokenizer.from_file(VOCAB_PATH)
BOS_ID = tokenizer.token_to_id("<|bos|>")
EOT_ID = tokenizer.token_to_id("<|endoftext|>")

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
    ids = tokenizer.encode(text).ids
    ids = [BOS_ID] + tokenizer.encode(text).ids + [EOT_ID]
    out = {"ids": ids, "len": len(ids)}
    
    return out

# --- Main Function ---
def main():
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



if __name__ == "__main__":
    main()