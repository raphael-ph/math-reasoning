import time
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

# internal imports
from ..utils.logger import get_logger

_logger = get_logger("hf_tokenizer", level="DEBUG")

VOCAB_SIZE = 12257 
DATASET_PATH = "data/corpus/final_training_corpus.txt"
MODEL_SAVE_PATH = "fast_tokenizer.json"

def train_fast_tokenizer():
    _logger.info(f"--- Starting Rust-based training (Target Vocab: {VOCAB_SIZE}) ---")
    start_time = time.time()

    # Initialize a BPE Tokenizer
    # We use ByteLevel BPE, which is the standard for GPT-2/3/4 models.
    # It handles all unicode characters by falling back to bytes.
    tokenizer = Tokenizer(models.BPE())
    
    # Pre-tokenization (How to split the text before BPE)
    # This uses the standard GPT-2 byte-level pre-tokenizer.
    # It handles the "space" logic automatically.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # Decoding (How to go back to text)
    tokenizer.decoder = decoders.ByteLevel()

    # Define the Trainer
    # special_tokens matches what you had. 
    # The library handles the merge math in Rust (C++ speed).
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<|endoftext|>"], 
        show_progress=True
    )

    _logger.info(f"Reading {DATASET_PATH} and training...")
    try:
        tokenizer.train([DATASET_PATH], trainer)
    except Exception as e:
        _logger.info(f"\nError: Could not find file '{DATASET_PATH}'. Please check the path.")
        return

    end_time = time.time()
    elapsed = end_time - start_time

    # Save
    tokenizer.save(MODEL_SAVE_PATH)

    _logger.info("\n" + "="*40)
    _logger.info(f"DONE!")
    _logger.info(f"Total Time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    _logger.info(f"Vocab Size: {tokenizer.get_vocab_size()}")
    _logger.info(f"Saved to:   {MODEL_SAVE_PATH}")
    _logger.info("="*40)

if __name__ == "__main__":
    train_fast_tokenizer()