# This here is a script to train the tokenizer remotely, using VastAI
# Eventually, I should probably add this to a directory with all the remote jobs. For now, this will work.

import os
import sys
import glob
import shutil
from pathlib import Path

# Ensure we can import our modules
sys.path.append(os.getcwd())

# internal imports
from src.preprocessing.tokenizer import Tokenizer
from src.utils.logger import get_logger

logger = get_logger("remote_entrypoint")

def main():
    logger.info("--- [Remote] Starting Training Job ---")
    
    # Setup Paths
    # On VastAI, we work in /workspace/ or the current dir
    REMOTE_WORK_DIR = Path(os.getcwd())
    CORPUS_PATH = REMOTE_WORK_DIR / "data/corpus/final_training_corpus.txt"
    
    # Ensure output directory exists (using current dir for simplicity remotely)
    OUTPUT_DIR = REMOTE_WORK_DIR / "output"
    OUTPUT_DIR.mkdir(exist_ok=True)

    if not CORPUS_PATH.exists():
        logger.info(f"ERROR: Corpus not found at {CORPUS_PATH}")
        sys.exit(1)

    # Initialize Tokenizer
    tokenizer = Tokenizer()
    
    # Force the output path to our known remote directory
    tokenizer.vocab_output_path = OUTPUT_DIR 

    # Train
    logger.info("--- [Remote] Loading Corpus ---")
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    
    logger.info(f"--- [Remote] Training on {len(text)} characters ---")
    tokenizer.train(text, verbose=True)

    # Post-Processing
    logger.info("--- [Remote] Standardizing Output Filename ---")
    
    # Find the most recently created config file in the output dir
    list_of_files = glob.glob(str(OUTPUT_DIR / "tokenizer_config_*.txt"))
    
    if not list_of_files:
        logger.info("ERROR: No config file generated!")
        sys.exit(1)
        
    latest_file = max(list_of_files, key=os.path.getctime)
    logger.info(f"Found generated artifact: {latest_file}")
    
    # Rename/Copy to a static name that the runner expects
    static_name = REMOTE_WORK_DIR / "final_tokenizer_artifact.json"
    shutil.copy(latest_file, static_name)
    
    logger.info(f"--- [Remote] Artifact ready at: {static_name} ---")

if __name__ == "__main__":
    main()