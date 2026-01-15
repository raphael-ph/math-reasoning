from tokenizers import Tokenizer
from pathlib import Path
from tqdm import tqdm  

# --- Config ---
TOKENIZER_PATH = "data/vocab/fast_tokenizer.json"
DATASET_PATH = "data/corpus/final_training_corpus.txt"

def count_tokens():
    # 1. Load your trained tokenizer
    print(f"Loading tokenizer from {TOKENIZER_PATH}...")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    
    # 2. Setup counters
    total_tokens = 0
    total_lines = 0
    
    # 3. Stream the file line-by-line to save RAM
    #    We use a buffer size (e.g., 1MB) to read efficiently
    print(f"Counting tokens in {DATASET_PATH}...")
    
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        # Tqdm wraps the file iterator to show a progress bar
        # We estimate lines based on file size for the progress bar (rough guess: 100 bytes/line)
        file_size = Path(DATASET_PATH).stat().st_size
        with tqdm(total=file_size, unit="B", unit_scale=True, desc="Processing") as pbar:
            
            # Read in batches of lines for speed (processing 1 by 1 is slow)
            batch_lines = []
            batch_size = 1000 
            
            for line in f:
                pbar.update(len(line.encode('utf-8'))) # Update progress bar by bytes
                batch_lines.append(line)
                
                if len(batch_lines) >= batch_size:
                    # Encode batch (fast Rust implementation)
                    # We only need the ids, not attention masks
                    encodings = tokenizer.encode_batch(batch_lines)
                    
                    for enc in encodings:
                        total_tokens += len(enc.ids)
                        
                    total_lines += len(batch_lines)
                    batch_lines = [] # Clear buffer
            
            # Process remaining lines
            if batch_lines:
                encodings = tokenizer.encode_batch(batch_lines)
                for enc in encodings:
                    total_tokens += len(enc.ids)
                total_lines += len(batch_lines)

    print("\n" + "="*30)
    print(f"Total Lines:  {total_lines:,}")
    print(f"Total Tokens: {total_tokens:,}")
    print("="*30)
    
    # Quick Check for your config
    print(f"Avg tokens/line: {total_tokens / total_lines:.2f}")
    return total_tokens

if __name__ == "__main__":
    count_tokens()