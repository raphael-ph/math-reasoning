# internal imports
from src.utils.logger import get_logger
from src.preprocessing.loader import RepoLoader, HuggingFaceLoader, CorpusBlender

logger = get_logger("main", level="INFO")

def main():
    # Configuration
    LEAN_OUTPUT = "data/raw/corpus_lean_raw.txt"
    NL_OUTPUT = "data/raw/corpus_english_raw.txt"
    FINAL_OUTPUT = "data/corpus/final_training_corpus.txt"
    
    # Extract Lean (Code)
    # Using the RepoLoader you wrote
    lean_loader = RepoLoader(
        repo_url="https://github.com/leanprover-community/mathlib4.git",
        clone_dir="temp_mathlib_clone",
        output_file=LEAN_OUTPUT,
        delete_after=True
    )
    lean_loader.run()
    
    # Extract Proof-Pile (English/Latex)
    # Using 'hoskinson-center/proof-pile' (a standard large math corpus)
    nl_loader = HuggingFaceLoader(
        dataset_name="hoskinson-center/proof-pile", 
        split="train", 
        output_file=NL_OUTPUT,
        max_samples=30_000 # Adjust this to match size of Lean corpus approx
    )
    nl_loader.run()
    
    # Mix them (Bilingual Data)
    blender = CorpusBlender(
        file_a=LEAN_OUTPUT,
        file_b=NL_OUTPUT,
        output_file=FINAL_OUTPUT
    )
    blender.run()
    
    logger.info("\nREADY FOR TOKENIZER TRAINING!")

if __name__ == "__main__":
    main()