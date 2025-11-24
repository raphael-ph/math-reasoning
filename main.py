from preprocessing.pre_training.loader import RepoLoader

def main():
    # --- Example Usage ---
    # NOTE: The 'Mathlib' subdirectory is where most of the code lives in the mathlib4 repo.
    # If you wanted to include other files (like tests), you could set repo_start_path to '.'
    loader = RepoLoader(
        repo_url="https://github.com/leanprover-community/mathlib4.git",
        clone_dir="nanolean_mathlib_source",
        output_file="data/mathlib_corpus.txt",
        delete_after=True, # Change to False if you want to inspect the cloned files
        repo_start_path="Mathlib" 
    )

    loader.run()


if __name__ == "__main__":
    main()
