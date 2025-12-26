# --- Loading the entire Mathlib4 repository ---
# For the pretrianing phase, we'll use the Mathlib4 repo as 
# the training corpus: https://github.com/leanprover-community/mathlib4
#
# Strategy is to use all files that end on .lean and generate a 
# huge corpus, so the model knows how to speak lean.

import os
import shutil
import subprocess
from pathlib import Path
from typing import List

# internal imports
from ..utils.logger import get_logger

# set up logging
_logger = get_logger("RepoLoader", level="DEBUG")

class RepoLoader:
    def __init__(self, repo_url: str,
                 clone_dir: str,
                 output_file: str,
                 delete_after: bool = True, # Set default to True for cleanup
                 file_extension: str = ".lean",
                 repo_start_path: str = "Mathlib"):
        """The loader class for extracting a corpus from a repository.
        
        Reads all files of a given extension in a repository and output it to a single .txt file.

        Parameters:
            repo_url (str): the repository url (e.g., https://github.com/leanprover-community/mathlib4.git)
            clone_dir (str): the local directory name to clone the repo to
            output_file (str): the output file name for the complete corpus
            delete_after (bool): True if you wish to delete the cloned repo after the corpus is extracted
            file_extension (str): The file extension to search for (default: ".lean")
            repo_start_path (str): The subdirectory within the clone to start the search (e.g., "Mathlib")
        """
        self.repo_url = repo_url
        self.clone_dir = clone_dir
        self.output_file = output_file
        self.delete_after = delete_after
        self.file_extension = file_extension
        # Use Pathlib to construct the path where the main code files reside
        self.start_path = Path(clone_dir) / repo_start_path

    def _clone_repo(self) -> bool:
        """Clones the repository if the target directory doesn't exist."""
        if Path(self.clone_dir).exists():
            _logger.info(f"Directory '{self.clone_dir}' already exists. Skipping clone.")
            return True
        
        _logger.info(f"Cloning {self.repo_url} into {self.clone_dir} (shallow clone)...")
        try:
            # Use --depth 1 for a shallow clone to save time and space
            subprocess.run(["git", "clone", "--depth", "1", self.repo_url, self.clone_dir], check=True, capture_output=True, text=True)
            _logger.info("Cloning complete.")
            return True
        except subprocess.CalledProcessError as e:
            _logger.info(f"Error during git clone: {e.stderr}")
            return False

    def _get_lean_files(self) -> List[Path]:
        """Recursively finds all files with the specified extension."""
        if not self.start_path.exists():
            _logger.info(f"Error: The main code directory '{self.start_path}' was not found.")
            return []

        # Use Path.rglob() for recursive search with a specific pattern
        pattern = f"*{self.file_extension}"
        _logger.info(f"Searching for files with pattern '{pattern}' in '{self.start_path}'...")
        
        # rglob returns a generator, which is memory efficient
        return list(self.start_path.rglob(pattern))

    def _write_corpus(self, file_paths: List[Path]):
        """Concatenates the contents of the files into the output file."""
        _logger.info(f"Writing contents of {len(file_paths)} files to '{self.output_file}'...")
        total_chars = 0
        
        with open(self.output_file, 'w', encoding='utf-8') as outfile:
            for file_path in file_paths:
                try:
                    # Add a unique separator for better training context
                    outfile.write(f"\n\n#### START_FILE: {file_path.name} ####\n\n")
                    
                    content = file_path.read_text(encoding='utf-8')
                    outfile.write(content)
                    total_chars += len(content)
                except Exception as e:
                    _logger.info(f"Warning: Could not read {file_path} - {e}")
                    continue

        _logger.info(f"--- CORPUS EXTRACTION COMPLETE ---")
        _logger.info(f"Total character count written: {total_chars:,}")
        _logger.info(f"Output file: {os.path.abspath(self.output_file)}")

    def run(self):
        """Executes the cloning, extraction, and optional cleanup process."""
        if not self._clone_repo():
            return
        
        file_paths = self._get_lean_files()
        if file_paths:
            self._write_corpus(file_paths)
        
        if self.delete_after:
            _logger.info(f"Cleaning up: Deleting directory '{self.clone_dir}'...")
            try:
                shutil.rmtree(self.clone_dir)
                _logger.info("Cleanup complete.")
            except OSError as e:
                _logger.info(f"Error during cleanup (could not delete {self.clone_dir}): {e}")

