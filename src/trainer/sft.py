# --- Supervised Fine-Tuning ---
# This script show an implementation of a Supervised Fine-Tuning pipeline. 
# For the Formalizer Model, the main plan is to "align", as per http://arxiv.org/abs/2203.02155, the outputs of the model.
#
# The Formalizer Model is responsible for correctly converting natural language, informal mathematical resolutions to formal, sympy
# representations.

# general imports
from pathlib import Path
import pyarrow.dataset as ds
import glob

# torch imports
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR

# hf imports
from tokenizers import Tokenizer

# internal imports
from ..utils.logger import get_logger
from .base import BaseTrainer

# set-up logging
_logger = get_logger("formalizer_posttraining", level="DEBUG")

class SFTFormalizerDataset(Dataset):
    """Implements the Dataset for supervised fine tuning"""
    def __init__(self, corpus_path: Path, tokenizer: Tokenizer = None, context_size: int = None):
        super().__init__()
        self.tokenizer = tokenizer
        self.context_size = context_size

        # loading dataset
        file_list = glob.glob(f"{corpus_path}/*.parquet", recursive=True)
        _logger.debug("Parquet file list: ")
        for f in file_list:
            _logger.debug(f)
        self.dataset = ds.dataset(file_list, format="parquet").to_table()
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):


        out = {
            "query": self.dataset["answer"][index].as_py(),             # dataset question
            "sympy": self.dataset["output"][index].as_py(),             # sympy code translation
            "code_output": self.dataset["code_output"][index].as_py(),  # code execution output
        }

        return out

# --- SFTTrainer ---
class SFTTrainer(BaseTrainer):
    """Implements the Supervised Fine-Tuning technique"""

if __name__ == "__main__":
    CORPUS_PATH = "data/posttraining/metamath_sympy"
    dataset = SFTFormalizerDataset(CORPUS_PATH)

    print(len(dataset))
    print(dataset[0]["sympy"])
