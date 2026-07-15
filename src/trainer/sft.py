# --- Supervised Fine-Tuning ---
# This script show an implementation of a Supervised Fine-Tuning pipeline. 
# For the Formalizer Model, the main plan is to "align", as per http://arxiv.org/abs/2203.02155, the outputs of the model.
#
# The Formalizer Model is responsible for correctly converting natural language, informal mathematical resolutions to formal, sympy
# representations.

# general imports
import glob
from pathlib import Path
import pyarrow.dataset as ds
from typing import Tuple

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
    def __init__(self, corpus_path: Path, tokenizer: Tokenizer, context_size: int):
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

    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.tensor, str]:

        item = {
            "query": self.dataset["answer"][index].as_py(),             # dataset question
            "sympy": self.dataset["output"][index].as_py(),             # sympy code translation
            "code_output": self.dataset["code_output"][index].as_py(),  # code execution output
        }

        query_tokens = self.tokenizer.encode(item["query"])
        sympy_tokens = self.tokenizer.encode(item["sympy"])
        
        pad_id = self.tokenizer.token_to_id("<|pad|>")
        query_ids = query_tokens.ids[:self.context_size - 1]
        sympy_ids = sympy_tokens.ids
        if len(query_ids) < self.context_size + 1:
            query_ids = query_ids + [pad_id] * (self.context_size + 1 - len(query_ids))

        input_ids = torch.tensor(query_ids, dtype=torch.long)
        label_ids = torch.tensor(sympy_ids, dtype=torch.long)

        return input_ids, label_ids, item["code_output"]

# --- SFTTrainer ---
class SFTTrainer(BaseTrainer):
    """Implements the Supervised Fine-Tuning technique"""

if __name__ == "__main__":
