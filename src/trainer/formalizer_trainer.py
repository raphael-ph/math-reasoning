# --- Formalizer Trainer ---
# Complete implementation of the formalizer training job, with the evaluation loop,
# checkpoints, best model, etc. This will rely havily on MLflow SDK: https://mlflow.org/docs/latest/ml/deep-learning/pytorch/

from pathlib import Path

# internal methods
# from .base import BaseTrainer, BaseTrainerConfig
from ..preprocessing.hf_tokenizer import Tokenizer
from ..utils.logger import get_logger

# torch imports
import torch
from torch.utils.data import Dataset

# setting up the logging
_logger = get_logger("formalizer_training", level="DEBUG")

# --- Formalizer Dataset Class ---
# Generate the training Dataset for the formalizer
class FormalizerDataset(Dataset):
    def __init__(self, corpus_path: Path, tokenizer, context_size: int):
        super().__init__()
        self.context_size = context_size
        with open(corpus_path, "r", encoding="utf-8") as f:
            text = f.read()
    
        _logger.info("Tokenizing corpus...")
        self.data = tokenizer.encode(text, return_tensors="pt").squeeze()
        _logger.info(f"Tokenization complete. Total tokens: {len(self.data)}")

    def __len__(self):
        return (len(self.data) - 1) // self.context_size

    def __getitem__(self, index):
        start_idx = index * self.context_size
        end_idx = start_idx + self.context_size
        input_ids = self.data[start_idx : end_idx]
        target_ids = self.data[start_idx + 1 : end_idx + 1]
        
        return input_ids, target_ids

# --- Formalizer Trainer ---
# class FormalizerTrainer(BaseTrainer):
#     def train(self):
#         pass