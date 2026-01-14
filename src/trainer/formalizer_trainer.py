# --- Formalizer Trainer ---
# Complete implementation of the formalizer training job, with the evaluation loop,
# checkpoints, best model, etc. This will rely havily on MLflow SDK: https://mlflow.org/docs/latest/ml/deep-learning/pytorch/

from pathlib import Path
from typing import Optional

# internal methods
from .base import BaseTrainer
from ..utils.logger import get_logger

# torch imports
import torch
from torch.utils.data import Dataset, DataLoader

# setting up the logging
_logger = get_logger("formalizer_training", level="DEBUG")

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
        
        # Validation to ensure we don't go out of bounds
        if end_idx + 1 > len(self.data):
             # Handle edge case or resize last batch
             start_idx = len(self.data) - self.context_size - 1
             end_idx = start_idx + self.context_size

        input_ids = self.data[start_idx : end_idx]
        target_ids = self.data[start_idx + 1 : end_idx + 1]
        
        return input_ids, target_ids

# --- Formalizer Trainer ---
class FormalizerTrainer(BaseTrainer):
    # We keep dataloaders as private attributes to avoid Pydantic validation issues
    _train_dataloader: Optional[DataLoader] = None
    _val_dataloader: Optional[DataLoader] = None

    def model_post_init(self, __context):
        self.model.to(self.config.device)

    def _setup_dataloaders(self):
        """Initialize dataloaders once to avoid overhead."""
        self._train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True # Faster transfer to CUDA
        )
        self._val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    def train(self):
        self._setup_dataloaders()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        
        # Create an infinite iterator for the training loop
        train_iter = iter(self._train_dataloader)
        
        _logger.info(f"Starting training on {self.config.device}...")
        
        for i in range(self.config.max_iters):
            # batch fetching
            try:
                xb, yb = next(train_iter)
            except StopIteration:
                # Restart iterator if dataset is exhausted
                train_iter = iter(self._train_dataloader)
                xb, yb = next(train_iter)

            # Move data to device
            xb = xb.to(self.config.device)
            yb = yb.to(self.config.device)
            
            # Evaluation
            if i % self.config.eval_interval == 0 and i > 0:
                losses = self._estimate_loss()
                _logger.info(f"Step {i}: train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")
            
            # Forward pass & Optimization
            logits, loss = self.model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    @torch.no_grad()
    def _estimate_loss(self):
        out = {}
        self.model.eval()
        
        # Re-use the loaders we created in _setup_dataloaders
        loaders = {
            'train': self._train_dataloader,
            'val': self._val_dataloader
        }
        
        for split, loader in loaders.items():
            losses = torch.zeros(self.config.eval_iters)
            loader_iter = iter(loader)
            
            for k in range(self.config.eval_iters):
                try:
                    X, Y = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(loader)
                    X, Y = next(loader_iter)
                
                X, Y = X.to(self.config.device), Y.to(self.config.device)
                
                _, loss = self.model(X, Y)
                losses[k] = loss.item()
                
            out[split] = losses.mean()
            
        self.model.train()
        return out