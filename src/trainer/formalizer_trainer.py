# --- Formalizer Trainer ---
# Complete implementation of the formalizer training job, with the evaluation loop,
# checkpoints, best model, etc. This will rely havily on MLflow SDK: https://mlflow.org/docs/latest/ml/deep-learning/pytorch/

from pathlib import Path
from typing import Optional

# mlflow imports
import mlflow
from mlflow.models import infer_signature

# internal methods
from .base import BaseTrainer, BaseTrainerConfig
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
        
        if end_idx + 1 > len(self.data):
             start_idx = len(self.data) - self.context_size - 1
             end_idx = start_idx + self.context_size

        input_ids = self.data[start_idx : end_idx]
        target_ids = self.data[start_idx + 1 : end_idx + 1]
        
        return input_ids, target_ids

class FormalizerTrainer(BaseTrainer):
    _train_dataloader: Optional[DataLoader] = None
    _val_dataloader: Optional[DataLoader] = None

    def model_post_init(self, __context):
        self.model.to(self.config.device)

    def _setup_dataloaders(self):
        self._train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
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
        train_iter = iter(self._train_dataloader)
        
        # Setup MLflow
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("Formalizer_Training")
        
        # we define how often to save the heavy model files.
        # This prevents saving a model every time you eval.
        checkpoint_interval = getattr(self.config, "checkpoint_interval", 1000)

        _logger.info(f"Starting training on {self.config.device}...")

        # Disable autologging for models so we can manually control checkpoint naming/frequency
        mlflow.pytorch.autolog(log_models=False, silent=True)

        with mlflow.start_run() as run:
            # Log config parameters
            mlflow.log_params(self.config.model_dump())
            
            for i in range(self.config.max_iters):
                # Fetch Batch
                try:
                    xb, yb = next(train_iter)
                except StopIteration:
                    train_iter = iter(self._train_dataloader)
                    xb, yb = next(train_iter)

                xb, yb = xb.to(self.config.device), yb.to(self.config.device)
                
                # Evaluation & Logging Loop
                if i % self.config.eval_interval == 0 and i > 0:
                    losses = self._estimate_loss()
                    train_loss = losses['train']
                    val_loss = losses['val']
                    
                    _logger.info(f"Step {i}: train loss: {train_loss:.4f}, val loss: {val_loss:.4f}")
                    
                    # Log metrics FIRST so they are associated with this step
                    mlflow.log_metrics({
                        "val_loss": val_loss, 
                        "train_loss": train_loss
                    }, step=i)

                    # Checkpointing
                    # We save the model if we hit the checkpoint interval.
                    # This allows `mlflow.search_logged_models` to find it later.
                    if i % checkpoint_interval == 0:
                        _logger.info(f"Logging checkpoint at step {i}")
                        
                        # create an input signature for easier inference later
                        signature = infer_signature(
                            xb.cpu().numpy(), 
                            self.model(xb, yb)[0].detach().cpu().numpy()
                        )
                        
                        mlflow.pytorch.log_model(
                            pytorch_model=self.model,
                            artifact_path=f"checkpoint_step_{i}",
                            signature=signature,
                            # Input example helps with model serving later
                            input_example=xb[:1].cpu().numpy() 
                        )

                # Optimization
                logits, loss = self.model(xb, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

    @torch.no_grad()
    def _estimate_loss(self):
        out = {}
        self.model.eval()
        loaders = {'train': self._train_dataloader, 'val': self._val_dataloader}
        
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
            out[split] = losses.mean().item()
            
        self.model.train()
        return out