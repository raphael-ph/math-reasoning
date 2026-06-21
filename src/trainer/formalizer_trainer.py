# --- Formalizer Trainer ---
# Complete implementation of the formalizer training job, with the evaluation loop,
# checkpoints, best model, etc. This will rely havily on MLflow SDK: https://mlflow.org/docs/latest/ml/deep-learning/pytorch/

import time
import json
import mmap
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# mlflow imports
import mlflow
from mlflow.models import infer_signature

# internal methods
from src.trainer.base import BaseTrainer
from src.utils.logger import get_logger
from src.preprocessing.hf_tokenizer import Tokenizer
from src.preprocessing.fim import apply_line_level_fim

# torch imports
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR

# setting up the logging
_logger = get_logger("formalizer_training", level="DEBUG")

class FormalizerDataset(Dataset):
    def __init__(self, corpus_path: Path, chunk_indices_path: Path, tokenizer: Tokenizer, context_size: int):
        super().__init__()
        self.context_size = context_size
        self.tokenizer = tokenizer
        self.corpus = np.memmap(corpus_path, dtype=np.uint16, mode='r')

        # We simply load the samples in the initialization of the Dataset.
        # FIM strategy application will be done in the __getitem__ method, on the fly. Specially for the FIM strategy
        # this is good because we ensure the model sees different configurations of PSM during training.
        self.indices = np.load(chunk_indices_path)

        _logger.info(f"Loaded {len(self.indices)/1e6:.02f}M chunks.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        # First we have to get the chunk. We'll select a chunk and get it from the corpus file
        ix = self.indices[index]
        start = ix * self.context_size
        tokens = self.corpus[start:start+self.context_size+1]
        
        text = self.tokenizer.decode(tokens.tolist())
        
        # Apply Dynamic FIM
        fim_text = apply_line_level_fim(text)
        encodings = self.tokenizer.encode(
            fim_text,
        )
        ids = encodings.ids[:self.context_size + 1]

        # pad if shorter than context_size + 1
        pad_id = self.tokenizer.token_to_id("<|pad|>")
        if len(ids) < self.context_size + 1:
            ids = ids + [pad_id] * (self.context_size + 1 - len(ids))

        full_tensor = torch.tensor(ids, dtype=torch.long)
        input_ids = full_tensor[:-1]  # 0 to N-1
        target_ids = full_tensor[1:]  # 1 to N
        
        return input_ids, target_ids
    
class FormalizerTrainer(BaseTrainer):
    _train_dataloader: Optional[DataLoader] = None
    _val_dataloader: Optional[DataLoader] = None

    def model_post_init(self, __context):
        self.model.to(self.config.device)
        # self.model = torch.compile(self.model)

    def _setup_dataloaders(self):
        self._train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=False, # Already shuffled
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

    def lr_lambda(self, current_step: int):
        # linear warmup
        if current_step < self.config.warmup_steps:
            return float(current_step) / float(max(1, self.config.warmup_steps))
        
        # 2. Cosine Decay Phase
        # Calculate how far along we are between the end of warmup and max_steps
        progress = float(current_step - self.config.warmup_steps) / float(max(1, self.config.max_iters - self.config.warmup_steps))
        
        # Clip progress at 1.0 just in case the loop goes slightly over max_iters
        progress = min(1.0, progress)
        
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        
        # Establish a floor so the learning rate doesn't completely drop to 0
        # 0.1 means it decays down to 10% of 3e-4 (which is 3e-5)
        return max(0.1, cosine_decay)

    def train(self):
            self._setup_dataloaders()
            optimizer = torch.optim.AdamW(self.model.parameters(), 
                                          lr=self.config.learning_rate, 
                                          betas=(0.9, 0.95), # deepseek values
                                          weight_decay=0.1,  # deepseek values 
                                          )
            lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=self.lr_lambda)
            train_iter = iter(self._train_dataloader)
            
            mlflow.set_tracking_uri("sqlite:///mlruns.db")
            mlflow.set_experiment("Formalizer_Training")
            
            checkpoint_interval = getattr(self.config, "checkpoint_interval", 1000)

            final_model_path = Path(self.config.final_model_path)
            final_model_path.parent.mkdir(parents=True, exist_ok=True)
            best_model_path = final_model_path.parent / "best_model.pt"

            _logger.info(f"Starting training on {self.config.device}...")
            mlflow.pytorch.autolog(log_models=False, silent=True)

            best_val_loss = float("inf")

            # --- TIMER START ---
            # We start the clock just before the loop begins
            start_time = time.time() 

            with mlflow.start_run() as run:
                mlflow.log_params(self.config.model_dump())
                
                for i in range(self.config.max_iters):
                    try:
                        xb, yb = next(train_iter)
                    except StopIteration:
                        train_iter = iter(self._train_dataloader)
                        xb, yb = next(train_iter)

                    xb, yb = xb.to(self.config.device), yb.to(self.config.device)
                    
                    # Evaluation & Logging Loop 
                    if (i == 0) or (i % self.config.eval_interval == 0 and i > 0):
                        losses = self._estimate_loss()
                        
                        # --- ETA CALCULATION ---
                        current_time = time.time()
                        elapsed_seconds = current_time - start_time
                        if i > 0:
                            # Avoid division by zero
                            avg_time_per_step = elapsed_seconds / i 
                            remaining_steps = self.config.max_iters - i
                            eta_seconds = remaining_steps * avg_time_per_step
                            
                            # Formatting nicely as HH:MM:SS
                            eta_str = str(timedelta(seconds=int(eta_seconds)))
                        else:
                            eta_str = "Calculating..."

                        elapsed_str = str(timedelta(seconds=int(elapsed_seconds)))
                        current_lr = optimizer.param_groups[0]['lr']

                        # Log to Console with ETA
                        _logger.info(
                            f"Step {i}/{self.config.max_iters} | "
                            f"Loss: {losses['train_loss']:.4f} | "
                            f"Val PPL: {losses['val_ppl']:.2f} | "
                            f"Elapsed: {elapsed_str} | ETA: {eta_str}"
                        )
                        
                        mlflow.log_metrics({
                            "learning_rate": current_lr,
                            "val_loss": losses['val_loss'], 
                            "train_loss": losses['train_loss'],
                            "val_ppl": losses['val_ppl'],
                            "train_ppl": losses['train_ppl']
                        }, step=i)

                        if losses["val_loss"] < best_val_loss:
                            best_val_loss = losses["val_loss"]
                            torch.save(self.model.state_dict(), best_model_path)  # fast local save
                            _logger.info(f"New best model (val_loss: {best_val_loss:.4f})")
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits, loss = self.model(xb, yb)

                    if i > 0 and i % checkpoint_interval == 0:
                        _logger.info(f"Saving checkpoint at step {i}")
                        checkpoint_model_path = final_model_path.parent / f"checkpoint_{i}.pt"
                        torch.save(self.model.state_dict(), checkpoint_model_path)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    optimizer.step()
                    lr_scheduler.step()
                
                # saving model 
                _logger.info(f"Training complete, saving final model")
                torch.save(self.model.state_dict(), final_model_path)
                mlflow.log_metric("best_val_loss", best_val_loss)

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
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, loss = self.model(X, Y)
                losses[k] = loss.item()
            
            mean_loss = losses.mean()
            
            # Save Loss
            out[f"{split}_loss"] = mean_loss.item()
            # Calculate Perplexity (PPL)
            # Perplexity is given by 2^H. Since Pytorch's implementation uses ln instead of log,
            # it is necessary to adapt to torch.exp()
            out[f"{split}_ppl"] = torch.exp(mean_loss).item()
            
        self.model.train()
        return out