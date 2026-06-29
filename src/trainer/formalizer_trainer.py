# --- Formalizer Trainer ---
# Complete implementation of the formalizer training job, with the evaluation loop,
# checkpoints, best model, etc. This will rely havily on MLflow SDK: https://mlflow.org/docs/latest/ml/deep-learning/pytorch/

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# mlflow imports
import mlflow
from mlflow.models import infer_signature

# internal methods
from .base import BaseTrainer
from ..utils.logger import get_logger
from ..preprocessing.hf_tokenizer import Tokenizer
from ..preprocessing.fim import apply_line_level_fim

# torch imports
import torch
from torch.utils.data import Dataset, DataLoader

# setting up the logging
_logger = get_logger("formalizer_training", level="DEBUG")

class FormalizerDataset(Dataset):
    def __init__(self, corpus_path: Path, tokenizer: Tokenizer, context_size: int):
        super().__init__()
        self.context_size = context_size
        self.tokenizer = tokenizer
        with open(corpus_path, "r", encoding="utf-8") as f:
            raw_content = f.read()

        # We simply load the samples in the initialization of the Dataset. All the tokenization and 
        # FIM strategy application will be done in the __getitem__ method, on the fly. Specially for the FIM strategy
        # this is good because we ensure the model sees different configurations of PSM during training.
        self.samples = raw_content.split("<|endoftext|>") # use the special EOT token to split each sample from the corpus
        self.samples = [s.strip() for s in self.samples if s.strip()] # filter empty strings

        _logger.info(f"Loaded {len(self.samples)} documents.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # Get the raw text for this specific document
        text = self.samples[index]
        
        # Apply Dynamic FIM
        fim_text = apply_line_level_fim(text)
        encodings = self.tokenizer.encode(
            fim_text, 
            max_length=self.context_size + 1, 
            truncation=True, 
            padding="max_length"
        )
        full_tensor = torch.tensor(encodings.ids, dtype=torch.long)
        input_ids = full_tensor[:-1]  # 0 to N-1
        target_ids = full_tensor[1:]  # 1 to N
        
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

    def train(
        self,
        start_step: int = 0,
        resume_run_id: Optional[str] = None,
        optimizer_state_dict: Optional[dict] = None,
    ):
        self._setup_dataloaders()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)

        train_iter = iter(self._train_dataloader)

        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("Formalizer_Training")

        checkpoint_interval = getattr(self.config, "checkpoint_interval", 1000)
        checkpoint_dir = Path("models/formalizer")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        _logger.info(f"Starting training on {self.config.device} from step {start_step}...")
        mlflow.pytorch.autolog(log_models=False, silent=True)

        start_time = time.time()

        with mlflow.start_run(run_id=resume_run_id) as run:
            if resume_run_id is None:
                mlflow.log_params(self.config.model_dump())
            else:
                mlflow.set_tag("resumed_from_step", start_step)

            for i in range(start_step, self.config.max_iters):
                try:
                    xb, yb = next(train_iter)
                except StopIteration:
                    train_iter = iter(self._train_dataloader)
                    xb, yb = next(train_iter)

                xb, yb = xb.to(self.config.device), yb.to(self.config.device)

                # Evaluation & Logging Loop
                if i % self.config.eval_interval == 0 and i > start_step:
                    losses = self._estimate_loss()

                    # ETA is relative to this session's start, not total training time
                    current_time = time.time()
                    elapsed_seconds = current_time - start_time
                    steps_done = i - start_step
                    avg_time_per_step = elapsed_seconds / max(steps_done, 1)
                    remaining_steps = self.config.max_iters - i
                    eta_seconds = remaining_steps * avg_time_per_step

                    eta_str = str(timedelta(seconds=int(eta_seconds)))
                    elapsed_str = str(timedelta(seconds=int(elapsed_seconds)))

                    _logger.info(
                        f"Step {i}/{self.config.max_iters} | "
                        f"Loss: {losses['train_loss']:.4f} | "
                        f"Val PPL: {losses['val_ppl']:.2f} | "
                        f"Elapsed: {elapsed_str} | ETA: {eta_str}"
                    )

                    mlflow.log_metrics({
                        "val_loss": losses['val_loss'],
                        "train_loss": losses['train_loss'],
                        "val_ppl": losses['val_ppl'],
                        "train_ppl": losses['train_ppl']
                    }, step=i)

                if i > start_step and i % checkpoint_interval == 0:
                    _logger.info(f"Saving checkpoint at step {i}")

                    # Local checkpoint: model + optimizer state + run_id for future resumption
                    checkpoint_path = checkpoint_dir / f"checkpoint_step_{i}.pt"
                    torch.save({
                        "step": i,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "run_id": run.info.run_id,
                    }, checkpoint_path)
                    _logger.info(f"Local checkpoint saved: {checkpoint_path}")

                    # MLflow artifact checkpoint
                    signature = infer_signature(
                        xb.cpu().numpy(),
                        self.model(xb, yb)[0].detach().cpu().numpy()
                    )
                    mlflow.pytorch.log_model(
                        pytorch_model=self.model,
                        artifact_path=f"checkpoint_step_{i}",
                        signature=signature,
                        input_example=xb[:1].cpu().numpy()
                    )

                logits, loss = self.model(xb, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

    def resume_from_checkpoint(self, checkpoint_dir: Optional[Path] = None) -> None:
        """Find the latest local checkpoint in checkpoint_dir and resume training from it."""
        if checkpoint_dir is None:
            checkpoint_dir = Path("models/formalizer")

        checkpoints = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

        latest = max(checkpoints, key=lambda p: int(p.stem.split("_")[-1]))
        step = int(latest.stem.split("_")[-1])
        _logger.info(f"Resuming from {latest} (step {step})")

        checkpoint = torch.load(latest, map_location=self.config.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        run_id = checkpoint.get("run_id")
        optimizer_state = checkpoint.get("optimizer_state_dict")

        if run_id:
            _logger.info(f"Continuing MLflow run: {run_id}")

        self.train(
            start_step=step,
            resume_run_id=run_id,
            optimizer_state_dict=optimizer_state,
        )

    def resume_from_mlflow_artifact(self, run_id: str, step: int) -> None:
        """Resume training from a checkpoint logged as an MLflow artifact.

        Use this when local .pt files are unavailable (e.g. the run crashed on a remote
        machine but the artifact was already uploaded). Optimizer state is not available
        from MLflow artifacts, so AdamW momentum terms will be cold-started at this step.

        Example:
            trainer.resume_from_mlflow_artifact(run_id="abc123", step=455000)
        """
        artifact_uri = f"runs:/{run_id}/checkpoint_step_{step}"
        _logger.info(f"Loading model from MLflow artifact: {artifact_uri}")

        loaded_model = mlflow.pytorch.load_model(artifact_uri, map_location=self.config.device)
        self.model.load_state_dict(loaded_model.state_dict())
        self.model.to(self.config.device)

        _logger.info(f"Model loaded. Resuming training from step {step}, MLflow run {run_id}")
        self.train(start_step=step, resume_run_id=run_id)

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
            
            mean_loss = losses.mean()
            
            # Save Loss
            out[f"{split}_loss"] = mean_loss.item()
            # Calculate Perplexity (PPL)
            # Perplexity is given by 2^H. Since Pytorch's implementation uses ln instead of log,
            # it is necessary to adapt to torch.exp()
            out[f"{split}_ppl"] = torch.exp(mean_loss).item()
            
        self.model.train()
        return out