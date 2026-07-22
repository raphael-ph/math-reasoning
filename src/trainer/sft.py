# --- Supervised Fine-Tuning ---
# This script show an implementation of a Supervised Fine-Tuning pipeline. 
# For the Formalizer Model, the main plan is to "align", as per http://arxiv.org/abs/2203.02155, the outputs of the model.
#
# The Formalizer Model is responsible for correctly converting natural language, informal mathematical resolutions to formal, sympy
# representations.

# general imports
import glob
import time
import numpy as np
import pyarrow as pa
from pathlib import Path
from datetime import timedelta
from typing import Tuple, Optional, Literal

# pyarrow imports
import pyarrow.dataset as ds
import pyarrow.compute as pc

# torch imports
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR

# hf imports
from tokenizers import Tokenizer

# mlflow imports
import mlflow
from mlflow.models import infer_signature

# internal imports
from ..utils.logger import get_logger
from .base import BaseTrainer

# set-up logging
_logger = get_logger("formalizer_posttraining", level="DEBUG")

SHUFFLING_SEED = 42

class SFTFormalizerDataset(Dataset):
    """Implements the Dataset for supervised fine tuning"""
    def __init__(self, corpus_path: Path, 
                 tokenizer: Tokenizer, 
                 context_size: int, 
                 split: Literal["train", "val"],
                 train_size: int = 15000, 
                 val_size: int = 3000):
        super().__init__()
        self.tokenizer = tokenizer
        self.context_size = context_size

        # loading dataset
        file_list = glob.glob(f"{corpus_path}/*.parquet", recursive=True)
        _logger.debug("Parquet file list: ")
        for f in file_list:
            _logger.debug(f)
        self.dataset = ds.dataset(file_list, format="parquet").to_table()

        # shuffle dataset
        indices = self.__shuffle_table(self.dataset)

        train_indices = indices[:train_size]
        val_indices = indices[train_size : train_size + val_size]

        # save both indices to disk
        output_path = Path("data/posttraining/metamath_sympy/sft")
        output_path.mkdir(parents=True, exist_ok=True)

        train_indices_path = Path(f"{output_path}/train_indices.npy")
        val_indices_path = Path(f"{output_path}/val_indices.npy")

        # Ensure paths are not overwritten
        if not train_indices_path.exists():
            np.save(train_indices_path, train_indices)

        if not val_indices_path.exists():
            np.save(val_indices_path, val_indices)

        # return dataset
        if split == "train":
            idx = np.load("data/posttraining/metamath_sympy/sft/train_indices.npy")
            self.dataset = pc.take(self.dataset, idx)
        elif split == "val":
            idx = np.load("data/posttraining/metamath_sympy/sft/val_indices.npy")
            self.dataset = pc.take(self.dataset, idx)
    
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

    # --- Helper ---
    def __shuffle_table(self, table: pa.Table) -> np.array:
        """Shuffles the parquet table and returns the shuffled indices as np.arr"""
        indices = np.array(range(table.num_rows))

        random_generator = np.random.default_rng(seed=SHUFFLING_SEED)
        random_generator.shuffle(indices)

        return indices

# --- SFTTrainer ---
class SFTTrainer(BaseTrainer):
    """Implements the Supervised Fine-Tuning technique"""
    _train_dataloader: Optional[DataLoader]
    _val_dataloader: Optional[DataLoader]

    def model_post_init(self, __context):
        self.model.to(self.config.device)

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
            shuffle=False, # Already shuffled
            num_workers=4,
            pin_memory=True
        )

    def lr_lambda(self, current_step: int):
        # linear warmup
        if current_step < self.config.warmup_steps:
            return float(current_step) / float(max(1, self.config.warmup_steps))

        # Cosine Decay Phase
        progress = float(current_step - self.config.warmup_steps) / float(max(1, self.config.max_iters - self.config.warmup_steps))
        progress = min(1.0, progress)
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))

        # Floor so LR doesn't drop to 0 — decays to 10% of base LR
        return max(0.1, cosine_decay)

    def train(
            self,
            start_step: int = 0,
            resume_run_id: Optional[str] = None,
            optimizer_state_dict: Optional[dict] = None,
        ):
            self._setup_dataloaders()
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
            )
            if optimizer_state_dict is not None:
                optimizer.load_state_dict(optimizer_state_dict)
    
            # When last_epoch >= 0 (resume), PyTorch requires initial_lr to already exist
            # in param groups — it's normally set during a fresh init (last_epoch=-1).
            # Must use config base LR, not group['lr']: if an optimizer state dict was
            # loaded, group['lr'] is the scaled LR at checkpoint time, which would cause
            # LambdaLR to double-apply the schedule.
            if start_step > 0:
                for group in optimizer.param_groups:
                    group['initial_lr'] = self.config.learning_rate
            lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=self.lr_lambda, last_epoch=start_step - 1)
    
            train_iter = iter(self._train_dataloader)
    
            mlflow.set_tracking_uri("sqlite:///mlruns.db")
            mlflow.set_experiment("Formalizer_Training")
    
            checkpoint_interval = getattr(self.config, "checkpoint_interval", 1000)
            checkpoint_dir = Path("models/formalizer")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
            final_model_path = Path(self.config.final_model_path)
            final_model_path.parent.mkdir(parents=True, exist_ok=True)
            best_model_path = checkpoint_dir / "best_model.pt"
    
            _logger.info(f"Starting training on {self.config.device} from step {start_step}...")
            mlflow.pytorch.autolog(log_models=False, silent=True)
    
            best_val_loss = float("inf")
            device_type = self.config.device.split(":")[0]
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
                    is_eval_step = (i == 0 and start_step == 0) or (i % self.config.eval_interval == 0 and i > start_step)
                    if is_eval_step:
                        losses = self._estimate_loss()
    
                        current_time = time.time()
                        elapsed_seconds = current_time - start_time
                        steps_done = i - start_step
                        avg_time_per_step = elapsed_seconds / max(steps_done, 1)
                        remaining_steps = self.config.max_iters - i
                        eta_seconds = remaining_steps * avg_time_per_step
    
                        eta_str = str(timedelta(seconds=int(eta_seconds)))
                        elapsed_str = str(timedelta(seconds=int(elapsed_seconds)))
                        current_lr = optimizer.param_groups[0]['lr']
    
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
                            torch.save(self.model.state_dict(), best_model_path)
                            _logger.info(f"New best model (val_loss: {best_val_loss:.4f})")
    
                    if i > start_step and i % checkpoint_interval == 0:
                        _logger.info(f"Saving checkpoint at step {i}")
    
                        checkpoint_path = checkpoint_dir / f"checkpoint_step_{i}.pt"
                        torch.save({
                            "step": i,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "run_id": run.info.run_id,
                        }, checkpoint_path)
                        _logger.info(f"Local checkpoint saved: {checkpoint_path}")
    
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
    
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        _, loss = self.model(xb, yb)
    
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    lr_scheduler.step()
    
                _logger.info("Training complete, saving final model")
                torch.save(self.model.state_dict(), final_model_path)
                mlflow.log_metric("best_val_loss", best_val_loss)
    
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

    def resume_from_state_dict(self, checkpoint_path: Path, run_id: str, step: int) -> None:
        """Resume training from a local .pt checkpoint file.

        Handles both plain state dicts (torch.save(model.state_dict(), path)) and
        our full checkpoint format (dict with model_state_dict / optimizer_state_dict keys).

        Example:
            trainer.resume_from_state_dict(Path("models/formalizer/checkpoint_455000.pt"), run_id="abc123", step=455000)
        """
        _logger.info(f"Loading state dict from {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=self.config.device, weights_only=False)

        if isinstance(state, dict) and "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])
            optimizer_state = state.get("optimizer_state_dict")
        else:
            self.model.load_state_dict(state)
            optimizer_state = None

        self.model.to(self.config.device)
        torch.cuda.empty_cache()
        _logger.info(f"Model loaded. Resuming training from step {step}, MLflow run {run_id}")
        self.train(start_step=step, resume_run_id=run_id, optimizer_state_dict=optimizer_state)

    def resume_from_mlflow_artifact(self, run_id: str, step: int) -> None:
        """Resume training from a checkpoint logged as an MLflow artifact.

        Use this when the checkpoint was uploaded to MLflow (not saved locally).
        Optimizer state is not available from MLflow artifacts, so AdamW momentum
        terms will be cold-started at this step.

        Example:
            trainer.resume_from_mlflow_artifact(run_id="abc123", step=455000)
        """
        mlflow.set_tracking_uri("sqlite:///mlruns.db")

        artifact_uri = f"runs:/{run_id}/checkpoint_{step}"
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

if __name__ == "__main__":
    import json
    # --- CONFIGURATION AND GLOBAL VARIABLES ---
    # Paths
    TOKENIZER_PATH = "data/vocab/tokenizer_vocab.json"
    VOCAB_METADATA_PATH = "data/corpus/metadata.json"
    CORPUS_PATH = "data/corpus/corpus.bin"
    TRAIN_INDICES = "data/corpus/train_indices.npy"
    VAL_INDICES = "data/corpus/val_indices.npy"
    FINAL_MODEL_PATH = "models/formalizer/final_model.pt"

    # ---------------- Global Vars --------------------
    with open(VOCAB_METADATA_PATH, "rb") as file:
        f = file.read()
        vocab_config = json.loads(f)

    VOCAB_SIZE = vocab_config["vocab_size"]
    CONTEXT_SIZE = vocab_config["context_size"]
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    logger = get_logger("trainer")
    # -------------------------------------------------
    CORPUS_PATH = "data/posttraining/metamath_sympy"
    dataset = SFTFormalizerDataset(CORPUS_PATH, tokenizer, CONTEXT_SIZE)

    print(len(dataset))
    input_ids, label_ids, code_out = dataset[0]

    print(input_ids.tolist())
    print(tokenizer.decode(label_ids.tolist()))
    print(code_out)