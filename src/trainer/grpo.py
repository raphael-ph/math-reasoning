# --- GRPO (Group Relative Policy Optimization) ---
# Implements GRPO training: for each prompt, sample a group of completions from the
# current policy, score them with a pluggable reward function (src/rewards), turn the
# group's rewards into advantages (no critic/value model needed), and update the policy
# with a clipped surrogate objective plus a KL penalty against a frozen reference model.
# See https://arxiv.org/abs/2402.03300 (DeepSeekMath).

import glob
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

# pyarrow imports
import pyarrow.dataset as ds
import pyarrow.compute as pc

# numpy
import numpy as np
import pandas as pd

# torch imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR

# hf imports
from tokenizers import Tokenizer

# mlflow imports
import mlflow

# pydantic
from pydantic import Field

# internal imports
from . import grpo_math
from .base import BaseTrainer, BaseTrainerConfig
from ..rewards.base import RewardFn
from ..utils.logger import get_logger

# set-up logging
_logger = get_logger("grpo", level="INFO")

EOS_TOKEN = "<|endoftext|>"
PAD_TOKEN = "<|pad|>"

class GRPOConfig(BaseTrainerConfig):
    """Extends BaseTrainerConfig with GRPO-specific hyperparameters."""
    group_size: int = Field(..., description="Number of completions (G) sampled per prompt")
    max_new_tokens: int = Field(..., description="Max tokens to generate per completion during rollout")
    top_p: Optional[float] = Field(default=None, description="Nucleus sampling threshold for rollout generation; None disables top-p filtering")
    temperature: float = Field(default=1.0, description="Sampling temperature for rollout generation")
    kl_coef: float = Field(default=0.04, description="Weight (beta) of the KL penalty against the reference model")
    clip_epsilon: float = Field(default=0.2, description="PPO-style clip range for the policy ratio")
    num_inner_epochs: int = Field(default=1, description="Number of gradient updates reusing the same rollout batch (mu)")
    advantage_eps: float = Field(default=1e-4, description="Added to the group reward std to avoid divide-by-zero when a group's rewards are identical")

class GRPOPromptDataset(Dataset):
    """Prompt-only dataset for GRPO rollouts.

    Unlike SFTFormalizerDataset, __getitem__ returns raw text rather than fixed-shape
    padded tensors: GRPO tokenizes per-prompt at rollout time (replicas within a group
    must share an identical prompt length, which dataset-level padding to a batch-wide
    max wouldn't preserve), and the completion is generated, not read from the dataset.

    Train/val row selection is NOT computed here — it's read from
    data/posttraining/metamath_sympy/grpo/{train,val}_indices.npy, produced once by
    `python -m src.preprocessing.split_dataset` (see that module's docstring). That
    script coordinates GRPO's split with SFT's and a reserved benchmark holdout from a
    single shuffle, so none of the three can ever overlap — this class used to shuffle
    and persist its own split independently, which could not provide that guarantee.
    """
    def __init__(
        self,
        corpus_path: Path,
        split: Literal["train", "val"],
        prompt_column: str = "answer",
    ):
        super().__init__()
        self.prompt_column = prompt_column

        file_list = glob.glob(f"{corpus_path}/*.parquet", recursive=True)
        _logger.debug("Parquet file list: ")
        for f in file_list:
            _logger.debug(f)
        self.dataset = ds.dataset(file_list, format="parquet").to_table()

        indices_path = Path(f"data/posttraining/metamath_sympy/grpo/{split}_indices.npy")
        if not indices_path.exists():
            raise FileNotFoundError(
                f"{indices_path} not found — run `python -m src.preprocessing.split_dataset` "
                "first to generate the coordinated SFT/GRPO/benchmark split."
            )
        idx = np.load(indices_path)
        self.dataset = pc.take(self.dataset, idx)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[str, Dict[str, Any]]:
        row = {col: self.dataset[col][index].as_py() for col in self.dataset.column_names}
        prompt = row.pop(self.prompt_column)

        # matches the SFT prompt template up to (and including) the <|assistant|> anchor;
        # generation continues from there, so whatever comes after is the completion
        prompt_text = f"<|bos|> <|user|> {prompt} <|assistant|>"

        # everything but the prompt itself — whatever a given reward function needs
        # (e.g. an expected numeric answer) is looked up from here by column name
        metadata = row

        return prompt_text, metadata

def prompt_collate_fn(batch: List[Tuple[str, Dict[str, Any]]]) -> List[Tuple[str, Dict[str, Any]]]:
    """Identity collate — rollout is per-prompt, so batches stay a plain list of
    (prompt_text, metadata) pairs rather than being stacked into a tensor."""
    return batch

class GRPORollout:
    """One prompt's rollout: group_size completions sharing an identical (unpadded) length."""
    def __init__(
        self,
        sequences: torch.Tensor,
        completion_mask: torch.Tensor,
        rewards: torch.Tensor,
        diagnostics: List[Optional[Dict[str, Any]]],
    ):
        self.sequences = sequences            # (G, T) — prompt + generated tokens, no padding
        self.completion_mask = completion_mask  # (G, T) bool — True at completion positions (up to & incl. first EOS)
        self.rewards = rewards                # (G,)
        # one entry per completion, or None if reward_fn doesn't expose `last_diagnostics`
        # (see SympyRewardFn) — full traceability (completion text, executes/correct,
        # error, stdout, predicted vs. expected) for MLflow logging, not used in the loss.
        self.diagnostics = diagnostics

class GRPOTrainer(BaseTrainer):
    """Implements Group Relative Policy Optimization.

    Unlike SFTTrainer/FormalizerTrainer, there's no fixed-shape (xb, yb) batch: each step
    rolls out group_size completions per sampled prompt, scores them with reward_fn, turns
    the group's rewards into advantages, and only then builds a padded tensor batch for the
    clipped-surrogate + KL-penalty update.
    """
    model_config = BaseTrainer.model_config
    config: GRPOConfig = Field(..., description="GRPO training run configuration hyperparameters")
    ref_model: nn.Module = Field(..., description="Frozen reference model used for the KL penalty")
    reward_fn: RewardFn = Field(..., description="(prompt_text, completion_text, metadata) -> reward")
    tokenizer: Tokenizer = Field(..., description="Tokenizer shared by rollout and the reference/policy models")

    _train_dataloader: Optional[DataLoader] = None
    _val_dataloader: Optional[DataLoader] = None
    _eos_token_id: Optional[int] = None
    _pad_token_id: Optional[int] = None

    def model_post_init(self, __context):
        self.model.to(self.config.device)
        self.ref_model.to(self.config.device)
        self.ref_model.eval()
        self.ref_model.requires_grad_(False)

        self._eos_token_id = self.tokenizer.token_to_id(EOS_TOKEN)
        self._pad_token_id = self.tokenizer.token_to_id(PAD_TOKEN)

    def _setup_dataloaders(self):
        self._train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,  # dataset is pre-shuffled at index-selection time
            collate_fn=prompt_collate_fn,
        )
        self._val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=prompt_collate_fn,
        )

    def lr_lambda(self, current_step: int):
        # linear warmup
        if current_step < self.config.warmup_steps:
            return float(current_step) / float(max(1, self.config.warmup_steps))

        # cosine decay, floored so LR doesn't drop to 0 — decays to 10% of base LR
        progress = float(current_step - self.config.warmup_steps) / float(max(1, self.config.max_iters - self.config.warmup_steps))
        progress = min(1.0, progress)
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        return max(0.1, cosine_decay)

    @torch.no_grad()
    def _rollout_one_prompt(self, prompt_text: str, metadata: Dict[str, Any]) -> GRPORollout:
        """Samples group_size completions for a single prompt and scores them."""
        device_type = self.config.device.split(":")[0]

        ids = self.tokenizer.encode(prompt_text).ids
        # left-truncate so the prompt leaves room for max_new_tokens — mirrors the
        # direction="left" truncation SFT applies via enable_truncation()
        max_prompt_len = self.config.context_size - self.config.max_new_tokens
        if len(ids) > max_prompt_len:
            ids = ids[-max_prompt_len:]
        prompt_len = len(ids)

        prompt_ids = torch.tensor(ids, dtype=torch.long, device=self.config.device)
        prompt_ids = prompt_ids.unsqueeze(0).repeat(self.config.group_size, 1)  # (G, T_prompt)

        self.model.eval()
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            sequences, _ = self.model.generate(
                prompt_ids,
                self.config.max_new_tokens,
                top_p=self.config.top_p,
                eos_token_id=self._eos_token_id,
                temperature=self.config.temperature,
            )

        # per-row: find the first EOS in the generated span (not the prompt) so a
        # sequence that stopped early doesn't get credit/loss for tokens after it,
        # and so the reward function only ever sees the actual completion
        gen_part = sequences[:, prompt_len:]  # (G, T_gen)
        is_eos = gen_part == self._eos_token_id
        has_eos = is_eos.any(dim=1)
        first_eos_idx = torch.where(
            has_eos,
            is_eos.float().argmax(dim=1),
            torch.full((gen_part.shape[0],), gen_part.shape[1] - 1, device=gen_part.device, dtype=torch.long),
        )
        gen_positions = torch.arange(gen_part.shape[1], device=gen_part.device).unsqueeze(0)
        gen_mask = gen_positions <= first_eos_idx.unsqueeze(1)  # inclusive of the EOS token itself

        completion_mask = torch.cat(
            [torch.zeros(self.config.group_size, prompt_len, dtype=torch.bool, device=self.config.device), gen_mask],
            dim=1,
        )

        rewards = torch.zeros(self.config.group_size, device=self.config.device)
        diagnostics: List[Optional[Dict[str, Any]]] = []
        for g in range(self.config.group_size):
            # decoded text excludes the EOS token itself — reward_fn should see the
            # actual completion content, not the special token that ends it
            completion_ids = gen_part[g, : first_eos_idx[g].item()].tolist()
            completion_text = self.tokenizer.decode(completion_ids)
            rewards[g] = float(self.reward_fn(prompt_text, completion_text, metadata))
            # duck-typed: only reward functions that opt into traceability (e.g.
            # SympyRewardFn) expose this; plain RewardFn callables leave it absent
            diagnostics.append(getattr(self.reward_fn, "last_diagnostics", None))

        return GRPORollout(sequences=sequences, completion_mask=completion_mask, rewards=rewards, diagnostics=diagnostics)

    @staticmethod
    def _diagnostic_records(
        step: int,
        split: Literal["train", "val"],
        prompt_batch: List[Tuple[str, Dict[str, Any]]],
        rollouts: List[GRPORollout],
    ) -> List[Dict[str, Any]]:
        """Flattens one step's rollouts into per-completion rows for MLflow's
        rollout-traceability table: step, split, prompt, plus everything reward_fn's
        last_diagnostics carried (completion text, executes/correct, error, stdout,
        predicted vs. expected). Skips completions whose reward_fn didn't expose
        diagnostics at all."""
        records = []
        for (prompt_text, _metadata), rollout in zip(prompt_batch, rollouts):
            for diag in rollout.diagnostics:
                if diag is None:
                    continue
                records.append({"step": step, "split": split, "prompt": prompt_text, **diag})
        return records

    @staticmethod
    def _diagnostic_rates(records: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
        """Aggregate execute/correct rates from a batch of diagnostic records — cheap
        scalar summary of the full traceability table, logged every eval step regardless
        of whether the full table is inspected."""
        if not records:
            return None
        n = len(records)
        return {
            "execute_rate": sum(1.0 for r in records if r["executes"]) / n,
            "correct_rate": sum(1.0 for r in records if r["correct"]) / n,
        }

    def _pad_and_stack(self, rollouts: List[GRPORollout]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Right-pads each rollout's (G, T) sequences to the batch-wide max length and
        stacks them into one (num_prompts * G, T_max) batch. Safe to right-pad here
        (unlike during generation) since this is a finished, teacher-forced sequence —
        causal attention never lets a real token attend into trailing padding."""
        max_len = max(r.sequences.shape[1] for r in rollouts)

        sequences_batch = []
        mask_batch = []
        for r in rollouts:
            pad_amount = max_len - r.sequences.shape[1]
            if pad_amount > 0:
                seq_pad = torch.full((r.sequences.shape[0], pad_amount), self._pad_token_id, dtype=torch.long, device=r.sequences.device)
                mask_pad = torch.zeros((r.completion_mask.shape[0], pad_amount), dtype=torch.bool, device=r.completion_mask.device)
                sequences_batch.append(torch.cat([r.sequences, seq_pad], dim=1))
                mask_batch.append(torch.cat([r.completion_mask, mask_pad], dim=1))
            else:
                sequences_batch.append(r.sequences)
                mask_batch.append(r.completion_mask)

        return torch.cat(sequences_batch, dim=0), torch.cat(mask_batch, dim=0)

    def _sequence_logprobs(self, model: nn.Module, sequences: torch.Tensor) -> torch.Tensor:
        """Per-token log-probs of the realized next tokens, via one teacher-forced pass."""
        device_type = self.config.device.split(":")[0]
        input_ids = sequences[:, :-1]
        target_ids = sequences[:, 1:]

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, _ = model(input_ids, targets=None)
        # upcast before log_softmax — the ratio exp(new - old) downstream is precision
        # sensitive, more than a bf16 forward pass alone can guarantee
        logits = logits.float()

        return grpo_math.sequence_logprobs(logits, target_ids)

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

        if start_step > 0:
            for group in optimizer.param_groups:
                group['initial_lr'] = self.config.learning_rate
        lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=self.lr_lambda, last_epoch=start_step - 1)

        train_iter = iter(self._train_dataloader)

        mlflow.set_tracking_uri("sqlite:///mlruns.db")
        mlflow.set_experiment("Formalizer_GRPO")

        checkpoint_interval = getattr(self.config, "checkpoint_interval", 1000)
        final_model_path = Path(self.config.final_model_path)
        checkpoint_dir = final_model_path.parent
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_model_path = checkpoint_dir / "best_model.pt"

        _logger.info(f"Starting GRPO training on {self.config.device} from step {start_step}...")

        best_mean_reward = float("-inf")
        start_time = time.time()

        with mlflow.start_run(run_id=resume_run_id) as run:
            if resume_run_id is None:
                mlflow.log_params(self.config.model_dump())
            else:
                mlflow.set_tag("resumed_from_step", start_step)

            for i in range(start_step, self.config.max_iters):
                try:
                    prompt_batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self._train_dataloader)
                    prompt_batch = next(train_iter)

                rollouts = [self._rollout_one_prompt(prompt, metadata) for prompt, metadata in prompt_batch]

                rewards = torch.stack([r.rewards for r in rollouts], dim=0)  # (B, G)
                advantages = grpo_math.group_advantages(rewards, eps=self.config.advantage_eps).reshape(-1)  # (B*G,)

                sequences_batch, completion_mask_batch = self._pad_and_stack(rollouts)
                target_mask = completion_mask_batch[:, 1:]

                self.model.train()
                with torch.no_grad():
                    old_logprobs = self._sequence_logprobs(self.model, sequences_batch)
                    ref_logprobs = self._sequence_logprobs(self.ref_model, sequences_batch)

                last_metrics = {}
                for _ in range(self.config.num_inner_epochs):
                    new_logprobs = self._sequence_logprobs(self.model, sequences_batch)
                    loss, last_metrics = grpo_math.grpo_loss(
                        old_logprobs, new_logprobs, ref_logprobs, advantages, target_mask,
                        clip_epsilon=self.config.clip_epsilon, kl_coef=self.config.kl_coef,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                lr_scheduler.step()

                mean_reward = rewards.mean().item()

                is_eval_step = (i == 0 and start_step == 0) or (i % self.config.eval_interval == 0 and i > start_step)
                if is_eval_step:
                    eval_metrics, val_records = self._estimate_reward(step=i)

                    train_records = self._diagnostic_records(i, "train", prompt_batch, rollouts)
                    step_records = train_records + val_records
                    if step_records:
                        mlflow.log_table(data=pd.DataFrame(step_records), artifact_file="rollout_traceability.json")

                    current_time = time.time()
                    elapsed_seconds = current_time - start_time
                    steps_done = i - start_step
                    avg_time_per_step = elapsed_seconds / max(steps_done, 1)
                    eta_seconds = (self.config.max_iters - i) * avg_time_per_step

                    _logger.info(
                        f"Step {i}/{self.config.max_iters} | "
                        f"Loss: {loss.item():.4f} | "
                        f"Train reward: {mean_reward:.4f} | "
                        f"Val reward: {eval_metrics['val_reward_mean']:.4f} | "
                        f"KL: {last_metrics['mean_kl']:.4f} | "
                        f"Elapsed: {timedelta(seconds=int(elapsed_seconds))} | ETA: {timedelta(seconds=int(eta_seconds))}"
                    )

                    step_metrics = {
                        "learning_rate": optimizer.param_groups[0]['lr'],
                        "loss": loss.item(),
                        "train_reward_mean": mean_reward,
                        "val_reward_mean": eval_metrics["val_reward_mean"],
                        "val_reward_std": eval_metrics["val_reward_std"],
                        "mean_kl": last_metrics["mean_kl"],
                        "clip_fraction": last_metrics["clip_fraction"],
                        "mean_ratio": last_metrics["mean_ratio"],
                    }
                    # only present when reward_fn exposes diagnostics (e.g. SympyRewardFn) —
                    # a plain RewardFn callable still trains fine, just without these
                    train_rates = self._diagnostic_rates(train_records)
                    if train_rates is not None:
                        step_metrics["train_execute_rate"] = train_rates["execute_rate"]
                        step_metrics["train_correct_rate"] = train_rates["correct_rate"]
                    if "val_execute_rate" in eval_metrics:
                        step_metrics["val_execute_rate"] = eval_metrics["val_execute_rate"]
                        step_metrics["val_correct_rate"] = eval_metrics["val_correct_rate"]

                    mlflow.log_metrics(step_metrics, step=i)

                    if eval_metrics["val_reward_mean"] > best_mean_reward:
                        best_mean_reward = eval_metrics["val_reward_mean"]
                        torch.save(self.model.state_dict(), best_model_path)
                        _logger.info(f"New best model (val_reward_mean: {best_mean_reward:.4f})")

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

            _logger.info("Training complete, saving final model")
            torch.save(self.model.state_dict(), final_model_path)
            mlflow.log_metric("best_val_reward_mean", best_mean_reward)

    def resume_from_checkpoint(self, checkpoint_dir: Optional[Path] = None) -> None:
        """Find the latest local checkpoint in checkpoint_dir and resume training from it."""
        if checkpoint_dir is None:
            checkpoint_dir = Path(self.config.final_model_path).parent

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

    @torch.no_grad()
    def _estimate_reward(self, step: int) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        """Rolls out eval_iters prompt batches from val_dataset with no gradient update.

        Returns (metrics, diagnostic_records) — the records are the same per-completion
        traceability rows _diagnostic_records produces for the training batch, tagged
        split="val", so a step's full rollout table covers both train and val rollouts.
        """
        self.model.eval()
        val_iter = iter(self._val_dataloader)

        all_rewards = []
        all_records: List[Dict[str, Any]] = []
        for _ in range(self.config.eval_iters):
            try:
                prompt_batch = next(val_iter)
            except StopIteration:
                val_iter = iter(self._val_dataloader)
                prompt_batch = next(val_iter)

            rollouts = [self._rollout_one_prompt(prompt, metadata) for prompt, metadata in prompt_batch]
            all_rewards.extend(r.rewards for r in rollouts)
            all_records.extend(self._diagnostic_records(step, "val", prompt_batch, rollouts))

        all_rewards = torch.cat(all_rewards)
        self.model.train()

        metrics = {
            "val_reward_mean": all_rewards.mean().item(),
            "val_reward_std": all_rewards.std().item(),
        }
        rates = self._diagnostic_rates(all_records)
        if rates is not None:
            metrics["val_execute_rate"] = rates["execute_rate"]
            metrics["val_correct_rate"] = rates["correct_rate"]

        return metrics, all_records
