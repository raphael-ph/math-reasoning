# --- GRPO (Group Relative Policy Optimization) ---
# Implements GRPO training: for each prompt, sample a group of completions from the
# current policy, score them with a pluggable reward function (src/rewards), turn the
# group's rewards into advantages (no critic/value model needed), and update the policy
# with a clipped surrogate objective plus a KL penalty against a frozen reference model.
# See https://arxiv.org/abs/2402.03300 (DeepSeekMath).

import glob
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

# pyarrow imports
import pyarrow.dataset as ds
import pyarrow.compute as pc

# numpy
import numpy as np

# torch imports
from torch.utils.data import Dataset
from pydantic import Field

# internal imports
from ..utils.logger import get_logger
from .base import BaseTrainerConfig

# set-up logging
_logger = get_logger("grpo", level="INFO")

SHUFFLING_SEED = 42

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
    """
    def __init__(
        self,
        corpus_path: Path,
        split: Literal["train", "val"],
        prompt_column: str = "answer",
        train_size: int = 15000,
        val_size: int = 3000,
    ):
        super().__init__()
        self.prompt_column = prompt_column

        file_list = glob.glob(f"{corpus_path}/*.parquet", recursive=True)
        _logger.debug("Parquet file list: ")
        for f in file_list:
            _logger.debug(f)
        self.dataset = ds.dataset(file_list, format="parquet").to_table()

        indices = self.__shuffle_indices(np.arange(len(self.dataset)))
        train_indices = indices[:train_size]
        val_indices = indices[train_size : train_size + val_size]

        # kept in its own output dir (rather than reusing SFT's train/val_indices.npy)
        # since GRPO is a separate training stage with its own split boundary
        output_path = Path("data/posttraining/metamath_sympy/grpo")
        output_path.mkdir(parents=True, exist_ok=True)

        train_indices_path = output_path / "train_indices.npy"
        val_indices_path = output_path / "val_indices.npy"

        if not train_indices_path.exists():
            np.save(train_indices_path, train_indices)
        if not val_indices_path.exists():
            np.save(val_indices_path, val_indices)

        if split == "train":
            idx = np.load(train_indices_path)
        elif split == "val":
            idx = np.load(val_indices_path)
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

    # --- Helper ---
    def __shuffle_indices(self, indices: np.ndarray) -> np.ndarray:
        """Shuffles the given (absolute) row indices with a fixed seed"""
        indices = indices.copy()

        random_generator = np.random.default_rng(seed=SHUFFLING_SEED)
        random_generator.shuffle(indices)

        return indices

def prompt_collate_fn(batch: List[Tuple[str, Dict[str, Any]]]) -> List[Tuple[str, Dict[str, Any]]]:
    """Identity collate — rollout is per-prompt, so batches stay a plain list of
    (prompt_text, metadata) pairs rather than being stacked into a tensor."""
    return batch
