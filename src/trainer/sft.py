# --- Supervised Fine-Tuning ---
# This script show an implementation of a Supervised Fine-Tuning pipeline. 
# For the Formalizer Model, the main plan is to "align", as per http://arxiv.org/abs/2203.02155, the outputs of the model.
#
# The Formalizer Model is responsible for correctly converting natural language, informal mathematical resolutions to formal, sympy
# representations.

# general imports
import glob
import numpy as np
import pyarrow as pa
from pathlib import Path
import pyarrow.dataset as ds
import pyarrow.compute as pc
from typing import Tuple, Optional, Literal

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

    def train(self):

    

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