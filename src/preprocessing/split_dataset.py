"""
Single source of truth for the corpus-wide split across SFT, GRPO, and a reserved
benchmark holdout — generates all five index files together from one shuffle, so none
of them can ever overlap by construction.

Without this, SFTFormalizerDataset and GRPOPromptDataset each independently shuffle and
persist their own train/val split with zero awareness of each other, and nothing has
ever reserved a held-out benchmark set at all — meaning a row used in SFT training could
just as easily end up in GRPO's split, or in whatever set is eventually used to report
final benchmark numbers.

Must run BEFORE SFTFormalizerDataset or GRPOPromptDataset are ever instantiated — both
only compute+persist their own split if the expected index file doesn't already exist
(see their "don't overwrite if exists" guards), so pre-writing everything here means
they simply load what this script produced instead of computing their own.

Usage:
    python -m src.preprocessing.split_dataset
    python -m src.preprocessing.split_dataset --benchmark-holdout-size 3000
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
from tokenizers import Tokenizer

from ..utils.logger import get_logger

_logger = get_logger("split_dataset")

CORPUS_PATH = Path("data/posttraining/metamath_sympy")
TOKENIZER_PATH = Path("data/vocab/tokenizer_vocab.json")
VOCAB_METADATA_PATH = Path("data/corpus/metadata.json")

SFT_SPLIT_DIR = CORPUS_PATH / "sft"
GRPO_SPLIT_DIR = CORPUS_PATH / "grpo"
BENCHMARK_SPLIT_DIR = CORPUS_PATH / "benchmark"

SPLIT_SEED = 1337

DEFAULT_SFT_TRAIN_SIZE = 15_000
DEFAULT_SFT_VAL_SIZE = 3_000
DEFAULT_GRPO_TRAIN_SIZE = 15_000
DEFAULT_GRPO_VAL_SIZE = 3_000
# Comparable scale to SFT/GRPO's own val splits — gives roughly +-2% margin of error on
# an accuracy proportion at 95% CI, judged sufficient for the execution-rate/accuracy
# eval this holdout is for (see research/DECISION_LOG.md, evaluation methodology entry).
DEFAULT_BENCHMARK_HOLDOUT_SIZE = 2_000

_SPLIT_FILES = (
    SFT_SPLIT_DIR / "train_indices.npy",
    SFT_SPLIT_DIR / "val_indices.npy",
    GRPO_SPLIT_DIR / "train_indices.npy",
    GRPO_SPLIT_DIR / "val_indices.npy",
    BENCHMARK_SPLIT_DIR / "holdout_indices.npy",
)


def _load_corpus_table(corpus_path: Path):
    file_list = glob.glob(f"{corpus_path}/*.parquet", recursive=True)
    return ds.dataset(file_list, format="parquet").to_table()


def _eligible_indices(table, tokenizer: Tokenizer, context_size: int) -> np.ndarray:
    """Rows whose `<|assistant|> {sympy}` completion alone fits in context_size+1
    tokens — mirrors SFTFormalizerDataset's own eligibility filter exactly (same
    tokenizer-truncation trick: enabling truncation first means the post-truncation
    length equals min(raw_length, context_size+1), so the `<` check below correctly
    identifies "fits without truncation" either way). A row that fails this would
    break the <|assistant|>-anchor mask lookup in SFTFormalizerDataset.__getitem__ if
    it ever ended up in SFT's split — applied here to the whole pool up front (not
    just SFT's slice of it) so every split draws from one single universe of usable
    rows, rather than SFT and GRPO/benchmark silently living in two different ones.
    """
    tokenizer.enable_truncation(max_length=context_size + 1, direction="left")
    fits_mask = np.array([
        len(tokenizer.encode(f" <|assistant|> {sympy}").ids) < context_size + 1
        for sympy in table["output"].to_pylist()
    ])
    return np.nonzero(fits_mask)[0]


def build_all_splits(
    sft_train_size: int = DEFAULT_SFT_TRAIN_SIZE,
    sft_val_size: int = DEFAULT_SFT_VAL_SIZE,
    grpo_train_size: int = DEFAULT_GRPO_TRAIN_SIZE,
    grpo_val_size: int = DEFAULT_GRPO_VAL_SIZE,
    benchmark_holdout_size: int = DEFAULT_BENCHMARK_HOLDOUT_SIZE,
) -> None:
    existing = [p for p in _SPLIT_FILES if p.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite already-committed split files: "
            f"{[str(p) for p in existing]}. Delete them first if you actually want to "
            "regenerate the splits (this invalidates any SFT/GRPO training or "
            "benchmark results already run against them)."
        )

    context_size = json.loads(VOCAB_METADATA_PATH.read_bytes())["context_size"]
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    table = _load_corpus_table(CORPUS_PATH)
    eligible = _eligible_indices(table, tokenizer, context_size)
    skipped = table.num_rows - len(eligible)
    _logger.info(
        f"{len(eligible):,}/{table.num_rows:,} rows eligible "
        f"({skipped:,} skipped: completion alone exceeds context_size)"
    )

    needed = benchmark_holdout_size + sft_train_size + sft_val_size + grpo_train_size + grpo_val_size
    if len(eligible) < needed:
        raise ValueError(
            f"Not enough eligible rows: need {needed:,} total "
            f"(benchmark {benchmark_holdout_size:,} + SFT {sft_train_size:,}+{sft_val_size:,} "
            f"+ GRPO {grpo_train_size:,}+{grpo_val_size:,}), only {len(eligible):,} available."
        )

    rng = np.random.default_rng(seed=SPLIT_SEED)
    shuffled = eligible.copy()
    rng.shuffle(shuffled)

    cursor = 0

    def take(n: int) -> np.ndarray:
        nonlocal cursor
        chunk = shuffled[cursor:cursor + n]
        cursor += n
        return chunk

    # Reserved first, before either training stage gets anything — protected
    # regardless of how SFT/GRPO's own sizes might change in a future run.
    benchmark_holdout = take(benchmark_holdout_size)
    sft_train = take(sft_train_size)
    sft_val = take(sft_val_size)
    grpo_train = take(grpo_train_size)
    grpo_val = take(grpo_val_size)

    splits = {
        "benchmark_holdout": (benchmark_holdout, BENCHMARK_SPLIT_DIR / "holdout_indices.npy"),
        "sft_train": (sft_train, SFT_SPLIT_DIR / "train_indices.npy"),
        "sft_val": (sft_val, SFT_SPLIT_DIR / "val_indices.npy"),
        "grpo_train": (grpo_train, GRPO_SPLIT_DIR / "train_indices.npy"),
        "grpo_val": (grpo_val, GRPO_SPLIT_DIR / "val_indices.npy"),
    }

    # Disjoint by construction (sequential slices of one shuffled array) — verify
    # rather than just trust the arithmetic, since this guarantee is the whole point.
    names = list(splits)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = set(splits[names[i]][0].tolist()) & set(splits[names[j]][0].tolist())
            if overlap:
                raise AssertionError(f"{names[i]} and {names[j]} overlap on {len(overlap)} rows")

    for name, (indices, path) in splits.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, indices)
        _logger.info(f"{name}: {len(indices):,} rows -> {path}")

    _logger.info("Verified: all five splits are pairwise disjoint.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the SFT, GRPO, and benchmark-holdout splits together from "
                     "one coordinated shuffle, so none of them can ever overlap."
    )
    parser.add_argument("--sft-train-size", type=int, default=DEFAULT_SFT_TRAIN_SIZE)
    parser.add_argument("--sft-val-size", type=int, default=DEFAULT_SFT_VAL_SIZE)
    parser.add_argument("--grpo-train-size", type=int, default=DEFAULT_GRPO_TRAIN_SIZE)
    parser.add_argument("--grpo-val-size", type=int, default=DEFAULT_GRPO_VAL_SIZE)
    parser.add_argument("--benchmark-holdout-size", type=int, default=DEFAULT_BENCHMARK_HOLDOUT_SIZE)
    args = parser.parse_args()

    build_all_splits(
        sft_train_size=args.sft_train_size,
        sft_val_size=args.sft_val_size,
        grpo_train_size=args.grpo_train_size,
        grpo_val_size=args.grpo_val_size,
        benchmark_holdout_size=args.benchmark_holdout_size,
    )
