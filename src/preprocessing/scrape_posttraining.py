"""
Dataset scraping script for formalizer post-training (SFT + GRPO).

Downloads tfshaman/metamath_sympy_v1 and rebuilds it into a 3-column dataset:
  - answer:      unchanged natural-language chain-of-thought (SFT/GRPO input)
  - output:      cleaned SymPy program extracted from the [|Sympy|] blocks of
                  the original `output` column (the [|Text|] blocks are
                  dropped), terminated with the existing <|endoftext|> token
  - code_output: unchanged ground-truth numeric result, kept for later
                  verification (e.g. executing the generated SymPy program
                  and comparing against it as a GRPO reward signal)

The source `question` and `data_type` columns are dropped.

Usage:
    # Full run
    python -m src.preprocessing.scrape_posttraining --output_dir ./data/posttraining/metamath_sympy

    # Smoke test on a subset
    python -m src.preprocessing.scrape_posttraining --output_dir ./data/posttraining/metamath_sympy --limit 500

    # Resume from last checkpoint
    python -m src.preprocessing.scrape_posttraining --output_dir ./data/posttraining/metamath_sympy --resume
"""

import argparse
import json
import math
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm

# internal imports
from ..utils.logger import get_logger

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ID          = "tfshaman/metamath_sympy_v1"
NAME             = "metamath_sympy"
SHARD_SIZE       = 20_000   # records per parquet shard
CHECKPOINT_EVERY = 10_000   # save checkpoint every N rows processed

EOT_TOKEN = "<|endoftext|>"  # existing special token (tokenizer.py SPECIAL_TOKENS /
                              # hf_tokenizer.py special_tokens), used the same way
                              # memmap_builder.py terminates pretraining documents

SYMPY_BLOCK_RE = re.compile(
    r"\[\|\s*sympy\s*\|\]\s*(.*?)\s*\[\|\s*endofblock\s*\|\]",
    re.IGNORECASE | re.DOTALL,
)

_logger = get_logger("scrape_posttraining")


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_sympy_output(raw_output: str) -> str | None:
    """Extract only the [|Sympy|]...[|EndOfBlock|] blocks from a raw `output`
    field, drop the [|Text|] blocks, and terminate with the EOT token.
    Returns None if no Sympy block could be found.
    """
    blocks = SYMPY_BLOCK_RE.findall(raw_output or "")
    if not blocks:
        return None
    code = "\n".join(block.strip() for block in blocks)
    return f"{code}\n{EOT_TOKEN}"


# ---------------------------------------------------------------------------
# Checkpoint / manifest
# ---------------------------------------------------------------------------

def checkpoint_path(dataset_dir: Path) -> Path:
    return dataset_dir / f"{NAME}.checkpoint.json"


def load_checkpoint(dataset_dir: Path) -> dict:
    path = checkpoint_path(dataset_dir)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"last_idx": 0, "shard_idx": 0, "written": 0,
            "skipped_no_sympy": 0, "skipped_no_code_output": 0,
            "completed": False}


def save_checkpoint(dataset_dir: Path, state: dict) -> None:
    with open(checkpoint_path(dataset_dir), "w") as f:
        json.dump(state, f, indent=2)


def save_manifest(dataset_dir: Path, state: dict, total_rows: int) -> None:
    manifest = {
        "dataset":        REPO_ID,
        "total_rows_src": total_rows,
        "rows_written":   state["written"],
        "skipped_no_sympy":       state["skipped_no_sympy"],
        "skipped_no_code_output": state["skipped_no_code_output"],
        "completed":      state["completed"],
    }
    with open(dataset_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


# ---------------------------------------------------------------------------
# Shard writer (output / answer / code_output records)
# ---------------------------------------------------------------------------

class RecordShardWriter:
    def __init__(self, output_dir: Path, prefix: str, start_shard: int = 0,
                 start_written: int = 0):
        self.output_dir = output_dir
        self.prefix      = prefix
        self.shard_idx   = start_shard
        self.written      = start_written
        self.buffer: dict[str, list] = {"output": [], "answer": [], "code_output": []}

    def add(self, output: str, answer: str, code_output: float) -> None:
        self.buffer["output"].append(output)
        self.buffer["answer"].append(answer)
        self.buffer["code_output"].append(code_output)
        self.written += 1
        if len(self.buffer["output"]) >= SHARD_SIZE:
            self._flush()

    def finalize(self) -> int:
        if self.buffer["output"]:
            self._flush()
        return self.written

    def _flush(self):
        path = self.output_dir / f"{self.prefix}_shard_{self.shard_idx:04d}.parquet"
        pq.write_table(pa.table(self.buffer), path, compression="snappy")
        _logger.info(f"    Flushed shard {self.shard_idx:04d} "
                     f"({len(self.buffer['output']):,} records) → {path.name}")
        self.buffer = {"output": [], "answer": [], "code_output": []}
        self.shard_idx += 1


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

def scrape_metamath_sympy(output_dir: Path, resume: bool, limit: int | None = None) -> int:
    dataset_dir = output_dir
    dataset_dir.mkdir(parents=True, exist_ok=True)

    ckpt = load_checkpoint(dataset_dir) if resume else \
           {"last_idx": 0, "shard_idx": 0, "written": 0,
            "skipped_no_sympy": 0, "skipped_no_code_output": 0,
            "completed": False}

    if ckpt.get("completed", False):
        _logger.info(f"  Already completed ({ckpt['written']:,} rows) — skipping.")
        return ckpt["written"]

    _logger.info(f"  Loading {REPO_ID} (train split) ...")
    ds = load_dataset(REPO_ID, split="train")
    total = len(ds) if limit is None else min(limit, len(ds))

    start_idx = ckpt["last_idx"]
    if resume and start_idx > 0:
        _logger.info(f"  Resuming from row {start_idx:,} / {total:,}")

    writer = RecordShardWriter(dataset_dir, NAME, ckpt["shard_idx"], ckpt["written"])
    skipped_no_sympy       = ckpt["skipped_no_sympy"]
    skipped_no_code_output = ckpt["skipped_no_code_output"]
    rows_since_ckpt = 0

    with tqdm(desc=NAME, total=total, initial=start_idx, unit=" rows") as pbar:
        for idx in range(start_idx, total):
            row = ds[idx]

            code_output = row["code_output"]
            if code_output is None or (isinstance(code_output, float) and math.isnan(code_output)):
                skipped_no_code_output += 1
                pbar.update(1)
                continue

            cleaned = clean_sympy_output(row["output"])
            if cleaned is None:
                skipped_no_sympy += 1
                pbar.update(1)
                continue

            writer.add(output=cleaned, answer=row["answer"], code_output=float(code_output))

            rows_since_ckpt += 1
            pbar.update(1)
            pbar.set_postfix(
                written=writer.written,
                skipped=skipped_no_sympy + skipped_no_code_output,
            )

            if rows_since_ckpt >= CHECKPOINT_EVERY:
                save_checkpoint(dataset_dir, {
                    "last_idx":                idx + 1,
                    "shard_idx":                writer.shard_idx,
                    "written":                  writer.written,
                    "skipped_no_sympy":         skipped_no_sympy,
                    "skipped_no_code_output":   skipped_no_code_output,
                    "completed":                False,
                })
                rows_since_ckpt = 0

    written = writer.finalize()
    final_state = {
        "last_idx":                total,
        "shard_idx":                writer.shard_idx,
        "written":                  written,
        "skipped_no_sympy":         skipped_no_sympy,
        "skipped_no_code_output":   skipped_no_code_output,
        "completed":                True,
    }
    save_checkpoint(dataset_dir, final_state)
    save_manifest(dataset_dir, final_state, total)

    _logger.info(f"  Done. {written:,} rows written | "
                 f"Skipped (no sympy): {skipped_no_sympy} | "
                 f"Skipped (no code_output): {skipped_no_code_output}")
    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scrape the metamath_sympy_v1 dataset for math formalizer post-training"
    )
    parser.add_argument("--output_dir", type=str, default="./data/posttraining/metamath_sympy")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N rows (for a smoke test)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    _logger.info("\n" + "=" * 60)
    _logger.info("Post-training Data Scraper — MetaMath SymPy")
    _logger.info("=" * 60)
    _logger.info(f"Source     : {REPO_ID}")
    _logger.info(f"Output dir : {output_dir}")
    _logger.info(f"Resume     : {args.resume}")
    _logger.info(f"Limit      : {args.limit if args.limit else 'none (full dataset)'}")

    written = scrape_metamath_sympy(output_dir, resume=args.resume, limit=args.limit)

    _logger.info("\n" + "=" * 60)
    _logger.info("SCRAPING COMPLETE")
    _logger.info(f"  Rows written → {written:,}")
    _logger.info(f"  Manifest     → {output_dir / 'manifest.json'}")
    _logger.info("=" * 60)


if __name__ == "__main__":
    main()
