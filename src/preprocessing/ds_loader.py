"""
Dataset scraping script for formalizer pre-training.
Downloads and saves to Parquet:
  1. Python-Edu        (HuggingFaceTB/smollm-corpus)          ~4B tokens  40%
  2. Stack-Edu Python  (HuggingFaceTB/stack-edu)              ~3.5B tokens 35%
  3. Algebraic-Stack   (EleutherAI/proof-pile-2, Python only) ~1.5B tokens 15%
  4. Arxiv Math        (EleutherAI/proof-pile-2, arxiv)       ~1B tokens   10%

Strategy:
  - Python datasets: broad Python fluency, SymPy files get a soft priority boost
    (written twice into the buffer = ~2x sampling weight, not a hard filter)
  - Arxiv: filtered to math-heavy documents only (min 10 LaTeX math pattern hits)
  - All output saved as Parquet shards with snappy compression
  - Resume support: skips already-completed shards if --resume is passed

Requirements:
    pip install datasets boto3 pyarrow tqdm regex

Usage:
    # Full run
    python scrape_datasets.py --output_dir /data/pretraining --max_tokens 10_000_000_000

    # Single dataset
    python scrape_datasets.py --output_dir /data/pretraining --datasets algebraic_stack

    # Resume interrupted run
    python scrape_datasets.py --output_dir /data/pretraining --resume
"""

import argparse
import gzip
import json
import re
from pathlib import Path

import boto3
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import ClientError
from datasets import load_dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Budget proportions (must sum to 1.0)
BUDGET_PROPORTIONS = {
    "python_edu":      0.40,
    "stack_edu":       0.35,
    "algebraic_stack": 0.15,
    "arxiv_math":      0.10,
}

# Records per Parquet shard — lower on memory-constrained machines
SHARD_SIZE = 50_000

# Minimum document length in characters
MIN_DOC_CHARS = 100

# SymPy soft boost: SymPy files are written twice, giving ~2x sampling weight
# without excluding non-SymPy Python files
SYMPY_SOFT_BOOST = True

# LaTeX math environment regex for arxiv filtering
LATEX_MATH_RE = re.compile(
    r"(\\begin\{(equation|align|theorem|lemma|proof|proposition|corollary"
    r"|definition|gather|multline)\}"
    r"|\\frac\{|\\int_|\\sum_|\\prod_"
    r"|\\mathbb\{|\\mathcal\{"
    r"|\$\$)",
    re.IGNORECASE,
)

# Minimum LaTeX math hits to keep an arxiv document
LATEX_MIN_HITS = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_tokens_approx(text: str) -> int:
    """1 token ≈ 4 characters."""
    return len(text) // 4


def is_sympy_file(text: str) -> bool:
    """True if the file explicitly imports SymPy."""
    return bool(re.search(r"^\s*(import sympy|from sympy)", text, re.MULTILINE))


def is_math_heavy_latex(text: str) -> bool:
    """True if the document has enough LaTeX math patterns to be useful."""
    return len(LATEX_MATH_RE.findall(text)) >= LATEX_MIN_HITS


def load_manifest(output_dir: Path) -> dict:
    path = output_dir / "manifest.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"datasets": {}}


def save_manifest(output_dir: Path, manifest: dict) -> None:
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def count_existing_shards(dataset_dir: Path, prefix: str) -> int:
    return len(list(dataset_dir.glob(f"{prefix}_shard_*.parquet")))


# ---------------------------------------------------------------------------
# Shard writer
# ---------------------------------------------------------------------------

class ShardWriter:
    """
    Buffers text records and flushes to Parquet shards.
    start_shard allows resuming from an existing shard count.
    """

    def __init__(self, output_dir: Path, prefix: str, start_shard: int = 0):
        self.output_dir = output_dir
        self.prefix = prefix
        self.shard_idx = start_shard
        self.buffer: list[str] = []
        self.total_tokens = 0

    def add(self, text: str) -> int:
        tokens = count_tokens_approx(text)
        self.buffer.append(text)
        self.total_tokens += tokens
        if len(self.buffer) >= SHARD_SIZE:
            self._flush()
        return tokens

    def finalize(self) -> int:
        if self.buffer:
            self._flush()
        return self.total_tokens

    def _flush(self):
        path = self.output_dir / f"{self.prefix}_shard_{self.shard_idx:04d}.parquet"
        pq.write_table(
            pa.table({"text": self.buffer}),
            path,
            compression="snappy",
        )
        print(f"    Flushed shard {self.shard_idx:04d} "
              f"({len(self.buffer):,} records) → {path.name}")
        self.buffer.clear()
        self.shard_idx += 1


# ---------------------------------------------------------------------------
# S3 downloader (shared by Python-Edu and Stack-Edu)
# ---------------------------------------------------------------------------

def make_s3_downloader():
    """Downloads file content from Software Heritage public S3."""
    s3 = boto3.client("s3", region_name="us-east-1")

    def download(blob_id: str) -> str | None:
        try:
            obj = s3.get_object(Bucket="softwareheritage", Key=f"content/{blob_id}")
            with gzip.GzipFile(fileobj=obj["Body"]) as f:
                return f.read().decode("utf-8", errors="ignore")
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return None
            raise

    return download


# ---------------------------------------------------------------------------
# Dataset 1: Python-Edu
# ---------------------------------------------------------------------------

def scrape_python_edu(output_dir: Path, token_budget: int, resume: bool) -> int:
    print("\n" + "=" * 60)
    print("[1/4] Python-Edu  (HuggingFaceTB/smollm-corpus)")
    print(f"      Budget: {token_budget / 1e9:.2f}B tokens")
    print("=" * 60)

    dataset_dir = output_dir / "python_edu"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    start_shard = count_existing_shards(dataset_dir, "python_edu") if resume else 0
    skip_n = start_shard * SHARD_SIZE
    if resume and start_shard > 0:
        print(f"  Resuming from shard {start_shard} (skipping ~{skip_n:,} records)")

    download = make_s3_downloader()
    writer = ShardWriter(dataset_dir, "python_edu", start_shard)
    ds = load_dataset("HuggingFaceTB/smollm-corpus", "python-edu",
                      split="train", streaming=True)

    skipped = 0
    sympy_count = 0
    seen = 0

    with tqdm(desc="Python-Edu", unit=" docs") as pbar:
        for example in ds:
            if writer.total_tokens >= token_budget:
                break
            if seen < skip_n:
                seen += 1
                continue
            seen += 1

            text = download(example["blob_id"])
            if not text or len(text.strip()) < MIN_DOC_CHARS:
                skipped += 1
                continue

            has_sympy = is_sympy_file(text)
            if has_sympy:
                sympy_count += 1

            writer.add(text)
            if SYMPY_SOFT_BOOST and has_sympy and writer.total_tokens < token_budget:
                writer.add(text)

            pbar.update(1)
            pbar.set_postfix(tokens=f"{writer.total_tokens/1e9:.2f}B",
                             sympy=sympy_count, skipped=skipped)

    total = writer.finalize()
    print(f"  Done. {total/1e9:.3f}B tokens | SymPy: {sympy_count} | Skipped: {skipped}")
    return total


# ---------------------------------------------------------------------------
# Dataset 2: Stack-Edu Python
# ---------------------------------------------------------------------------

def scrape_stack_edu(output_dir: Path, token_budget: int, resume: bool) -> int:
    print("\n" + "=" * 60)
    print("[2/4] Stack-Edu Python  (HuggingFaceTB/stack-edu)")
    print(f"      Budget: {token_budget / 1e9:.2f}B tokens")
    print("=" * 60)

    dataset_dir = output_dir / "stack_edu_python"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    start_shard = count_existing_shards(dataset_dir, "stack_edu_python") if resume else 0
    skip_n = start_shard * SHARD_SIZE
    if resume and start_shard > 0:
        print(f"  Resuming from shard {start_shard} (skipping ~{skip_n:,} records)")

    download = make_s3_downloader()
    writer = ShardWriter(dataset_dir, "stack_edu_python", start_shard)
    ds = load_dataset("HuggingFaceTB/stack-edu", "Python",
                      split="train", streaming=True)

    skipped = 0
    sympy_count = 0
    seen = 0

    with tqdm(desc="Stack-Edu Python", unit=" docs") as pbar:
        for example in ds:
            if writer.total_tokens >= token_budget:
                break
            if seen < skip_n:
                seen += 1
                continue
            seen += 1

            text = download(example["blob_id"])
            if not text or len(text.strip()) < MIN_DOC_CHARS:
                skipped += 1
                continue

            # Must look like Python
            if "def " not in text and "import " not in text and "class " not in text:
                skipped += 1
                continue

            has_sympy = is_sympy_file(text)
            if has_sympy:
                sympy_count += 1

            writer.add(text)
            if SYMPY_SOFT_BOOST and has_sympy and writer.total_tokens < token_budget:
                writer.add(text)

            pbar.update(1)
            pbar.set_postfix(tokens=f"{writer.total_tokens/1e9:.2f}B",
                             sympy=sympy_count, skipped=skipped)

    total = writer.finalize()
    print(f"  Done. {total/1e9:.3f}B tokens | SymPy: {sympy_count} | Skipped: {skipped}")
    return total


# ---------------------------------------------------------------------------
# Dataset 3: Algebraic-Stack (Python only)
# ---------------------------------------------------------------------------

def scrape_algebraic_stack(output_dir: Path, token_budget: int, resume: bool) -> int:
    print("\n" + "=" * 60)
    print("[3/4] Algebraic-Stack Python  (EleutherAI/proof-pile-2)")
    print(f"      Budget: {token_budget / 1e9:.2f}B tokens")
    print("=" * 60)

    dataset_dir = output_dir / "algebraic_stack_python"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    start_shard = count_existing_shards(dataset_dir, "algebraic_stack_python") if resume else 0
    skip_n = start_shard * SHARD_SIZE
    if resume and start_shard > 0:
        print(f"  Resuming from shard {start_shard} (skipping ~{skip_n:,} records)")

    writer = ShardWriter(dataset_dir, "algebraic_stack_python", start_shard)
    ds = load_dataset("EleutherAI/proof-pile-2", "algebraic-stack",
                      split="train", streaming=True)

    skipped_lang = 0
    skipped_empty = 0
    sympy_count = 0
    seen = 0

    with tqdm(desc="Algebraic-Stack", unit=" docs") as pbar:
        for example in ds:
            if writer.total_tokens >= token_budget:
                break
            if seen < skip_n:
                seen += 1
                continue
            seen += 1

            # Python only
            meta = example["meta"]
            if isinstance(meta, str):
                meta = json.loads(meta)
            lang = meta.get("language", meta.get("lang", "")).lower()
            if lang != "python":
                skipped_lang += 1
                continue

            text = example["text"]
            if not text or len(text.strip()) < MIN_DOC_CHARS:
                skipped_empty += 1
                continue

            has_sympy = is_sympy_file(text)
            if has_sympy:
                sympy_count += 1

            writer.add(text)
            if SYMPY_SOFT_BOOST and has_sympy and writer.total_tokens < token_budget:
                writer.add(text)

            pbar.update(1)
            pbar.set_postfix(tokens=f"{writer.total_tokens/1e9:.2f}B",
                             sympy=sympy_count,
                             skipped=skipped_lang + skipped_empty)

    total = writer.finalize()
    print(f"  Done. {total/1e9:.3f}B tokens | SymPy: {sympy_count} "
          f"| Skipped lang: {skipped_lang} | Skipped empty: {skipped_empty}")
    return total


# ---------------------------------------------------------------------------
# Dataset 4: Arxiv Math
# ---------------------------------------------------------------------------

def scrape_arxiv_math(output_dir: Path, token_budget: int, resume: bool) -> int:
    print("\n" + "=" * 60)
    print("[4/4] Arxiv Math  (EleutherAI/proof-pile-2, arxiv subset)")
    print(f"      Budget: {token_budget / 1e9:.2f}B tokens")
    print("=" * 60)

    dataset_dir = output_dir / "arxiv_math"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    start_shard = count_existing_shards(dataset_dir, "arxiv_math") if resume else 0
    skip_n = start_shard * SHARD_SIZE
    if resume and start_shard > 0:
        print(f"  Resuming from shard {start_shard} (skipping ~{skip_n:,} records)")

    writer = ShardWriter(dataset_dir, "arxiv_math", start_shard)
    ds = load_dataset("EleutherAI/proof-pile-2", "arxiv",
                      split="train", streaming=True)

    skipped_not_math = 0
    skipped_empty = 0
    seen = 0

    with tqdm(desc="Arxiv Math", unit=" docs") as pbar:
        for example in ds:
            if writer.total_tokens >= token_budget:
                break
            if seen < skip_n:
                seen += 1
                continue
            seen += 1

            text = example["text"]
            if not text or len(text.strip()) < MIN_DOC_CHARS:
                skipped_empty += 1
                continue

            if not is_math_heavy_latex(text):
                skipped_not_math += 1
                continue

            writer.add(text)
            pbar.update(1)
            pbar.set_postfix(tokens=f"{writer.total_tokens/1e9:.2f}B",
                             skipped=skipped_not_math + skipped_empty)

    total = writer.finalize()
    print(f"  Done. {total/1e9:.3f}B tokens "
          f"| Skipped (not math): {skipped_not_math} "
          f"| Skipped (empty): {skipped_empty}")
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SCRAPERS = {
    "python_edu":      scrape_python_edu,
    "stack_edu":       scrape_stack_edu,
    "algebraic_stack": scrape_algebraic_stack,
    "arxiv_math":      scrape_arxiv_math,
}


def main():
    parser = argparse.ArgumentParser(
        description="Scrape pretraining datasets for math formalizer model"
    )
    parser.add_argument("--output_dir", type=str, default="./pretraining_data")
    parser.add_argument("--max_tokens", type=int, default=10_000_000_000,
                        help="Total approximate token budget (default 10B)")
    parser.add_argument("--datasets", nargs="+",
                        choices=list(SCRAPERS.keys()) + ["all"],
                        default=["all"])
    parser.add_argument("--resume", action="store_true",
                        help="Skip existing shards and continue from where left off")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    to_run = list(SCRAPERS.keys()) if "all" in args.datasets else args.datasets
    budgets = {name: int(args.max_tokens * BUDGET_PROPORTIONS[name]) for name in to_run}

    print("\n" + "=" * 60)
    print("Pre-training Data Scraper — Math Formalizer")
    print("=" * 60)
    print(f"Output dir  : {output_dir}")
    print(f"Total budget: {args.max_tokens / 1e9:.1f}B tokens")
    print(f"Datasets    : {', '.join(to_run)}")
    print(f"Resume      : {args.resume}")
    print(f"SymPy boost : {SYMPY_SOFT_BOOST}")
    for name in to_run:
        print(f"  {name:<25} budget: {budgets[name]/1e9:.2f}B "
              f"({BUDGET_PROPORTIONS[name]*100:.0f}%)")

    manifest = load_manifest(output_dir)
    totals = {}

    for name in to_run:
        tokens = SCRAPERS[name](output_dir, budgets[name], args.resume)
        totals[name] = tokens
        manifest["datasets"][name] = {"tokens_approx": tokens}
        save_manifest(output_dir, manifest)  # save after each dataset for safety

    # Final summary
    print("\n" + "=" * 60)
    print("SCRAPING COMPLETE — Summary")
    print("=" * 60)
    grand_total = sum(totals.values())
    for name, tokens in totals.items():
        pct = (tokens / grand_total * 100) if grand_total > 0 else 0
        print(f"  {name:<25} {tokens/1e9:.3f}B  ({pct:.1f}%)")
    print(f"  {'TOTAL':<25} {grand_total/1e9:.3f}B tokens")
    print("=" * 60)

    manifest["total_tokens_approx"] = grand_total
    manifest["output_dir"] = str(output_dir)
    save_manifest(output_dir, manifest)
    print(f"\nManifest → {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()