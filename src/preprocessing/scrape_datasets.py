"""
Dataset scraping script for formalizer pre-training.

Downloads and saves to Parquet:
  1. Python-Edu        (HuggingFaceTB/smollm-corpus)          ~4B tokens  40%
  2. Stack-Edu Python  (HuggingFaceTB/stack-edu)              ~3.5B tokens 35%
  3. Algebraic-Stack   (EleutherAI/proof-pile-2, Python only) ~1.5B tokens 15%
  4. Arxiv Math        (EleutherAI/proof-pile-2, arxiv)       ~1B tokens   10%

Key design decisions:
  - Python-Edu and Stack-Edu: metadata parquet downloaded first, then S3 in
    true parallel via ThreadPoolExecutor (no streaming bottleneck)
  - Algebraic-Stack and Arxiv: streamed directly from HuggingFace (no S3)
  - Checkpoints saved every CHECKPOINT_EVERY docs so interrupted runs resume
    from the last checkpoint, not the beginning
  - SymPy files written twice for ~2x soft sampling boost
  - Arxiv filtered to math-heavy docs (LATEX_MIN_HITS pattern matches)

Requirements:
    pip install datasets boto3 pyarrow pandas tqdm huggingface_hub

Usage:
    # Full run
    python scrape_datasets.py --output_dir /data/pretraining --max_tokens 10_000_000_000

    # Single dataset
    python scrape_datasets.py --output_dir /data/pretraining --datasets algebraic_stack

    # Resume from last checkpoint
    python scrape_datasets.py --output_dir /data/pretraining --resume
"""

import argparse
import gzip
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError
from datasets import load_dataset
from huggingface_hub import snapshot_download
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BUDGET_PROPORTIONS = {
    "python_edu":      0.40,
    "stack_edu":       0.35,
    "algebraic_stack": 0.15,
    "arxiv_math":      0.10,
}

SHARD_SIZE        = 50_000   # records per parquet shard
MIN_DOC_CHARS     = 100      # skip documents shorter than this
SYMPY_SOFT_BOOST  = True     # write sympy files twice (~2x weight)
S3_WORKERS        = 64       # parallel S3 threads
S3_BATCH_SIZE     = 512      # docs per parallel batch
CHECKPOINT_EVERY  = 10_000   # save checkpoint every N docs processed

LATEX_MATH_RE = re.compile(
    r"(\\begin\{(equation|align|theorem|lemma|proof|proposition|corollary"
    r"|definition|gather|multline)\}"
    r"|\\frac\{|\\int_|\\sum_|\\prod_"
    r"|\\mathbb\{|\\mathcal\{"
    r"|\$\$)",
    re.IGNORECASE,
)
LATEX_MIN_HITS = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_tokens_approx(text: str) -> int:
    return len(text) // 3


def is_sympy_file(text: str) -> bool:
    return bool(re.search(r"^\s*(import sympy|from sympy)", text, re.MULTILINE))


def is_math_heavy_latex(text: str) -> bool:
    return len(LATEX_MATH_RE.findall(text)) >= LATEX_MIN_HITS


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def checkpoint_path(dataset_dir: Path, name: str) -> Path:
    return dataset_dir / f"{name}.checkpoint.json"


def load_checkpoint(dataset_dir: Path, name: str) -> dict:
    path = checkpoint_path(dataset_dir, name)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"last_idx": 0, "total_tokens": 0, "shard_idx": 0,
            "sympy_count": 0, "skipped": 0, "completed": False}


def save_checkpoint(dataset_dir: Path, name: str, state: dict) -> None:
    with open(checkpoint_path(dataset_dir, name), "w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def load_manifest(output_dir: Path) -> dict:
    path = output_dir / "manifest.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"datasets": {}}


def save_manifest(output_dir: Path, manifest: dict) -> None:
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


# ---------------------------------------------------------------------------
# Shard writer
# ---------------------------------------------------------------------------

class ShardWriter:
    def __init__(self, output_dir: Path, prefix: str, start_shard: int = 0,
                 start_tokens: int = 0):
        self.output_dir   = output_dir
        self.prefix       = prefix
        self.shard_idx    = start_shard
        self.buffer: list[str] = []
        self.total_tokens = start_tokens

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
        pq.write_table(pa.table({"text": self.buffer}), path, compression="snappy")
        print(f"    Flushed shard {self.shard_idx:04d} "
              f"({len(self.buffer):,} records) → {path.name}")
        self.buffer.clear()
        self.shard_idx += 1


# ---------------------------------------------------------------------------
# S3 client pool
# ---------------------------------------------------------------------------

def make_s3_clients(n: int) -> list:
    return [
        boto3.session.Session().client(
            "s3",
            region_name="us-east-1",
            config=Config(signature_version=UNSIGNED),
        )
        for _ in range(n)
    ]


def fetch_blob(args) -> tuple[int, str | None]:
    """Fetch a single blob from S3. args = (idx, blob_id, client)"""
    idx, blob_id, client = args
    try:
        obj = client.get_object(
            Bucket="softwareheritage", Key=f"content/{blob_id}"
        )
        with gzip.GzipFile(fileobj=obj["Body"]) as f:
            return idx, f.read().decode("utf-8", errors="ignore")
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            return idx, None
        raise


def parallel_s3_fetch(
    blob_ids: list[str],
    clients: list,
    start_idx: int = 0,
) -> list[tuple[int, str | None]]:
    """
    Fetch blob_ids in parallel.
    Returns list of (original_index, text|None) sorted by original_index.
    """
    import itertools
    client_cycle = itertools.cycle(clients)
    args = [(start_idx + i, bid, next(client_cycle))
            for i, bid in enumerate(blob_ids)]

    results = []
    with ThreadPoolExecutor(max_workers=S3_WORKERS) as executor:
        futures = {executor.submit(fetch_blob, a): a for a in args}
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda x: x[0])
    return results


# ---------------------------------------------------------------------------
# Metadata downloader
# ---------------------------------------------------------------------------

def download_metadata(repo_id: str, patterns: list[str], local_dir: Path) -> Path:
    """Download only the metadata parquet files from a HuggingFace dataset."""
    print(f"  Downloading metadata from {repo_id} ...")
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=str(local_dir),
    )
    return Path(path)


def load_parquet_metadata(meta_dir: Path, pattern: str) -> pd.DataFrame:
    """Load all parquet files matching pattern into a single DataFrame."""
    files = sorted(meta_dir.rglob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files found matching {pattern} in {meta_dir}")
    print(f"  Loading {len(files)} parquet file(s) ...")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"  Total rows: {len(df):,}")
    return df


# ---------------------------------------------------------------------------
# Core S3 scraper (shared logic for Python-Edu and Stack-Edu)
# ---------------------------------------------------------------------------

def scrape_s3_dataset(
    name: str,
    output_dir: Path,
    token_budget: int,
    resume: bool,
    repo_id: str,
    meta_patterns: list[str],
    meta_parquet_glob: str,
    extra_filter=None,        # optional fn(text) -> bool
) -> int:
    dataset_dir = output_dir / name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    meta_dir    = output_dir / f"{name}_meta"

    # FIX 1: always define ckpt, then check completed
    ckpt = load_checkpoint(dataset_dir, name) if resume else \
           {"last_idx": 0, "total_tokens": 0, "shard_idx": 0,
            "sympy_count": 0, "skipped": 0, "completed": False}

    if ckpt.get("completed", False):
        print(f"  Already completed ({ckpt['total_tokens']/1e6:.0f}M tokens) — skipping.")
        return ckpt["total_tokens"]

    # Download metadata parquet if not already present
    if not meta_dir.exists() or not any(meta_dir.rglob("*.parquet")):
        download_metadata(repo_id, meta_patterns, meta_dir)
    else:
        print(f"  Metadata already downloaded at {meta_dir}")

    df = load_parquet_metadata(meta_dir, meta_parquet_glob)

    start_idx    = ckpt["last_idx"]
    sympy_count  = ckpt["sympy_count"]
    skipped      = ckpt["skipped"]

    if resume and start_idx > 0:
        print(f"  Resuming from doc {start_idx:,} / {len(df):,}")

    writer  = ShardWriter(dataset_dir, name, ckpt["shard_idx"], ckpt["total_tokens"])
    clients = make_s3_clients(S3_WORKERS)

    blob_ids = df["blob_id"].tolist()
    total    = len(blob_ids)
    docs_since_ckpt = 0

    with tqdm(desc=name, total=total, initial=start_idx, unit=" docs") as pbar:
        for batch_start in range(start_idx, total, S3_BATCH_SIZE):
            if writer.total_tokens >= token_budget:
                break

            batch_ids = blob_ids[batch_start: batch_start + S3_BATCH_SIZE]
            results   = parallel_s3_fetch(batch_ids, clients, start_idx=batch_start)

            for idx, text in results:
                if writer.total_tokens >= token_budget:
                    break
                if not text or len(text.strip()) < MIN_DOC_CHARS:
                    skipped += 1
                    continue
                if extra_filter and not extra_filter(text):
                    skipped += 1
                    continue

                has_sympy = is_sympy_file(text)
                if has_sympy:
                    sympy_count += 1

                writer.add(text)
                if SYMPY_SOFT_BOOST and has_sympy and writer.total_tokens < token_budget:
                    writer.add(text)

                docs_since_ckpt += 1

            pbar.update(len(batch_ids))
            pbar.set_postfix(
                tokens=f"{writer.total_tokens/1e6:.0f}M",
                sympy=sympy_count,
                skipped=skipped,
            )

            # Checkpoint
            if docs_since_ckpt >= CHECKPOINT_EVERY:
                save_checkpoint(dataset_dir, name, {
                    "last_idx":     batch_start + S3_BATCH_SIZE,
                    "total_tokens": writer.total_tokens,
                    "shard_idx":    writer.shard_idx,
                    "sympy_count":  sympy_count,
                    "skipped":      skipped,
                    "completed":    False,
                })
                docs_since_ckpt = 0

    total_tokens = writer.finalize()

    # FIX 2: mark completed in final checkpoint
    save_checkpoint(dataset_dir, name, {
        "last_idx":     total,
        "total_tokens": total_tokens,
        "shard_idx":    writer.shard_idx,
        "sympy_count":  sympy_count,
        "skipped":      skipped,
        "completed":    True,
    })

    print(f"  Done. {total_tokens/1e6:.0f}M tokens | "
          f"SymPy: {sympy_count} | Skipped: {skipped}")

    # Clean up metadata — no longer needed once shards are written
    if meta_dir.exists():
        import shutil
        shutil.rmtree(meta_dir)
        print(f"  Cleaned up metadata dir: {meta_dir}")

    return total_tokens


# ---------------------------------------------------------------------------
# Dataset 1: Python-Edu
# ---------------------------------------------------------------------------

def scrape_python_edu(output_dir: Path, token_budget: int, resume: bool) -> int:
    print("\n" + "=" * 60)
    print("[1/4] Python-Edu  (HuggingFaceTB/smollm-corpus)")
    print(f"      Budget: {token_budget/1e9:.2f}B tokens")
    print("=" * 60)
    return scrape_s3_dataset(
        name          = "python_edu",
        output_dir    = output_dir,
        token_budget  = token_budget,
        resume        = resume,
        repo_id       = "HuggingFaceTB/smollm-corpus",
        meta_patterns = ["python-edu/*.parquet"],
        meta_parquet_glob = "*.parquet",
    )


# ---------------------------------------------------------------------------
# Dataset 2: Stack-Edu Python
# ---------------------------------------------------------------------------

def _is_python(text: str) -> bool:
    return "def " in text or "import " in text or "class " in text


def scrape_stack_edu(output_dir: Path, token_budget: int, resume: bool) -> int:
    print("\n" + "=" * 60)
    print("[2/4] Stack-Edu Python  (HuggingFaceTB/stack-edu)")
    print(f"      Budget: {token_budget/1e9:.2f}B tokens")
    print("=" * 60)
    return scrape_s3_dataset(
        name          = "stack_edu",
        output_dir    = output_dir,
        token_budget  = token_budget,
        resume        = resume,
        repo_id       = "HuggingFaceTB/stack-edu",
        meta_patterns = ["Python/*.parquet"],
        meta_parquet_glob = "*.parquet",
        extra_filter  = _is_python,
    )


# ---------------------------------------------------------------------------
# Dataset 3: Algebraic-Stack (Python shards, streamed from HF)
# ---------------------------------------------------------------------------

def scrape_algebraic_stack(output_dir: Path, token_budget: int, resume: bool) -> int:
    print("\n" + "=" * 60)
    print("[3/4] Algebraic-Stack Python  (EleutherAI/proof-pile-2)")
    print(f"      Budget: {token_budget/1e9:.2f}B tokens")
    print("=" * 60)

    dataset_dir = output_dir / "algebraic_stack"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    ckpt = load_checkpoint(dataset_dir, "algebraic_stack") if resume else \
           {"last_idx": 0, "total_tokens": 0, "shard_idx": 0,
            "sympy_count": 0, "skipped_lang": 0, "skipped_empty": 0,
            "completed": False}

    if ckpt.get("completed", False):
        print(f"  Already completed ({ckpt['total_tokens']/1e6:.0f}M tokens) — skipping.")
        return ckpt["total_tokens"]

    skip_n        = ckpt["last_idx"]
    sympy_count   = ckpt.get("sympy_count", 0)
    skipped_lang  = ckpt.get("skipped_lang", 0)
    skipped_empty = ckpt.get("skipped_empty", 0)

    if resume and skip_n > 0:
        print(f"  Resuming from doc {skip_n:,}")

    writer = ShardWriter(dataset_dir, "algebraic_stack",
                         ckpt["shard_idx"], ckpt["total_tokens"])
    ds     = load_dataset("EleutherAI/proof-pile-2", "algebraic-stack",
                          split="train", streaming=True, trust_remote_code=True)

    seen            = 0
    docs_since_ckpt = 0

    with tqdm(desc="Algebraic-Stack", unit=" docs") as pbar:
        for example in ds:
            if writer.total_tokens >= token_budget:
                break
            if seen < skip_n:
                seen += 1
                continue
            seen += 1

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

            docs_since_ckpt += 1
            pbar.update(1)
            pbar.set_postfix(
                tokens=f"{writer.total_tokens/1e6:.0f}M",
                sympy=sympy_count,
                skipped=skipped_lang + skipped_empty,
            )

            if docs_since_ckpt >= CHECKPOINT_EVERY:
                save_checkpoint(dataset_dir, "algebraic_stack", {
                    "last_idx":      seen,
                    "total_tokens":  writer.total_tokens,
                    "shard_idx":     writer.shard_idx,
                    "sympy_count":   sympy_count,
                    "skipped_lang":  skipped_lang,
                    "skipped_empty": skipped_empty,
                    "completed":     False,
                })
                docs_since_ckpt = 0

    total_tokens = writer.finalize()
    save_checkpoint(dataset_dir, "algebraic_stack", {
        "last_idx":      seen,
        "total_tokens":  total_tokens,
        "shard_idx":     writer.shard_idx,
        "sympy_count":   sympy_count,
        "skipped_lang":  skipped_lang,
        "skipped_empty": skipped_empty,
        "completed":     True,
    })
    print(f"  Done. {total_tokens/1e6:.0f}M tokens | SymPy: {sympy_count} "
          f"| Skipped lang: {skipped_lang} | Skipped empty: {skipped_empty}")
    return total_tokens


# ---------------------------------------------------------------------------
# Dataset 4: Arxiv Math (streamed from HF)
# ---------------------------------------------------------------------------

def scrape_arxiv_math(output_dir: Path, token_budget: int, resume: bool) -> int:
    print("\n" + "=" * 60)
    print("[4/4] Arxiv Math  (EleutherAI/proof-pile-2, arxiv subset)")
    print(f"      Budget: {token_budget/1e9:.2f}B tokens")
    print("=" * 60)

    dataset_dir = output_dir / "arxiv_math"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    ckpt = load_checkpoint(dataset_dir, "arxiv_math") if resume else \
           {"last_idx": 0, "total_tokens": 0, "shard_idx": 0,
            "skipped_not_math": 0, "skipped_empty": 0, "completed": False}

    if ckpt.get("completed", False):
        print(f"  Already completed ({ckpt['total_tokens']/1e6:.0f}M tokens) — skipping.")
        return ckpt["total_tokens"]

    skip_n           = ckpt["last_idx"]
    skipped_not_math = ckpt.get("skipped_not_math", 0)
    skipped_empty    = ckpt.get("skipped_empty", 0)

    if resume and skip_n > 0:
        print(f"  Resuming from doc {skip_n:,}")

    writer = ShardWriter(dataset_dir, "arxiv_math",
                         ckpt["shard_idx"], ckpt["total_tokens"])
    ds     = load_dataset("EleutherAI/proof-pile-2", "arxiv",
                          split="train", streaming=True, trust_remote_code=True)

    seen            = 0
    docs_since_ckpt = 0

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
            docs_since_ckpt += 1
            pbar.update(1)
            pbar.set_postfix(
                tokens=f"{writer.total_tokens/1e6:.0f}M",
                skipped=skipped_not_math + skipped_empty,
            )

            if docs_since_ckpt >= CHECKPOINT_EVERY:
                save_checkpoint(dataset_dir, "arxiv_math", {
                    "last_idx":          seen,
                    "total_tokens":      writer.total_tokens,
                    "shard_idx":         writer.shard_idx,
                    "skipped_not_math":  skipped_not_math,
                    "skipped_empty":     skipped_empty,
                    "completed":         False,
                })
                docs_since_ckpt = 0

    total_tokens = writer.finalize()
    save_checkpoint(dataset_dir, "arxiv_math", {
        "last_idx":         seen,
        "total_tokens":     total_tokens,
        "shard_idx":        writer.shard_idx,
        "skipped_not_math": skipped_not_math,
        "skipped_empty":    skipped_empty,
        "completed":        True,
    })
    print(f"  Done. {total_tokens/1e6:.0f}M tokens "
          f"| Skipped (not math): {skipped_not_math} "
          f"| Skipped (empty): {skipped_empty}")
    return total_tokens


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
    parser.add_argument("--output_dir",  type=str, default="./pretraining_data")
    parser.add_argument("--max_tokens",  type=int, default=10_000_000_000,
                        help="Total approximate token budget (default 10B)")
    parser.add_argument("--datasets",    nargs="+",
                        choices=list(SCRAPERS.keys()) + ["all"],
                        default=["all"])
    parser.add_argument("--resume",      action="store_true",
                        help="Resume from last checkpoint")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    to_run  = list(SCRAPERS.keys()) if "all" in args.datasets else args.datasets
    budgets = {
        name: int(args.max_tokens * BUDGET_PROPORTIONS[name])
        for name in to_run
    }

    print("\n" + "=" * 60)
    print("Pre-training Data Scraper — Math Formalizer")
    print("=" * 60)
    print(f"Output dir      : {output_dir}")
    print(f"Total budget    : {args.max_tokens/1e9:.1f}B tokens")
    print(f"Datasets        : {', '.join(to_run)}")
    print(f"Resume          : {args.resume}")
    print(f"SymPy boost     : {SYMPY_SOFT_BOOST}")
    print(f"S3 workers      : {S3_WORKERS}  |  Batch: {S3_BATCH_SIZE}")
    print(f"Checkpoint every: {CHECKPOINT_EVERY:,} docs")
    for name in to_run:
        print(f"  {name:<25} {budgets[name]/1e9:.2f}B  "
              f"({BUDGET_PROPORTIONS[name]*100:.0f}%)")

    manifest = load_manifest(output_dir)
    totals   = {}

    for name in to_run:
        tokens       = SCRAPERS[name](output_dir, budgets[name], args.resume)
        totals[name] = tokens
        manifest["datasets"][name] = {"tokens_approx": tokens}
        save_manifest(output_dir, manifest)

    print("\n" + "=" * 60)
    print("SCRAPING COMPLETE — Summary")
    print("=" * 60)
    grand_total = sum(totals.values())
    for name, tokens in totals.items():
        pct = (tokens / grand_total * 100) if grand_total > 0 else 0
        print(f"  {name:<25} {tokens/1e6:.0f}M  ({pct:.1f}%)")
    print(f"  {'TOTAL':<25} {grand_total/1e9:.2f}B tokens")
    print("=" * 60)

    manifest["total_tokens_approx"] = grand_total
    manifest["output_dir"]          = str(output_dir)
    save_manifest(output_dir, manifest)
    print(f"\nManifest → {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()