"""
Benchmark contamination check for the Formalizer's training data.

Checks whether the SFT/GRPO training corpus (a MetaMathQA-derived dataset) contains
questions that are exact or near-duplicate matches of questions in the benchmarks used
for final held-out evaluation of the trained model.

Method (see CONTAMINATION_REPORT.md for the full writeup and results):
  1. Normalize question text (lowercase, strip punctuation, collapse whitespace).
  2. Exact match on the normalized string.
  3. Near-duplicate match via word-4-gram Jaccard similarity, restricted to candidate
     pairs sharing at least one "rare" word (an inverted index over words that appear
     in <5% of the training corpus) so the O(N*M) comparison stays tractable at
     N ~ 185k training rows.

Usage:
    uv run python research/contamination_check/check_contamination.py
    uv run python research/contamination_check/check_contamination.py --ngram-n 5 --threshold 0.7
    uv run python research/contamination_check/check_contamination.py --benchmarks gsm8k math500

Requires the `datasets` package (already a project dependency, see pyproject.toml) and
network access to Hugging Face Hub on first run (datasets are cached locally after that).
"""

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from datasets import load_dataset

# ---------------------------------------------------------------------------
# Training corpus + benchmark registry
# ---------------------------------------------------------------------------

TRAIN_DATASET_ID = "tfshaman/metamath_sympy_v1"
TRAIN_SPLIT = "train"
TRAIN_QUESTION_COLUMN = "question"  # the clean question text, not the CoT `answer` column


@dataclass
class BenchmarkSpec:
    name: str
    repo_id: str
    split: str
    question_column: str
    # optional filter applied to each row (e.g. restrict BLUEX to math-tagged questions);
    # receives the full row dict, returns True to keep it
    row_filter: Optional[Callable[[dict], bool]] = None
    notes: str = ""


def _bluex_is_math(row: dict) -> bool:
    subjects = row.get("subject") or []
    return any("math" in s.lower() or "matemática" in s.lower() or "matematica" in s.lower() for s in subjects)


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "gsm8k": BenchmarkSpec(
        name="GSM8K (test)",
        repo_id="openai/gsm8k",
        split="test",
        question_column="question",
        notes="'main' config passed via load_dataset config arg, handled specially below.",
    ),
    "math500": BenchmarkSpec(
        name="MATH-500",
        repo_id="HuggingFaceH4/MATH-500",
        split="test",
        question_column="problem",
    ),
    "enem": BenchmarkSpec(
        name="ENEM Challenge",
        repo_id="eduagarcia/enem_challenge",
        split="train",
        question_column="question",
        notes="Portuguese-language, all subjects (not math-only); dataset only ships a 'train' split.",
    ),
    "bluex": BenchmarkSpec(
        name="BLUEX (USP/FUVEST + UNICAMP)",
        repo_id="portuguese-benchmark-datasets/BLUEX",
        split="questions",
        question_column="question",
        row_filter=_bluex_is_math,
        notes="Filtered to rows whose 'subject' list contains a math tag; split is named 'questions', not 'train'.",
    ),
}

# GSM8K requires an explicit config name ("main") in addition to the split.
_LOAD_DATASET_CONFIG: dict[str, str] = {
    "gsm8k": "main",
}


# ---------------------------------------------------------------------------
# Text normalization + near-duplicate detection
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9À-ÿ ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def build_rare_word_index(train_norms: list[str], max_doc_frequency: float = 0.05) -> dict[str, set[int]]:
    """Inverted index word -> {row indices}, restricted to words appearing in fewer
    than `max_doc_frequency` of training rows. Used to cheaply find near-duplicate
    candidates without an O(N*M) full comparison: two questions about "similar
    triangles" share only common words and would never become candidates, but two
    questions sharing a rare proper noun / distinctive phrase will."""
    doc_count: dict[str, int] = defaultdict(int)
    for t in train_norms:
        for w in set(t.split()):
            doc_count[w] += 1

    limit = max(1, int(max_doc_frequency * len(train_norms)))
    index: dict[str, set[int]] = defaultdict(set)
    for i, t in enumerate(train_norms):
        for w in set(t.split()):
            if len(w) > 3 and doc_count[w] <= limit:
                index[w].add(i)
    return index


def best_match(
    query_norm: str,
    train_norms: list[str],
    index: dict[str, set[int]],
    ngram_n: int,
    max_candidates: int = 3000,
) -> tuple[Optional[int], float]:
    words = query_norm.split()
    candidates: set[int] = set()
    for w in words:
        if w in index:
            candidates |= index[w]
        if len(candidates) > max_candidates:
            break
    if not candidates:
        return None, 0.0

    query_ngrams = word_ngrams(words, ngram_n)
    if not query_ngrams:
        return None, 0.0

    best_i, best_score = None, 0.0
    for i in candidates:
        train_ngrams = word_ngrams(train_norms[i].split(), ngram_n)
        if not train_ngrams:
            continue
        inter = len(query_ngrams & train_ngrams)
        union = len(query_ngrams | train_ngrams)
        score = inter / union if union else 0.0
        if score > best_score:
            best_score = score
            best_i = i
    return best_i, best_score


# ---------------------------------------------------------------------------
# Per-benchmark check
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    key: str
    name: str
    total: int
    exact_matches: int
    near_duplicates: list[dict] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def near_duplicate_count(self) -> int:
        return len(self.near_duplicates)

    def to_summary_row(self) -> dict:
        if self.error:
            return {"benchmark": self.name, "total": None, "exact_matches": None,
                     "near_duplicates": None, "near_duplicate_pct": None, "error": self.error}
        pct = 100 * self.near_duplicate_count / self.total if self.total else 0.0
        return {
            "benchmark": self.name,
            "total": self.total,
            "exact_matches": self.exact_matches,
            "near_duplicates": self.near_duplicate_count,
            "near_duplicate_pct": round(pct, 2),
            "error": None,
        }


def check_benchmark(
    key: str,
    spec: BenchmarkSpec,
    train_questions: list[str],
    train_norms: list[str],
    train_exact_set: set[str],
    index: dict[str, set[int]],
    ngram_n: int,
    threshold: float,
) -> BenchmarkResult:
    config = _LOAD_DATASET_CONFIG.get(key)
    try:
        ds = load_dataset(spec.repo_id, config, split=spec.split) if config else load_dataset(spec.repo_id, split=spec.split)
    except Exception as e:  # noqa: BLE001 - report load failures as part of results, don't crash the run
        return BenchmarkResult(key=key, name=spec.name, total=0, exact_matches=0, error=str(e))

    rows = ds.to_list() if spec.row_filter else None
    if spec.row_filter:
        questions = [row[spec.question_column] for row in rows if spec.row_filter(row)]
    else:
        questions = ds[spec.question_column]

    exact_matches = 0
    near_duplicates = []
    for q in questions:
        qn = normalize(q)
        if qn in train_exact_set:
            exact_matches += 1
            continue
        match_idx, score = best_match(qn, train_norms, index, ngram_n)
        if match_idx is not None and score >= threshold:
            near_duplicates.append({
                "benchmark_question": q,
                "train_question": train_questions[match_idx],
                "jaccard_score": round(score, 4),
            })

    return BenchmarkResult(
        key=key, name=spec.name, total=len(questions),
        exact_matches=exact_matches, near_duplicates=near_duplicates,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-dataset", default=TRAIN_DATASET_ID)
    parser.add_argument("--train-split", default=TRAIN_SPLIT)
    parser.add_argument("--train-column", default=TRAIN_QUESTION_COLUMN)
    parser.add_argument("--benchmarks", nargs="+", choices=list(BENCHMARKS.keys()), default=list(BENCHMARKS.keys()),
                         help="Which benchmarks to check (default: all)")
    parser.add_argument("--ngram-n", type=int, default=4, help="Word n-gram size for Jaccard similarity")
    parser.add_argument("--threshold", type=float, default=0.6, help="Jaccard similarity threshold to flag a near-duplicate")
    parser.add_argument("--max-doc-frequency", type=float, default=0.05,
                         help="Words appearing in more than this fraction of training rows are excluded from the blocking index")
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "results"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading training corpus: {args.train_dataset} [{args.train_split}] ...")
    train_ds = load_dataset(args.train_dataset, split=args.train_split)
    train_questions = train_ds[args.train_column]
    train_norms = [normalize(q) for q in train_questions]
    train_exact_set = set(train_norms)
    print(f"  {len(train_norms):,} training questions loaded")

    print("Building inverted (rare-word) index for near-duplicate candidate lookup ...")
    index = build_rare_word_index(train_norms, max_doc_frequency=args.max_doc_frequency)

    summary_rows = []
    for key in args.benchmarks:
        spec = BENCHMARKS[key]
        print(f"\nChecking {spec.name} ({spec.repo_id}, split={spec.split}) ...")
        result = check_benchmark(
            key, spec, train_questions, train_norms, train_exact_set, index,
            ngram_n=args.ngram_n, threshold=args.threshold,
        )
        summary_rows.append(result.to_summary_row())

        if result.error:
            print(f"  FAILED TO LOAD: {result.error}")
            continue

        print(f"  total={result.total}  exact={result.exact_matches}  "
              f"near_dup(jaccard>={args.threshold})={result.near_duplicate_count}")

        if result.near_duplicates:
            out_path = output_dir / f"{key}_near_duplicates.json"
            with open(out_path, "w") as f:
                json.dump(result.near_duplicates, f, indent=2, ensure_ascii=False)
            print(f"  wrote {len(result.near_duplicates)} near-duplicate pairs -> {out_path}")

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "train_dataset": args.train_dataset,
            "train_split": args.train_split,
            "train_rows": len(train_norms),
            "ngram_n": args.ngram_n,
            "jaccard_threshold": args.threshold,
            "results": summary_rows,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nWrote summary -> {summary_path}")


if __name__ == "__main__":
    main()
