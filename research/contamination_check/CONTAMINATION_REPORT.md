# Benchmark Contamination Check

**Date:** 2026-09-19
**Scope:** Formalizer model training pipeline (SFT + GRPO), math-reasoning master's thesis project.

## 1. Objective

The Formalizer model is fine-tuned (SFT) and further trained (GRPO) on a dataset derived
from MetaMathQA, then evaluated on five external held-out benchmarks: **GSM8K**,
**MATH-500**, **ENEM** (Brazilian national high-school exam), and **USP/FUVEST** +
**UNICAMP** (Brazilian university entrance exams, via the BLUEX dataset).

Before trusting any accuracy number reported against those benchmarks, we need to rule
out **data contamination**: training questions that are identical or near-identical to
benchmark questions, which would inflate reported performance without reflecting real
generalization. This document records the exact data sources, methodology, results, and
caveats of that check, so the check is reproducible and citable as part of the thesis's
evaluation methodology section.

## 2. Data sources

| Role | Dataset | HF repo ID | Split | Question column | Rows checked |
|---|---|---|---|---|---|
| Training corpus | MetaMathQA (sympy-augmented) | `tfshaman/metamath_sympy_v1` | `train` | `question` | 185,804 |
| Benchmark | GSM8K | `openai/gsm8k` (config `main`) | `test` | `question` | 1,319 |
| Benchmark | MATH-500 | `HuggingFaceH4/MATH-500` | `test` | `problem` | 500 |
| Benchmark | ENEM Challenge | `eduagarcia/enem_challenge` | `train`\* | `question` | 1,432 |
| Benchmark | BLUEX (USP/FUVEST + UNICAMP) | `portuguese-benchmark-datasets/BLUEX` | `questions`\* | `question` | 230 (math-tagged subset of 1,422 total) |

\* Both `eduagarcia/enem_challenge` and `portuguese-benchmark-datasets/BLUEX` only ship
their evaluation data under a split literally named `train` / `questions` respectively —
this is not a training-data split of ours, just how the dataset authors named it.

**Why compare against `question`, not `answer`:** `tfshaman/metamath_sympy_v1` (and the
scrape in `src/preprocessing/scrape_posttraining.py`) carries three relevant text
columns — `question` (the clean problem statement), `answer` (informal natural-language
chain-of-thought solution, used as the SFT/GRPO prompt), and `output` (the extracted
sympy program). Benchmark contamination is a property of the *problem statement*
matching, not the solution text, so `question` is the correct column to compare against.
(`answer`/CoT text was not checked — a leaked solution without the matching question
being memorized verbatim is a much weaker signal and out of scope here.)

**Training corpus composition** (`data_type` column, confirms standard MetaMathQA
augmentation categories, all sourced from GSM8K/MATH **train** splits per the MetaMath
paper's methodology):

| data_type | rows |
|---|---|
| GSM_AnsAug | 62,890 |
| GSM_Rephrased | 62,391 |
| MATH_AnsAug | 17,653 |
| GSM_SV | 17,432 |
| MATH_Rephrased | 12,319 |
| GSM_FOBAR | 7,829 |
| MATH_SV | 3,284 |
| MATH_FOBAR | 2,006 |

**Not checked:** Alvorada-Bench (a broader FUVEST/UNICAMP dataset found during the HF
dataset search) — BLUEX was sufficient to answer the USP/UNICAMP contamination question
as originally asked; Alvorada-Bench could be added later with the same script if it
becomes the actual eval benchmark.

## 3. Methodology

1. **Normalization** (`normalize()` in `check_contamination.py`): lowercase, strip
   everything except alphanumerics/accented Latin letters and spaces, collapse
   whitespace. Strips LaTeX/markdown punctuation noise (`$`, `\`, etc.) so formatting
   differences alone don't hide or fake a match.
2. **Exact match**: is the normalized benchmark question present verbatim (post
   normalization) in the set of normalized training questions.
3. **Near-duplicate match**: word-4-gram Jaccard similarity between the benchmark
   question and its best-matching training question.
   - **Word n-grams over character n-grams**: math problem statements are short (a
     few sentences) and mostly differ from each other in *which numbers/variables/target
     quantity* they ask about — the surrounding sentence structure is often reused
     near-verbatim within MATH in particular. Word n-grams are sensitive to exactly this
     kind of "same words, different order/values" reuse without being thrown off by
     unrelated punctuation/LaTeX noise the way character n-grams would be.
   - **n = 4, threshold = 0.6**: chosen empirically as a middle ground — low enough to
     catch near-verbatim reuse with a handful of words changed, high enough that
     generic shared math phrasing ("what is the value of", "how many ways are there
     to") doesn't trigger false positives on its own. Both are CLI flags
     (`--ngram-n`, `--threshold`); no result in this report is sensitive to small
     changes in either (spot-checked at n=3/n=5 and threshold=0.5/0.7 without
     qualitatively different findings).
   - **Blocking / candidate generation**: an exhaustive O(train_size × benchmark_size)
     comparison is intractable at 185k training rows. Instead, an inverted index maps
     each "rare" word (appearing in <5% of training rows, `--max-doc-frequency`) to the
     training row indices containing it. For each benchmark question, only training
     rows sharing at least one rare word are compared — this only misses a genuine
     near-duplicate pair if every single word two problems share is a common word
     (extremely unlikely for any real near-duplicate, since duplicated problems share
     rare content words like proper nouns, specific quantities-as-words, or distinctive
     phrasing).

Implementation: `check_contamination.py`. See §6 for exact reproduction commands.

## 4. Results

| Benchmark | Rows checked | Exact matches | Near-duplicates (Jaccard ≥ 0.6) | Near-dup % |
|---|---:|---:|---:|---:|
| GSM8K (test) | 1,319 | 0 | 0 | 0.00% |
| MATH-500 | 500 | 0 | 8 | 1.60% |
| ENEM Challenge | 1,432 | 0 | 0 | 0.00% |
| BLUEX — USP/FUVEST + UNICAMP (math subset) | 230 | 0 | 0 | 0.00% |
| BLUEX — full (all subjects, sanity check) | 1,422 | 0 | 0 | 0.00% |

Raw data: [`results/summary.json`](results/summary.json),
[`results/math500_near_duplicates.json`](results/math500_near_duplicates.json) (all 8
pairs, with Jaccard scores).

**Verdicts:**
- **GSM8K — clean.** No exact or near-duplicate matches out of 1,319 test questions.
- **ENEM — clean.** No matches. Different language and domain (Brazilian Portuguese,
  general high-school exam) from the English, grade-school/competition-math training
  corpus — no plausible contamination mechanism.
- **BLUEX (USP/UNICAMP) — clean.** No matches, checked both the math-tagged subset and
  the full multi-subject set as a sanity check. Same reasoning as ENEM: different
  language/domain.
- **MATH-500 — mild overlap, assessed as template reuse, not leakage.** 8/500 (1.6%)
  questions matched a training question at Jaccard ≥ 0.6. See §5 for the full analysis —
  the finding is a known property of the MATH dataset (train/test share problem
  templates), not evidence that our training corpus contains verbatim MATH-500 test
  items.

## 5. MATH-500 near-duplicates: detailed findings

All 8 pairs (full text in `results/math500_near_duplicates.json`) follow the same
pattern: **identical problem setup and (for geometry problems) identical Asymptote
diagram code, with a different target quantity, changed numeric literal, or a
trigonometric identity swap.** Three representative examples:

1. **Same sine-graph diagram, different target variable** (score 0.86)
   Benchmark: *"...Find the smallest possible value of $c$."*
   Training: *"...Find $d$."* — identical graph, identical Asymptote code, different
   variable asked for.

2. **Same anagram-counting structure, different word** (score 0.82)
   Benchmark: *"...arrange the letters of the word ELLIPSE."*
   Training: *"...arrange the letters of the word MADAM."* — same problem type and
   phrasing, different input word (and therefore a different combinatorics answer).

3. **Same diagram/coordinates, different region asked about** (score 0.92)
   Benchmark: *"...Determine the area of quadrilateral $DBEF$."*
   Training: *"...Determine the area of $\triangle DBC$."* — identical geometric setup
   and Asymptote diagram, different region.

**Assessment:** this is consistent with MATH's own well-documented structure — the
dataset (Hendrycks et al.) reuses problem templates, diagrams, and setups across many
variants, some landing in `train` and some in `test`. Since MetaMathQA's `MATH_AnsAug` /
`MATH_Rephrased` / `MATH_SV` / `MATH_FOBAR` rows are built from MATH's `train` split, a
training row sharing 60-90% of its wording with a MATH-500 (`test`) question is exactly
what you'd expect from that pre-existing template reuse *within the MATH dataset itself*
— it is not evidence that MetaMathQA (or this project's scrape of it) pulled in `test`
items. None of the 8 pairs are the *same* problem (same target quantity + same numeric
literal); all differ in the one detail that actually determines the answer.

**Recommendation:** treat MATH-500 accuracy as valid for reporting, but disclose this
finding in the thesis's evaluation/limitations section. If stricter isolation is wanted,
the 8 flagged training rows could be excluded from the training set (their row indices
in the training corpus are recoverable by re-running the check — not persisted here
since only the matched pairs, not full row indices, were saved by design; see §7).

## 6. Caveats

- **Automated contamination detection is imperfect.** Public discussion around
  MetaMathQA specifically notes that automated contamination-detection tools have
  flagged possible GSM8K overlap before, while the dataset authors maintain (and this
  check corroborates for GSM8K specifically, at 0 exact/near-dup matches out of 1,319)
  that only the GSM8K/MATH **train** splits were used. No single detection method
  (including the n-gram approach used here) is a proof of absence — only a level of
  confidence bounded by the method's sensitivity (n-gram size, threshold, normalization
  choices).
- **"Rephrased" augmentation is contamination-adjacent even when train-sourced.** A
  chunk of the training corpus (`GSM_Rephrased`, `MATH_Rephrased`, ~74.7k rows) consists
  of LLM-rephrased versions of GSM8K/MATH train questions. This is legitimate
  data-augmentation, not contamination, by construction (it's sourced from `train`, not
  `test`) — but it does mean the model has seen many stylistic variants of "how GSM8K/MATH
  problems are phrased," which should be kept in mind when interpreting *why* the model
  performs well on GSM8K/MATH-family benchmarks specifically vs. the Brazilian exams (a
  genuinely distinct distribution).
- **Question-only comparison.** This check only compares problem statements. It does not
  check whether the *solution/answer* text or numeric answer alone was memorized without
  the full question matching (a much weaker and harder-to-define contamination signal,
  considered out of scope here).
- **Threshold choice is a judgment call**, documented and parameterized (§3) rather than
  derived from a formal calibration — appropriate for a screening check, worth
  mentioning as a limitation if cited directly as a proof of non-contamination.

## 7. How to reproduce

Requires the `datasets` package (already a project dependency — see root
`pyproject.toml`) and network access to the Hugging Face Hub on first run (datasets are
cached locally under `~/.cache/huggingface` afterwards).

```bash
# from the repo root
uv run python research/contamination_check/check_contamination.py

# re-run with different near-duplicate sensitivity
uv run python research/contamination_check/check_contamination.py --ngram-n 5 --threshold 0.7

# check only specific benchmarks
uv run python research/contamination_check/check_contamination.py --benchmarks gsm8k math500
```

Outputs are written to `results/`: `summary.json` (per-benchmark counts) and one
`<benchmark>_near_duplicates.json` per benchmark with any near-duplicate hits (only
`math500_near_duplicates.json` exists currently, since it's the only benchmark with
hits above threshold). Neither the downloaded training corpus nor the benchmark
datasets themselves are written into the repo — only these derived result files.
