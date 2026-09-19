# Contamination Check

Checks whether the Formalizer's SFT/GRPO training corpus (`tfshaman/metamath_sympy_v1`,
a MetaMathQA-derived dataset) contains questions that overlap with the benchmarks used
for final evaluation: GSM8K, MATH-500, ENEM, and USP/FUVEST + UNICAMP (via BLUEX). See
[`CONTAMINATION_REPORT.md`](CONTAMINATION_REPORT.md) for the full methodology, results, and analysis — it's the
main document here and is written to be citable directly in the thesis.

To reproduce: `uv run python check_contamination.py` from this directory (or from the
repo root with the full path). Requires the `datasets` package (already a project
dependency) and network access on first run. Results land in `results/`.
