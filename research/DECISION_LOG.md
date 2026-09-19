# Decision Log

A running record of the scientific and engineering decisions behind the Formalizer
model (informal natural-language math solutions → executable sympy code) and its
post-training pipeline (SFT + GRPO). The goal is to make the *why* behind each
direction — including things that were tried, rejected, or deferred — traceable for
the thesis write-up, not just the *what* (which git history already covers).

Entries are chronological (oldest first). Each entry is written close to when the
decision was actually made. Entries dated before this log existed (2026-09-19) are
reconstructed from `TODO.md` and commit messages, so they're necessarily thinner than
entries written contemporaneously from here on.

---

## 2026-07-22 — SFT prompt format and masking

**Context.** `SFTFormalizerDataset` (`src/trainer/sft.py`) needed to turn
(natural-language answer, sympy program) pairs into single sequences the decoder-only
`Transformer` could train on — the model expects one `idx` + a same-shape, pre-shifted
`targets`, with a flat per-position cross-entropy and no `ignore_index` support at the
time.

**Decisions:**
- Format each example as `<|bos|> <|user|> {answer} <|assistant|> {sympy} <|endoftext|>`,
  reusing the `<|user|>`/`<|assistant|>` special tokens that already existed in the
  tokenizer but had never been wired into any training example.
- Build the loss mask anchored on the `<|assistant|>` token position: mask everything at
  or before it (the prompt) plus trailing padding, keep the completion live. Masked
  positions get `target = -100`, and `F.cross_entropy` gained an `ignore_index=-100`.
- Truncation policy: never truncate away the completion. Rows whose completion alone
  (`<|assistant|> {sympy}`) doesn't fit in `context_size` are excluded from the
  train/val shuffle pool entirely, rather than risking a truncation that cuts the
  `<|assistant|>` anchor itself.

**Also fixed in this pass:** `SFTFormalizerDataset.__getitem__` returning a 3-tuple
(`query, sympy, code_output`) while the training loop unpacked 2 values; a missing
`split` arg in the `__main__` smoke test.

*(Reconstructed from `TODO.md`'s 2026-07-22 entry — see git history for the actual diffs.)*

---

## 2026-09-15 — GRPO scaffolding lands, reward function identified as the blocker

**Context.** With SFT training working, the GRPO stage (`src/trainer/grpo.py`,
`src/trainer/grpo_math.py`) was built out: `GRPOPromptDataset` (prompt-only, per-row
metadata carried through to reward scoring), the clipped-surrogate + KL-penalty loss,
group-relative advantage normalization, and the full rollout/train loop.

**Decision.** `RewardFn` (`src/rewards/base.py`) was deliberately left as just a type
alias — `(prompt_text, completion_text, metadata) -> float` — with no concrete
implementation and no base class, since it's the one piece of the pipeline meant to be
swapped freely per task. `GRPOTrainer` is fully wired and ready to train, but cannot
actually run until a real reward function exists. Logged as the "must do" item in
`TODO.md`.

---

## 2026-09-19 — Found and fixed a real bug: SFT completions never import sympy

**Symptom (reported by the user, not caught by any test).** The SFT-tuned model
generates sympy code using `sp.symbols(...)` etc. but never emits `import sympy as sp`
— the generated code doesn't run.

**Investigation.** Loaded the actual scraped shard
(`data/posttraining/metamath_sympy/metamath_sympy_shard_0000.parquet`) and checked: only
25/500 rows (5%) contain `import sympy as sp` anywhere in the `output` column; the rest
jump straight into `sp.symbols(...)`. This traces to the upstream source dataset
(`tfshaman/metamath_sympy_v1`, itself a MetaMathQA derivative) being inconsistent about
including the import line in its `[|Sympy|]` blocks — not to anything
`scrape_posttraining.py` was stripping out. Also confirmed no alternate import styles
(e.g. `from sympy import ...`) exist in the source data, so a single canonical-line fix
is safe.

**Decision: fix at the data-prep layer (`src/preprocessing/scrape_posttraining.py`),
not in the GRPO reward function.** `clean_sympy_output()` now prepends
`import sympy as sp` to the extracted code block when it's missing, before the
`<|endoftext|>` terminator is appended (idempotent — doesn't duplicate the import if
already present). Confirmed via inspection that `src/trainer/grpo.py`/`grpo_math.py`
never read the `output` column at all (GRPO only uses `answer` as the prompt and
`code_output` for reward scoring), so this fix only affects SFT and needed no GRPO-side
change.

**Decision: don't patch the already-scraped 500-row shard in place.** The user runs
training on a separate SSH machine with the full dataset; re-running
`make scrape-metamath-sympy` there (pulling the fix) was simpler and more correct than
maintaining a patched-in-place copy that could drift from what the scraper would now
produce from scratch.

**Decision: version-bump the SFT output path and mlflow experiment (v2 → v3) instead of
overwriting.** `SFTTrainer.train()` always fine-tunes fresh from the pretrained base
checkpoint (never resumes from a prior SFT checkpoint), so re-running with corrected
data doesn't need a "resume" step — but `scripts/train_sft.py`'s `FINAL_MODEL_PATH` and
the `mlflow.set_experiment(...)` name were still pointing at the same
`models/sft/formalizer_v2/` directory and `Formalizer_Finetuning_v2` experiment as the
buggy run. Since `resume_from_checkpoint()` picks "latest checkpoint by step number" in
that directory, leaving the path unchanged risked a fresh (correct-data) checkpoint
sitting next to a stale (buggy-data) one and being ambiguous later. Bumped both to `v3`,
following the same pattern already used once before for a similar reason
(`fef7e75 chore: change experiment name for v2 training`). This also has the side
benefit of keeping the v2 (no-import) model around as a baseline for comparison.

**Status:** SFT-v3 training kicked off on the remote SSH machine with the corrected
data (in progress as of this entry).

---

## 2026-09-19 — Evaluation methodology: no harness exists yet, deferred building a new benchmark

**Finding.** There is currently no quantitative evaluation of the Formalizer model at
all — `main.py`'s only "eval" is a single hardcoded prompt qualitatively comparing base
vs. SFT output. No held-out test split has ever been carved out: both
`SFTFormalizerDataset` and `GRPOPromptDataset` split off `train_size + val_size` rows
and silently discard the remainder (already flagged in `TODO.md`: *"Revisit GRPO's
remainder split — still nothing computed/persisted for it"*).

**Decision: the right eval metric is execution rate + numeric-match accuracy**, in the
program-aided-reasoning tradition (PAL / Program-of-Thoughts) — not a novel choice, and
directly reusable as an evaluation harness once a held-out `test_indices.npy` is
persisted (not yet done).

**Caveat surfaced and accepted:** numeric match alone has a false-positive risk (right
answer for the wrong/degenerate reason). Worth complementing with a small,
manually-inspected sample rather than trusting the automatic metric alone.

**Decision: don't build a new benchmark right now.** The user considered creating a
dedicated benchmark for "formalizing already-solved informal math CoT into executable
sympy" specifically (there isn't an existing one for this exact task). Assessed as a
defensible thesis contribution in principle, but the biggest scope-creep risk on the
table given the reward function / GRPO loop wasn't working yet. Deferred to a smaller
(~50-100 problem) hand-verified diagnostic set, to be built later as a complement to
the held-out metamath split — not attempted yet.

---

## 2026-09-19 — Benchmark contamination check

Full methodology, script, and results saved separately at
[`research/contamination_check/`](contamination_check/) (see
`CONTAMINATION_REPORT.md`). Summary: before trusting final accuracy numbers against
GSM8K, MATH-500, ENEM, and BLUEX (USP/FUVEST + UNICAMP), checked whether
`tfshaman/metamath_sympy_v1`'s training questions leak into any of those benchmarks'
test questions (exact match + word-4-gram Jaccard near-duplicate detection, full
185,804-row train split vs. each benchmark).

**Result:** GSM8K, ENEM, and BLUEX come back clean (0 matches). MATH-500 shows 8/500
(1.6%) near-duplicates — inspected and assessed as MATH's own well-documented train/test
template reuse (same diagram/problem setup, different target variable or numeric
literal), not evidence of corpus leakage. Recommendation: report MATH-500 accuracy as
valid, disclose the finding in the thesis's limitations section.

---

## 2026-09-19 — GRPO reward function design

**Starting point.** `RewardFn` needed a concrete implementation:
`(prompt_text, completion_text, metadata) -> float`, where `metadata` carries the
dataset row's `code_output` (ground-truth numeric answer).

**Considered: multi-objective reward via MO-GRPO.** The user proposed combining two
signals — "does the code execute" and "is the answer correct" — inspired by
DeepSeek-R1's `accuracy_reward + format_reward`, and specifically wanted to try
**MO-GRPO** (Ichihara et al., *"Mitigating Reward Hacking of Group Relative Policy
Optimization on Multi-Objective Problems,"* arXiv:2509.22047 — verified as a real paper
via web search after an initial ASR mis-transcription of the title as "Group Loyalty").
MO-GRPO's fix for combining multiple reward objectives: normalize each objective's
advantage separately within the group (by that objective's own mean/std) before
summing, so no objective dominates the policy gradient purely by having higher raw
variance.

**Nuance flagged, not yet resolved either way:** "executes" and "correct" aren't
independent/competing objectives the way MO-GRPO's benchmark domains are (bandits,
translation adequacy-vs-fluency) — they're hierarchical (`correct ⊆ executes`, you can't
be correct without executing). Whether MO-GRPO's variance-equalization machinery earns
its keep on a *nested* objective pair, versus a plain shaped/summed reward already
getting the same effect for free, is an open question — flagged as a potential thesis
angle in its own right (testing the paper's method outside the setting it was designed
for) rather than assumed away.

**Correction made along the way:** the process-reward formula the user initially quoted
(per-step rewards `r_index(j)_i` indexed by reasoning-step boundaries, normalized by
group mean/std) is from the original GRPO/DeepSeekMath paper's **process supervision**
setting (a PRM scoring each intermediate step of one output) — a different concept from
DeepSeek-R1's simpler **outcome-level** `accuracy_reward + format_reward` sum, which is
what the user actually wanted to replicate. Verified DeepSeek-R1's actual formulation via
its paper (arXiv:2501.12948, §2.3.1) before proceeding, rather than trusting recall.
Building a real PRM would need step-level ground-truth labels that don't exist for this
task — not pursued.

**Decision: implement the vanilla (non-MO-GRPO) version now**, deferring the
multi-objective treatment. Two-part reasoning:
1. Getting GRPO training running at all (currently fully blocked with no reward
   function) is more urgent than getting the reward-combination scheme right on the
   first attempt.
2. A vanilla summed reward is itself a legitimate experimental condition — the natural
   "does MO-GRPO's normalization actually help here" comparison needs a naive baseline
   to compare against anyway.

**Execution sandbox: in-process `exec()`, not subprocess.** Chosen for performance —
GRPO rollouts execute `group_size × batch_size` generated programs per training step,
and subprocess-per-sample would re-pay Python-interpreter-plus-sympy-import startup
cost on every single one. Mitigated the safety cost of in-process execution with: a
builtins allowlist (no file/network/process access, no `eval`/`exec`/`compile`/introspection
builtins), a custom restricted `__import__` that only permits `sympy` (blocks
`import os` etc. while still letting the model's own `import sympy as sp` line succeed),
and a 5-second wall-clock timeout via `SIGALRM` to kill infinite loops. Explicitly scoped
as "safe enough for scoring your own model's own rollouts on your own machine," not a
general-purpose untrusted-code sandbox.

**Correctness check: parse-and-compare with tolerance, not exact string match.** The
last non-empty line of captured stdout is parsed via `sympy.sympify(...).evalf()` (not
bare `float()`) so answers like `144/2` or an unevaluated sympy expression still resolve
correctly, then compared to `metadata["code_output"]` via `math.isclose` (relative
tolerance `1e-4`, absolute `1e-6`). Rejected plain string-equality because sympy/float
formatting differences (`72` vs `72.0` vs a fraction) would otherwise produce false
negatives on answers that are actually correct.

**Interface: single combined float, not a dict of components.** First implementation
had `sympy_reward()` return `{"executes": ..., "correct": ...}` to keep the door open
for MO-GRPO later. Simplified to a single summed float
(0 = doesn't run, 1 = runs but wrong, 2 = runs and correct) after the user pointed out
nothing currently consumes the per-component values — the dict was premature
abstraction for a "vanilla" reward with no MO-GRPO consumer yet. Conforms directly to
the existing `RewardFn` type; `GRPOTrainer`/`grpo_math.py` needed no changes at all.

**Verified by hand** (`src/rewards/sympy_execution.py`): correct-answer case, wrong-answer
case, syntax error, runtime error (undefined name), infinite loop (confirmed ~5s
timeout), blocked `import os`, fraction-valued printed output, and no state leakage
between repeated calls to the same executor.

**Still open / next step:** wire `sympy_reward` into an actual GRPO training script
(mirroring `scripts/train_sft.py`), and later — once vanilla GRPO training is validated
— revisit the MO-GRPO comparison as its own experimental arm.
