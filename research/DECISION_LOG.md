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

---

## 2026-09-19 — Full rollout traceability in MLflow for GRPO

**Motivation.** GRPO is much more opaque to debug than SFT: a low reward could mean the
code crashed, timed out, ran but got the wrong answer, or something else entirely, and
none of that is visible from a scalar loss/reward curve alone. The user wanted full
traceability — for every generated completion, whether it executed, what error it hit if
not, whether the answer was correct, and the raw generated text — logged to MLflow so
training failures are diagnosable after the fact, not just observable as "reward is low."

**Decision: derive diagnostics from the same single execution already used for the
reward, not a second one.** `sympy_reward()` (the plain `RewardFn`) is untouched — same
signature, same behavior, same one execution per completion. A new `SympyRewardFn`
class wraps the identical scoring logic (`_score_completion`, shared by both) but keeps
the full `ScoredCompletion` result around as `last_diagnostics` after each call. Chosen
over changing `RewardFn`'s return type to a dict/tuple (which would have meant touching
`GRPOTrainer`'s loss-relevant code path just to carry logging metadata through it) —
`GRPOTrainer` reads `last_diagnostics` via `getattr(self.reward_fn, "last_diagnostics",
None)` right after calling the reward function, so a plain float-returning `RewardFn`
still works with zero changes, and only reward functions that opt into traceability pay
for it.

**Decisions on logging volume** (both explicitly chosen over the more expensive
alternative, to keep MLflow storage from ballooning over a full training run):
- **Cadence: only at eval steps** (reusing the existing `eval_interval` cadence already
  used for `val_reward_mean` etc.), not every single training step. A step generates
  `batch_size × group_size` completions; logging full text every step across thousands
  of steps was judged not worth the storage for the added granularity.
- **Scope: log every completion, correct or not** — not just failures. Rejected
  logging only incorrect/failed completions (which would have matched the literal
  ask more narrowly) because seeing *successful* completions matters too, e.g. to catch
  the model finding a degenerate way to get the right number.

**Implementation:** `mlflow.log_table(artifact_file="rollout_traceability.json")` (appends
rows across repeated calls within a run) with one row per completion — `step`, `split`
(train/val), `prompt`, and every `ScoredCompletion` field. Cheap aggregate metrics
(`train/val_execute_rate`, `train/val_correct_rate`) are logged every eval step
regardless, alongside the existing reward metrics, so the trend line survives even
without opening the full table.

---

## 2026-09-19 — GRPO training script and hyperparameter sanity check

**Script.** `scripts/train_grpo.py`, mirroring `scripts/train_sft.py`'s structure:
loads the tokenizer/vocab metadata, builds a `GRPOConfig`, constructs two independent
`Transformer` instances from the same starting checkpoint (the trained policy and the
frozen reference model used for the KL penalty — separate instances so training the
policy can't affect the reference), wires up `SympyRewardFn`, and hands everything to
`GRPOTrainer`. Also added `make grpo-formalizer`, mirroring the existing
`sft-formalizer` target.

**Decision: start GRPO from `final_model.pt`, not `best_model.pt`.** SFT
(`scripts/train_sft.py`) was deliberately trained to overfit (12k steps, "force
overfit" — see the 2026-07-22 and earlier entries), following the InstructGPT paper's
finding that a later, overfit SFT checkpoint outperforms the lowest-validation-loss one
as the starting point for downstream RL. `best_model.pt` tracks lowest val loss and
would most likely pick an early, pre-overfit checkpoint here — the wrong one for this
project's chosen approach. `final_model.pt` (the actual last-step checkpoint) is correct.

**Caution flagged and respected: do not run this script against the local corpus.**
The corpus present on this laptop is still the 500-row smoke-test scrape.
`GRPOPromptDataset` persists its train/val split indices with the same "don't
overwrite if exists" guard as `SFTFormalizerDataset` — running the script here first
would silently poison the split for when the full corpus is scraped on the SSH
machine (the exact footgun already identified for SFT earlier in this log). The script
was written and sanity-checked (`py_compile` only) without ever being imported or
executed, specifically to avoid triggering `GRPOPromptDataset`'s module-level dataset
construction against the wrong data.

**Hyperparameter sanity check — one real bug caught, one judgment call revised:**

- **`max_new_tokens`: 256 → 768 (bug, caught empirically, not by inspection).**
  Tokenized the actual scraped shard's `output` column and measured completion
  length: median 346, p95 647, max 1094 tokens. A 256-token budget would have
  truncated 78% of completions before they ever reached `print()`/EOS — meaning the
  vast majority of rollouts would score reward=0 for running out of token budget, not
  for being mathematically wrong, making the reward signal nearly uninformative for
  the policy gradient despite the pipeline appearing to "work." 768 leaves only ~1.8%
  truncated, and still leaves 256 tokens of prompt budget (`context_size` 1024 - 768),
  comfortably above the prompt side's own p95 (226 tokens, measured the same way).
  This is the kind of bug that wouldn't show up as a crash or an obviously-bad metric
  early on — worth remembering to sanity-check length budgets against actual data
  length distributions, not round numbers, on any future context-window-constrained
  pipeline change.
- **`learning_rate`: 1e-5 → 1e-6.** Initially picked 1e-5 (an order of magnitude below
  SFT's 3e-5) without a specific empirical basis. Revised down to match DeepSeekMath's
  own published GRPO actor LR exactly (1e-6), on the reasoning that RL updates
  compound more riskily than SFT's (via the KL penalty and future rollouts building on
  a bad update), and there's no project-specific data yet to justify deviating from
  the reference implementation's value.
- **Left as first-run starting points, explicitly not tuned:** `group_size=8`
  (DeepSeekMath uses 64, but that's large-scale — 8 is a compute-constrained starting
  point, larger groups would give a better advantage estimate), `batch_size=4`
  (untested against real GPU step-time on the SSH machine), `max_iters=1_000` (at
  batch_size=4, only ~27% of one epoch over the default 15k-row train split — likely
  enough to validate the pipeline runs at all, likely not enough for real
  convergence). `top_p=0.9`/`temperature=0.8` kept consistent with the values already
  used elsewhere in the repo (`main.py`'s generation config). `kl_coef=0.04`,
  `clip_epsilon=0.2`, `num_inner_epochs=1`, `advantage_eps=1e-4` left at `GRPOConfig`'s
  defaults, which are themselves taken from the DeepSeekMath paper.

**Next step:** run this on the SSH machine once SFT-v3 finishes and the full corpus is
confirmed scraped there, treat it as a first smoke test of the pipeline mechanics
(does it run end-to-end, does the reward signal look sane in the MLflow traceability
table) rather than a real training run, then revisit `max_iters`/`group_size` once
real throughput numbers exist.

---

## 2026-09-19 — Coordinated SFT/GRPO/benchmark split, and correcting an RL-data-size assumption

**Problem found.** `SFTFormalizerDataset` and `GRPOPromptDataset` each independently
shuffle and persist their own train/val split — different seeds, different pools,
zero awareness of each other. Nothing has ever reserved a held-out benchmark set at
all (the "still nothing computed/persisted for it" gap flagged in `TODO.md` back on
2026-07-22, still unresolved). Left as-is, a row used in SFT training could just as
easily land in GRPO's split, or worse, in whatever set eventually gets used to report
final benchmark numbers — silently invalidating exactly the kind of eval integrity
the contamination check earlier today was trying to establish for the *external*
benchmarks (GSM8K/MATH-500/ENEM/BLUEX). This is the same class of problem, just
internal to our own corpus.

**Complication:** SFT-v3 was already training on the SSH machine at this point, using
a split it had already committed to disk (persisted the moment
`SFTFormalizerDataset` is constructed, before training even starts) — computed under
the old, uncoordinated, per-class scheme. Two options: (a) treat SFT's already-fixed
split as immutable and carve GRPO's split plus a benchmark holdout only out of
whatever it didn't touch, or (b) stop the in-progress run and regenerate all splits
together from scratch. **Decision: (b).** The user opted to stop training rather than
retrofit around a split that was never designed with a benchmark holdout in mind —
cleaner to have one coordinated source of truth than to carry the old scheme's gap
forward indefinitely.

**Implementation:** `src/preprocessing/split_dataset.py` (`make split-dataset`) —
a single script that must run once, after `scrape-metamath-sympy` and before either
`sft-formalizer` or `grpo-formalizer`. It shuffles the whole eligible pool exactly
once (same eligibility filter as `SFTFormalizerDataset`'s own — a row whose
`<|assistant|> {sympy}` completion alone exceeds `context_size` is excluded from
every split, not just SFT's, so there's one single universe of "usable" rows project-
wide) and slices off, in order: the benchmark holdout first (protected regardless of
how SFT/GRPO's own sizes change later), then SFT train/val, then GRPO train/val.
Disjointness across all five resulting sets is asserted programmatically before
anything is written, not just assumed from the slicing arithmetic. Both
`SFTFormalizerDataset` and `GRPOPromptDataset` needed zero code changes — both already
had "compute-and-persist only if the file doesn't exist yet, otherwise just load it"
logic, so pre-writing the files here means they simply load this script's output.

**Split sizes chosen: `sft_train=15000, sft_val=3000, grpo_train=15000, grpo_val=3000,
benchmark_holdout=2000`** — i.e., GRPO's split kept the same scale as SFT's. This
followed a literature check that changed *why* that's the right call, not the number
itself:

- **Initial framing (revised): "GRPO/RL needs less data because it averages over a
  group" — checked and rejected.** Group averaging (GRPO's group-relative advantage
  estimate) is about how many *samples per prompt* you need for a low-variance
  baseline (DeepSeekMath uses `group_size=64` — see below), not about how many
  *unique prompts* your training set needs. These are different axes; no claim in the
  paper ties the former to needing fewer of the latter.
- **InstructGPT precedent points the other way:** SFT used ~13k prompts, PPO used
  ~31k (arXiv:2203.02155) — the RL phase used ~2.4x *more* unique prompts than SFT,
  not fewer.
- **DeepSeekMath's actual RL-stage subset, quoted directly by the user:** *"The
  training data of RL are chain-of-thought-format questions related to GSM8K and MATH
  from the SFT data, which consists of around 144K questions. We exclude other SFT
  questions to investigate the impact of RL on benchmarks that lack data throughout
  the RL phase."* This is a **domain restriction** (keep only GSM8K/MATH-CoT
  questions from a broader, multi-domain SFT mix; deliberately exclude other SFT
  domains to test whether RL gains transfer to benchmarks the RL phase never touched),
  not a quantity-reduction decision, and not "less data because of averaging" either.
  It doesn't map onto our corpus cleanly: `metamath_sympy` is already single-domain —
  there's no "other SFT domain" for GRPO to exclude the way DeepSeekMath excluded
  non-math domains, so this precedent is closer to *not directly applicable* to our
  split-sizing question than to supporting either a bigger or smaller GRPO split.
- **Net conclusion:** neither paper's RL-vs-SFT data-size ratio transfers cleanly to
  this project (different task, different reason for their ratio in each case), so
  there's no literature-derived answer here — `grpo_train_size = sft_train_size` was
  kept as a neutral default in the absence of evidence either way, not because either
  paper actually endorses it.

**Other numbers confirmed or corrected against the user's exact DeepSeekMath quote**
(*"we set the learning rate of the policy model as 1e-6. The KL coefficient is 0.04.
For each question, we sample 64 outputs. The max length is set to 1024, and the
training batch size is 1024."*):
- `learning_rate=1e-6` and `kl_coef=0.04` — already matched, confirmed correct.
- `max_length=1024` — matches our `context_size` exactly, confirming our config
  terminology lines up with the paper's.
- `group_size=64`, and "training batch size 1024" resolves to **16 prompts/step**
  (1024 total sampled sequences ÷ 64 samples/question) — both far larger than our
  defaults (`group_size=8`, `batch_size=4`, an 8x and 4x gap respectively). Already
  flagged as a first-run, compute-constrained starting point in the earlier
  hyperparameter sanity-check entry; recorded here again specifically tied to the
  paper's exact numbers, so the gap reads as an acknowledged tradeoff in the thesis,
  not an oversight.

**Also fixed while writing this entry:** an earlier edit to this log had
accidentally orphaned a paragraph (the `mlflow.log_table` implementation detail) at
the very end of the file, disconnected from the "Full rollout traceability" section
it belonged to — moved back into place. Worth double-checking this log's structure
after any edit that inserts near existing content, since a misplaced paragraph in a
document meant to be citable is worse than a missing one.

**Operational sequencing on the SSH machine:** stop the in-progress SFT-v3 run;
delete its old `data/posttraining/metamath_sympy/sft/{train,val}_indices.npy` (computed
under the pre-coordination scheme); run `make split-dataset`; restart `make
sft-formalizer` (now loads the coordinated split); later, `make grpo-formalizer`.

---

## 2026-09-19 — Removed the now-dead per-class splitting logic from SFT/GRPO

**Prompted by a reproducibility concern, not a bug report.** With
`split_dataset.py` now the single source of truth for all three splits,
`SFTFormalizerDataset` and `GRPOPromptDataset` still contained their own original
shuffle-and-persist-if-missing logic — now permanently dead in practice (the split
files always exist by the time either class is constructed, so the "compute fresh"
branch never executes again) but still fully present and readable in the source,
including two separately-defined `SHUFFLING_SEED = 42` constants in `sft.py` and
`grpo.py` that no longer influence anything. Left alone, this is exactly the kind of
thing that confuses a reader (or a thesis committee, or future-self) trying to
reproduce the pipeline: which seed actually determined the split that produced the
reported results — 42 (visible in both files) or 1337 (`split_dataset.py`'s, the one
that actually ran)? The dead code doesn't just look confusing, it's actively
misleading about provenance.

**Also a real (if minor) performance cost, not just a readability one:**
`SFTFormalizerDataset.__init__`'s dead branch included tokenizing every row's
`<|assistant|> {sympy}` text to compute `fits_mask` — full-corpus tokenization work
that ran on every construction and was then discarded once the "file already exists"
check hit, silently, with no log message calling out that the result was unused.

**Change:** removed the eligibility/shuffle/persist blocks and the now-unreachable
`__shuffle_indices` helpers from both classes entirely. Both now simply require
`data/posttraining/metamath_sympy/{sft,grpo}/{train,val}_indices.npy` to already exist
(raising `FileNotFoundError` with a pointer to `split_dataset.py` if not) and load
whichever split the `split` argument asks for. Also dropped the now-unused
`train_size`/`val_size` constructor parameters from both classes — no caller
(`scripts/train_sft.py`, `scripts/train_grpo.py`, `sft.py`'s own `__main__` smoke
test) passed them explicitly, so removing them is not a breaking change to any actual
call site, just a tightening of the interface to match what the class now actually
does.

**Verification:** `git stash` + `pytest tests` confirmed the 7 pre-existing test
failures (all in `test_formalizer_trainer.py`/`test_transformer.py`, unrelated to
`sft.py`/`grpo.py` — no test in this repo exercises `SFTFormalizerDataset` or
`GRPOPromptDataset` at all) are unchanged before and after this edit — nothing broke,
nothing was newly covered either.

**General lesson for this pipeline going forward:** whenever a new coordinating
script (like `split_dataset.py`) supersedes logic that used to live inside a
consumer class, sweep the codebase for the old logic and remove it in the same pass —
don't leave two versions of the truth sitting side by side, even when the old one is
provably inert. Prompted directly by the user's concern about the codebase staying
reproducible and unambiguous for the thesis, not something caught incidentally.
