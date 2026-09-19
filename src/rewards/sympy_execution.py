# --- Sympy Execution Reward ---
# Executes a Formalizer completion (model-generated sympy code) once, in a restricted
# sandbox, and derives two reward signals from that single run:
#   - "executes": did the code run to completion with no exception/timeout
#   - "correct":  does its printed output numerically match the dataset row's
#                 ground-truth answer (metadata["code_output"])
# Good enough for scoring your own model's own GRPO rollouts on your own machine — this
# is not a general-purpose untrusted-code sandbox (no memory/CPU limits, only a
# wall-clock timeout and a builtins/import allowlist). Never point this at code from an
# adversarial or otherwise untrusted source.

import builtins
import contextlib
import io
import math
import signal
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import sympy

EXEC_TIMEOUT_SECONDS = 5
REL_TOL = 1e-4
ABS_TOL = 1e-6

# Everything the executed code can reach through __builtins__. No file/network/process
# access, no eval/exec/compile, no globals()/getattr()-style introspection that could be
# used to climb back out to the real builtins. Exception names are included so a
# try/except in generated code doesn't NameError on e.g. ZeroDivisionError.
_ALLOWED_BUILTIN_NAMES = (
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "isinstance",
    "len", "list", "max", "min", "pow", "print", "range", "round", "set", "sorted",
    "str", "sum", "tuple", "type", "zip",
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "ZeroDivisionError", "ArithmeticError", "OverflowError", "StopIteration",
)


def _restricted_import(name: str, *args, **kwargs):
    """Only `sympy` (and its submodules) may be imported from sandboxed code — this is
    what lets the completion's own `import sympy as sp` line succeed despite `sp` also
    being pre-bound in the namespace below, while still blocking `import os` etc."""
    if name.split(".")[0] != "sympy":
        raise ImportError(f"import of '{name}' is not allowed in sandboxed execution")
    return builtins.__import__(name, *args, **kwargs)


_SAFE_BUILTINS: Dict[str, Any] = {name: getattr(builtins, name) for name in _ALLOWED_BUILTIN_NAMES}
_SAFE_BUILTINS["__import__"] = _restricted_import


class _Timeout(Exception):
    pass


def _alarm_handler(_signum, _frame):
    raise _Timeout()


@dataclass
class ExecutionResult:
    ran: bool                # executed to completion within the timeout, no exception
    stdout: str               # captured stdout — whatever the code printed
    error: Optional[str]      # exception message, or "timeout" — set iff ran is False


def run_sympy_completion(code: str, timeout: int = EXEC_TIMEOUT_SECONDS) -> ExecutionResult:
    """Executes model-generated sympy code in a restricted namespace, capturing stdout.

    Uses SIGALRM for the wall-clock timeout, so this only works in the main thread on a
    Unix-like OS — true for GRPOTrainer's sequential rollout loop, not necessarily true
    if this is ever called from a worker thread/process.
    """
    namespace: Dict[str, Any] = {"sp": sympy, "__builtins__": _SAFE_BUILTINS}
    buffer = io.StringIO()

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout)
    try:
        with contextlib.redirect_stdout(buffer):
            exec(code, namespace)
        return ExecutionResult(ran=True, stdout=buffer.getvalue(), error=None)
    except _Timeout:
        return ExecutionResult(ran=False, stdout=buffer.getvalue(), error="timeout")
    except BaseException as e:
        # BaseException (not Exception): generated code raising SystemExit/KeyboardInterrupt
        # must still be scored as "didn't run", never propagate out of the reward function.
        return ExecutionResult(ran=False, stdout=buffer.getvalue(), error=f"{type(e).__name__}: {e}")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _parse_last_number(stdout: str) -> Optional[float]:
    """Parses the last non-empty printed line as a number. Goes through sympy.sympify
    first (not bare float()) so answers like a fraction ("144/2") or an unevaluated
    sympy expression still resolve to their numeric value."""
    lines = [line.strip() for line in stdout.strip().splitlines() if line.strip()]
    if not lines:
        return None
    try:
        return float(sympy.sympify(lines[-1]).evalf())
    except Exception:
        return None


@dataclass
class ScoredCompletion:
    """Everything worth knowing about why a completion scored the way it did — the
    full diagnostic trail behind a single reward value."""
    completion_text: str
    executes: bool
    correct: bool
    error: Optional[str]
    stdout: str
    predicted: Optional[float]
    expected: float
    reward: float


def _score_completion(completion_text: str, metadata: Dict[str, Any]) -> ScoredCompletion:
    """Executes the completion once and derives everything downstream from that one run:
    DeepSeek-R1-style summed reward (0.0 doesn't run, 1.0 runs but wrong, 2.0 runs and
    correct) plus the full diagnostic trail for callers that want it."""
    result = run_sympy_completion(completion_text)
    expected = float(metadata["code_output"])

    predicted = None
    correct = False
    if result.ran:
        predicted = _parse_last_number(result.stdout)
        if predicted is not None:
            correct = math.isclose(predicted, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL)

    reward = (1.0 if result.ran else 0.0) + (1.0 if correct else 0.0)

    return ScoredCompletion(
        completion_text=completion_text,
        executes=result.ran,
        correct=correct,
        error=result.error,
        stdout=result.stdout,
        predicted=predicted,
        expected=expected,
        reward=reward,
    )


def sympy_reward(_prompt_text: str, completion_text: str, metadata: Dict[str, Any]) -> float:
    """Plain RewardFn: executes the completion once and returns the combined reward
    (0.0 / 1.0 / 2.0), with no diagnostics kept around. Use SympyRewardFn instead when
    the caller (e.g. GRPOTrainer) wants traceability into *why* a completion scored the
    way it did, without executing the code a second time to get that detail.
    """
    return _score_completion(completion_text, metadata).reward


class SympyRewardFn:
    """Stateful RewardFn — same callable signature/contract as sympy_reward (still
    conforms to RewardFn), but after each call `last_diagnostics` holds the full
    ScoredCompletion (as a dict) for that call: completion text, executes/correct
    flags, error, raw stdout, predicted vs. expected value. Callers that want full
    traceability (e.g. GRPOTrainer logging to MLflow) read this attribute immediately
    after calling — it's overwritten on the next call, so it's only valid for the
    single completion just scored, not a history.
    """
    def __init__(self):
        self.last_diagnostics: Optional[Dict[str, Any]] = None

    def __call__(self, _prompt_text: str, completion_text: str, metadata: Dict[str, Any]) -> float:
        scored = _score_completion(completion_text, metadata)
        self.last_diagnostics = asdict(scored)
        return scored.reward
