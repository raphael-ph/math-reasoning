# --- Reward Function Interface ---
# GRPO's advantage comes entirely from this function — it's the one piece of the pipeline
# meant to be swapped out per task. No base class: it's a plain function, not an object
# with state, so any callable matching this signature works.

from typing import Any, Callable, Dict

# (prompt_text, completion_text, metadata) -> reward
#   prompt_text: the exact text the model was conditioned on (post any truncation applied
#     at rollout time), not re-derived from the dataset row.
#   completion_text: decoded model output, already cut at the first EOS token — never the
#     raw/padded generation tensor.
#   metadata: the dataset row's remaining fields (everything but the prompt itself), keyed
#     by column name — whatever a specific reward implementation needs to check correctness
#     (e.g. an expected numeric answer) lives here.
# Returns a scalar reward. No fixed range is assumed — group_advantages normalizes within
# each prompt's group regardless of scale.
RewardFn = Callable[[str, str, Dict[str, Any]], float]
