# --- GRPO Math ---
# Pure-tensor building blocks for GRPO: group-relative advantages, per-token log-probs,
# and the clipped surrogate + KL-penalty loss. No model/dataset/mlflow coupling here on
# purpose, so this can be unit-tested with plain tensors.

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def group_advantages(rewards: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Normalizes rewards within each prompt's group of completions.

    Args:
        rewards: (B, G) — B prompts, G sampled completions per prompt.
        eps: added to the std so a group with identical rewards (std=0) doesn't blow up.

    Returns:
        (B, G) advantages, zero-mean and unit-variance within each group.
    """
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True)

    return (rewards - mean) / (std + eps)


def sequence_logprobs(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Gathers per-token log-probabilities of the actual next tokens.

    Args:
        logits: (N, T-1, V) raw model logits over the shifted input (positions 0..T-2).
        target_ids: (N, T-1) the actual next-token ids (positions 1..T-1).

    Returns:
        (N, T-1) log-prob assigned by the model to each realized target token.
    """
    log_probs = F.log_softmax(logits, dim=-1)

    return log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)


def grpo_loss(
    old_logprobs: torch.Tensor,
    new_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_epsilon: float,
    kl_coef: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Clipped GRPO surrogate objective with a KL penalty against the reference model.

    Args:
        old_logprobs: (N, T-1) log-probs under the policy at rollout time (no grad).
        new_logprobs: (N, T-1) log-probs under the current policy (has grad).
        ref_logprobs: (N, T-1) log-probs under the frozen reference model (no grad).
        advantages: (N,) group-normalized reward per sequence, broadcast over its tokens.
        mask: (N, T-1) bool, True at completion-token positions (prompt/pad excluded).
        clip_epsilon: PPO-style clip range.
        kl_coef: weight (beta) of the KL penalty term.

    Returns:
        (loss, metrics) where loss is a scalar tensor ready for .backward(), and metrics
        is a dict of detached floats for logging.
    """
    ratio = torch.exp(new_logprobs - old_logprobs)
    adv = advantages.unsqueeze(-1)

    surrogate1 = ratio * adv
    surrogate2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv
    policy_term = torch.min(surrogate1, surrogate2)

    # k3 estimator (Schulman): unbiased, always >= 0, low variance vs. the naive
    # (ref_logprobs - new_logprobs) difference.
    kl = torch.exp(ref_logprobs - new_logprobs) - (ref_logprobs - new_logprobs) - 1

    per_token = policy_term - kl_coef * kl

    mask_f = mask.float()
    # per-sequence normalization first (DeepSeekMath GRPO) so a long completion doesn't
    # out-weight a short one just by having more terms in the sum; clamp guards the
    # degenerate case of a fully-masked row (shouldn't happen — completions always
    # include at least the EOS token — but costs nothing to guard).
    token_counts = mask_f.sum(dim=-1).clamp(min=1.0)
    seq_values = (per_token * mask_f).sum(dim=-1) / token_counts
    loss = -seq_values.mean()

    with torch.no_grad():
        kl_seq = (kl * mask_f).sum(dim=-1) / token_counts
        mean_kl = kl_seq.mean().item()

        clipped = ((ratio - 1.0).abs() > clip_epsilon).float()
        valid_tokens = mask_f.sum().clamp(min=1.0)
        clip_fraction = ((clipped * mask_f).sum() / valid_tokens).item()
        mean_ratio = ((ratio * mask_f).sum() / valid_tokens).item()

    metrics = {
        "mean_kl": mean_kl,
        "clip_fraction": clip_fraction,
        "mean_ratio": mean_ratio,
    }

    return loss, metrics
