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


def sequence_logprobs(logits: torch.Tensor, target_ids: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
    """Gathers per-token log-probabilities of the actual next tokens.

    Args:
        logits: (N, T-1, V) raw model logits over the shifted input (positions 0..T-2).
        target_ids: (N, T-1) the actual next-token ids (positions 1..T-1).
        chunk_size: rows (flattened over N*T) processed per chunk.

    Returns:
        (N, T-1) log-prob assigned by the model to each realized target token, fp32.

    log_softmax runs in fp32 on every row, same precision as computing it over the
    whole (N, T-1, V) tensor at once — this is NOT a bf16-then-upcast shortcut (that
    was tried and measured to introduce ~0.05 max log-prob error on realistic vocab
    sizes, which by itself is a ~5-6% worst-case error in the downstream
    exp(new_logprobs - old_logprobs) ratio GRPO's clipping depends on — unacceptable).
    Instead, this processes `chunk_size` rows at a time: the (N, T-1, V) logits tensor
    (often the single largest tensor in a GRPO training step) is never upcast to fp32
    in one piece, only one chunk of it at a time, capping this operation's peak memory
    at chunk_size * V * 4 bytes regardless of how large N*T is.
    """
    N, T = target_ids.shape
    V = logits.shape[-1]
    flat_logits = logits.reshape(N * T, V)
    flat_targets = target_ids.reshape(N * T)

    chunks = []
    for start in range(0, flat_logits.shape[0], chunk_size):
        chunk_logits = flat_logits[start:start + chunk_size].float()
        chunk_targets = flat_targets[start:start + chunk_size]
        chunk_log_probs = F.log_softmax(chunk_logits, dim=-1)
        chunks.append(chunk_log_probs.gather(-1, chunk_targets.unsqueeze(-1)).squeeze(-1))

    return torch.cat(chunks, dim=0).reshape(N, T)


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
