"""This is the script to run GRPO post-training for the Formalizer Model.

Starts from an SFT checkpoint (the model already knows the <|user|>/<|assistant|>
prompt format and how to emit sympy code — GRPO is what pushes it towards actually
running and getting the right answer). Both the trained policy and the frozen
reference model (used for the KL penalty) are initialized from the same checkpoint.
"""
import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from src.models.transformer import Transformer
from src.rewards.sympy_execution import SympyRewardFn
from src.trainer.grpo import GRPOConfig, GRPOPromptDataset, GRPOTrainer
from src.utils.logger import get_logger

# --- CONFIGURATION AND GLOBAL VARIABLES ---
# Paths
TOKENIZER_PATH = "data/vocab/tokenizer_vocab.json"
VOCAB_METADATA_PATH = "data/corpus/metadata.json"
CORPUS_PATH = "data/posttraining/metamath_sympy"
FINAL_MODEL_PATH = "models/grpo/formalizer_v1/final_model.pt"
# Not best_model.pt: SFT was deliberately trained to overfit (see TODO.md / scripts/train_sft.py,
# 12k steps "to force overfit", following the InstructGPT paper's finding that the
# lowest-val-loss checkpoint underperforms a later, overfit one for downstream RL).
# best_model.pt tracks lowest val loss and would likely pick an early, pre-overfit
# checkpoint here — final_model.pt is the actual last-step model this project wants.
DEFAULT_SFT_MODEL_PATH = "models/sft/formalizer_v3/final_model.pt"

# ---------------- Global Vars --------------------
with open(VOCAB_METADATA_PATH, "rb") as file:
    f = file.read()
    vocab_config = json.loads(f)

VOCAB_SIZE = vocab_config["vocab_size"]
CONTEXT_SIZE = vocab_config["context_size"]
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
logger = get_logger("grpo_trainer")
# -------------------------------------------------

config = GRPOConfig(
    vocab_size=VOCAB_SIZE,
    context_size=CONTEXT_SIZE,
    n_embeddings=912,
    n_heads=12,
    n_layer=12,
    # GRPO rollouts (a generate() call per prompt, group_size times) are far more
    # expensive per step than SFT's teacher-forced batches, hence far fewer steps,
    # a much smaller batch_size (prompts/step), and a smaller LR than SFT's 3e-5 —
    # this is fine-tuning an already-fine-tuned model, not training from scratch.
    max_iters=1_000,
    eval_iters=5,
    eval_interval=50,
    checkpoint_interval=100,
    # Cut from 4 after a CUDA OOM on a 16GB card — batch_size (prompts/step) is the
    # cheaper thing to shrink vs. group_size: it only means fewer distinct prompts per
    # step (recoverable with more steps), not a noisier per-prompt advantage estimate.
    batch_size=2,
    # DeepSeekMath's own GRPO actor LR (1e-6) — no empirical basis of our own yet to
    # deviate from it, and erring conservative matters more here than for SFT: a bad
    # policy update compounds across the KL penalty and future rollouts.
    learning_rate=1e-6,
    warmup_steps=20,
    device="cuda",
    final_model_path=FINAL_MODEL_PATH,
    # GRPO-specific
    group_size=8,
    # Measured directly against the scraped corpus (metamath_sympy_shard_0000.parquet):
    # completion token lengths are median=346, p95=647, max=1094 — a 256-token budget
    # would truncate 78% of completions before they ever reach print()/EOS, making
    # nearly every reward 0 for running out of tokens rather than for being wrong.
    # 768 leaves only ~1.8% truncated, at the cost of capping the prompt side at 256
    # tokens (context_size 1024 - 768) — comfortably above the prompt's own p95 (226).
    max_new_tokens=768,
    top_p=0.9,
    temperature=0.8,
)

train_ds = GRPOPromptDataset(corpus_path=CORPUS_PATH, split="train")
val_ds = GRPOPromptDataset(corpus_path=CORPUS_PATH, split="val")


def load_sft_model(checkpoint_path: Path) -> Transformer:
    """Builds a fresh Transformer and loads SFT weights — used twice, once for the
    trained policy and once for the frozen reference model, so each gets its own
    independent copy of the weights rather than sharing tensors."""
    model = Transformer(
        vocab_size=config.vocab_size,
        emb_dim=config.n_embeddings,
        context_size=config.context_size,
        n_heads=config.n_heads,
        n_layers=config.n_layer,
    )

    state = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
    state_dict = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    model.load_state_dict(state_dict)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sft-model",
        type=str,
        default=DEFAULT_SFT_MODEL_PATH,
        help="Path to the SFT checkpoint both the policy and reference model start from",
    )
    args = parser.parse_args()

    sft_model_path = Path(args.sft_model)
    logger.info(f"Loading policy + reference model from {sft_model_path}")
    policy_model = load_sft_model(sft_model_path)
    ref_model = load_sft_model(sft_model_path)

    total = sum(p.numel() for p in policy_model.parameters())
    logger.info(f"Total parameters: {total/1e6:.1f}M")

    reward_fn = SympyRewardFn()

    trainer = GRPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        reward_fn=reward_fn,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=config,
    )

    trainer.train()
