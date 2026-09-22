"""Evaluates a Formalizer checkpoint against the reserved benchmark holdout.

Works against any checkpoint (SFT or GRPO) — architecture is the same, only the
weights differ. Runs one completion per prompt (greedy by default: a reported
benchmark number should be reproducible, not sampled), executes each generated sympy
program through the same sandboxed scorer GRPO training uses (SympyRewardFn), and
logs full per-row traceability plus aggregate execute-rate/accuracy metrics to
MLflow — same rollout_traceability pattern GRPOTrainer uses, so results are directly
comparable across checkpoints/runs in the MLflow UI.

Usage:
    python -m scripts.run_benchmark --checkpoint models/grpo/formalizer_v1/best_model.pt
    python -m scripts.run_benchmark --checkpoint models/sft/formalizer_v3/final_model.pt --limit 100
"""
import argparse
import glob
import json
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import torch
from tokenizers import Tokenizer

from src.models.transformer import Transformer
from src.rewards.sympy_execution import SympyRewardFn
from src.utils.logger import get_logger

_logger = get_logger("run_benchmark")

TOKENIZER_PATH = "data/vocab/tokenizer_vocab.json"
VOCAB_METADATA_PATH = "data/corpus/metadata.json"
CORPUS_PATH = "data/posttraining/metamath_sympy"
HOLDOUT_INDICES_PATH = Path("data/posttraining/metamath_sympy/benchmark/holdout_indices.npy")

EOS_TOKEN = "<|endoftext|>"
# Same rationale as GRPO's rollout budget (scripts/train_grpo.py): measured against the
# actual corpus, completion lengths are median 346 / p95 647 / max 1094 tokens.
DEFAULT_MAX_NEW_TOKENS = 768
N_EMBEDDINGS, N_HEADS, N_LAYER = 912, 12, 12  # matches every other script in this repo


def load_model(checkpoint_path: Path, vocab_size: int, context_size: int, device: str) -> Transformer:
    model = Transformer(
        vocab_size=vocab_size,
        emb_dim=N_EMBEDDINGS,
        context_size=context_size,
        n_heads=N_HEADS,
        n_layers=N_LAYER,
    )
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def generate_completion(
    model: Transformer,
    tokenizer: Tokenizer,
    eos_token_id: int,
    prompt_text: str,
    max_new_tokens: int,
    context_size: int,
    device: str,
    top_p: Optional[float],
    temperature: float,
) -> str:
    """One prompt, one completion — no group sampling, this is eval not rollout."""
    ids = tokenizer.encode(prompt_text).ids
    # left-truncate so the prompt leaves room for max_new_tokens, same as GRPO's rollout
    max_prompt_len = context_size - max_new_tokens
    if len(ids) > max_prompt_len:
        ids = ids[-max_prompt_len:]
    prompt_len = len(ids)

    prompt_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    device_type = device.split(":")[0]
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        sequences, _ = model.generate(
            prompt_ids, max_new_tokens, top_p=top_p, temperature=temperature, eos_token_id=eos_token_id,
        )

    gen_part = sequences[0, prompt_len:]
    is_eos = gen_part == eos_token_id
    first_eos_idx = is_eos.float().argmax().item() if is_eos.any() else gen_part.shape[0] - 1
    completion_ids = gen_part[:first_eos_idx].tolist()

    return tokenizer.decode(completion_ids)


def run_benchmark(
    checkpoint_path: Path,
    limit: Optional[int] = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    top_p: Optional[float] = None,
    temperature: float = 0.0,
    device: str = "cuda",
) -> dict:
    if not HOLDOUT_INDICES_PATH.exists():
        raise FileNotFoundError(
            f"{HOLDOUT_INDICES_PATH} not found — run `python -m src.preprocessing.split_dataset` first."
        )

    vocab_config = json.loads(Path(VOCAB_METADATA_PATH).read_bytes())
    vocab_size = vocab_config["vocab_size"]
    context_size = vocab_config["context_size"]

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    eos_token_id = tokenizer.token_to_id(EOS_TOKEN)

    file_list = glob.glob(f"{CORPUS_PATH}/*.parquet", recursive=True)
    table = ds.dataset(file_list, format="parquet").to_table()
    idx = np.load(HOLDOUT_INDICES_PATH)
    if limit is not None:
        idx = idx[:limit]
    table = pc.take(table, idx)

    model = load_model(checkpoint_path, vocab_size, context_size, device)
    reward_fn = SympyRewardFn()

    _logger.info(f"Evaluating {checkpoint_path} on {table.num_rows:,} benchmark holdout rows")

    records = []
    for i in range(table.num_rows):
        answer = table["answer"][i].as_py()
        code_output = table["code_output"][i].as_py()
        prompt_text = f"<|bos|> <|user|> {answer} <|assistant|>"

        completion_text = generate_completion(
            model, tokenizer, eos_token_id, prompt_text, max_new_tokens, context_size, device, top_p, temperature,
        )
        reward_fn(prompt_text, completion_text, {"code_output": code_output})
        records.append({"row": int(idx[i]), "prompt": answer, **reward_fn.last_diagnostics})

        if (i + 1) % 50 == 0 or (i + 1) == table.num_rows:
            _logger.info(f"  {i + 1}/{table.num_rows} done")

    df = pd.DataFrame(records)
    metrics = {
        "execute_rate": df["executes"].mean(),
        "accuracy": df["correct"].mean(),
        "mean_reward": df["reward"].mean(),
    }

    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("Formalizer_Benchmark")
    with mlflow.start_run(run_name=checkpoint_path.stem):
        mlflow.log_params({
            "checkpoint_path": str(checkpoint_path),
            "n_rows": len(df),
            "max_new_tokens": max_new_tokens,
            "top_p": top_p,
            "temperature": temperature,
        })
        mlflow.log_metrics(metrics)
        mlflow.log_table(data=df, artifact_file="benchmark_results.json")

    _logger.info(
        f"execute_rate={metrics['execute_rate']:.4f} "
        f"accuracy={metrics['accuracy']:.4f} "
        f"mean_reward={metrics['mean_reward']:.4f}"
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Formalizer checkpoint against the benchmark holdout")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N holdout rows")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy decoding (default, reproducible)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_benchmark(
        checkpoint_path=Path(args.checkpoint),
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        device=args.device,
    )
