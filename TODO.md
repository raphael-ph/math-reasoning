# TODO — 2026-07-22

## Done today

- `SFTFormalizerDataset` (`src/trainer/sft.py`): added `train`/`val` split.
  Shuffles row indices with a fixed seed, slices 15k train / 3k val, persists
  `train_indices.npy` / `val_indices.npy` under `data/posttraining/metamath_sympy/sft/`
  (skips re-saving if the files already exist), applies the right slice via
  `pc.take` based on the `split` arg.
- `SFTTrainer` (`src/trainer/sft.py`): training loop, LR scheduler, checkpointing,
  MLflow logging, and resume paths copied over from `FormalizerTrainer`.
- GRPO split and the 500-row smoke-test dataset explicitly parked for now —
  full dataset now available on the SSH machine.

## Must do tomorrow

### 1. Implement prompt/completion masking (blocker — SFT doesn't work without this)

The model (`src/models/transformer.py::Transformer.forward`) is decoder-only:
it expects one sequence `idx` and a same-shape, already-shifted `targets`, and
computes a flat per-position cross-entropy with no `ignore_index`. Right now
`SFTFormalizerDataset` hands it two unrelated sequences (query, sympy) — this
needs to become one concatenated, shifted, masked sequence:

- [ ] Format each example as `<|user|> {answer} <|assistant|> {sympy} <|endoftext|>`
      and tokenize as a single sequence. (`<|user|>`/`<|assistant|>` already
      exist as reserved special tokens in `hf_tokenizer.py` — never wired into
      any training example yet.)
- [ ] Shift by one for `input_ids`/`target_ids`, same as `FormalizerDataset` does
      for pretraining.
- [ ] Build the loss mask anchored on the `<|assistant|>` token position:
      mask everything at/before it (prompt), keep everything after it live
      (completion, through `<|endoftext|>`). Also mask trailing padding.
- [ ] Set masked target positions to `-100`.
- [ ] Add `ignore_index=-100` to the `F.cross_entropy` call in
      `Transformer.forward`.
- [ ] Decide truncation policy for prompt+completion > `context_size` —
      must never truncate away the completion.
- [ ] Decide whether to prepend `<|bos|>` for consistency with the pretraining
      format.

### 2. Fix crash bugs in `SFTTrainer` (some may resolve once #1 changes `__getitem__`)

- [ ] `xb, yb = next(train_iter)` / `X, Y = next(loader_iter)` unpack 2 values
      but the dataset currently returns a 3-tuple (`code_output` unused) —
      `ValueError` on first batch.
- [ ] `__main__` block: `SFTFormalizerDataset(CORPUS_PATH, tokenizer, CONTEXT_SIZE)`
      is missing the now-required `split` arg.

### 3. Once full dataset lands locally (or training points at the SSH machine's copy)

- [ ] Re-run the split — `train_indices.npy`/`val_indices.npy` on disk right
      now reflect the 500-row smoke test. The "don't overwrite if exists"
      guard means they won't regenerate automatically; delete them first.
- [ ] Revisit GRPO's remainder split (`indices[train_size + val_size:]`) —
      still nothing computed/persisted for it.
