train:
	echo "Generating updated requirements.txt"
	uv pip compile pyproject.toml -o requirements.txt
	python3 -m src.utils.runner

resume-training:
ifndef STEP
	$(error STEP is required. Usage: make resume-training STEP=455000 RUN=<mlflow-run-id>)
endif
ifndef RUN
	$(error RUN is required. Usage: make resume-training STEP=455000 RUN=<mlflow-run-id>)
endif
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run -m scripts.resume_formalizer --step $(STEP) --run $(RUN)

test:
	pytest tests --verbose

train-tokenizer:
	uv run -m src.preprocessing.hf_tokenizer

scrape-datasets:
	uv run -m src.preprocessing.scrape_datasets --output_dir ./data/pretraining --max_tokens 12_000_000_000

run-memmap:
	uv run -m src.preprocessing.memmap_builder

# --- Post-training (SFT + GRPO) ---
scrape-metamath-sympy:
	uv run -m src.preprocessing.scrape_posttraining --output_dir ./data/posttraining/metamath_sympy

# Generates the SFT, GRPO, and benchmark-holdout splits together from one coordinated
# shuffle. Must run after scrape-metamath-sympy and BEFORE sft-formalizer/grpo-formalizer
# — refuses to run if any split file already exists (delete data/posttraining/metamath_sympy/{sft,grpo,benchmark}/ first if you actually want to regenerate).
split-dataset:
	uv run -m src.preprocessing.split_dataset

# --- Training Formalizer ---
run-formalizer-training:
	uv run -m scripts.train_formalizer

# --- SFT ---
BASE_MODEL ?= models/formalizer/best_model.pt

sft-formalizer:
	uv run -m scripts.train_sft --base-model $(BASE_MODEL)

# --- GRPO ---
SFT_MODEL ?= models/sft/formalizer_v3/final_model.pt

grpo-formalizer:
	uv run -m scripts.train_grpo --sft-model $(SFT_MODEL)

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000