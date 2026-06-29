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
	python3 src/trainer/resume_formalizer_training_job.py --step $(STEP) --run $(RUN)

test:
	pytest tests --verbose

train-tokenizer:
	uv run -m src.preprocessing.hf_tokenizer

scrape-datasets:
	uv run -m src.preprocessing.scrape_datasets --output_dir ./data/pretraining --max_tokens 12_000_000_000

run-memmap:
	uv run -m src.preprocessing.memmap_builder

# --- Training Formalizer ---
run-formalizer-training:
	uv run -m scripts.train_formalizer

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000