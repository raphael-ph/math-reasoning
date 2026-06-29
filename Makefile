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