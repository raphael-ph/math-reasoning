train:
	echo "Generating updated requirements.txt"
	uv pip compile pyproject.toml -o requirements.txt
	python3 -m src.utils.runner

test:
	pytest tests --verbose

train-tokenizer:
	uv run -m src.preprocessing.hf_tokenizer

scrape-datasets:
	uv run -m src.preprocessing.scrape_datasets --output_dir ./data/pretraining --max_tokens 12_000_000_000

run-memmap:
	uv run -m src.preprocessing.memmap_builder