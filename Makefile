train:
	uv export --no-hashes --no-dev --format requirements.txt > requirements.txt
	python3 -m src.utils.runner