train:
	echo "Generating updated requirements.txt"
	uv pip compile pyproject.toml -o requirements.txt
	python3 -m src.utils.runner