.PHONY: env install format lint test run clean

env:
	python3 -m venv venv
	@echo "Virtual environment 'venv' created. Run 'source venv/bin/activate'."

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install black ruff pytest

format:
	python3 -m black .

lint:
	python3 -m ruff check --fix .

test:
	python3 tests/test_api.py

train:
	python3 src/models/train.py

visualize:
	python3 src/evaluation/visualize.py

run:
	bash start_api.sh

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
