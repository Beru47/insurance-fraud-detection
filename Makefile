.PHONY: install data train evaluate serve test lint

PY ?= python

install:
	$(PY) -m pip install -e ".[dev]"

data:
	$(PY) -m fraudlab.cli data --n 50000 --seed 42

train: data
	$(PY) -m fraudlab.cli train

evaluate:
	$(PY) -m fraudlab.cli evaluate

serve:
	$(PY) -m uvicorn fraudlab.serve:app --host 0.0.0.0 --port 8000

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check . && $(PY) -m ruff format --check .
