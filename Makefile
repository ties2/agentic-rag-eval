export PYTHONPATH := src:.
# One word per task. `make help` lists them.
.PHONY: help install ingest eval serve test lint type check up down

help:
	@grep -E '^[a-zA-Z_-]+:.*?# .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?# "}{printf "  %-10s %s\n", $$1, $$2}'

install:  # install package + dev tools (mock mode needs nothing else)
	pip install -e ".[dev]"

ingest:  # build the vector index from data/corpus
	python scripts/ingest.py

eval:  # run the evaluation harness and write reports/scorecard.md
	python scripts/evaluate.py 

serve:  # run the API at http://localhost:8000/docs
	uvicorn ragkit.api.app:app --reload --app-dir src

test:  # run the test suite (fully offline in mock mode)
	pytest -q

lint:  # static lint
	ruff check .

type:  # static type check
	mypy

check: lint type test  # everything CI runs

up:  # start qdrant + mlflow + api via docker
	docker compose up --build

down:
	docker compose down -v
