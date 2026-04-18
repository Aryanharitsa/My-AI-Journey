# Activate the venv before running any target:
#   source .venv/bin/activate
# Or create it first:
#   make venv
#
# Note: ~/.cache is root-owned on this machine, so uv is pointed at ~/.uv-cache
# via UV_CACHE_DIR. Override by exporting your own UV_CACHE_DIR before running.

UV_CACHE_DIR ?= $(HOME)/.uv-cache
export UV_CACHE_DIR

.PHONY: venv install dev smoke test lint format clean download-beir

venv:
	uv venv --python 3.11 .venv
	@echo "Venv created. Run: source .venv/bin/activate"

install:
	uv pip install -e .

dev:
	uv pip install -e ".[dev]"

smoke:
	python -m vitruvius.cli smoke --cpu

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/
	mypy src/vitruvius --ignore-missing-imports

format:
	ruff format src/ tests/
	black src/ tests/

download-beir:
	python scripts/download_beir.py --datasets nfcorpus scifact fiqa

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
