.PHONY: install-uv install install-dev install-cuda fix check typecheck test test-verbose clean all

# Variables
PYTHON ?= python
UV ?= uv
#UV_PIP = $(UV) pip install --python $(PYTHON)
UV_PIP = $(UV) pip install --system

install-uv:
	$(PYTHON) -m pip install uv

install:
	$(UV_PIP) -e .

install-dev:
	$(UV_PIP) --torch-backend cpu -e ".[dev,tcc,video]"

install-cuda:
	$(UV_PIP) --torch-backend cu121 torch
	$(UV_PIP) -e ".[tcc-cuda]"

fix:
	ruff format .
	ruff check --fix .

typecheck:
	mypy src tests

test:
	pytest -n auto tests/

test-verbose:
	pytest -n auto -v tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_tmp" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

check:
	ruff format --check .
	ruff check .
	mypy src tests
	pytest -n auto tests/

all: fix check
