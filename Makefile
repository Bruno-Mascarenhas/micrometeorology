.PHONY: install install-dev fix check typecheck test test-verbose clean all

# Variables
PYTHON = python3
PIP = pip

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install -e ".[dev,tcc]"

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
