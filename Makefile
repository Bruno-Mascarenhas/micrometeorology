.PHONY: install install-dev lint lint-fix format format-check typecheck test test-verbose clean all check

# Variables
PYTHON = python3
PIP = pip

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install -e ".[dev,tcc]"

lint:
	ruff check .

lint-fix:
	ruff check --fix .

format:
	ruff format .

format-check:
	ruff format --check .

typecheck:
	mypy src tests

test:
	pytest -n auto tests/

test-verbose:
	pytest -n auto -v tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Full CI-equivalent check
check: format-check lint typecheck test

# Format + lint-fix + test
all: format lint-fix typecheck test
