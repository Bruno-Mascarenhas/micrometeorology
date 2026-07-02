.PHONY: install-uv install install-dev install-cuda fix check typecheck test test-verbose audit lock-check clean all

# Variables
PYTHON ?= python
UV ?= uv
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

# Mirrors the CI vulnerability gate so advisory failures surface before a push.
# dev+video is the widest auditable set (torch ships from the PyTorch index).
audit:
	$(UV) export --frozen --extra dev --extra video --format requirements-txt --no-emit-project -o requirements-audit.txt
	uvx pip-audit --strict --disable-pip -r requirements-audit.txt
	rm -f requirements-audit.txt

# Fails when uv.lock is out of sync with pyproject.toml (offline, fast).
lock-check:
	$(UV) lock --check

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_tmp" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

check: lock-check
	ruff format --check .
	ruff check .
	mypy src tests
	pytest -n auto tests/

all: fix check
