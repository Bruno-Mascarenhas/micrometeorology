.PHONY: install-uv install install-dev install-cuda fix check typecheck test test-verbose audit lock-check bench clean all

# Variables
PYTHON ?= python
UV ?= uv
UV_PIP = $(UV) pip install --system

install-uv:
	$(PYTHON) -m pip install uv

install:
	$(UV_PIP) -e .

install-dev:
	$(UV_PIP) --torch-backend cpu -e ".[dev,tcc,video,allsky]"

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
# dev+video+allsky is the widest auditable set. torch and torchvision ship from
# the PyTorch index (as +cpu local versions), not PyPI, so pip-audit cannot
# resolve them — both are excluded via --no-emit-package while the rest of the
# allsky extra (safetensors, tensorboard, tqdm, imageio-ffmpeg) gets audited.
audit:
	$(UV) export --frozen --extra dev --extra video --extra allsky --no-emit-package torch --no-emit-package torchvision --format requirements-txt --no-emit-project -o requirements-audit.txt
	uvx pip-audit --strict --disable-pip -r requirements-audit.txt
	rm -f requirements-audit.txt

# Fails when uv.lock is out of sync with pyproject.toml (offline, fast).
lock-check:
	$(UV) lock --check

# Synthetic perf harnesses for the solrad hot paths (no data/ needed).
bench:
	$(PYTHON) benchmarks/solrad_correction/loading.py --rows 10000 --features 16
	$(PYTHON) benchmarks/solrad_correction/preprocessing.py --rows 20000 --features 24
	$(PYTHON) benchmarks/solrad_correction/sequence_dataloader.py --rows 50000 --features 24 --sequence-length 24
	$(PYTHON) benchmarks/solrad_correction/artifact_checkpoint.py --hidden-size 32 --layers 2

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
