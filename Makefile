.PHONY: install-uv require-conda install install-dev install-cuda fix check typecheck test test-verbose audit lock-check bench clean all

# Variables
PYTHON ?= python
UV ?= uv
UV_PIP = $(UV) pip install --system
TORCH_BACKEND ?= cu130
TORCH_VERSION ?= 2.13.0

install-uv:
	$(PYTHON) -m pip install uv

require-conda:
	@test -n "$(CONDA_PREFIX)" || (echo "Activate the micrometeorology Conda environment first." && exit 1)

install: require-conda
	UV_PROJECT_ENVIRONMENT="$(CONDA_PREFIX)" $(UV) sync --locked --inexact

install-dev: require-conda
	UV_PROJECT_ENVIRONMENT="$(CONDA_PREFIX)" $(UV) sync --locked --inexact --extra dev --extra tcc --extra video --extra allsky

install-cuda: require-conda
	UV_PROJECT_ENVIRONMENT="$(CONDA_PREFIX)" $(UV) sync --locked --inexact --extra tcc-cuda --extra allsky --no-install-package torch
	$(UV_PIP) --reinstall --torch-backend $(TORCH_BACKEND) "torch==$(TORCH_VERSION)"

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
# dev+video+allsky is the widest auditable set. torch ships from the PyTorch
# index (as a +cpu local version), not PyPI, so pip-audit cannot resolve it — it
# is excluded via --no-emit-package while the rest of the allsky extra
# (safetensors, tensorboard, imageio-ffmpeg) gets audited.
audit:
	$(UV) export --frozen --extra dev --extra video --extra allsky --no-emit-package torch --format requirements-txt --no-emit-project -o requirements-audit.txt
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
