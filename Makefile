.PHONY: install-uv require-conda install install-dev install-cuda fix check typecheck test test-verbose audit lock-check bench clean all

# Variables
CONDA_ENV_NAME ?= micrometeorology
PYTHON ?= $(CONDA_PREFIX)/bin/python
UV ?= uv
UV_PIP = $(UV) pip install --system

# Lint, type and test through the $(CONDA_ENV_NAME) ENVIRONMENT, never through
# PATH and never through a second one. pyproject relies on ruff 0.16's expanded
# default rule set (see the `extend-select` comment there), so an older ruff —
# the miniforge base build, say — disagrees about which rules exist: it reports
# this tree's real `# noqa: BLE001` / `# noqa: TRY004` directives as unused
# RUF100 and `make fix` DELETES them, exiting 0, after which CI's pinned ruff
# fails on violations the developer never wrote.
#
# `--no-sync` pins the resolution without touching the environment. Naming the
# environment explicitly is what stops `uv run` from creating its own in .venv/
# and running everything there: that one is invisible to `make install-dev`, so
# it drifts, and the ruff it holds is not the ruff the floor above pins.
RUN = UV_PROJECT_ENVIRONMENT="$(CONDA_PREFIX)" $(UV) run --no-sync
TORCH_BACKEND ?= cu130
TORCH_VERSION ?= 2.13.0

install-uv: require-conda
	$(PYTHON) -m pip install uv

# Checks the env by NAME, not merely that some env is active: the miniforge
# `base` env satisfies "CONDA_PREFIX is set" and carries none of this project,
# which is how `make bench` came to run against a python without numpy.
require-conda:
	@test "$(notdir $(CONDA_PREFIX))" = "$(CONDA_ENV_NAME)" || \
		(echo "Activate the $(CONDA_ENV_NAME) Conda environment first (active: $(if $(CONDA_PREFIX),$(notdir $(CONDA_PREFIX)),none))." && exit 1)

install: require-conda
	UV_PROJECT_ENVIRONMENT="$(CONDA_PREFIX)" $(UV) sync --locked --inexact

install-dev: require-conda
	UV_PROJECT_ENVIRONMENT="$(CONDA_PREFIX)" $(UV) sync --locked --inexact --extra dev --extra tcc --extra video --extra allsky

install-cuda: require-conda
	UV_PROJECT_ENVIRONMENT="$(CONDA_PREFIX)" $(UV) sync --locked --inexact --extra tcc-cuda --extra allsky --no-install-package torch
	$(UV_PIP) --reinstall --torch-backend $(TORCH_BACKEND) "torch==$(TORCH_VERSION)"

fix: require-conda
	$(RUN) ruff format .
	$(RUN) ruff check --fix .

typecheck: require-conda
	$(RUN) mypy src tests

test: require-conda
	$(RUN) pytest -n auto tests/

test-verbose: require-conda
	$(RUN) pytest -n auto -v tests/

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
bench: require-conda
	$(RUN) python benchmarks/solrad_correction/loading.py --rows 10000 --features 16
	$(RUN) python benchmarks/solrad_correction/preprocessing.py --rows 20000 --features 24
	$(RUN) python benchmarks/solrad_correction/sequence_dataloader.py --rows 50000 --features 24 --sequence-length 24
	$(RUN) python benchmarks/solrad_correction/artifact_checkpoint.py --hidden-size 32 --layers 2

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_tmp" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

check: require-conda lock-check
	$(RUN) ruff format --check .
	$(RUN) ruff check .
	$(RUN) mypy src tests
	$(RUN) pytest -n auto tests/

all: fix check
