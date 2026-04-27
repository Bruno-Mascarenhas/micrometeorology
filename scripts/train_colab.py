"""Google Colab / remote GPU wrapper for ``solrad-run`` experiments.

Preferred usage in Colab:

    python scripts/train_colab.py --config configs/tcc/experiments/lstm_hourly.yaml \
        --output-dir /content/drive/MyDrive/LabMiM/experiments \
        --device cuda --amp --num-workers 2

This script intentionally delegates to the package experiment runner so Colab
runs produce the same artifacts as local ``solrad-run`` runs.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from solrad_correction.config import ExperimentConfig
from solrad_correction.experiments.runner import run_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ColabTrain")


def parse_args() -> argparse.Namespace:
    """Parse Colab wrapper arguments."""
    parser = argparse.ArgumentParser(description="Run a solrad-run experiment in Colab")
    parser.add_argument("--config", required=True, help="Experiment YAML config")
    parser.add_argument("--name", default=None, help="Override experiment name")
    parser.add_argument("--output-dir", default=None, help="Drive-backed experiment output dir")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resume", default=None, help="Path to checkpoints/last.pt")
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--validate-config", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args()


def load_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Load config and apply Colab runtime overrides."""
    cfg = ExperimentConfig.from_yaml(args.config)
    if args.name:
        cfg.name = args.name
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.device:
        cfg.runtime.device = args.device
    if args.num_workers is not None:
        cfg.runtime.num_workers = args.num_workers
    if args.pin_memory is not None:
        cfg.runtime.pin_memory = args.pin_memory
    if args.amp is not None:
        cfg.runtime.amp = args.amp
    if args.compile is not None:
        cfg.runtime.torch_compile = args.compile
    if args.resume:
        cfg.runtime.resume = args.resume
    if args.limit_rows is not None:
        cfg.runtime.limit_rows = args.limit_rows
    if args.profile:
        cfg.runtime.profile = True
    return cfg


def main() -> None:
    args = parse_args()
    cfg = load_config_from_args(args)
    cfg.validate()

    if args.print_config:
        print(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False, default=str))
        return

    if args.validate_config:
        logger.info("Config is valid.")
        return

    logger.info("Experiment: %s", cfg.name)
    logger.info("Model: %s", cfg.model.model_type)
    logger.info("Output: %s", Path(cfg.experiment_dir).resolve())
    run_experiment(cfg)


if __name__ == "__main__":
    main()
