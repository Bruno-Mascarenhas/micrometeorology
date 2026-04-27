"""Tests for the Colab training wrapper."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from scripts.train_colab import load_config_from_args


def test_colab_script_loads_config_and_applies_runtime_overrides():
    scratch = Path("scratch")
    scratch.mkdir(exist_ok=True)
    config_path = scratch / "colab_wrapper_config.yaml"
    try:
        config_path.write_text(
            yaml.safe_dump({"name": "colab_base", "model": {"model_type": "lstm"}}),
            encoding="utf-8",
        )
        args = argparse.Namespace(
            config=str(config_path),
            name="colab_override",
            output_dir="scratch/colab-output",
            device="cpu",
            num_workers=0,
            pin_memory=False,
            amp=False,
            compile=False,
            resume=None,
            limit_rows=20,
            profile=True,
        )

        cfg = load_config_from_args(args)

        assert cfg.name == "colab_override"
        assert cfg.output_dir == "scratch/colab-output"
        assert cfg.runtime.device == "cpu"
        assert cfg.runtime.num_workers == 0
        assert cfg.runtime.pin_memory is False
        assert cfg.runtime.amp is False
        assert cfg.runtime.torch_compile is False
        assert cfg.runtime.limit_rows == 20
        assert cfg.runtime.profile is True
    finally:
        config_path.unlink(missing_ok=True)
