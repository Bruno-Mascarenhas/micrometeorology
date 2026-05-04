"""Compatibility entry point for Colab GPU solrad training.

Examples
--------
Run directly using Python:
    python scripts/train_colab.py --config configs/tcc/experiments/lstm_hourly.yaml

Run and save to Google Drive:
    python scripts/train_colab.py --config configs/tcc/experiments/lstm_hourly.yaml --output-dir /content/drive/MyDrive/outputs/
"""

from __future__ import annotations

from solrad_correction.cli_colab import load_colab_config, main, run_colab_cli

__all__ = ["load_colab_config", "main", "run_colab_cli"]


if __name__ == "__main__":
    main()
