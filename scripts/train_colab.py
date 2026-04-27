"""Compatibility entry point for Colab GPU solrad training."""

from __future__ import annotations

from solrad_correction.cli_colab import load_colab_config, main, run_colab_cli

__all__ = ["load_colab_config", "main", "run_colab_cli"]


if __name__ == "__main__":
    main()
