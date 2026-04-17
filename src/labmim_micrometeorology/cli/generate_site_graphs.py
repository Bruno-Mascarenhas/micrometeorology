"""CLI: Generate site-facing graphs from processed sensor data.

Usage::

    python scripts/generate_site_graphs.py --input data/hourly/sensor_data.csv \\
        --output output/site_graphs/ --variables Temp1 RH1 WS_WVT
"""

from __future__ import annotations

import click
import matplotlib.pyplot as plt
import pandas as pd

from labmim_micrometeorology.common.logging import setup_logging
from labmim_micrometeorology.common.paths import ensure_dir


@click.command()
@click.option("--input", "-i", "input_path", required=True, help="Processed sensor CSV file.")
@click.option("--output", "-o", "output_dir", required=True, help="Output directory for graphs.")
@click.option("--variables", "-v", multiple=True, required=True, help="Columns to plot.")
@click.option("--last-days", default=7, type=int, help="Number of recent days to plot.")
@click.option("--log-level", default="INFO", help="Logging level.")
def main(
    input_path: str,
    output_dir: str,
    variables: tuple[str, ...],
    last_days: int,
    log_level: str,
) -> None:
    """Generate time-series graphs for the LabMiM website."""
    setup_logging(log_level)
    out = ensure_dir(output_dir)

    df = pd.read_csv(input_path, parse_dates=[0], index_col=0)

    # Filter to last N days
    if last_days > 0:
        cutoff = df.index.max() - pd.Timedelta(days=last_days)
        df = df[df.index >= cutoff]

    for var in variables:
        if var not in df.columns:
            click.echo(f"⚠ Column '{var}' not found — skipping")
            continue

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df.index, df[var], linewidth=0.8)
        ax.set_ylabel(var)
        ax.set_title(f"{var} — Last {last_days} days")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        fig.savefig(out / f"{var}_last_{last_days}d.png", dpi=150)
        plt.close(fig)
        click.echo(f"✓ {var}")

    click.echo(f"\n✓ Graphs saved to {out}")


if __name__ == "__main__":
    main()
