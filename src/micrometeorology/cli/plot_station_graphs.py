"""CLI: Generate site-facing graphs from processed sensor data.

Examples
--------
Generate temperature and humidity graphs:
    labmim-site-graphs -i data/hourly/sensor_data.csv -o output/site_graphs/ -v Temp1 -v RH1

Generate wind speed graph for the last 14 days:
    labmim-site-graphs -i data/hourly/sensor_data.csv -o output/site_graphs/ -v WS_WVT --last-days 14
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import pandas as pd
import typer

from micrometeorology.common.logging import setup_logging
from micrometeorology.common.paths import ensure_dir

app = typer.Typer(rich_markup_mode="markdown", no_args_is_help=True)


@app.command()
def run(
    input_path: Annotated[
        Path, typer.Option("-i", "--input", help="Processed sensor CSV file.", exists=True)
    ],
    output_dir: Annotated[
        Path, typer.Option("-o", "--output", help="Output directory for graphs.")
    ],
    variables: Annotated[list[str], typer.Option("-v", "--variables", help="Columns to plot.")],
    last_days: Annotated[int, typer.Option(help="Number of recent days to plot.")] = 7,
    log_level: Annotated[str, typer.Option(help="Logging level.")] = "INFO",
) -> None:
    """Generate time-series graphs for the LabMiM website."""
    setup_logging(log_level)
    out = ensure_dir(output_dir)

    df = pd.read_csv(input_path, parse_dates=[0], index_col=0)

    if last_days > 0:
        cutoff = df.index.max() - pd.Timedelta(days=last_days)
        df = df[df.index >= cutoff]

    for var in variables:
        if var not in df.columns:
            typer.echo(f"Warning: Column '{var}' not found -- skipping")
            continue

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df.index, df[var], linewidth=0.8)
        ax.set_ylabel(var)
        ax.set_title(f"{var} -- Last {last_days} days")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        fig.savefig(out / f"{var}_last_{last_days}d.png", dpi=150)
        plt.close(fig)
        typer.echo(f">> {var}")

    typer.echo(f"\n>> Graphs saved to {out}")


def main() -> None:
    """Module entry point (``python -m micrometeorology.cli.plot_station_graphs``)."""
    app()


if __name__ == "__main__":
    main()
