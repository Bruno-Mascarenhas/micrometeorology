"""Rewrite the lab's solar-geometry CSV as a parquet beside it.

Run once, and again whenever the lab replaces the CSV. The CSV stays the source
of truth; this only adds the derived form the pipeline prefers to read.

Usage
-----
::

    uv run python scripts/converter_teorica.py --data data
"""

import time
from pathlib import Path
from typing import Annotated

import typer

from micrometeorology.sensors.calibration import (
    SHADE_RING_FACTOR_FILE,
    SHADE_RING_FACTOR_PARQUET,
    solar_geometry_to_parquet,
)

app = typer.Typer(rich_markup_mode="markdown", no_args_is_help=True)

DEFAULT_DATA_DIR = Path("data")


@app.command()
def run(
    data_dir: Annotated[
        Path, typer.Option("--data", help="Directory holding the CSV.")
    ] = DEFAULT_DATA_DIR,
) -> None:
    """Write the derived parquet beside the lab's solar-geometry CSV."""
    source = data_dir / SHADE_RING_FACTOR_FILE
    destination = data_dir / SHADE_RING_FACTOR_PARQUET
    started = time.perf_counter()
    solar_geometry_to_parquet(source, destination)
    typer.echo(
        f"{destination}  {destination.stat().st_size / 1e6:.1f} MB "
        f"(de {source.stat().st_size / 1e6:.0f} MB) em {time.perf_counter() - started:.2f} s"
    )


def main() -> None:
    """Entry point for ``python scripts/converter_teorica.py``."""
    app()


if __name__ == "__main__":
    main()
