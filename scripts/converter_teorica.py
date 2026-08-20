"""Rewrite the lab's solar-geometry CSV as a parquet beside it.

Run once, and again whenever the lab replaces the CSV. The CSV stays the source
of truth; this only adds the derived form the pipeline prefers to read.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from micrometeorology.sensors.calibration import (
    SHADE_RING_FACTOR_FILE,
    SHADE_RING_FACTOR_PARQUET,
    solar_geometry_to_parquet,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"), help="Directory holding the CSV.")
    args = parser.parse_args()

    origem = args.data / SHADE_RING_FACTOR_FILE
    destino = args.data / SHADE_RING_FACTOR_PARQUET
    inicio = time.perf_counter()
    solar_geometry_to_parquet(origem, destino)
    print(
        f"{destino}  {destino.stat().st_size / 1e6:.1f} MB "
        f"(de {origem.stat().st_size / 1e6:.0f} MB) em {time.perf_counter() - inicio:.2f} s"
    )


if __name__ == "__main__":
    main()
