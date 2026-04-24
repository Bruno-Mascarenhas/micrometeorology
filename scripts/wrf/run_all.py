"""Automate the processing of WRF output files."""

import glob
import logging
import subprocess
from pathlib import Path

from micrometeorology.common.logging import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the WRF post-processing pipeline on all available data."""
    setup_logging(level="INFO")

    data_dir = Path("/home/models/WRF/wrf-op/d-output/20260330/wrf01")
    output_json = Path("output/wrf_json")
    output_geojson = Path("output/wrf_geojson")
    output_figures = Path("output/wrf_figures")

    output_json.mkdir(parents=True, exist_ok=True)
    output_geojson.mkdir(parents=True, exist_ok=True)
    output_figures.mkdir(parents=True, exist_ok=True)

    wrf_files = sorted(glob.glob(str(data_dir / "wrfout*")))
    logger.info("Found %d wrfout files: %s", len(wrf_files), wrf_files)

    if not wrf_files:
        logger.warning("No wrfout files found in %s/", data_dir)
        return

    for f in wrf_files:
        logger.info("=" * 60)
        logger.info("Processing file: %s", f)
        logger.info("=" * 60)

        # 1. GeoJSON and JSON export
        cmd_export = [
            "python",
            "-m",
            "micrometeorology.cli.export_wrf_geojson",
            "--dataset",
            f,
            "-o",
            str(output_json),
            "-g",
            str(output_geojson),
        ]
        logger.info("Running: %s", " ".join(cmd_export))
        try:
            subprocess.run(cmd_export, check=True)
        except subprocess.CalledProcessError as e:
            logger.error("Error running export for %s: %s", f, e)

        # 2. Render maps
        # cmd_render = [
        #     "python",
        #     "-m",
        #     "micrometeorology.cli.render_wrf_maps",
        #     "--dataset",
        #     f,
        #     "-o",
        #     str(output_figures),
        # ]
        # logger.info("Running: %s", " ".join(cmd_render))
        # try:
        #     subprocess.run(cmd_render, check=True)
        # except subprocess.CalledProcessError as e:
        #     logger.error("Error running render for %s: %s", f, e)


if __name__ == "__main__":
    main()
