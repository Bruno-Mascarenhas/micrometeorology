"""CLI entry points for the labmim-micrometeorology package.

These are thin wrappers that delegate to the Click commands defined in
the ``scripts/micromet/`` directory.  They are registered as console
entry points in ``pyproject.toml``.
"""

from __future__ import annotations


def wrf_figures() -> None:
    """Entry point for ``labmim-wrf-figures``."""
    from scripts.micromet.process_wrf_figures import main

    main()


def wrf_geojson() -> None:
    """Entry point for ``labmim-wrf-geojson``."""
    from scripts.micromet.process_wrf_geojson import main

    main()


def sensor_process() -> None:
    """Entry point for ``labmim-sensor-process``."""
    from scripts.micromet.process_sensor_data import main

    main()


def site_graphs() -> None:
    """Entry point for ``labmim-site-graphs``."""
    from scripts.micromet.generate_site_graphs import main

    main()


def comparison() -> None:
    """Entry point for ``labmim-comparison``."""
    from scripts.micromet.run_comparison import main

    main()


def metrics() -> None:
    """Entry point for ``labmim-metrics``."""
    from scripts.micromet.run_metrics import main

    main()
