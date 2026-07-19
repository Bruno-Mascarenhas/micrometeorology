"""Command-line entry points for the micrometeorology toolkit.

Each module in this package builds its own ``typer`` application and exposes a
``main()`` callable wired to a console script in ``pyproject.toml`` (for
example ``labmim-wrf-figures`` -> :mod:`render_wrf_maps`). The package holds no
shared state; import the specific command module you need.
"""
