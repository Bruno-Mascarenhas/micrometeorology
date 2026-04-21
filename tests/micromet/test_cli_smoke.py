import pytest
from click.testing import CliRunner

from micrometeorology.cli.compare_wrf_observations import main as comparison_main
from micrometeorology.cli.compute_metrics import main as metrics_main
from micrometeorology.cli.export_wrf_geojson import main as wrf_geojson_main
from micrometeorology.cli.ingest_sensor_data import main as sensor_process_main
from micrometeorology.cli.plot_station_graphs import main as site_graphs_main
from micrometeorology.cli.render_wrf_maps import main as wrf_figures_main
from solrad_correction.cli import run_experiment_cli


@pytest.fixture
def runner():
    """Returns a Click CLI runner."""
    return CliRunner()


@pytest.mark.parametrize(
    ("command_fn", "name"),
    [
        (wrf_figures_main, "labmim-wrf-figures"),
        (wrf_geojson_main, "labmim-wrf-geojson"),
        (sensor_process_main, "labmim-sensor-process"),
        (site_graphs_main, "labmim-site-graphs"),
        (comparison_main, "labmim-comparison"),
        (metrics_main, "labmim-metrics"),
        (run_experiment_cli, "solrad-run"),
    ],
)
def test_cli_help(runner, command_fn, name):
    """Smoke test to ensure every CLI command can import and display its help text."""
    result = runner.invoke(command_fn, ["--help"])
    assert result.exit_code == 0, f"Command {name} failed: {result.output}"
    assert "Usage:" in result.output
