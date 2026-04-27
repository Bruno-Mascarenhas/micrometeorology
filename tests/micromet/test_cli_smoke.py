import pytest
from click.testing import CliRunner

from micrometeorology.cli.compare_wrf_observations import main as comparison_main
from micrometeorology.cli.compute_metrics import main as metrics_main
from micrometeorology.cli.export_wrf_geojson import main as wrf_geojson_main
from micrometeorology.cli.ingest_sensor_data import main as sensor_process_main
from micrometeorology.cli.plot_station_graphs import main as site_graphs_main
from micrometeorology.cli.render_wrf_maps import main as wrf_figures_main
from micrometeorology.cli.run_wrf_pipeline import main as wrf_pipeline_main
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
        (wrf_pipeline_main, "labmim-wrf-pipeline"),
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


def test_wrf_geojson_help_exposes_reader_and_worker_backends(runner):
    result = runner.invoke(wrf_geojson_main, ["--help"])

    assert result.exit_code == 0, result.output
    assert "--reader" in result.output
    assert "--chunks" in result.output
    assert "--worker-backend" in result.output
    assert "--tmp-dir" in result.output
    assert "[default: auto]" in result.output


def test_wrf_figures_help_exposes_lazy_reader_options(runner):
    result = runner.invoke(wrf_figures_main, ["--help"])

    assert result.exit_code == 0, result.output
    assert "--reader" in result.output
    assert "--chunks" in result.output
    assert "--worker-backend" in result.output
    assert "--tmp-dir" in result.output
    assert "[default: auto]" in result.output


def test_wrf_pipeline_help_exposes_reader_and_json_worker_backends(runner):
    result = runner.invoke(wrf_pipeline_main, ["--help"])

    assert result.exit_code == 0, result.output
    assert "--reader" in result.output
    assert "--chunks" in result.output
    assert "--json-worker-backend" in result.output
    assert "--figure-worker-backend" in result.output
    assert "--tmp-dir" in result.output
    assert "[default: auto]" in result.output
