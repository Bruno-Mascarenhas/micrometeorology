import pytest
from typer.testing import CliRunner

from micrometeorology.cli.compare_wrf_observations import app as comparison_app
from micrometeorology.cli.compute_metrics import app as metrics_app
from micrometeorology.cli.export_wrf_geojson import app as wrf_geojson_app
from micrometeorology.cli.ingest_sensor_data import app as sensor_process_app
from micrometeorology.cli.plot_station_graphs import app as site_graphs_app
from micrometeorology.cli.render_wrf_maps import app as wrf_figures_app
from micrometeorology.cli.run_wrf_pipeline import app as wrf_pipeline_app
from solrad_correction.cli import app as solrad_app


@pytest.fixture
def runner():
    """Returns a Typer CLI runner."""
    return CliRunner()


@pytest.mark.parametrize(
    ("command_app", "name"),
    [
        (wrf_figures_app, "labmim-wrf-figures"),
        (wrf_geojson_app, "labmim-wrf-geojson"),
        (wrf_pipeline_app, "labmim-wrf-pipeline"),
        (sensor_process_app, "labmim-sensor-process"),
        (site_graphs_app, "labmim-site-graphs"),
        (comparison_app, "labmim-comparison"),
        (metrics_app, "labmim-metrics"),
        (solrad_app, "solrad-run"),
    ],
)
def test_cli_help(runner, command_app, name):
    """Smoke test to ensure every CLI command can import and display its help text."""
    result = runner.invoke(command_app, ["--help"])
    assert result.exit_code == 0, f"Command {name} failed: {result.output}"
    assert "Usage:" in result.output


def test_wrf_geojson_help_exposes_reader_and_worker_backends(runner):
    result = runner.invoke(wrf_geojson_app, ["--help"])

    assert result.exit_code == 0, result.output
    assert "--reader" in result.output
    assert "--chunks" in result.output
    assert "--worker-backend" in result.output
    assert "--tmp-dir" in result.output


def test_wrf_figures_help_exposes_lazy_reader_options(runner):
    result = runner.invoke(wrf_figures_app, ["--help"])

    assert result.exit_code == 0, result.output
    assert "--reader" in result.output
    assert "--chunks" in result.output
    assert "--worker-backend" in result.output
    assert "--tmp-dir" in result.output


def test_wrf_pipeline_help_exposes_reader_and_json_worker_backends(runner):
    result = runner.invoke(wrf_pipeline_app, ["--help"], env={"COLUMNS": "120"})

    assert result.exit_code == 0, result.output
    assert "--reader" in result.output
    assert "--chunks" in result.output
    assert "--json-worker-backend" in result.output
    assert "--figure-worker-backend" in result.output
    assert "--tmp-dir" in result.output
