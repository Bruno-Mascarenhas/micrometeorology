"""CLI: Produce the LabMiM monitoring-page graphs from a processed sensor CSV.

This module is the **producer** for the nine fixed-name PNGs consumed by the
LabMiM public monitoring page (``https://labmim.if.ufba.br/monitoring.html``)
in the read-only sibling site repository (``site-labmim``).  The page hard-codes
the image names under ``assets/graphs/``; this CLI writes exactly those names so
a run overwrites them in place.  The consumer is **external** (cron/manual copy)
and therefore invisible to any reverse-import analysis of this repository.

The nine-image contract (``site`` command)::

    temperatura.png      <- AirT1_C_Avg    (line,    Temperatura do Ar)
    umidade.png          <- RH1            (line,    Umidade Relativa do Ar)
    pressao.png          <- BP1_mbar_Avg   (line,    Pressao Atmosferica)
    precipitacao.png     <- PL01_mm_Tot    (bar,     Precipitacao)
    velocidade.png       <- WS_ms          (line,    Velocidade do Vento)
    direcao.png          <- WindDir        (scatter, Direcao do Vento, 0-360)
    balanco.png          <- Net_Wm2_Avg    (line + optional CM3/CG3 components)
    radiacao_difusa.png  <- PSP_Wm2_Avg    (line,    Radiacao Difusa)
    radiacao_par.png     <- PAR_Wm2_Avg    (line,    Radiacao PAR)

Column names are **overridable** (a logger change must not require a code edit):
per-graph via repeatable ``--col KEY=COLUMN`` options, or in bulk via a small
YAML passed with ``--config`` (keys ``columns`` and ``balance_components``).

Examples
--------
Generate all nine monitoring-page PNGs for the last 7 days straight into a
checkout of the site (operational default target is ``site/assets/graphs/``)::

    labmim-site-graphs site -i data/hourly/sensor_data.csv \
        -o ../site-labmim/site/assets/graphs

Point at a renamed logger column without touching code::

    labmim-site-graphs site -i data/hourly/sensor_data.csv -o out/ \
        --col temperatura=AirT2_C_Avg --last-days 14

Ad-hoc per-variable graphs (secondary generic command, legacy filenames)::

    labmim-site-graphs columns -i data/hourly/sensor_data.csv -o out/ \
        -v AirT1_C_Avg -v RH1 --last-days 14
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import matplotlib

matplotlib.use("Agg")  # headless, no display — safe on a cron server

import matplotlib.pyplot as plt
import pandas as pd
import typer
import yaml

from micrometeorology.common.logging import setup_logging
from micrometeorology.common.paths import ensure_dir
from micrometeorology.sensors.plotting import (
    add_labmim_watermark,
    add_timestamp_label,
    add_top_legend,
    create_figure,
    save_figure,
    setup_date_axis,
)
from micrometeorology.sensors.wind import wind_direction_from_components

app = typer.Typer(rich_markup_mode="markdown", no_args_is_help=True)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Contract specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphSpec:
    """One entry of the nine-image monitoring-page contract.

    Attributes
    ----------
    key:
        Logical name, also the ``columns`` config key and the ``--col`` token.
    filename:
        Fixed output PNG name the site reads (never varies per run).
    ylabel:
        Portuguese y-axis label including units, matching the page language.
    kind:
        Plot style: ``"line"``, ``"scatter"`` (direction), ``"bar"``
        (precipitation), or ``"balance"`` (net radiation + components).
    ylim:
        Optional fixed y-limits reproducing the legacy graphs' framing.
    """

    key: str
    filename: str
    ylabel: str
    kind: str
    ylim: tuple[float, float] | None = None


# Ordered exactly as the monitoring page lays the cards out.
GRAPH_SPECS: tuple[GraphSpec, ...] = (
    GraphSpec("temperatura", "temperatura.png", "Temperatura do Ar (°C)", "line", (10, 40)),
    GraphSpec("umidade", "umidade.png", "Umidade Relativa do Ar (%)", "line", (0, 100)),
    GraphSpec("pressao", "pressao.png", "Pressão Atmosférica (hPa)", "line"),
    GraphSpec("precipitacao", "precipitacao.png", "Precipitação (mm)", "bar"),
    GraphSpec("velocidade", "velocidade.png", "Velocidade do Vento (m/s)", "line", (0, 15)),
    GraphSpec("direcao", "direcao.png", "Direção do Vento (°)", "scatter", (0, 360)),
    GraphSpec("balanco", "balanco.png", "Balanço de Radiação (W/m²)", "balance"),
    GraphSpec("radiacao_difusa", "radiacao_difusa.png", "Radiação Difusa (W/m²)", "line"),
    GraphSpec("radiacao_par", "radiacao_par.png", "Radiação PAR (W/m²)", "line"),
)

# Default column mapping: the processed-CSV column each contract graph reads.
# These are the hourly-export column names of ``labmim-sensor-process``
# (``sensors.export.export_csv``); override per-logger via --config / --col.
DEFAULT_COLUMNS: dict[str, str] = {
    "temperatura": "AirT1_C_Avg",
    "umidade": "RH1",
    "pressao": "BP1_mbar_Avg",
    "precipitacao": "PL01_mm_Tot",
    "velocidade": "WS_ms",
    "direcao": "WindDir",
    "balanco": "Net_Wm2_Avg",
    "radiacao_difusa": "PSP_Wm2_Avg",
    "radiacao_par": "PAR_Wm2_Avg",
}

# Optional radiation-balance components (CNR1 four-stream), plotted on the
# ``balanco`` graph when present.  The upward (``*_up``) channels are drawn
# negated, matching the legacy ``graficos1_UFBA_v5.py`` sign convention.
DEFAULT_BALANCE_COMPONENTS: dict[str, str] = {
    "sw_down": "CM3Up_Wm2_Avg",
    "sw_up": "CM3Dn_Wm2_Avg",
    "lw_down": "CG3Up_Wm2_Avg",
    "lw_up": "CG3Dn_Wm2_Avg",
}

# Fallback U/V component columns used to reconstruct wind direction when the
# direct ``direcao`` column is absent (see ``sensors.wind``).
DEFAULT_DIRECTION_COMPONENTS: tuple[str, str] = ("u", "v")


# ---------------------------------------------------------------------------
# Configuration resolution
# ---------------------------------------------------------------------------


def load_graph_config(
    config_path: Path | None,
    overrides: list[str] | None = None,
) -> tuple[dict[str, str], dict[str, str], tuple[str, str]]:
    """Resolve the column mapping from defaults, a YAML file, and CLI overrides.

    Precedence (lowest to highest): :data:`DEFAULT_COLUMNS` →
    ``--config`` YAML → ``--col KEY=COLUMN`` options.

    Parameters
    ----------
    config_path:
        Optional YAML with top-level ``columns`` (logical → column) and
        ``balance_components`` (channel → column) mappings, plus an optional
        ``direction_components`` ``[u, v]`` pair.
    overrides:
        ``"KEY=COLUMN"`` strings; ``KEY`` must be one of the nine contract keys.

    Returns
    -------
    tuple
        ``(columns, balance_components, direction_components)``.

    Raises
    ------
    ValueError
        If an override is malformed or names an unknown contract key.
    """
    columns = dict(DEFAULT_COLUMNS)
    balance = dict(DEFAULT_BALANCE_COMPONENTS)
    direction_components = DEFAULT_DIRECTION_COMPONENTS

    if config_path is not None:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        columns.update(data.get("columns", {}) or {})
        balance.update(data.get("balance_components", {}) or {})
        raw_dir = data.get("direction_components")
        if raw_dir:
            if len(raw_dir) != 2:
                raise ValueError("direction_components must be a [u, v] pair")
            direction_components = (str(raw_dir[0]), str(raw_dir[1]))

    for item in overrides or []:
        key, sep, value = item.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not value:
            raise ValueError(f"Invalid --col override {item!r}; expected KEY=COLUMN")
        if key not in DEFAULT_COLUMNS:
            raise ValueError(f"Unknown --col key {key!r}; valid keys: {', '.join(DEFAULT_COLUMNS)}")
        columns[key] = value

    return columns, balance, direction_components


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


def load_hourly_csv(input_path: Path, last_days: int) -> pd.DataFrame:
    """Load a processed hourly CSV and clip it to the most recent window.

    Parameters
    ----------
    input_path:
        CSV whose first column is the timestamp index — the default
        (``include_datetime_columns=False``) export of ``labmim-sensor-process``.
    last_days:
        Keep only rows within ``last_days`` of the newest timestamp. A value
        ``<= 0`` disables the clip and keeps the whole file.

    Returns
    -------
    pandas.DataFrame
        The (optionally clipped) frame with a sorted ``DatetimeIndex``.
    """
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    if last_days > 0 and not df.empty:
        cutoff = df.index.max() - pd.Timedelta(days=last_days)
        df = df.loc[df.index >= cutoff]

    return df


# ---------------------------------------------------------------------------
# Per-kind renderers
# ---------------------------------------------------------------------------


def _plot_line(ax: plt.Axes, series: pd.Series, *, label: str) -> None:
    """Draw an hourly line with small markers (temperature, humidity, ...)."""
    ax.plot(series.index, series.to_numpy(), "o-", markersize=3, linewidth=1.0, label=label)


def _plot_scatter(ax: plt.Axes, series: pd.Series, *, label: str) -> None:
    """Scatter wind direction as dots on a fixed 0-360 axis.

    Limitation
    ----------
    Direction is circular (359° and 1° are adjacent); a connecting line
    would draw a spurious full-range sweep across the wrap. Dots avoid that.
    """
    ax.plot(series.index, series.to_numpy() % 360.0, "o", markersize=4, color="black", label=label)
    ax.set_yticks([0, 90, 180, 270, 360])


def _plot_bar(ax: plt.Axes, series: pd.Series, *, label: str) -> None:
    """Draw hourly precipitation accumulation as bars."""
    width = 0.9 / 24.0  # matplotlib date units are days; ~90% of one hour
    ax.bar(series.index, series.to_numpy(), width=width, color="tab:blue", label=label)


def _plot_balance(
    ax: plt.Axes,
    net: pd.Series,
    components: dict[str, pd.Series],
) -> None:
    """Draw net radiation plus any available four-stream components.

    Formula
    -------
    Net radiation ``Rn = (SW_down - SW_up) + (LW_down - LW_up)``. Upward
    channels are plotted negated so the stacked lines visually sum toward
    ``Rn``, following the legacy ``graficos1_UFBA_v5.py`` convention.
    """
    ax.plot(net.index, net.to_numpy(), "p-", color="black", label="Rn")
    styling = {
        "sw_down": ("SW_dw", "red", 1.0),
        "sw_up": ("SW_up", "blue", -1.0),
        "lw_down": ("LW_dw", "green", 1.0),
        "lw_up": ("LW_up", "orange", -1.0),
    }
    for channel, (label, color, sign) in styling.items():
        series = components.get(channel)
        if series is not None:
            ax.plot(series.index, sign * series.to_numpy(), "-", color=color, label=label)


# ---------------------------------------------------------------------------
# Contract driver
# ---------------------------------------------------------------------------


def _resolve_direction_series(
    df: pd.DataFrame,
    direction_column: str,
    components: tuple[str, str],
) -> pd.Series | None:
    """Return the wind-direction series, reconstructing it from U/V if needed.

    Uses the direct ``direction_column`` when present; otherwise, if both U/V
    component columns exist, reconstructs direction via
    :func:`micrometeorology.sensors.wind.wind_direction_from_components`.
    Returns ``None`` when neither source is available.
    """
    if direction_column in df.columns:
        return df[direction_column]
    u_col, v_col = components
    if u_col in df.columns and v_col in df.columns:
        logger.info(
            "Direction column %r absent; reconstructing from U/V (%s, %s)",
            direction_column,
            u_col,
            v_col,
        )
        direction = wind_direction_from_components(df[u_col].to_numpy(), df[v_col].to_numpy())
        return pd.Series(direction, index=df.index, name=direction_column)
    return None


def render_site_graphs(
    df: pd.DataFrame,
    output_dir: Path,
    columns: dict[str, str],
    balance_components: dict[str, str],
    direction_components: tuple[str, str],
) -> tuple[list[Path], list[str]]:
    """Render every contract graph whose source column is present.

    Parameters
    ----------
    df:
        Hourly frame with a ``DatetimeIndex`` (already clipped to the window).
    output_dir:
        Directory receiving the fixed-name PNGs (created if missing).
    columns:
        Logical-key → CSV-column mapping (see :func:`load_graph_config`).
    balance_components:
        Channel → CSV-column mapping for the optional radiation-balance streams.
    direction_components:
        ``(u, v)`` fallback columns for wind-direction reconstruction.

    Returns
    -------
    tuple
        ``(written_paths, missing_keys)`` — one entry in ``missing_keys`` per
        contract graph whose primary column was absent (its PNG is skipped).
    """
    out = ensure_dir(output_dir)
    label_dt = df.index.max() if not df.empty else None

    written: list[Path] = []
    missing: list[str] = []

    for spec in GRAPH_SPECS:
        column = columns[spec.key]

        if spec.kind == "scatter":
            series = _resolve_direction_series(df, column, direction_components)
            if series is None:
                logger.warning(
                    "Column %r (and U/V fallback) not found -- skipping %s",
                    column,
                    spec.filename,
                )
                missing.append(spec.key)
                continue
        elif column not in df.columns:
            logger.warning("Column %r not found -- skipping %s", column, spec.filename)
            missing.append(spec.key)
            continue
        else:
            series = df[column]

        fig, ax = create_figure()
        try:
            if spec.kind == "line":
                _plot_line(ax, series, label=column)
            elif spec.kind == "scatter":
                _plot_scatter(ax, series, label=column)
            elif spec.kind == "bar":
                _plot_bar(ax, series, label=column)
            elif spec.kind == "balance":
                present = {
                    channel: df[col]
                    for channel, col in balance_components.items()
                    if col in df.columns
                }
                _plot_balance(ax, series, present)

            if spec.ylim is not None:
                ax.set_ylim(*spec.ylim)
            setup_date_axis(ax)
            ax.set_ylabel(spec.ylabel, fontsize=12)
            if label_dt is not None:
                add_timestamp_label(ax, label_dt)
            add_labmim_watermark(ax)
            if ax.get_legend_handles_labels()[0]:
                add_top_legend(ax, ncol=4)
            written.append(save_figure(fig, out / spec.filename))
        finally:
            plt.close(fig)

    return written, missing


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def site(
    input_path: Annotated[
        Path, typer.Option("-i", "--input", help="Processed hourly sensor CSV.", exists=True)
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "-o",
            "--output",
            help=(
                "Directory for the nine contract PNGs. Operationally point this "
                "at the site checkout's `site/assets/graphs/`."
            ),
        ),
    ] = Path("output/site_graphs"),
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            help="YAML overriding column names (keys `columns`, `balance_components`).",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    col: Annotated[
        list[str] | None,
        typer.Option("--col", help="Per-graph column override `KEY=COLUMN` (repeatable)."),
    ] = None,
    last_days: Annotated[
        int, typer.Option("--last-days", help="Days back from the newest timestamp.")
    ] = 7,
    strict: Annotated[
        bool,
        typer.Option("--strict", help="Exit non-zero if any contract column is missing."),
    ] = False,
    log_level: Annotated[str, typer.Option(help="Logging level.")] = "INFO",
) -> None:
    """Generate the nine LabMiM monitoring-page PNGs from a processed CSV.

    Writes fixed-name images (`temperatura.png`, `umidade.png`, ...) that the
    `site-labmim` monitoring page reads by exact name. A missing source column
    logs a warning and skips only that image, still exiting 0 -- unless
    `--strict` is given, which turns any missing contract column into a
    non-zero exit.
    """
    setup_logging(log_level)

    columns, balance_components, direction_components = load_graph_config(config, col)
    df = load_hourly_csv(input_path, last_days)

    if df.empty:
        typer.echo("[!] No rows in the requested window -- nothing to plot.")
        raise typer.Exit(code=1 if strict else 0)

    written, missing = render_site_graphs(
        df, output_dir, columns, balance_components, direction_components
    )

    for path in written:
        typer.echo(f"  [ok] {path.name}")
    if missing:
        typer.echo(f"[!] Skipped (missing column): {', '.join(missing)}")
    typer.echo(f"\n>> {len(written)} graph(s) saved to {output_dir}")

    if strict and missing:
        raise typer.Exit(code=1)


@app.command()
def columns(
    input_path: Annotated[
        Path, typer.Option("-i", "--input", help="Processed sensor CSV file.", exists=True)
    ],
    output_dir: Annotated[
        Path, typer.Option("-o", "--output", help="Output directory for graphs.")
    ],
    variables: Annotated[list[str], typer.Option("-v", "--variables", help="Columns to plot.")],
    last_days: Annotated[
        int, typer.Option("--last-days", help="Number of recent days to plot.")
    ] = 7,
    log_level: Annotated[str, typer.Option(help="Logging level.")] = "INFO",
) -> None:
    """Generic per-variable time-series graphs (ad-hoc, legacy filenames).

    Preserves the original behaviour: one line graph per requested column,
    written as `{column}_last_{N}d.png`. Unknown columns warn and are skipped.
    """
    setup_logging(log_level)
    out = ensure_dir(output_dir)

    df = load_hourly_csv(input_path, last_days)

    for var in variables:
        if var not in df.columns:
            typer.echo(f"Warning: Column '{var}' not found -- skipping")
            continue

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df.index, df[var].to_numpy(), linewidth=0.8)
        ax.set_ylabel(var)
        ax.set_title(f"{var} -- Last {last_days} days")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(out / f"{var}_last_{last_days}d.png", dpi=150)
        plt.close(fig)
        typer.echo(f">> {var}")

    typer.echo(f"\n>> Graphs saved to {out}")


def main() -> None:
    """Console-script entry point (pyproject: ``labmim-site-graphs``)."""
    app()


if __name__ == "__main__":
    main()
