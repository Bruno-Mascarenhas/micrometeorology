"""CLI: Produce the LabMiM monitoring-page graphs from a processed sensor CSV.

Producer for the nine fixed-name PNGs the LabMiM monitoring page
(``https://labmim.if.ufba.br/monitoring.html``, read-only sibling repository
``site-labmim``) hard-codes under ``assets/graphs/``; a run overwrites them in
place. The consumer is external (cron/manual copy), so no reverse-import
analysis of this repository can see it.

The nine-image contract (``site`` command)::

    temperatura.png      <- T -> AirT1_C_Avg         (line,    Temperatura do Ar)
    umidade.png          <- ur -> RH1                (line,    Umidade Relativa do Ar)
    pressao.png          <- pressure -> BP1_mbar_Avg (line,    Pressao Atmosferica)
    precipitacao.png     <- precip -> PL01_mm_Tot    (bar,     Precipitacao)
    velocidade.png       <- WS -> WS_ms              (line,    Velocidade do Vento)
    direcao.png          <- WD -> WindDir            (scatter, Direcao do Vento, 0-360)
    balanco.png          <- Net_CNR1 -> Net_Wm2_Avg  (line + optional CM3/CG3 components)
    radiacao_difusa.png  <- Sw_dif -> PSP_Wm2_Avg    (line,    Radiacao Difusa)
    radiacao_par.png     <- Sw_par -> PAR_Wm2_Avg    (line,    Radiacao PAR)

The left name of each chain is the unified archive column, the right one the raw
logger column; the first present in the frame is the one plotted.

Column names are **overridable** (a logger change must not require a code edit):
per-graph via repeatable ``--col KEY=COLUMN`` options, or in bulk via a small
YAML passed with ``--config`` (keys ``columns`` and ``balance_components``).

Examples
--------
All nine PNGs for the last 7 days, straight into a site checkout (the
operational target is ``site/assets/graphs/``)::

    labmim-site-graphs site -i data/hourly/sensor_data.csv \
        -o ../site-labmim/site/assets/graphs

Point at a renamed logger column without touching code::

    labmim-site-graphs site -i data/hourly/sensor_data.csv -o out/ \
        --col temperatura=AirT2_C_Avg --last-days 14

Ad-hoc per-variable graphs (secondary generic command, legacy filenames)::

    labmim-site-graphs columns -i data/hourly/sensor_data.csv -o out/ \
        -v AirT1_C_Avg -v RH1 --last-days 14
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import matplotlib

matplotlib.use("Agg")  # headless, no display — safe on a cron server

import matplotlib.pyplot as plt
import pandas as pd
import typer
import yaml
from matplotlib.typing import ColorType

from micrometeorology.common.logging import setup_logging
from micrometeorology.common.paths import ensure_dir

# Imported, not re-typed: a second literal of the same ordered tuple defeats
# what makes a new extraction variable a data change rather than a code change.
from micrometeorology.sensors.monitoring import _WRF_RAIN_CANDIDATES
from micrometeorology.sensors.plotting import (
    BALANCE_COMPONENT_COLORS,
    add_labmim_watermark,
    add_timestamp_label,
    add_top_legend,
    create_figure,
    save_figure,
    setup_date_axis,
)
from micrometeorology.sensors.wind import wind_direction_from_components
from micrometeorology.wrf.columns import (
    PSFC_HPA,
    RH_PCT,
    SWDDIF_W_M2,
    SWDOWN_W_M2,
    T2_C,
    WIND_DIR_DEG,
    WIND_SPEED_M_S,
)

app = typer.Typer(rich_markup_mode="markdown", no_args_is_help=True)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
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
    kind: Literal["line", "scatter", "bar", "balance"]
    ylim: tuple[float, float] | None = None


# 90% of one hour, in matplotlib's date units (days).
HOURLY_BAR_WIDTH_DAYS = 0.9 / 24.0

# Ordered exactly as the monitoring page lays the cards out.
GRAPH_SPECS: tuple[GraphSpec, ...] = (
    GraphSpec("temperatura", "temperatura.png", "Temperatura do Ar (°C)", "line", (10, 40)),
    # 105, not 100: the model's RH is uncapped and exceeds 100% in 314 of the
    # 24,816 hours the extraction carries (max 101.6), which a 0-100 frame would
    # clip unmarked. MONITORING_CHARTS declares the same 105.
    GraphSpec("umidade", "umidade.png", "Umidade Relativa do Ar (%)", "line", (0, 105)),
    GraphSpec("pressao", "pressao.png", "Pressão Atmosférica (hPa)", "line"),
    GraphSpec("precipitacao", "precipitacao.png", "Precipitação (mm)", "bar"),
    GraphSpec("velocidade", "velocidade.png", "Velocidade do Vento (m/s)", "line", (0, 15)),
    GraphSpec("direcao", "direcao.png", "Direção do Vento (°)", "scatter", (0, 360)),
    GraphSpec("balanco", "balanco.png", "Balanço de Radiação (W/m²)", "balance"),
    GraphSpec("radiacao_difusa", "radiacao_difusa.png", "Radiação Difusa (W/m²)", "line"),
    GraphSpec("radiacao_par", "radiacao_par.png", "Radiação PAR (W/m²)", "line"),
)

# Candidates each contract graph reads, best first: the chain head is the
# UNIFIED name ``sensor_switches`` builds (``labmim-archive``), the tail the raw
# logger column ``labmim-sensor-process`` exports. A raw column is ONE
# instrument while the graph's quantity changes instrument over the record — the
# shade ring was off the PSP from 2019-09 to 2025-05 (calibrations.yaml, Sw_dif
# era map) with the CMP21 carrying diffuse, so a window inside those six years
# resolved through ``PSP_Wm2_Avg`` publishes near-global irradiance under
# "Radiação Difusa". A historical window replayed through the non-unifying
# sensor-process path still hits that raw tail; feed it from the archive's
# hourly frame, or override the chain with --config / --col.
DEFAULT_COLUMNS: dict[str, tuple[str, ...]] = {
    "temperatura": ("T", "AirT1_C_Avg"),
    "umidade": ("ur", "RH1"),
    "pressao": ("pressure", "BP1_mbar_Avg"),
    "precipitacao": ("precip", "PL01_mm_Tot"),
    "velocidade": ("WS", "WS_ms"),
    "direcao": ("WD", "WindDir"),
    "balanco": ("Net_CNR1", "Net_Wm2_Avg"),
    "radiacao_difusa": ("Sw_dif", "PSP_Wm2_Avg"),
    "radiacao_par": ("Sw_par", "PAR_Wm2_Avg"),
}

# Optional radiation-balance components (CNR1 four-stream), plotted on the
# ``balanco`` graph when present.  The upward (``*_up``) channels are drawn
# negated, matching the legacy ``graficos1_UFBA_v5.py`` sign convention.
DEFAULT_BALANCE_COMPONENTS: dict[str, tuple[str, ...]] = {
    "sw_down": ("Sw_dw", "CM3Up_Wm2_Avg"),
    "sw_up": ("Sw_up", "CM3Dn_Wm2_Avg"),
    # The ``Cr`` spellings are the FLUXES; plain ``CG3*_Wm2_Avg`` is the
    # pyrgeometers' raw thermopile signal, missing the sigma*T_body^4 case term:
    # over 2022-07-01..08 it averages -41.7 and +0.9 W/m2 against fluxes of
    # +405.7 and +448.2. The case term cancels in the NET (-42.55 W/m2 either
    # way), so the raw columns still sum visually to Rn while both are wrong by
    # ~447 W/m2 — closure is no evidence the right columns were picked.
    "lw_down": ("Lw_dw", "CG3Up_Wm2Cr_Avg"),
    "lw_up": ("Lw_up", "CG3Dn_Wm2Cr_Avg"),
}

# Fallback U/V component columns used to reconstruct wind direction when the
# direct ``direcao`` column is absent (see ``sensors.wind``).
DEFAULT_DIRECTION_COMPONENTS: tuple[str, str] = ("u", "v")

# The model layer's colour. Validated against the site's chart palette so the
# PNGs and the interactive page agree on what "WRF" looks like.
_WRF_COLOR = "#e07a1f"

# Candidate WRF column names per contract graph, in priority order. A tuple, not
# a single name, because ``series_operacional.dat`` gains variables over time:
# precipitation is absent today; the overlay appears the day it lands, no edit.
DEFAULT_WRF_COLUMNS: dict[str, tuple[str, ...]] = {
    "temperatura": (T2_C,),
    "umidade": (RH_PCT,),
    "pressao": (PSFC_HPA,),
    "precipitacao": _WRF_RAIN_CANDIDATES,
    "velocidade": (WIND_SPEED_M_S,),
    "direcao": (WIND_DIR_DEG,),
    "radiacao_difusa": (SWDDIF_W_M2,),
    # Incoming shortwave only: the four-stream balance would need the model's
    # upwelling terms, which this graph has never drawn.
    "balanco": (SWDOWN_W_M2,),
    # No PAR in the point extraction.
    "radiacao_par": (),
}


def _as_chain(value: object) -> tuple[str, ...]:
    """Normalise one configured mapping to a candidate chain, best first.

    A scalar replaces the whole default chain: an operator retargeting a renamed
    logger means that column, not that column behind the shipped alternatives.
    """
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list | tuple):
        return tuple(str(item) for item in value)
    raise ValueError(f"Column mapping must be a name or a list of names, got {value!r}")


def resolve_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    """First candidate the frame carries, or ``None`` if it carries none."""
    return next((name for name in candidates if name in frame.columns), None)


def load_graph_config(
    config_path: Path | None,
    overrides: list[str] | None = None,
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]], tuple[str, str]]:
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
        columns.update(
            {key: _as_chain(value) for key, value in (data.get("columns", {}) or {}).items()}
        )
        balance.update(
            {
                key: _as_chain(value)
                for key, value in (data.get("balance_components", {}) or {}).items()
            }
        )
        raw_dir = data.get("direction_components")
        if raw_dir:
            if not isinstance(raw_dir, list | tuple) or len(raw_dir) != 2:
                raise ValueError("direction_components must be a [u, v] pair")
            direction_components = (str(raw_dir[0]), str(raw_dir[1]))

    for item in overrides or []:
        key, sep, value = item.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not value:
            raise ValueError(f"Invalid --col override {item!r}; expected KEY=COLUMN")
        if key not in DEFAULT_COLUMNS:
            raise ValueError(f"Unknown --col key {key!r}; valid keys: {', '.join(DEFAULT_COLUMNS)}")
        columns[key] = _as_chain(value)

    return columns, balance, direction_components


def _read_time_indexed(path: Path) -> pd.DataFrame:
    """Read a Parquet or CSV frame and return it on a sorted ``DatetimeIndex``.

    Parameters
    ----------
    path:
        ``.parquet`` from ``labmim-archive``, or the CSV ``labmim-sensor-process``
        exports with the timestamp in its first column.

    Returns
    -------
    pandas.DataFrame
        ``(N, C)``, indexed by naive station-local stamps in ascending order.
    """
    frame = (
        pd.read_parquet(path)
        if path.suffix == ".parquet"
        else pd.read_csv(path, index_col=0, parse_dates=True)
    )
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index)
    return frame.sort_index()


def load_hourly_csv(input_path: Path, last_days: int) -> pd.DataFrame:
    """Load a processed hourly frame and clip it to the most recent window.

    Parameters
    ----------
    input_path:
        CSV whose first column is the timestamp index (the default
        ``include_datetime_columns=False`` export of ``labmim-sensor-process``)
        or ``station_hourly.parquet`` from ``labmim-archive``. Only the archive
        frame carries the unified, era-mapped names, so only it names the right
        instrument for a historical radiation window.
    last_days:
        Keep only rows within ``last_days`` of the newest timestamp. A value
        ``<= 0`` disables the clip and keeps the whole file.

    Returns
    -------
    pandas.DataFrame
        The (optionally clipped) frame with a sorted ``DatetimeIndex``.
    """
    df = _read_time_indexed(input_path)

    if last_days > 0 and not df.empty:
        cutoff = df.index.max() - pd.Timedelta(days=last_days)
        df = df.loc[df.index >= cutoff]

    return df


def _plot_line(ax: plt.Axes, series: pd.Series, *, label: str, over_raw: bool = False) -> None:
    """Draw the hourly line (temperature, humidity, ...).

    Markers are dropped when a raw layer sits underneath: at 12 raw samples per
    hour they cover exactly the points that layer exists to expose.
    """
    style = "-" if over_raw else "o-"
    ax.plot(
        series.index,
        series.to_numpy(),
        style,
        markersize=3,
        linewidth=1.6 if over_raw else 1.0,
        zorder=2,
        label=label,
    )


def _plot_raw(ax: plt.Axes, series: pd.Series, *, label: str, dots: bool = True) -> None:
    """Draw the raw logger samples as the recessive layer under the hourly line.

    Low-contrast and unconnected on purpose: at equal weight this context layer
    would hide the hourly mean it exists to be compared against.
    """
    style = "." if dots else "-"
    ax.plot(
        series.index,
        series.to_numpy(),
        style,
        markersize=2.6,
        linewidth=0.8,
        color="0.52",
        alpha=0.75,
        zorder=1,
        label=label,
    )


def _plot_wrf(
    ax: plt.Axes,
    series: pd.Series,
    *,
    label: str,
    dots: bool = False,
    color: ColorType | None = None,
) -> None:
    """Draw the model series, visually distinct from anything measured.

    The balance chart separates three things onto three channels: HUE for the
    physical family, solid/dashed for the flux direction, DOTTED for the model —
    hence the overlay borrows the hue of the series it mirrors (``color``).

    Direction passes ``dots=True``: joining 350 deg to 10 deg would sweep the
    axis through a bearing that never occurred.
    """
    if dots:
        ax.plot(
            series.index,
            series.to_numpy() % 360.0,
            "s",
            markersize=3,
            markerfacecolor="none",
            color=color or _WRF_COLOR,
            zorder=3,
            label=label,
        )
        return
    ax.plot(
        series.index,
        series.to_numpy(),
        linestyle=(0, (1.5, 1.5)),
        linewidth=1.7,
        color=color or _WRF_COLOR,
        zorder=3,
        label=label,
    )


def _plot_scatter(ax: plt.Axes, series: pd.Series, *, label: str) -> None:
    """Scatter wind direction as dots on a fixed 0-360 axis.

    Direction is circular (359° and 1° are adjacent), so a connecting line would
    draw a spurious full-range sweep across the wrap.
    """
    ax.plot(series.index, series.to_numpy() % 360.0, "o", markersize=4, color="black", label=label)
    ax.set_yticks([0, 90, 180, 270, 360])


def _plot_bar(ax: plt.Axes, series: pd.Series, *, label: str) -> None:
    """Draw hourly precipitation accumulation as bars."""
    ax.bar(
        series.index,
        series.to_numpy(),
        width=HOURLY_BAR_WIDTH_DAYS,
        color="tab:blue",
        label=label,
    )


def _plot_balance(
    ax: plt.Axes,
    net: pd.Series | None,
    components: dict[str, pd.Series],
) -> None:
    """Draw net radiation plus any available four-stream components.

    ``Rn = (SW_down - SW_up) + (LW_down - LW_up)``: the upward channels are
    plotted negated so the lines visually sum toward ``Rn``, following the
    legacy ``graficos1_UFBA_v5.py`` convention.  *net* is ``None`` when the
    logger's own net column carries nothing over the window, which leaves the
    four measured components to speak for the balance on their own.
    """
    if net is not None:
        ax.plot(net.index, net.to_numpy(), "p-", color="black", label="Rn")
    styling = {
        "sw_down": ("SW_dw", 1.0),
        "sw_up": ("SW_up", -1.0),
        "lw_down": ("LW_dw", 1.0),
        "lw_up": ("LW_up", -1.0),
    }
    for channel, (label, sign) in styling.items():
        series = components.get(channel)
        if series is not None:
            ax.plot(
                series.index,
                sign * series.to_numpy(),
                "-",
                color=BALANCE_COMPONENT_COLORS[channel],
                label=label,
            )


def _resolve_direction_series(
    df: pd.DataFrame,
    direction_column: str,
    components: tuple[str, str],
) -> pd.Series | None:
    """Return the wind-direction series, reconstructing it from U/V if needed.

    Falls back to
    :func:`micrometeorology.sensors.wind.wind_direction_from_components` when
    the direct column is absent; ``None`` when neither source exists.
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


def _wrf_series(
    wrf: pd.DataFrame | None, key: str, candidates: dict[str, tuple[str, ...]]
) -> tuple[pd.Series | None, str | None]:
    """First model column present for this graph, with the name that resolved."""
    if wrf is None or wrf.empty:
        return None, None
    name = resolve_column(wrf, candidates.get(key, ()))
    return (None, None) if name is None else (wrf[name], name)


def render_site_graphs(
    df: pd.DataFrame,
    output_dir: Path,
    columns: Mapping[str, Sequence[str]],
    balance_components: Mapping[str, Sequence[str]],
    direction_components: tuple[str, str],
    *,
    raw: pd.DataFrame | None = None,
    wrf: pd.DataFrame | None = None,
) -> tuple[list[Path], list[str], list[str]]:
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
    raw:
        Optional high-frequency record drawn as the recessive layer beneath the
        hourly line, on the column that already resolved against ``df`` so one
        chart stays one quantity.
    wrf:
        Optional model frame drawn on top, hourly, on a time index.

    Returns
    -------
    tuple
        ``(written_paths, missing_keys, empty_keys)`` — ``missing_keys`` names
        the contract graphs whose source column was absent (PNG skipped), and
        ``empty_keys`` those whose station column carried no finite value over
        the window (drawn from the remaining layers, or not written at all).
    """
    out = ensure_dir(output_dir)
    label_dt = df.index.max() if not df.empty else None

    written: list[Path] = []
    missing: list[str] = []
    empty: list[str] = []

    for spec in GRAPH_SPECS:
        candidates = columns[spec.key]
        # Resolved on the HOURLY frame and reused for the other layers, so one
        # chart is one quantity — not the archive's ``Sw_dif`` under raw PSP.
        column = resolve_column(df, candidates) or candidates[0]

        if spec.kind == "scatter":
            series = _resolve_direction_series(df, column, direction_components)
            if series is None:
                logger.warning(
                    "Columns %s (and U/V fallback) not found -- skipping %s",
                    ", ".join(repr(name) for name in candidates),
                    spec.filename,
                )
                missing.append(spec.key)
                continue
        elif column not in df.columns:
            logger.warning(
                "Columns %s not found -- skipping %s",
                ", ".join(repr(name) for name in candidates),
                spec.filename,
            )
            missing.append(spec.key)
            continue
        else:
            series = df[column]

        fig, ax = create_figure()
        drawn_raw = raw is not None and column in raw.columns
        try:
            if raw is not None and drawn_raw:
                _plot_raw(
                    ax,
                    raw[column] % 360.0 if spec.kind == "scatter" else raw[column],
                    label="bruto 5 min",
                    dots=spec.kind != "bar",
                )

            # A column that EXISTS but holds no finite value over the window is
            # skipped like an absent one: it draws nothing yet still registers a
            # legend entry, which reads as a line off the scale.
            present = (
                {
                    channel: df[resolved_component]
                    for channel, chain in balance_components.items()
                    if (resolved_component := resolve_column(df, chain)) is not None
                    and df[resolved_component].notna().any()
                }
                if spec.kind == "balance"
                else {}
            )
            station_drawn = series.notna().any()
            if not station_drawn and not present:
                logger.warning(
                    "Column %r has no value over the plotted window -- "
                    "drawing %s without the station layer",
                    column,
                    spec.filename,
                )
                empty.append(spec.key)
            elif spec.kind == "line":
                _plot_line(ax, series, label=column, over_raw=drawn_raw)
            elif spec.kind == "scatter":
                _plot_scatter(ax, series, label=column)
            elif spec.kind == "bar":
                _plot_bar(ax, series, label=column)
            elif spec.kind == "balance":
                if not station_drawn:
                    logger.warning(
                        "Column %r has no value over the plotted window -- "
                        "drawing %s from the four components alone",
                        column,
                        spec.filename,
                    )
                _plot_balance(ax, series if station_drawn else None, present)
            else:
                raise ValueError(f"{spec.filename}: unknown graph kind {spec.kind!r}")

            model, resolved = _wrf_series(wrf, spec.key, DEFAULT_WRF_COLUMNS)
            if model is not None:
                _plot_wrf(
                    ax,
                    model,
                    label=f"WRF 1h ({resolved})",
                    dots=spec.kind == "scatter",
                    # On balance the overlay mirrors incoming shortwave: same hue.
                    color=BALANCE_COMPONENT_COLORS["sw_down"] if spec.kind == "balance" else None,
                )

            if spec.ylim is not None:
                ax.set_ylim(*spec.ylim)
            setup_date_axis(ax)
            ax.set_ylabel(spec.ylabel, fontsize=12)
            if label_dt is not None:
                add_timestamp_label(ax, label_dt)
            add_labmim_watermark(ax)
            handles = ax.get_legend_handles_labels()[0]
            if not handles:
                # An empty framed axis under a contract filename is
                # indistinguishable from a calm day, so the old image stays.
                logger.warning("%s: no layer had data -- not written", spec.filename)
                if spec.key not in empty:
                    # One entry per bare chart, not one per reason it is bare.
                    empty.append(spec.key)
                continue
            add_top_legend(ax, ncol=4)
            written.append(save_figure(fig, out / spec.filename))
        finally:
            plt.close(fig)

    return written, missing, empty


def _load_raw_layer(path: Path, hourly: pd.DataFrame) -> pd.DataFrame:
    """Load the raw record and clip it to the window the hourly frame covers."""
    frame = _read_time_indexed(path)
    clipped: pd.DataFrame = frame.loc[hourly.index.min() : hourly.index.max()]
    logger.info("raw layer: %d samples over the plotted window", len(clipped))
    return clipped


def _load_wrf(path: Path, hourly: pd.DataFrame) -> pd.DataFrame:
    """Load the model series through the shared defensive reader and clip it."""
    from micrometeorology.wrf.operational_record import read_wrf_series

    frame = read_wrf_series(
        path, consumes=[name for chain in DEFAULT_WRF_COLUMNS.values() for name in chain]
    )
    clipped: pd.DataFrame = frame.loc[hourly.index.min() : hourly.index.max()]
    logger.info("wrf layer: %d hours over the plotted window", len(clipped))
    return clipped


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
    raw_path: Annotated[
        Path | None,
        typer.Option(
            "--raw",
            help=(
                "Raw high-frequency record (parquet or CSV) drawn UNDER the hourly "
                "line, so the aggregation can be judged against what it came from."
            ),
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    wrf_path: Annotated[
        Path | None,
        typer.Option(
            "--wrf",
            help=(
                "series_operacional.dat; its column for each graph is drawn as a "
                "dashed overlay. Resolved by candidate name, so a variable the "
                "extraction gains later appears without a code change."
            ),
            exists=True,
            dir_okay=False,
        ),
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

    try:
        columns, balance_components, direction_components = load_graph_config(config, col)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    df = load_hourly_csv(input_path, last_days)

    if df.empty:
        typer.echo("[!] No rows in the requested window -- nothing to plot.")
        raise typer.Exit(code=1 if strict else 0)

    raw = _load_raw_layer(raw_path, df) if raw_path else None
    wrf = _load_wrf(wrf_path, df) if wrf_path else None

    written, missing, empty = render_site_graphs(
        df,
        output_dir,
        columns,
        balance_components,
        direction_components,
        raw=raw,
        wrf=wrf,
    )

    written_keys = {path.stem for path in written}
    for path in written:
        typer.echo(f"  [ok] {path.name}")
    if missing:
        typer.echo(f"[!] Skipped (missing column): {', '.join(missing)}")
    if empty:
        # Two states share this list: drawn from the other layers, or not
        # written at all. Split so a dead sensor reads apart from a dead run.
        bare = [key for key in empty if key not in written_keys]
        partial = [key for key in empty if key in written_keys]
        if partial:
            typer.echo(f"[!] Drawn without the station layer: {', '.join(partial)}")
        if bare:
            typer.echo(f"[!] Not written, no layer had data: {', '.join(bare)}")
    typer.echo(f"\n>> {len(written)} graph(s) saved to {output_dir}")

    # MISSING fails, empty does not: broken configuration versus broken
    # instrument. The Gill thermohygrometer railed in December 2025, so
    # temperature and humidity are empty in EVERY current window; failing on
    # that would hold the hourly cron non-zero and freeze the site until repair.
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
