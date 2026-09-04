"""The monitoring-window contract: which series each chart draws, from which source.

The monitoring page shows a rolling window of the station record in THREE
LAYERS:

1. the **raw** samples at the logger's own cadence, so the data can be inspected
   rather than only its summary;
2. the **hourly mean** drawn over them, so the aggregation can be judged against
   what it came from;
3. the **WRF** series for the same variable and period, wherever one exists.

This module owns only the catalogue and the payload shape. Reading the archive
belongs to :mod:`micrometeorology.sensors.archive`, aggregation to
:mod:`micrometeorology.sensors.aggregation`, and the JSON encoding rules are
shared with :mod:`micrometeorology.stats.climatology_export`.

Why the WRF column is a LIST of candidates
------------------------------------------
``series_operacional.dat`` gains variables over time. So every chart declares an
ORDERED TUPLE of candidate column names and the exporter resolves whichever one
the file actually carries; a variable the extraction does not write yet is
declared so the payload can say the layer is absent instead of rendering a
silently short legend. ``precip_mm`` is the live case: the column enters the tuple
and lights its chart with no change here.

The tuples carry only names the record can actually hold. A file still on the v1
schema is renamed to v2 by
:func:`~micrometeorology.wrf.operational_record.rename_v1_columns` inside the
readers, so no consumer ever sees a v1 spelling and listing one here would only
publish a name that cannot match into the page's "looked for" report.

Add a new candidate spelling to the tuple; do not add a branch.
"""

import logging
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from micrometeorology.wrf.columns import (
    GLW_W_M2,
    PRECIP_MM,
    PSFC_HPA,
    RH_PCT,
    SWDDIF_W_M2,
    SWDOWN_W_M2,
    T2_C,
    WIND_DIR_DEG,
    WIND_SPEED_M_S,
)

logger = logging.getLogger(__name__)

__all__ = [
    "MONITORING_CHARTS",
    "MonitoringChart",
    "MonitoringSeries",
    "resolve_wrf_column",
]

type SeriesHue = Literal["net", "shortwave", "longwave"]
type SeriesDirection = Literal["down", "up"]
type ChartKind = Literal["line", "bar", "scatter"]


@dataclass(frozen=True, slots=True)
class MonitoringSeries:
    """One line on a chart, and where its three layers come from.

    Attributes
    ----------
    id:
        Slug the page keys off.
    label:
        Portuguese caption for the legend.
    station:
        Unified column in the archive frames (both the raw and the hourly one
        carry the same name — that is the point of the unification step).
    wrf:
        Ordered candidate column names in ``series_operacional.dat``. The first
        one present wins; an empty tuple means the model has no counterpart and
        the page should say so rather than leave the legend short.
    hue:
        ``net`` / ``shortwave`` / ``longwave`` — which of the three validated
        hues the page paints this series with. The balance chart carries five
        series in three hues precisely so the pairs read as pairs.
    direction:
        ``down`` or ``up``, drawn as solid versus dashed. This is the secondary
        encoding that lets the balance chart use three hues instead of five,
        and it mirrors the physics rather than just distinguishing lines.
    """

    id: str
    label: str
    station: str
    wrf: tuple[str, ...] = ()
    hue: SeriesHue = "net"
    direction: SeriesDirection = "down"


@dataclass(frozen=True, slots=True)
class MonitoringChart:
    """One card of the monitoring page, matching one of the nine legacy PNGs.

    Attributes
    ----------
    id:
        The PNG stem the page replaces (``temperatura``, ``balanco``, ...), kept
        so the two products stay comparable during the migration.
    title:
        Portuguese heading of the card.
    unit:
        Physical unit every series on the card is drawn in, already the unit the
        archive publishes; the card never rescales.
    kind:
        ``line`` — raw as dots, hourly as a line;
        ``bar`` — hourly as bars over a raw accumulation line (precipitation);
        ``scatter`` — every layer as dots, because the quantity is circular and
        a connecting line would sweep across the 0/360 wrap.
    series:
        The lines of the card, in legend order.
    y_limits:
        Fixed frame, or ``None`` to autoscale. Taken from the legacy contract
        except where measurement showed the frame was clipping real values.
    caveats:
        Portuguese sentences the page prints with the card. They carry what the
        chart cannot show for itself -- a model bias, an instrument handover, a
        sign convention -- so a reader is not left to infer it from the shape of
        the lines.
    """

    id: str
    title: str
    unit: str
    kind: ChartKind
    series: tuple[MonitoringSeries, ...]
    y_limits: tuple[float, float] | None = None
    caveats: tuple[str, ...] = ()


# `labmim-wrf-series` writes this column; a record produced before it existed
# has no precipitation at all, which the tuple mechanism reports as an absent
# layer rather than an empty one.
#
# The wrfout names RAINC and RAINNC are deliberately NOT candidates: they are
# accumulated since the run's initialisation (see
# micrometeorology.wrf.variables.extract_rain_step, which differences them for
# exactly that reason), while this card's station layers are the rain of each
# interval. Nothing on the way to the chart differences a resolved column, so an
# accumulation would be drawn against per-interval totals under the same unit.
_WRF_RAIN_CANDIDATES = (PRECIP_MM,)

MONITORING_CHARTS: tuple[MonitoringChart, ...] = (
    MonitoringChart(
        id="temperatura",
        title="Temperatura do ar",
        unit="°C",
        kind="line",
        series=(MonitoringSeries("t", "Temperatura", "T", (T2_C,)),),
        y_limits=(10.0, 40.0),
    ),
    MonitoringChart(
        id="umidade",
        title="Umidade relativa",
        unit="%",
        kind="line",
        series=(MonitoringSeries("ur", "Umidade relativa", "ur", (RH_PCT,)),),
        # 0-105, not 0-100: the model value is published unclipped, so a WRF
        # hour CAN land above saturation and a 0-100 frame would push it off the
        # axis without a trace. The 322 such rows -- 314 distinct hours -- the v1
        # record carried were its specific-humidity bug, not the model.
        y_limits=(0.0, 105.0),
        caveats=(
            "A umidade do WRF é publicada sem recorte na saturação, então o eixo vai a 105% para que uma hora supersaturada apareça em vez de sumir. As 314 horas acima de 100% que o registro trazia até março de 2026 são erro da extração antiga, que aplicava a conversão de umidade específica sobre uma razão de mistura; a migração para o esquema v2 as corrige.",
        ),
    ),
    MonitoringChart(
        id="pressao",
        title="Pressão atmosférica",
        unit="hPa",
        kind="line",
        series=(MonitoringSeries("pressure", "Pressão", "pressure", (PSFC_HPA,)),),
        caveats=(
            "O WRF traz um deslocamento sistemático de cerca de +2 hPa em relação à estação, que é diferença de redução ao nível da estação e não ruído.",
        ),
    ),
    MonitoringChart(
        id="precipitacao",
        title="Precipitação",
        unit="mm",
        kind="bar",
        series=(MonitoringSeries("precip", "Precipitação", "precip", _WRF_RAIN_CANDIDATES),),
        caveats=(
            "A camada bruta é o acumulado de cada intervalo do pluviômetro (báscula de 0,254 mm) e a horária é a soma dele, não uma média.",
            "A camada do WRF é o incremento de cada hora de RAINC+RAINNC, já diferenciado na extração; as horas anteriores à entrada do produtor atual não têm essa coluna, porque a extração antiga nunca amostrou precipitação.",
        ),
    ),
    MonitoringChart(
        id="velocidade",
        title="Velocidade do vento",
        unit="m/s",
        kind="line",
        series=(MonitoringSeries("ws", "Velocidade", "WS", (WIND_SPEED_M_S,)),),
        y_limits=(0.0, 15.0),
        caveats=(
            "O WRF é sistematicamente mais ventoso que a estação — cerca de +1,3 m/s nesta janela, mais da metade da média observada. A linha do modelo não é uma correção da medida.",
        ),
    ),
    MonitoringChart(
        id="direcao",
        title="Direção do vento",
        unit="°",
        kind="scatter",
        series=(MonitoringSeries("wd", "Direção", "WD", (WIND_DIR_DEG,)),),
        y_limits=(0.0, 360.0),
        caveats=(
            "Todas as camadas são pontos, inclusive a do WRF: direção é circular, e uma linha ligando 350° a 10° varreria o gráfico inteiro passando por um rumo que nunca ocorreu.",
            "A média horária é vetorial ponderada pela velocidade — cada amostra do intervalo entra com o peso do vento que a produziu, de modo que uma calmaria não desloca o rumo da hora.",
            "O WRF traz o vento mais de leste do que a estação: em 12.618 horas pareadas com vento acima de 1,5 m/s o rumo médio do modelo é 104° contra 131° medidos, um desvio sistemático de cerca de 20°. Três em cada quatro horas caem dentro de 45° da medida, então isso é viés de sítio costeiro e não convenção invertida.",
        ),
    ),
    MonitoringChart(
        id="balanco",
        title="Balanço de radiação",
        unit="W/m²",
        kind="line",
        series=(
            MonitoringSeries("net", "Saldo (Rn)", "Net_CNR1", (), hue="net"),
            MonitoringSeries(
                "sw_down",
                "Onda curta ↓",
                "Sw_dw",
                (SWDOWN_W_M2,),
                hue="shortwave",
                direction="down",
            ),
            MonitoringSeries("sw_up", "Onda curta ↑", "Sw_up", (), hue="shortwave", direction="up"),
            MonitoringSeries(
                "lw_down",
                "Onda longa ↓",
                "Lw_dw",
                (GLW_W_M2,),
                hue="longwave",
                direction="down",
            ),
            MonitoringSeries("lw_up", "Onda longa ↑", "Lw_up", (), hue="longwave", direction="up"),
        ),
        caveats=(
            "As parcelas ascendentes vão em magnitude física, positivas. Nos PNGs antigos elas apareciam negadas, que era convenção de desenho para as parcelas somarem visualmente ao saldo.",
            "O WRF só entra nas duas parcelas incidentes. As ascendentes do modelo existem no arquivo (swupb_w_m2 e lwup_w_m2) mas não são desenhadas aqui; até março de 2026 elas dependiam de um albedo e de uma emissividade que a extração antiga gravava com 273,15 subtraídos; a migração para o esquema v2 refaz essas colunas em todo o registro.",
        ),
    ),
    MonitoringChart(
        id="radiacao_difusa",
        title="Radiação difusa",
        unit="W/m²",
        kind="line",
        series=(MonitoringSeries("sw_dif", "Difusa", "Sw_dif", (SWDDIF_W_M2,), hue="shortwave"),),
        caveats=(
            "A difusa migrou do CMP21 para o PSP em 14/05/2025; janelas anteriores a essa data vêm do outro instrumento.",
        ),
    ),
    MonitoringChart(
        id="radiacao_par",
        title="Radiação PAR",
        unit="W/m²",
        kind="line",
        series=(MonitoringSeries("par", "PAR", "Sw_par", (), hue="shortwave"),),
        caveats=(
            "A razão entre PAR e irradiância global medida após 2019 fica em torno de 0,26, onde a literatura espera cerca de 0,45 — há suspeita de erro de escala de aproximadamente 1,7 vezes nesta era.",
        ),
    ),
)


def resolve_wrf_column(series: MonitoringSeries, available: pd.Index) -> str | None:
    """First candidate WRF column that the extraction actually provides.

    Parameters
    ----------
    series:
        The chart series whose ``wrf`` tuple lists the candidate spellings, in
        preference order.
    available:
        Column index of the loaded ``series_operacional.dat``.

    Returns
    -------
    str or None
        The winning column name, or ``None`` when the model has no counterpart
        yet. The absence is a normal state, not an error: the page reports it,
        and the same call starts returning a name the moment the extraction
        script grows the column.
    """
    for candidate in series.wrf:
        if candidate in available:
            return candidate
    if series.wrf:
        logger.info("%s: no WRF column yet (looked for %s)", series.id, ", ".join(series.wrf))
    return None
