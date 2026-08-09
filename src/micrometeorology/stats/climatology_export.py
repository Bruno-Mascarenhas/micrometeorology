"""Build and write the ``labmim-climatology-v1`` artifacts the public site reads.

This module is the **producer** for the JSON the climatology page of the
read-only sibling site repository (``site-labmim``) fetches at runtime, the same
role :mod:`micrometeorology.wrf.jobs` plays for the WebGIS map data. The consumer
is external (FTP deploy), so nothing in this repository imports it back and no
reverse-import analysis will find the dependency.

The contract
------------
One ``manifest.json`` plus one file per variable, all inside the directory the
publication declares as ``dataset.paths.climatology`` (``site/Climatologia/``
today). The page reads the manifest first, then the file for whichever variable
the reader selects.

Everything the page draws is **precomputed here**, including the theoretical
density curve, so the drawn line provably belongs to the printed parameters
instead of to a second, divergent implementation of the same estimator in
JavaScript.

Both the bars and the curve are densities of the **continuous part alone**. A
point mass — wind calms, dry hours, the humidity saturation clip — is removed
from the sample by the caller, so the histogram already normalises over what is
left; scaling the curve by ``1 - sum(atom fractions)`` on top of that would put
the two on different normalisations and flatten the line against the axis. The
mass itself is published beside them as a printed probability, which is exactly
the hybrid density of Takle & Brown (1978) shown as its two pieces.

Why the data is not committed
-----------------------------
These artifacts are derived from the laboratory's own sensor archive, which is
not public. Like the WRF output they are gitignored in the site repository and
attached at deploy time; the page degrades to a "not published yet" state when
the directory is empty, which is what every development checkout and CI run
sees.

Relationship to the neighbouring modules
----------------------------------------
- :mod:`micrometeorology.stats.distributions` owns the mathematics (histograms,
  estimators, distances). This module owns only the *shape of the bytes*.
- :mod:`micrometeorology.stats.climatology` owns the time groupings that select
  a subset before it reaches here.
- Preparing the series themselves — merging the archive, masking sentinels,
  harmonising sensor eras, aggregating to hourly — belongs to the caller
  (``micrometeorology.cli.export_climatology``), not here.
"""

import json
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from micrometeorology.common.paths import ensure_dir
from micrometeorology.stats import distributions as dist

logger = logging.getLogger(__name__)

__all__ = [
    "CLIMATOLOGY_VARIABLES",
    "MANIFEST_FILENAME",
    "MANIFEST_FORMAT",
    "VARIABLE_FORMAT",
    "Atom",
    "VariableSpec",
    "build_manifest",
    "build_variable_payload",
    "write_json",
]

MANIFEST_FORMAT = "labmim-climatology-v1"
VARIABLE_FORMAT = "labmim-climatology-variable-v1"
MANIFEST_FILENAME = "manifest.json"

# Angles on the precomputed wind-rose overlay, from 0 to 360 inclusive. Only the
# rose needs a free grid: a histogram's curve is sampled at its own bin centres,
# which is both exact and smaller. 201 samples draw a smooth ring at any width a
# browser will render, and the production host serves uncompressed, so the count
# is a deliberate byte budget rather than a default.
CURVE_POINTS = 201

# Decimal places per field group. Densities are small numbers whose visual
# accuracy matters; counts are integers; statistics are printed to the reader.
_DENSITY_DECIMALS = 8
_STATISTIC_DECIMALS = 4
_FRACTION_DECIMALS = 6
_PARAMETER_DECIMALS = 6

# Sectors of the wind rose. 16 is the wind-atlas convention (22.5 deg each,
# centred on the compass points).
ROSE_SECTORS = 16

# Share of the samples the plotted x-window may leave outside on EACH side. The
# bins are unaffected; this only decides how far the axis is drawn. See
# _display_range.
_DISPLAY_TAIL = 0.001


@dataclass(frozen=True)
class Atom:
    """A point mass reported beside a continuous fit instead of inside it.

    Wind calms, dry hours and the relative-humidity saturation clip are all
    spikes that no continuous family can represent. Fitting over them biases the
    estimate (a Weibull shape pulled down by zeros is the classic case), so the
    caller removes them and this module prints their probability next to the
    conditional density that the bars and the curve both show.

    Attributes
    ----------
    id:
        Stable slug the page keys off.
    label:
        Portuguese caption, printed verbatim.
    fraction:
        Share of the subset the mass holds, in ``[0, 1]``.
    """

    id: str
    label: str
    fraction: float


@dataclass(frozen=True)
class VariableSpec:
    """One published variable: how it is binned, fitted and captioned.

    Attributes
    ----------
    id:
        Slug used for the file name and as the page's selector value.
    label, unit:
        Portuguese caption and unit, printed on the axis and in the tooltip.
    chart:
        ``"histogram"`` or ``"rose"``. The rose is not a histogram with a
        different skin: direction is circular, so it carries sector frequencies
        and circular statistics instead of bins and quantiles.
    family:
        Key of :data:`micrometeorology.stats.distributions.FAMILIES`, or ``None``
        for a variable the literature gives no canonical density (the payload
        then carries bars and statistics with no curve).
    family_label:
        Citation shown next to the curve, e.g. ``"Weibull (Justus et al., 1978)"``.
    edges:
        Frozen bin edges. Identical for every subset by construction — a
        per-subset width would make the summer and winter bars incomparable,
        which is the one thing this page exists to allow.
    fit_scale:
        Divide the sample by this before fitting, for a family whose support is
        not the variable's own unit (relative humidity is fitted as a beta on
        ``[0, 1]`` but binned in percent). The curve is transformed back, so the
        published density is always per unit of ``unit``.
    caveats:
        Portuguese sentences the page prints under the chart. These are the
        scientific qualifications a reviewer would demand; they travel with the
        data so the page cannot silently drop them.
    """

    id: str
    label: str
    unit: str
    chart: str
    family: str | None
    family_label: str
    edges: tuple[float, ...]
    fit_scale: float = 1.0
    caveats: tuple[str, ...] = ()


def _linear_edges(start: float, stop: float, step: float) -> tuple[float, ...]:
    """Inclusive edge set on a physical step, rounded away from float drift.

    ``np.arange`` on a fractional step accumulates error and can emit
    ``1024.9999999999998`` as the last edge, which then prints as a bin label
    nobody can read.
    """
    count = round((stop - start) / step)
    return tuple(round(start + index * step, 6) for index in range(count + 1))


# Tipping-bucket resolution of the LabMiM gauge: 0.254 mm is 0.01 inch. It is
# the smallest quantity the instrument can report, hence both the wet/dry
# threshold and the narrowest bin that is not an artefact of quantisation.
RAIN_BUCKET_MM = 0.254


def _rain_edges(
    bucket: float, stop: float, *, unit_bins: int, per_decade: int
) -> tuple[float, ...]:
    """Bin edges for a tipping-bucket record: aligned to the instrument's lattice.

    A tipping bucket can only report integer multiples of its volume, so hourly
    intensities do not live on a continuum — they sit on the lattice
    ``{bucket, 2*bucket, 3*bucket, ...}``. Plain logarithmic edges cut *between*
    lattice points at the low end, where the bins are narrower than the quantum
    itself: one bin swallows a lattice point and reports an enormous density
    while its neighbour reports zero, producing a comb that looks like signal
    and is entirely an artefact of the binning.

    So the first ``unit_bins`` edges are placed at half-integer multiples of the
    bucket, giving each low-intensity lattice point a bin of its own, and only
    then do the bins widen geometrically — every later edge still snapped to a
    half-integer multiple so no bin ever splits a lattice point.
    """
    edges = [round((index + 0.5) * bucket, 6) for index in range(unit_bins)]
    current = edges[-1]
    step = 10.0 ** (1.0 / per_decade)
    while current < stop:
        # Snapping to the lattice can leave the edge where it was; forcing at
        # least one bucket of growth keeps the sequence strictly increasing.
        candidate = max(current * step, current + bucket)
        snapped = round((round(candidate / bucket - 0.5) + 0.5) * bucket, 6)
        current = snapped if snapped > current else round(current + bucket, 6)
        edges.append(current)
    return tuple(edges)


CLIMATOLOGY_VARIABLES: tuple[VariableSpec, ...] = (
    VariableSpec(
        id="air_temperature",
        label="Temperatura do ar",
        unit="°C",
        chart="histogram",
        family="normal",
        family_label="Gaussiana (Wilks, 2019, cap. 4)",
        edges=_linear_edges(14.0, 40.0, 0.5),
        caveats=(
            "Valores horários de todo o registro: a distribuição é uma mistura sobre a hora do dia e a estação do ano, então o afastamento da gaussiana é o conteúdo do gráfico, não um defeito do ajuste.",
        ),
    ),
    VariableSpec(
        id="relative_humidity",
        label="Umidade relativa",
        unit="%",
        chart="histogram",
        family="beta",
        family_label="Beta (Raschke, 2011)",
        edges=_linear_edges(0.0, 100.0, 2.0),
        fit_scale=100.0,
        caveats=(
            "A beta é definida no intervalo aberto (0, 100): o acúmulo de amostras na saturação sai do ajuste e é informado à parte. Barras e curva são densidades condicionais ao restante.",
        ),
    ),
    VariableSpec(
        id="relative_humidity_wxt",
        label="Umidade relativa (WXT)",
        unit="%",
        chart="histogram",
        family="beta",
        family_label="Beta (Raschke, 2011)",
        edges=_linear_edges(0.0, 100.0, 2.0),
        fit_scale=100.0,
        caveats=(
            "Segundo higrômetro da torre, publicado ao lado do principal em vez de fundido com ele: os dois divergem cerca de 10 pontos de umidade relativa e não há sobreposição suficiente para decidir qual é o correto. A diferença é o que este par de gráficos existe para mostrar.",
            "Bordas de intervalo idênticas às do higrômetro principal, para que as duas distribuições sejam comparáveis no mesmo eixo.",
        ),
    ),
    VariableSpec(
        id="pressure",
        label="Pressão atmosférica",
        unit="hPa",
        chart="histogram",
        family="normal",
        family_label="Gaussiana (Wilks, 2019, cap. 4)",
        edges=_linear_edges(995.0, 1025.0, 0.5),
        caveats=(
            "Em latitudes tropicais a maré barométrica semidiurna S2 impõe uma oscilação determinística de cerca de 1 hPa (Dai e Wang, 1999), que alarga os ombros do histograma — nenhum modelo unimodal a explica.",
        ),
    ),
    VariableSpec(
        id="wind_speed",
        label="Velocidade do vento",
        unit="m/s",
        chart="histogram",
        family="weibull",
        family_label="Weibull (Justus et al., 1978)",
        edges=_linear_edges(0.0, 20.0, 0.5),
        caveats=(
            "A Weibull não representa massa em zero: as calmarias saem do ajuste e são informadas à parte. Barras e curva são densidades condicionais às horas com vento acima do limiar de partida.",
        ),
    ),
    VariableSpec(
        id="wind_direction",
        label="Direção do vento",
        unit="°",
        chart="rose",
        family="von_mises_mixture",
        family_label="Mistura de von Mises (Carta, Bueno e Ramírez, 2008)",
        edges=_linear_edges(0.0, 360.0, 360.0 / ROSE_SECTORS),
        caveats=(
            "Direção é uma grandeza circular: 0° e 360° são o mesmo rumo, então o gráfico é uma rosa dos ventos e as estatísticas são circulares. Média e desvio-padrão aritméticos de graus não têm significado aqui.",
            "As calmarias ficam fora da rosa: abaixo do limiar de partida do anemômetro a direção é indefinida.",
        ),
    ),
    VariableSpec(
        id="precipitation",
        label="Precipitação (horas chuvosas)",
        unit="mm/h",
        chart="histogram",
        family="gamma",
        family_label="Gama (Thom, 1958; Wilks, 2019)",
        edges=_rain_edges(RAIN_BUCKET_MM, 100.0, unit_bins=8, per_decade=6),
        caveats=(
            "Só as horas chuvosas entram no ajuste; a fração de horas secas é reportada à parte. Um histograma sobre todas as horas seria uma única barra em zero.",
            "O pluviômetro é de báscula com resolução de 0,254 mm, então as intensidades vivem numa grade discreta: os primeiros intervalos têm exatamente uma báscula de largura e só depois alargam, para que a grade do instrumento não vire um pente falso no histograma.",
            "A gama subestima a cauda das intensidades horárias extremas.",
        ),
    ),
    VariableSpec(
        id="par_early",
        label="Radiação PAR (até 2019-03)",
        unit="W/m²",
        chart="histogram",
        family=None,
        family_label="sem densidade teórica canônica",
        edges=_linear_edges(0.0, 800.0, 20.0),
        caveats=(
            "A literatura não atribui densidade canônica à PAR: o que ela modela é a RAZÃO entre PAR e irradiância global. As barras são descritivas e não há curva sobreposta.",
            "Publicada por era, sem correção. A razão PAR/global medida depois de 2019 fica em 0,25 a 0,28 onde se espera cerca de 0,45, o que sugere um erro de escala de aproximadamente 1,7 vezes que nenhuma calibração documentada explica. Separar as eras deixa o degrau visível em vez de escondido numa distribuição única.",
        ),
    ),
    VariableSpec(
        id="par_late",
        label="Radiação PAR (a partir de 2019-03)",
        unit="W/m²",
        chart="histogram",
        family=None,
        family_label="sem densidade teórica canônica",
        edges=_linear_edges(0.0, 800.0, 20.0),
        caveats=(
            "A literatura não atribui densidade canônica à PAR: o que ela modela é a RAZÃO entre PAR e irradiância global. As barras são descritivas e não há curva sobreposta.",
            "Era com suspeita de erro de escala de cerca de 1,7 vezes (razão PAR/global de 0,25 a 0,28 contra os ~0,45 esperados). Os valores estão como medidos, sem correção: comparar com a era anterior é o ponto.",
        ),
    ),
    VariableSpec(
        id="shortwave_down",
        label="Radiação de onda curta incidente",
        unit="W/m²",
        chart="histogram",
        family=None,
        family_label="sem densidade teórica canônica",
        edges=_linear_edges(0.0, 1400.0, 25.0),
        caveats=(
            "Restrito às horas com elevação solar acima de 10°. Sem esse corte metade do registro é noite e o histograma vira uma barra única em zero.",
            "A irradiância em W/m² não tem densidade canônica: o forçamento extraterrestre varia ao longo do dia e do ano, então a forma da distribuição é em boa parte geometria solar, não clima. A grandeza que a literatura modela é o índice de claridade, publicado à parte nesta mesma página.",
        ),
    ),
    VariableSpec(
        id="shortwave_up",
        label="Radiação de onda curta refletida",
        unit="W/m²",
        chart="histogram",
        family=None,
        family_label="sem densidade teórica canônica",
        edges=_linear_edges(0.0, 450.0, 10.0),
        caveats=(
            "Restrito às horas com elevação solar acima de 10°, pelo mesmo motivo da onda curta incidente.",
            "Publicada em magnitude física (positiva). No gráfico de balanço da página de monitoramento os canais ascendentes aparecem negados, que é convenção de desenho para as parcelas somarem visualmente ao saldo — aqui o valor é o medido.",
            "A forma desta distribuição é dominada pelo albedo da superfície vista pelo sensor, então ela diz mais sobre o entorno da torre do que sobre o clima.",
        ),
    ),
    VariableSpec(
        id="longwave_down",
        label="Radiação de onda longa incidente",
        unit="W/m²",
        chart="histogram",
        family=None,
        family_label="sem densidade teórica canônica",
        edges=_linear_edges(330.0, 540.0, 3.0),
        caveats=(
            "Todas as horas: ao contrário da onda curta, a atmosfera emite dia e noite, então não há corte por elevação solar a fazer.",
            "Os limites 0,4·sigma·Ta⁴ e sigma·Ta⁴ + 25, usados como controle de qualidade, são COEFICIENTES AJUSTÁVEIS do QCRad calibrados por sítio (Long e Shi, 2008, dão 0,58 a 0,80 e 11 a 23 em estações reais) — não constantes físicas, ao contrário do que sugere a forma da expressão.",
        ),
    ),
    VariableSpec(
        id="longwave_up",
        label="Radiação de onda longa emitida",
        unit="W/m²",
        chart="histogram",
        family=None,
        family_label="sem densidade teórica canônica",
        edges=_linear_edges(400.0, 640.0, 4.0),
        caveats=(
            "Todas as horas, em magnitude física (positiva). No balanço do monitoramento este canal aparece negado por convenção de desenho.",
            "A onda longa emitida é sigma·epsilon·Ts⁴ da superfície, mas ao contrário do que essa forma sugere a quarta potência quase não deforma a distribuição: com a temperatura de brilho medida aqui, ela sozinha produziria assimetria de 0,14, e a observada é 1,02. A assimetria vem de a população ser DOIS regimes — dia e noite —, não da potência.",
        ),
    ),
    VariableSpec(
        id="net_radiation_day",
        label="Saldo de radiação (dia)",
        unit="W/m²",
        chart="histogram",
        family=None,
        family_label="sem densidade teórica canônica",
        edges=_linear_edges(-100.0, 1000.0, 20.0),
        caveats=(
            "O saldo é uma mistura de dois regimes com escalas completamente diferentes, e juntar os dois num histograma só produz uma distribuição bimodal que nenhum modelo explica. Por isso dia e noite são publicados separados: aqui, as horas com elevação solar acima de 10°.",
            "Saldo diurno é dominado pela onda curta líquida, logo pela cobertura de nuvens. Nenhuma família paramétrica da literatura descreve essa mistura, então não há curva sobreposta.",
        ),
    ),
    VariableSpec(
        id="net_radiation_night",
        label="Saldo de radiação (noite)",
        unit="W/m²",
        chart="histogram",
        family="normal",
        family_label="Gaussiana (regime noturno de céu claro)",
        edges=_linear_edges(-120.0, 40.0, 2.0),
        caveats=(
            "Horas com o sol abaixo do horizonte. É o único recorte do saldo em que uma gaussiana é defensável: sem onda curta, o saldo é a perda líquida de onda longa, que num sítio fixo é estreita e quase simétrica.",
            "E espere que ela falhe: o registro medido é BIMODAL — noites de céu claro perdem cerca de 60 W/m² e noites nubladas ficam perto de 10, porque a nuvem devolve a onda longa que a superfície emite. São dois regimes de céu, não uma população. Nenhuma gaussiana representa isso, e é por essa razão que a distância do ajuste é impressa ao lado da curva em vez de escondida.",
            "Saldo noturno é onde o radiômetro tem a maior incerteza relativa (deriva térmica e contaminação de domo), então o modo estreito é em parte limitado pelo instrumento.",
        ),
    ),
    VariableSpec(
        id="clearness_index",
        label="Índice de claridade (Kt)",
        unit="",
        chart="histogram",
        family="hollands_huget",
        family_label="Hollands e Huget (1983)",
        edges=_linear_edges(0.0, 1.0, 0.02),
        caveats=(
            "A irradiância global em W/m² não tem densidade canônica: metade do registro é noite e o forçamento extraterrestre varia ao longo do dia. A grandeza da literatura é o índice de claridade Kt = global / extraterrestre, restrito às horas com elevação solar acima de 10°.",
            "Num litoral tropical úmido espera-se bimodalidade (céu encoberto perto de 0,25 e céu limpo perto de 0,65), que a curva unimodal de Hollands e Huget não reproduz.",
        ),
    ),
)


def _finite(value: float | None) -> float | None:
    """Map a non-finite number to ``None`` so the strict JSON writer accepts it.

    The writer uses ``allow_nan=False`` on purpose (a ``NaN`` token is invalid
    JSON and every browser parser rejects the whole file), and an empty subset
    legitimately produces NaN parameters. Converting at the boundary keeps that
    guard while letting a genuinely absent number travel as ``null``.
    """
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _rounded(value: float | None, decimals: int) -> float | None:
    number = _finite(value)
    return None if number is None else round(number, decimals)


def _rounded_list(values: NDArray, decimals: int) -> list[float | None]:
    return [_rounded(float(value), decimals) for value in np.asarray(values, dtype=float)]


def _describe(sample: NDArray) -> dict[str, float | None]:
    """Descriptive statistics of a cleaned sample, in the variable's own unit."""
    values = np.asarray(sample, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        keys = (
            "mean",
            "std",
            "min",
            "p01",
            "p25",
            "p50",
            "p75",
            "p99",
            "max",
            "skewness",
            "kurtosis",
        )
        return dict.fromkeys(keys)
    skewness, kurtosis = _shape_moments(values)
    percentiles = np.percentile(values, [1, 25, 50, 75, 99])
    return {
        "mean": _rounded(float(values.mean()), _STATISTIC_DECIMALS),
        "std": _rounded(
            float(values.std(ddof=1)) if values.size > 1 else float("nan"), _STATISTIC_DECIMALS
        ),
        "min": _rounded(float(values.min()), _STATISTIC_DECIMALS),
        "p01": _rounded(float(percentiles[0]), _STATISTIC_DECIMALS),
        "p25": _rounded(float(percentiles[1]), _STATISTIC_DECIMALS),
        "p50": _rounded(float(percentiles[2]), _STATISTIC_DECIMALS),
        "p75": _rounded(float(percentiles[3]), _STATISTIC_DECIMALS),
        "p99": _rounded(float(percentiles[4]), _STATISTIC_DECIMALS),
        "max": _rounded(float(values.max()), _STATISTIC_DECIMALS),
        "skewness": _rounded(skewness, _STATISTIC_DECIMALS),
        "kurtosis": _rounded(kurtosis, _STATISTIC_DECIMALS),
    }


def _shape_moments(values: NDArray) -> tuple[float, float]:
    r"""Bias-corrected sample skewness and excess kurtosis.

    Formula
    -------
    The Fisher-Pearson adjusted estimators — the convention pandas, Excel and
    every descriptive table the laboratory already publishes use:

    .. math::
        G_1 = \frac{\sqrt{n(n-1)}}{n-2}\, \frac{m_3}{m_2^{3/2}}, \qquad
        G_2 = \frac{n-1}{(n-2)(n-3)}\,\big[(n+1)g_2 + 6\big]

    with :math:`g_2 = m_4/m_2^2 - 3`. Written out here rather than delegated to
    ``pandas.Series.skew`` so this module needs no pandas import and the
    convention is legible next to the numbers it produces. Samples below four
    values, or with zero spread, yield NaN — the corrections divide by
    ``n - 3`` and by the variance.
    """
    n = int(values.size)
    if n < 4:
        return float("nan"), float("nan")
    deviation = values - values.mean()
    m2 = float(np.mean(deviation**2))
    if m2 <= 0.0:
        return float("nan"), float("nan")
    m3 = float(np.mean(deviation**3))
    m4 = float(np.mean(deviation**4))
    skewness = np.sqrt(n * (n - 1.0)) / (n - 2.0) * (m3 / m2**1.5)
    excess = m4 / (m2 * m2) - 3.0
    kurtosis = (n - 1.0) / ((n - 2.0) * (n - 3.0)) * ((n + 1.0) * excess + 6.0)
    return float(skewness), float(kurtosis)


def _histogram_subset(
    spec: VariableSpec,
    sample: NDArray,
    atoms: Sequence[Atom],
) -> dict[str, Any]:
    """Bars, statistics, fit, distances and the pre-scaled curve for one subset."""
    values = np.asarray(sample, dtype=float)
    binned = dist.histogram(values, spec.edges)
    payload: dict[str, Any] = {
        "n": binned.n,
        "counts": [int(count) for count in binned.counts],
        "density": _rounded_list(binned.density, _DENSITY_DECIMALS),
        "below": binned.below,
        "above": binned.above,
        "stats": _describe(values),
        "atoms": [
            {
                "id": atom.id,
                "label": atom.label,
                "fraction": _rounded(atom.fraction, _FRACTION_DECIMALS),
            }
            for atom in atoms
        ],
        "fit": None,
        "quality": None,
        "curve": None,
    }
    if spec.family is None or binned.n < 2:
        return payload

    fitted = dist.fit_distribution(spec.family, values / spec.fit_scale)
    if not all(np.isfinite(value) for value in fitted.params.values()):
        logger.warning("%s: %s fit did not converge on %d samples", spec.id, spec.family, binned.n)
        return payload

    payload["fit"] = {
        "family": fitted.family,
        "params": {
            name: _rounded(value, _PARAMETER_DECIMALS) for name, value in fitted.params.items()
        },
        "n": fitted.n,
    }
    quality = dist.goodness_of_fit(
        fitted,
        values / spec.fit_scale,
        binned=dist.histogram(
            values / spec.fit_scale, [edge / spec.fit_scale for edge in spec.edges]
        ),
    )
    payload["quality"] = {
        "ks_distance": _rounded(quality.ks_distance, _DENSITY_DECIMALS),
        # The gap is measured on the fitted scale, so it is converted back to the
        # unit the page prints beside it.
        "quantile_gap": _rounded(quality.quantile_gap * spec.fit_scale, _STATISTIC_DECIMALS),
        "density_r_squared": _rounded(quality.density_r_squared, 6),
        "n": quality.n,
        "n_effective": _rounded(quality.n_effective, 1),
        "lag1_autocorrelation": _rounded(quality.lag1_autocorrelation, 4),
    }

    # The curve is sampled at the BIN CENTRES, one value per bar, rather than on
    # a free grid. That is what lets the page draw it as a second dataset on the
    # same categorical axis as the bars: no interpolation in the browser, exact
    # alignment with every bar, and it works unchanged for the logarithmic bins
    # precipitation needs, where a linear abscissa would misplace the line.
    #
    # Two conversions here: /fit_scale puts the abscissa on the family's own
    # support, and the matching /fit_scale on the density puts the ordinate back
    # on a per-unit-of-`unit` basis so it overlays the histogram directly.
    density = dist.pdf(fitted, binned.centers / spec.fit_scale) / spec.fit_scale
    payload["curve"] = _rounded_list(density, _DENSITY_DECIMALS)
    return payload


def _rose_subset(
    sample: NDArray,
    atoms: Sequence[Atom],
    *,
    components: int,
) -> dict[str, Any]:
    """Sector frequencies, circular statistics and the mixture overlay."""
    values = np.asarray(sample, dtype=float)
    _centers, frequencies = dist.sector_frequencies(values, sectors=ROSE_SECTORS)
    circular = dist.circular_summary(values)
    payload: dict[str, Any] = {
        "n": int(circular["n"]),
        "frequencies": _rounded_list(frequencies, _FRACTION_DECIMALS),
        "circular": {
            "mean_direction": _rounded(circular["mean_direction"], 2),
            "resultant_length": _rounded(circular["resultant_length"], 4),
            "circular_variance": _rounded(circular["circular_variance"], 4),
        },
        "atoms": [
            {
                "id": atom.id,
                "label": atom.label,
                "fraction": _rounded(atom.fraction, _FRACTION_DECIMALS),
            }
            for atom in atoms
        ],
        "fit": None,
        "quality": None,
        "curve": None,
    }
    if payload["n"] < components:
        return payload

    mixture = dist.fit_von_mises_mixture(values, components=components)
    if not all(np.isfinite(weight) for weight in mixture.weights):
        return payload

    payload["fit"] = {
        "family": "von_mises_mixture",
        "params": {
            "weights": [_rounded(weight, _PARAMETER_DECIMALS) for weight in mixture.weights],
            "mu_degrees": [_rounded(mu, 3) for mu in mixture.mu_degrees],
            "kappa": [_rounded(kappa, _PARAMETER_DECIMALS) for kappa in mixture.kappa],
        },
        "n": mixture.n,
    }

    # The curve is expressed as "frequency an equally wide sector would hold", so
    # it is directly comparable with the plotted petals instead of living on a
    # per-radian scale the reader would have to convert. Unlike a histogram, the
    # rose overlay is a smooth closed ring, so it is sampled on a regular grid of
    # CURVE_POINTS angles from 0 to 360 inclusive — the page reconstructs the
    # abscissa as ``index * 360 / (CURVE_POINTS - 1)``.
    grid = np.linspace(0.0, 360.0, CURVE_POINTS)
    modelled = dist.von_mises_mixture_pdf(mixture, grid) * (360.0 / ROSE_SECTORS)
    payload["curve"] = _rounded_list(modelled, _DENSITY_DECIMALS)

    at_centers = dist.von_mises_mixture_pdf(mixture, _centers) * (360.0 / ROSE_SECTORS)
    residual = float(np.sum(np.square(frequencies - at_centers)))
    total = float(np.sum(np.square(frequencies - frequencies.mean())))
    payload["quality"] = {
        "density_r_squared": _rounded(1.0 - residual / total if total > 0.0 else float("nan"), 6),
        "n": mixture.n,
    }
    return payload


def _display_range(spec: VariableSpec, subsets: Mapping[str, Any]) -> list[int]:
    """First and last bin index worth plotting, shared by every subset.

    The bin edges cover the full physical range a variable could take, which is
    much wider than the range it actually occupies: the wind-speed axis runs to
    20 m/s so a gust is never silently dropped, while the station rarely exceeds
    6 m/s. Drawn literally, four fifths of the plot is empty and the populated
    bars are unreadably narrow.

    So the window is computed from the UNION of the populated bins across every
    subset and published once per variable. Two properties follow, and both
    matter: the bins themselves are untouched (nothing is re-binned or dropped,
    and the overflow tallies still describe the whole record), and summer,
    winter and model still share one x-axis, which is the comparison the page
    exists to make. A per-subset window would silently rescale the axis under
    the reader between two clicks.
    """
    bin_count = len(spec.edges) - 1
    first, last = bin_count, -1
    for subset in subsets.values():
        counts = list(subset.get("counts", ()))
        total = sum(counts)
        if total <= 0:
            continue
        # Trim by MASS, not by emptiness: a single gust three times the typical
        # speed puts one sample in a far bin and, on a "non-empty" rule, would
        # stretch the axis over a range that is blank for every other subset.
        # The tail it hides is still in the table, the statistics and p99.
        budget = total * _DISPLAY_TAIL
        running = 0.0
        low = 0
        for index, count in enumerate(counts):
            running += count
            if running > budget:
                low = index
                break
        running = 0.0
        high = bin_count - 1
        for index in range(bin_count - 1, -1, -1):
            running += counts[index]
            if running > budget:
                high = index
                break
        first, last = min(first, low), max(last, high)
    if last < first:
        return [0, bin_count - 1]
    # One empty bin of breathing room on each side so the outermost bar does not
    # sit flush against the axis.
    return [max(0, first - 1), min(bin_count - 1, last + 1)]


def build_variable_payload(
    spec: VariableSpec,
    samples: Mapping[str, NDArray],
    *,
    version: str,
    atoms: Mapping[str, Sequence[Atom]] | None = None,
    mixture_components: int = 2,
) -> dict[str, Any]:
    """Assemble the published payload for one variable across every subset.

    Parameters
    ----------
    spec:
        The variable being published.
    samples:
        Subset id -> the prepared sample for that subset, already merged,
        quality-controlled, aggregated to hourly and stripped of its point
        masses. An empty array is legitimate and produces an empty subset rather
        than an omitted one, so the page can say "no data" instead of silently
        offering fewer options in one variable than in another.
    version:
        Run stamp, repeated in every file so a half-updated directory is
        detectable from the browser.
    atoms:
        Subset id -> the point masses removed from that sample.
    mixture_components:
        Components of the von Mises mixture, for a ``rose`` variable. Fixed
        rather than chosen per subset: a component count that changes between
        summer and winter makes the two roses incomparable.

    Returns
    -------
    dict
        JSON-ready payload; every non-finite number is already ``None``.
    """
    per_subset = atoms or {}
    subsets: dict[str, Any] = {}
    for subset_id, sample in samples.items():
        subset_atoms = list(per_subset.get(subset_id, ()))
        if spec.chart == "rose":
            subsets[subset_id] = _rose_subset(sample, subset_atoms, components=mixture_components)
        else:
            subsets[subset_id] = _histogram_subset(spec, sample, subset_atoms)

    payload: dict[str, Any] = {
        "format": VARIABLE_FORMAT,
        "version": version,
        "id": spec.id,
        "label": spec.label,
        "unit": spec.unit,
        "chart": spec.chart,
        "family": spec.family,
        "family_label": spec.family_label,
        "caveats": list(spec.caveats),
        "subsets": subsets,
    }
    if spec.chart == "histogram":
        payload["display_range"] = _display_range(spec, subsets)
    if spec.chart == "rose":
        payload["sectors"] = [
            round(index * 360.0 / ROSE_SECTORS, 4) for index in range(ROSE_SECTORS)
        ]
    else:
        payload["edges"] = list(spec.edges)
    return payload


def build_manifest(
    *,
    version: str,
    generated_utc: str,
    station: Mapping[str, Any],
    period: Mapping[str, str],
    sources: Sequence[Mapping[str, Any]],
    subsets: Sequence[Mapping[str, str]],
    selector: Sequence[str],
    coverage: Mapping[str, Any],
    variables: Sequence[VariableSpec],
    caveats: Sequence[str] = (),
    package_version: str | None = None,
    commit: str | None = None,
) -> dict[str, Any]:
    """Assemble ``manifest.json``: what exists, over what period, with what caveats.

    ``selector`` is deliberately narrower than ``subsets``: every source-by-season
    combination is precomputed (it costs one more pass over data already in
    memory), while the page offers only the options it was designed around. A
    later redesign can widen the selector without regenerating anything.

    Raises
    ------
    ValueError
        If ``selector`` names a subset that ``subsets`` does not declare — the
        page would render a dead option whose only symptom is an empty chart.
    """
    declared = {subset["id"] for subset in subsets}
    unknown = [subset_id for subset_id in selector if subset_id not in declared]
    if unknown:
        raise ValueError(
            f"selector names undeclared subset(s): {unknown}; declared: {sorted(declared)}"
        )

    return {
        "format": MANIFEST_FORMAT,
        "version": version,
        "generated_utc": generated_utc,
        "package_version": package_version,
        "commit": commit,
        "station": dict(station),
        "period": dict(period),
        "sources": [dict(source) for source in sources],
        "subsets": [dict(subset) for subset in subsets],
        "selector": list(selector),
        "coverage": dict(coverage),
        "caveats": list(caveats),
        "variables": [
            {
                "id": spec.id,
                "label": spec.label,
                "unit": spec.unit,
                "chart": spec.chart,
                "family": spec.family,
                "family_label": spec.family_label,
                "file": f"{spec.id}.json",
            }
            for spec in variables
        ],
    }


def write_json(output_path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write one artifact atomically, in the encoding the site pipeline uses.

    Same contract as :mod:`micrometeorology.wrf.jobs`: serialise to a private
    ``.<name>.tmp-<pid>`` sibling and ``os.replace`` it into place, so a reader
    fetching the directory mid-run sees either the old file or the new one and
    never a truncated parse error. ``allow_nan=False`` is what makes that
    guarantee meaningful — ``NaN`` is not valid JSON and would fail in the
    browser instead of here.

    Raises
    ------
    ValueError
        If the payload still contains a non-finite float. Route every number
        through :func:`_finite` before calling this.
    """
    out = Path(output_path)
    ensure_dir(out.parent)
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    temporary = out.with_name(f".{out.name}.tmp-{os.getpid()}")
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(encoded)
        os.replace(temporary, out)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    logger.info("wrote %s (%d bytes)", out, len(encoded.encode("utf-8")))
    return out
