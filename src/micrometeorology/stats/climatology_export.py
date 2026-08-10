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
the hybrid density of [[takle1978]] shown as its two pieces.

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
import re
from collections.abc import Collection, Mapping, Sequence
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
    "REFERENCES",
    "VARIABLE_FORMAT",
    "Atom",
    "Reference",
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
        Share of the subset the mass holds, in ``[0, 1]``. The denominator is the
        subset BEFORE the mass was removed, which is not the ``n`` published
        beside it — that one counts what is left.
    count:
        The mass in samples. Published because the fraction alone is not
        resolvable on screen: a reader who multiplies "5,2 % calmarias" by the
        "66.345 observações" printed next to it gets 3.472, and the true figure
        is 3.663. One is a share of 70.008, the other is what survived.
    """

    id: str
    label: str
    fraction: float
    count: int = 0


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
        Citation shown next to the curve, e.g. ``"Weibull ([[justus1978]])"``.
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
    # Options the family cannot get from the sample, which the caller must supply
    # per subset. Empty means the family is estimated from the sample alone, as
    # every non-radiation variable is. Declaring it here is what makes a missing
    # covariate a build failure instead of a curve quietly fitted on the wrong
    # thing.
    fit_options: tuple[str, ...] = ()


@dataclass(frozen=True)
class Reference:
    """One bibliographic record, published so the page can link to the source.

    A caption that says only "(Justus et al., 1978)" tells a reader who already
    knows the field nothing they did not know and tells everyone else nothing at
    all. Carrying the full record plus a resolvable link is what turns a citation
    into something a student can actually go and read.

    Attributes
    ----------
    key:
        Slug used in ``[[key]]`` markers inside labels and caveats.
    short:
        What the sentence reads as, e.g. ``"Justus et al., 1978"``.
    citation:
        Authors, title, journal, volume, pages, year — the tooltip's contents.
    url:
        Where to go: a ``doi.org`` link, one per record, each resolved against
        Crossref and checked to land on the cited work.

        Every entry earns one. The ten that used to carry a Crossref TITLE
        SEARCH instead all had a findable DOI — the search was a fallback nobody
        had revisited, and it had rotted: ``search.crossref.org/?q=`` answers
        200, serves the real page, and the current interface ignores the query,
        so the box opened empty and nothing was searched. Every link checker
        passed it, status being all they inspect, and the failure showed only
        once the page had rendered. A resolver link cannot fail that way: it
        either redirects to the publisher or it does not exist.
    """

    key: str
    short: str
    citation: str
    url: str


# Every reference the page prints, in one place. The DOIs below carry a
# doi.org link only where the identifier was checked against the primary text or
# Crossref during the review that produced these curves; the rest resolve to a
# title search, so no link on this page can rot into a wrong paper.
REFERENCES: dict[str, Reference] = {
    reference.key: reference
    for reference in (
        Reference(
            "justus1978",
            "Justus et al., 1978",
            "Justus, C. G., Hargraves, W. R., Mikhail, A. & Graber, D. (1978). "
            "Methods for estimating wind speed frequency distributions. "
            "Journal of Applied Meteorology 17(3), 350-353.",
            "https://doi.org/10.1175/1520-0450(1978)017%3C0350:MFEWSF%3E2.0.CO;2",
        ),
        Reference(
            "thom1958",
            "Thom, 1958",
            "Thom, H. C. S. (1958). A note on the gamma distribution. "
            "Monthly Weather Review 86(4), 117-122.",
            "https://doi.org/10.1175/1520-0493(1958)086%3C0117:ANOTGD%3E2.0.CO;2",
        ),
        Reference(
            "wilks2019",
            "Wilks, 2019",
            "Wilks, D. S. (2019). Statistical Methods in the Atmospheric Sciences, "
            "4th edition. Elsevier. Chapter 4 covers the parametric families used here.",
            "https://doi.org/10.1016/B978-0-12-815823-4.00004-3",
        ),
        Reference(
            "raschke2011",
            "Raschke, 2011",
            "Raschke, M. (2011). Empirical behaviour of tests for the beta distribution "
            "and their application in environmental research. "
            "Stochastic Environmental Research and Risk Assessment 25, 79-89.",
            "https://doi.org/10.1007/s00477-010-0410-3",
        ),
        Reference(
            "carta2008",
            "Carta, Bueno e Ramirez, 2008",
            "Carta, J. A., Bueno, C. & Ramirez, P. (2008). Statistical modelling of "
            "directional wind speeds using mixtures of von Mises distributions: case study. "
            "Energy Conversion and Management 49(5), 897-907.",
            "https://doi.org/10.1016/j.enconman.2007.10.017",
        ),
        Reference(
            "hollands1983",
            "Hollands e Huget, 1983",
            "Hollands, K. G. T. & Huget, R. G. (1983). A probability density function for "
            "the clearness index, with applications. Solar Energy 30(3), 195-209.",
            "https://doi.org/10.1016/0038-092X(83)90149-4",
        ),
        Reference(
            "liu1960",
            "Liu e Jordan, 1960",
            "Liu, B. Y. H. & Jordan, R. C. (1960). The interrelationship and characteristic "
            "distribution of direct, diffuse and total solar radiation. Solar Energy 4(3), 1-19. "
            "The frequency curves the Hollands-Huget density was built to reproduce.",
            "https://doi.org/10.1016/0038-092X(60)90062-1",
        ),
        Reference(
            "pashiardis2021",
            "Pashiardis e Kalogirou, 2021",
            "Pashiardis, S. & Kalogirou, S. A. (2021). Characteristics of downward and upward "
            "longwave radiation at Athalassa, an inland location of the island of Cyprus. "
            "Applied Sciences 11(2), 719. Open access.",
            "https://doi.org/10.3390/app11020719",
        ),
        Reference(
            "freeman2006",
            "Freeman e Modarres, 2006",
            "Freeman, J. & Modarres, R. (2006). Inverse Box-Cox: the power-normal distribution. "
            "Statistics and Probability Letters 76(8), 764-772. "
            "Defines the single power-normal; the two-component mixture used here is not theirs.",
            "https://doi.org/10.1016/j.spl.2005.10.036",
        ),
        Reference(
            "hu2012",
            "Hu, Wang e Liu, 2012",
            "Hu, B., Wang, Y. & Liu, G. (2012). Relationship between net radiation and broadband "
            "solar radiation in the Tibetan Plateau. Advances in Atmospheric Sciences 29(1), "
            "135-143. Note the original works in MJ/m2 per day, not hourly W/m2.",
            "https://doi.org/10.1007/s00376-011-0221-6",
        ),
        Reference(
            "yang2008",
            "Yang, Dickinson e Liang, 2008",
            "Yang, F., Dickinson, R. E. & Liang, X.-Z. (2008). Solar radiation and its impact on "
            "land surface albedo parameterisation. Journal of Applied Meteorology and Climatology "
            "47(11), 2963-2982. Source of the zenith-angle albedo form used here.",
            "https://doi.org/10.1175/2008JAMC1843.1",
        ),
        Reference(
            "briegleb1992",
            "Briegleb, 1992",
            "Briegleb, B. P. (1992). Delta-Eddington approximation for solar radiation in the NCAR "
            "Community Climate Model. Journal of Geophysical Research 97(D7), 7603-7612. "
            "Origin of the d = 0,4 prescription for grassland, as transcribed by Yang et al. (2008).",
            "https://doi.org/10.1029/92JD00291",
        ),
        Reference(
            "idso1975",
            "Idso et al., 1975",
            "Idso, S. B., Jackson, R. D., Reginato, R. J., Kimball, B. A. & Nakayama, F. S. (1975). "
            "The dependence of bare soil albedo on soil water content. "
            "Journal of Applied Meteorology 14(1), 109-113.",
            "https://doi.org/10.1175/1520-0450(1975)014%3C0109:TDOBSA%3E2.0.CO;2",
        ),
        Reference(
            "custodio2021",
            "Custódio, Silva e Santos, 2021",
            "Custódio, L. L. M., Silva, B. B. da & Santos, C. A. C. dos (2021). Relationship between "
            "photosynthetically active radiation and global radiation in Petrolina and Brasilia, "
            "Brazil. Revista Brasileira de Engenharia Agricola e Ambiental 25(9), 612-619. Open access.",
            "https://doi.org/10.1590/1807-1929/agriambi.v25n9p612-619",
        ),
        Reference(
            "long2008",
            "Long e Shi, 2008",
            "Long, C. N. & Shi, Y. (2008). An automated quality assessment and control algorithm for "
            "surface radiation measurements (QCRad). The Open Atmospheric Science Journal 2, 23-37. "
            "Source of the site-tuned coefficient ranges quoted here.",
            "https://doi.org/10.2174/1874282300802010023",
        ),
        Reference(
            "takle1978",
            "Takle e Brown, 1978",
            "Takle, E. S. & Brown, J. M. (1978). Note on the use of Weibull statistics to characterize "
            "wind-speed data. Journal of Applied Meteorology 17(4), 556-559. "
            "The hybrid density this page reproduces by publishing point masses beside the curve.",
            "https://doi.org/10.1175/1520-0450(1978)017%3C0556:NOTUOW%3E2.0.CO;2",
        ),
    )
}

# Marker syntax inside labels and caveats: [[key]]. The page turns each one into
# a link carrying the full record; Python only has to guarantee that every marker
# resolves, which _assert_references does at import time.
_REFERENCE_MARKER = re.compile(r"\[\[([a-z0-9_]+)\]\]")

# Value syntax inside caveats: {{atom:<id>:count}}, {{atom:<id>:share}} and
# {{param:<name>:<decimals>}}, resolved against the subset the page prints the
# prose beside (``observed_all``).
#
# Prose used to carry these as literals, and a literal is a copy that stops
# being true. Three shipped wrong at once: the PAR caveats asserted 552, 24 and
# 18 point-mass hours beside published atoms of 538, 13 and 17, and the net
# radiation caveat quoted a = 0,775 / b = -24,4 / 36,2 W/m2 while the fit panel
# on the same screen printed 0,7730 / -23,59 / 35,61. Nothing was wrong with the
# numbers — masking 52 clock-corrupted days moved them, and only the sentence
# stayed behind. Interpolating makes the two agree by construction instead of by
# vigilance.
_VALUE_MARKER = re.compile(r"\{\{(atom|param):([A-Za-z0-9_]+):([A-Za-z0-9_]+)\}\}")


def _format_count(value: int) -> str:
    """Portuguese thousands separator, matching how the page prints counts."""
    return f"{value:,}".replace(",", ".")


def _format_decimal(value: float, decimals: int) -> str:
    """Portuguese decimal comma, at the precision the caveat's author chose."""
    return f"{value:.{decimals}f}".replace(".", ",")


def _resolve_values(text: str, subset: dict[str, Any], where: str) -> str:
    """Replace every ``{{...}}`` marker with the value the subset publishes.

    Raises on anything unresolvable — an unknown atom id, a parameter the fit
    does not carry, a malformed precision. A caveat that quotes a number nobody
    published is the defect this exists to remove, so it stops the export that
    writes the artifacts rather than reaching a reader.
    """

    def replace(match: re.Match[str]) -> str:
        kind, name, detail = match.groups()
        if kind == "atom":
            atoms = {atom["id"]: atom for atom in subset.get("atoms", [])}
            if name not in atoms:
                raise ValueError(
                    f"{where}: caveat cites atom {name!r}, published atoms are "
                    f"{sorted(atoms) or 'none'}"
                )
            atom = atoms[name]
            if detail == "count":
                return _format_count(int(atom["count"]))
            if detail == "share":
                return _format_decimal(100.0 * float(atom["fraction"]), 1)
            raise ValueError(f"{where}: unknown atom detail {detail!r}; use count or share")
        params = ((subset.get("fit") or {}).get("params")) or {}
        if name not in params:
            raise ValueError(
                f"{where}: caveat cites parameter {name!r}, the fit publishes "
                f"{sorted(params) or 'none'}"
            )
        if not detail.isdigit():
            raise ValueError(f"{where}: parameter precision {detail!r} is not a number of decimals")
        return _format_decimal(float(params[name]), int(detail))

    return _VALUE_MARKER.sub(replace, text)


def _assert_references() -> None:
    """Fail at import if any published sentence cites a reference that is not declared.

    A dangling ``[[key]]`` would reach the browser as literal brackets in the
    middle of a caption. Checking here means the typo stops the build that writes
    the artifacts, not the reader.
    """
    unknown: set[str] = set()
    for spec in CLIMATOLOGY_VARIABLES:
        for text in (spec.family_label, *spec.caveats):
            unknown |= {key for key in _REFERENCE_MARKER.findall(text) if key not in REFERENCES}
    if unknown:
        raise ValueError(
            f"undeclared reference marker(s) {sorted(unknown)}; add them to REFERENCES"
        )
    # And the reverse: a record must never contain a marker itself. A registry
    # entry whose `short` reads "[[hu2012]]" renders as literal brackets in the
    # middle of a caption and resolves to nothing, which is exactly what a
    # careless global rename of the prose citations once produced here.
    self_referential = sorted(
        key
        for key, reference in REFERENCES.items()
        if _REFERENCE_MARKER.search(f"{reference.short} {reference.citation} {reference.url}")
    )
    if self_referential:
        raise ValueError(
            f"reference record(s) {self_referential} contain a [[marker]] of their own"
        )


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


# Options every induced radiation curve needs from the caller. Named once so a
# spec cannot declare a subset of them by accident.
INDUCED = ("lam", "kt_max", "scales", "weights", "gain")

CLIMATOLOGY_VARIABLES: tuple[VariableSpec, ...] = (
    VariableSpec(
        id="air_temperature",
        label="Temperatura do ar",
        unit="°C",
        chart="histogram",
        family="normal",
        family_label="Gaussiana ([[wilks2019]], cap. 4)",
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
        family_label="Beta ([[raschke2011]])",
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
        family_label="Beta ([[raschke2011]])",
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
        family_label="Gaussiana ([[wilks2019]], cap. 4)",
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
        family_label="Weibull ([[justus1978]])",
        edges=_linear_edges(0.0, 20.0, 0.5),
        caveats=(
            "A Weibull não representa massa em zero: as calmarias saem do ajuste e são informadas à parte. Barras e curva são densidades condicionais às horas com vento acima do limiar de partida — é a densidade híbrida de [[takle1978]], mostrada como as suas duas peças. O limiar entra como calmaria: é o valor que o anemômetro reporta parado, não uma velocidade medida.",
        ),
    ),
    VariableSpec(
        id="wind_direction",
        label="Direção do vento",
        unit="°",
        chart="rose",
        family="von_mises_mixture",
        family_label="Mistura de von Mises ([[carta2008]])",
        edges=_linear_edges(0.0, 360.0, 360.0 / ROSE_SECTORS),
        caveats=(
            "Direção é uma grandeza circular: 0° e 360° são o mesmo rumo, então o gráfico é uma rosa dos ventos e as estatísticas são circulares. Média e desvio-padrão aritméticos de graus não têm significado aqui.",
            "As calmarias ficam fora da rosa: no limiar de partida do anemômetro, ou abaixo dele, o rumo é o da concha parada e não o do vento. A fração removida é informada à parte.",
        ),
    ),
    VariableSpec(
        id="precipitation",
        label="Precipitação (horas chuvosas)",
        unit="mm/h",
        chart="histogram",
        family="gamma",
        family_label="Gama ([[thom1958]]; [[wilks2019]])",
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
        family="compound_hollands_huget",
        family_label="[[hollands1983]] escalada pela fração PAR",
        edges=_linear_edges(0.0, 800.0, 20.0),
        fit_options=INDUCED,
        caveats=(
            "Restrita às horas com elevação solar acima de 10°, como toda variável de onda curta desta página. Até esta versão a PAR era a única exportada sobre TODAS as horas, e 38% dos valores publicados eram exatamente zero porque eram noite; a mesma curva contra a amostra sem o corte daria distância KS de 0,55.",
            "Mesma densidade induzida da onda curta incidente, com as escalas multiplicadas pela fração PAR média da era ({{param:gain:4}}, razão de energias com o denominador acima de 50 W/m²). A literatura não atribui densidade canônica à PAR em si — o que ela modela é a RAZÃO entre PAR e global —, e é justamente por isso que a curva entra por essa razão em vez de por um ajuste direto.",
            "[[custodio2021]] medem PAR/global diária de 0,50 em Petrolina e 0,44 em Brasília, e citam a faixa de 0,41 a 0,52 para a banda de 400 a 700 nm, que é a deste sensor. Os {{param:gain:4}} desta era estão dentro da faixa.",
            "Duas massas pontuais saem do ajuste e são impressas ao lado: {{atom:daytime_zero:count}} horas diurnas com PAR exatamente zero, concentradas entre agosto e outubro de 2018 e coincidindo com onda curta negativa, isto é, um período de zero do registrador; e {{atom:par_over_global:count}} horas com PAR/global acima de 0,6, fisicamente impossível por a PAR ser sub-banda da global.",
        ),
    ),
    VariableSpec(
        id="par_late",
        label="Radiação PAR (a partir de 2019-03)",
        unit="W/m²",
        chart="histogram",
        family="compound_hollands_huget",
        family_label="[[hollands1983]] escalada pela fração PAR",
        edges=_linear_edges(0.0, 800.0, 20.0),
        fit_options=INDUCED,
        caveats=(
            "Restrita às horas com elevação solar acima de 10°, como a era anterior. Sem o corte, 21% dos valores publicados eram zeros de noite.",
            "Mesma densidade induzida da era anterior; a ÚNICA coisa que difere entre as duas curvas é a fração PAR média: 0,4475 antes contra {{param:gain:4}} aqui. A razão entre as duas, 1,728, quantifica em quatro dígitos o erro de escala suspeito.",
            "[[custodio2021]] citam a faixa de 0,41 a 0,52 para a banda de 400 a 700 nm. Os {{param:gain:4}} desta era ficam bem abaixo, e nenhuma calibração documentada explica a diferença.",
            "Os valores estão como medidos, sem correção: comparar as duas eras é o ponto. Após o corte diurno esta era não tem nenhum zero; {{atom:par_over_global:count}} horas com PAR/global acima de 0,6 saem como massa pontual.",
        ),
    ),
    VariableSpec(
        id="shortwave_down",
        label="Radiação de onda curta incidente",
        unit="W/m²",
        chart="histogram",
        family="compound_hollands_huget",
        family_label="[[hollands1983]], densidade induzida",
        edges=_linear_edges(0.0, 1400.0, 25.0),
        fit_options=INDUCED,
        caveats=(
            "Restrito às horas com elevação solar acima de 10°. Sem esse corte metade do registro é noite e o histograma vira uma barra única em zero.",
            "As distâncias impressas ao lado da curva são as DESTE recorte; os números citados nas ressalvas vêm da análise de referência sobre o registro inteiro, então podem diferir na terceira casa.",
            "A curva NÃO é um ajuste ao fluxo: é a densidade que a lei do índice de claridade, já publicada nesta página, INDUZ sobre o fluxo por G = Kt · I0h, marginalizada sobre a irradiância extraterrestre observada em 60 faixas de igual contagem. Nenhum parâmetro novo é estimado — lambda e Kt_máx são herdados literalmente do ajuste do índice de claridade deste mesmo recorte.",
            "Por que não ajustar direto: a irradiância em W/m² não tem densidade canônica, porque o forçamento extraterrestre varia ao longo do dia e do ano e a forma da distribuição é em boa parte geometria solar, não clima. Mudar de variável é o que permite citar uma lei em vez de escolher a família que melhor couber.",
            "Uma beta de quatro parâmetros ajusta um pouco melhor — na análise de referência sobre o registro inteiro, distância KS de 0,0377 contra 0,0483 —, mas não tem respaldo na literatura para esta grandeza. A curva publicada é a que tem procedência, e a diferença fica registrada aqui.",
        ),
    ),
    VariableSpec(
        id="shortwave_up",
        label="Radiação de onda curta refletida",
        unit="W/m²",
        chart="histogram",
        family="compound_hollands_huget",
        family_label="[[hollands1983]] escalada pelo albedo",
        edges=_linear_edges(0.0, 450.0, 10.0),
        fit_options=INDUCED,
        caveats=(
            "Restrito às horas com elevação solar acima de 10°, pelo mesmo motivo da onda curta incidente, e à era a partir de 15/03/2019.",
            "A era anterior está fora porque o albedo medido muda de patamar: a mediana mensal diurna é 0,345 / 0,353 / 0,370 em dez/2018, jan/2019 e fev/2019 e cai para 0,155 em mar/2019, ficando entre 0,120 e 0,160 em todos os meses seguintes. A quebra fica dentro de uma interrupção de 17 dias que termina em 15/03/2019 — a mesma data que já separa as eras da PAR.",
            "Mesma densidade induzida da onda curta incidente, com todas as escalas multiplicadas pelo albedo médio da era (0,1323, razão de energias com o denominador acima de 50 W/m²). Um único escalar é estimado; lambda e Kt_máx seguem herdados.",
            "O albedo entra como constante porque foi medido como constante: ajustar a forma canônica com dependência do zênite dá d = 0,026 com R² de 0,003 neste sítio, contra o d = 0,4 que [[briegleb1992]] prescreve para pastagem, conforme transcrito por [[yang2008]]. A mediana por faixa de cosseno do zênite varia só de 0,122 a 0,133.",
            "Publicada em magnitude física (positiva). No gráfico de balanço da página de monitoramento os canais ascendentes aparecem negados, que é convenção de desenho para as parcelas somarem visualmente ao saldo — aqui o valor é o medido.",
            "A forma desta distribuição diz mais sobre o entorno da torre do que sobre o clima: o albedo de solo nu depende da umidade dos primeiros milímetros de superfície, que [[idso1975]] mediram variando de 0,00 a 0,18 num mesmo terreno conforme o teor de água.",
        ),
    ),
    VariableSpec(
        id="longwave_down",
        label="Radiação de onda longa incidente",
        unit="W/m²",
        chart="histogram",
        family="normal",
        family_label="Gaussiana ([[pashiardis2021]])",
        edges=_linear_edges(330.0, 540.0, 3.0),
        caveats=(
            "Todas as horas: ao contrário da onda curta, a atmosfera emite dia e noite, então não há corte por elevação solar a fazer.",
            "É a única das sete radiações desta página com densidade publicada para o fluxo bruto em si: [[pashiardis2021]] afirmam que a onda longa incidente segue a normal, enquanto a emitida segue uma normal com cauda positiva longa. O registro daqui é compatível — assimetria de -0,05 e curtose em excesso de -0,46.",
            "Uma mistura de duas gaussianas ajusta cerca de sete vezes melhor — na análise de referência, distância KS de 0,0051 contra 0,0352 —, mas é escolha estatística sem lei publicada por trás. A curva publicada é a citada, e a diferença fica registrada aqui.",
            "Os limites 0,4·sigma·Ta⁴ e sigma·Ta⁴ + 25, usados como controle de qualidade, são COEFICIENTES AJUSTÁVEIS do QCRad calibrados por sítio ([[long2008]], dão 0,58 a 0,80 e 11 a 23 em estações reais) — não constantes físicas, ao contrário do que sugere a forma da expressão.",
        ),
    ),
    VariableSpec(
        id="longwave_up",
        label="Radiação de onda longa emitida",
        unit="W/m²",
        chart="histogram",
        family="power_normal_mixture",
        family_label="Mistura de dois regimes (dia/noite); componente potência-normal ([[freeman2006]])",
        edges=_linear_edges(400.0, 640.0, 4.0),
        caveats=(
            "Todas as horas, em magnitude física (positiva). No balanço do monitoramento este canal aparece negado por convenção de desenho.",
            "A curva é a densidade que uma mistura de duas normais sobre a TEMPERATURA DE BRILHO induz no fluxo. A temperatura de brilho é Stefan-Boltzmann invertida a emissividade unitária, T = (L/sigma)^(1/4): é uma reetiquetagem exata, então a derivação não precisa supor nenhum valor de emissividade.",
            "A MISTURA NÃO É DA LITERATURA — [[freeman2006]] definem a potência-normal simples, não a mistura de duas; a composição segue a convenção que esta página já usa para o saldo noturno. O que a defende é que as componentes são IDENTIFICADAS, não escolhidas: rodando o algoritmo às cegas sobre a amostra inteira saem 298,06 K e 305,17 K, e ajustar separadamente as horas de noite e de dia dá 297,89 K e 304,29 K — as mesmas componentes a menos de 0,17 K e 0,87 K.",
            "A onda longa emitida é sigma·epsilon·Ts⁴ da superfície, mas ao contrário do que essa forma sugere a quarta potência quase não deforma a distribuição: com a temperatura de brilho medida aqui, ela sozinha produziria assimetria de 0,14, e a observada é 1,02. A assimetria vem de a população ser dois regimes, não da potência — que é exatamente por que a mistura, e não uma potência-normal simples, é o que se ajusta (KS 0,0135 contra 0,1138).",
        ),
    ),
    VariableSpec(
        id="net_radiation_day",
        label="Saldo de radiação (dia)",
        unit="W/m²",
        chart="histogram",
        family="compound_hollands_gaussian",
        family_label="[[hollands1983]] com [[hu2012]]",
        edges=_linear_edges(-100.0, 1000.0, 20.0),
        fit_options=(*INDUCED, "slope", "intercept", "residual_sd"),
        caveats=(
            "O saldo é uma mistura de dois regimes com escalas completamente diferentes, e juntar os dois num histograma só produz uma distribuição bimodal que nenhum modelo explica. Por isso dia e noite são publicados separados: aqui, as horas com elevação solar acima de 10°.",
            "A curva compõe dois resultados: a densidade de [[hollands1983]] para o índice de claridade, já publicada nesta página, e a relação linear entre saldo e onda curta incidente de [[hu2012]], reajustada localmente. Rn = a·Kt·I0h + b + erro, com a = {{param:slope:4}}, b = {{param:intercept:2}} W/m² e erro gaussiano de {{param:residual_sd:2}} W/m², marginalizado sobre a irradiância extraterrestre observada.",
            "A convolução gaussiana é licenciada por medida, não suposta: o resíduo da regressão tem assimetria 0,010 e curtose em excesso 0,063, com R² de 0,975.",
            "Atenção ao comparar com [[hu2012]]: o trabalho original é em MJ/m² por dia, não em W/m² horários. Os coeficientes acima são os desta estação, não os deles.",
            "Uma beta de quatro parâmetros ajusta cerca de duas vezes melhor — na análise de referência sobre o registro inteiro, distância KS de 0,0283 contra 0,0559 — e não tem lei publicada por trás. A curva publicada é a que tem procedência, e a diferença fica registrada aqui.",
        ),
    ),
    VariableSpec(
        id="net_radiation_night",
        label="Saldo de radiação (noite)",
        unit="W/m²",
        chart="histogram",
        family="normal",
        family_label="Gaussiana ([[wilks2019]]), regime noturno de céu claro",
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
        family_label="[[hollands1983]]",
        edges=_linear_edges(0.0, 1.0, 0.02),
        caveats=(
            "A irradiância global em W/m² não tem densidade canônica: metade do registro é noite e o forçamento extraterrestre varia ao longo do dia. A grandeza da literatura é o índice de claridade Kt = global / extraterrestre, restrito às horas com elevação solar acima de 10°. A densidade ajustada aqui foi construída para reproduzir as curvas de frequência de [[liu1960]].",
            "Num litoral tropical úmido espera-se bimodalidade (céu encoberto perto de 0,25 e céu limpo perto de 0,65), que a curva unimodal de [[hollands1983]] não reproduz.",
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


def _rounded_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Round a fit's parameters, including the list-valued ones.

    The induced radiation families carry the scale mixture in their parameters —
    sixty scales and sixty weights, and for net radiation nearly six thousand
    mixture means — so a scalar-only rounding raises on the first list. The rose
    already serialises list-valued parameters, so this is the existing shape
    rather than a new one.
    """
    rounded: dict[str, Any] = {}
    for name, value in params.items():
        if isinstance(value, (list, tuple, np.ndarray)):
            rounded[name] = [_rounded(float(item), _PARAMETER_DECIMALS) for item in value]
        else:
            rounded[name] = _rounded(float(value), _PARAMETER_DECIMALS)
    return rounded


def _finite_params(params: Mapping[str, Any]) -> bool:
    """Whether every parameter of a fit is finite, lists included."""
    for value in params.values():
        if isinstance(value, (list, tuple, np.ndarray)):
            if not np.all(np.isfinite(np.asarray(value, dtype=float))):
                return False
        elif not np.isfinite(float(value)):
            return False
    return True


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
    options: Mapping[str, Any] | None = None,
    *,
    curveless: bool = False,
) -> dict[str, Any]:
    """Bars, statistics, fit, distances and the pre-scaled curve for one subset."""
    values = np.asarray(sample, dtype=float)
    binned = dist.histogram(values, spec.edges)
    payload: dict[str, Any] = {
        # The WHOLE subset, so `n == sum(counts) + below + above` is an identity
        # the reader can check on screen. It used to be the binned count alone
        # while `stats` described the whole sample, which put three different
        # totals for one recorte on one panel: the clearness index printed
        # "34.934 observações" beside a maximum of 1,7382 that belongs to none of
        # the bars, and beside a fit whose own n was a third number again (the
        # family cuts at its fitted ceiling, which is legitimate and separate).
        "n": binned.n + binned.below + binned.above,
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
                "count": atom.count,
            }
            for atom in atoms
        ],
        "fit": None,
        "quality": None,
        "curve": None,
    }
    if spec.family is None or binned.n < 2:
        return payload

    # A family whose parameters are not estimable from the sample alone needs the
    # covariate the caller derived (the extraterrestrial-irradiance mixture every
    # induced radiation curve rides on). Declaring the requirement on the spec is
    # what turns a missing option into a loud failure instead of a silent
    # fallback to a curve fitted on the wrong thing.
    if spec.fit_options and not options:
        if not curveless:
            raise ValueError(
                f"{spec.id}: family {spec.family!r} requires fit options "
                f"{list(spec.fit_options)}, none supplied for this subset"
            )
        # The caller could not build the covariate and said so. Bars, counts,
        # statistics and atoms publish; only the curve is absent, which is a
        # state the page already renders. Blanking the sample to reach the
        # `binned.n < 2` early return instead would delete real measurements
        # over a missing model input.
        logger.warning("%s: no covariate for this subset, publishing bars without a curve", spec.id)
        return payload
    fitted = dist.fit_distribution(spec.family, values / spec.fit_scale, **(options or {}))
    if not _finite_params(fitted.params):
        logger.warning("%s: %s fit did not converge on %d samples", spec.id, spec.family, binned.n)
        return payload

    payload["fit"] = {
        "family": fitted.family,
        "params": _rounded_params(fitted.params),
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
                "count": atom.count,
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
    options: Mapping[str, Mapping[str, Any]] | None = None,
    curveless: Collection[str] = (),
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
    options:
        Subset id -> the fit options that subset's family needs, for the induced
        radiation curves. Per subset and not per variable because the inheritance
        is subset-matched: each recorte's induced curve rides on the clearness
        index fitted to that same recorte, exactly as every other variable on the
        page is fitted per subset.
    curveless:
        Subset ids for which the caller could not build the covariate. Those
        subsets publish bars with ``fit``, ``quality`` and ``curve`` left
        ``None`` instead of raising — an acknowledged absence, as distinct from
        options a caller simply forgot to pass, which stays a loud failure.
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
    per_subset_options = options or {}
    subsets: dict[str, Any] = {}
    for subset_id, sample in samples.items():
        subset_atoms = list(per_subset.get(subset_id, ()))
        if spec.chart == "rose":
            subsets[subset_id] = _rose_subset(sample, subset_atoms, components=mixture_components)
        else:
            subsets[subset_id] = _histogram_subset(
                spec,
                sample,
                subset_atoms,
                per_subset_options.get(subset_id),
                curveless=subset_id in curveless,
            )

    payload: dict[str, Any] = {
        "format": VARIABLE_FORMAT,
        "version": version,
        "id": spec.id,
        "label": spec.label,
        "unit": spec.unit,
        "chart": spec.chart,
        "family": spec.family,
        "family_label": spec.family_label,
        # Resolved against the recorte the page prints the prose beside, so a
        # number in a sentence is the same object as the number in the panel.
        "caveats": [
            _resolve_values(caveat, subsets.get("observed_all") or {}, f"{spec.id} caveat {index}")
            for index, caveat in enumerate(spec.caveats)
        ],
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
        # The bibliography, once, keyed by the [[key]] markers the labels and
        # caveats carry. Published in the manifest rather than repeated in every
        # variable file: sixteen records shared by sixteen variables would
        # otherwise be sixteen copies, and the page loads the manifest first
        # anyway. The page turns each marker into a link with the full record.
        "references": {
            key: {
                "short": reference.short,
                "citation": reference.citation,
                "url": reference.url,
            }
            for key, reference in sorted(REFERENCES.items())
        },
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


_assert_references()


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
