"""Diffuse fraction against clearness index: the Kt-Kd plane and its models.

Kt is the clearness index (measured global over extraterrestrial horizontal
irradiance) and Kd the **diffuse fraction** Hd/H — diffuse over GLOBAL, not over
the extraterrestrial term.  Every quantity here is an hourly mean, which is the
timescale the three published models were fitted on.

The models
----------
Only Marques Filho et al. (2016) is a function of Kt alone, so it is the one that
can be drawn as a single curve.  Lemos et al. (2017) and the BRL model of Ridley,
Boland & Lauret (2010) also read the apparent solar time, the solar elevation,
the daily clearness index and the persistence, so on a Kt-Kd plane they are not
functions at all: at one Kt they predict a spread.  They are therefore summarised
per Kt bin as a median with a p10-p90 envelope, never as a line — a line would
assert a determinism the model does not have.

The solar geometry is not recomputed here: :mod:`allsky.solar` owns it, and the
extraterrestrial term behind Kt is the same one the climatology exporter divides
by.

Timestamps are naive station-local, as they arrive from the datalogger's own
clock; the apparent solar time derived here takes its offset from the
``utc_offset_hours`` parameter, never from the host's zone.
"""

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from allsky.solar import hour_angle_deg, solar_elevation_deg
from micrometeorology.stats.daylight import elevation_bounds

__all__ = [
    "KTKD_SCHEMA",
    "MODEL_LABELS",
    "MODEL_REFERENCES",
    "apparent_solar_time_hours",
    "daily_clearness_index",
    "density_2d",
    "lemos_2017",
    "marques_filho_2016",
    "model_band",
    "persistence_index",
    "regression_scores",
    "ridley_brl_2010",
]

#: Published schema tag of the artifact the sky page reads.
KTKD_SCHEMA = "labmim-ktkd-v1"

#: Short display names. Published in the payload so the page prints the name the
#: producer chose, instead of deriving one by splitting the citation on its first
#: comma — a parser that turns a bibliography edit into a relabelled chart.
MODEL_LABELS: dict[str, str] = {
    "marques_filho_2016": "Marques Filho et al. (2016)",
    "lemos_2017": "Lemos et al. (2017)",
    "ridley_brl_2010": "BRL — Ridley et al. (2010)",
}

MODEL_REFERENCES: dict[str, str] = {
    "marques_filho_2016": (
        "Marques Filho, E. P., Oliveira, A. P., Vita, W. A., Mesquita, F. L. L., Codato, G., "
        "Escobedo, J. F., Cassol, M. & França, J. R. A. (2016). Global, diffuse and direct "
        "solar radiation at the surface in the city of Rio de Janeiro. "
        "Renewable and Sustainable Energy Reviews 54, 1210-1220."
    ),
    "lemos_2017": (
        "Lemos, L. F. L., Starke, A. R., Boland, J., Cardemil, J. M., Machado, R. D. & "
        "Colle, S. (2017). Assessment of solar radiation components in Brazil using the BRL "
        "model. Renewable Energy 108, 569-580."
    ),
    "ridley_brl_2010": (
        "Ridley, B., Boland, J. & Lauret, P. (2010). Modelling of diffuse solar fraction with "
        "multiple predictors. Renewable Energy 35(2), 478-483."
    ),
}


def apparent_solar_time_hours(
    timestamps: pd.DatetimeIndex, longitude: float, utc_offset_hours: float
) -> NDArray:
    """Apparent solar time in hours, from the hour angle the solar module derives.

    Parameters
    ----------
    timestamps:
        Naive station-local stamps, ``(N,)``.
    longitude, utc_offset_hours:
        Site longitude in degrees east and the fixed offset of the local clock.

    Returns
    -------
    numpy.ndarray
        Apparent solar time, ``(N,)``, in hours on ``[0, 24)``: noon is 12 by
        construction, since the hour angle is zero at solar noon.
    """
    angle = hour_angle_deg(timestamps, longitude, utc_offset_hours=utc_offset_hours)
    return np.asarray(12.0 + angle / 15.0, dtype=np.float64)


def daily_clearness_index(global_radiation: pd.Series, extraterrestrial: pd.Series) -> pd.Series:
    """Daily Kt broadcast back onto every hour of its own day.

    The ratio of the day's summed global to the day's summed extraterrestrial
    irradiance — a ratio of daily totals, not the mean of hourly ratios, which
    would weight a noisy sunrise hour like a bright midday one.

    Parameters
    ----------
    global_radiation, extraterrestrial:
        Hourly means aligned on the same index, W m-2.

    Returns
    -------
    pandas.Series
        Daily Kt on the input index; NaN for a day whose extraterrestrial total
        is zero (a polar-night case this site never sees, guarded anyway).
    """
    day = pd.DatetimeIndex(global_radiation.index).normalize()
    totals = pd.DataFrame({"g": global_radiation, "e": extraterrestrial}).groupby(day).sum()
    ratio = (totals["g"] / totals["e"]).replace([np.inf, -np.inf], np.nan)
    return ratio.reindex(day).set_axis(global_radiation.index)


def persistence_index(kt: pd.Series) -> pd.Series:
    """Persistence ψ: how clear the neighbouring hours were.

    The mean of the previous and next hour's Kt, falling back to the single
    available neighbour at the ends of a daylight run.  A gap in the record is
    not bridged: an hour whose neighbour is missing takes the neighbour it has,
    and an isolated hour takes its own Kt, so persistence never invents a
    continuity the record does not show.

    Parameters
    ----------
    kt:
        Hourly clearness index on a DatetimeIndex, ``(N,)``.

    Returns
    -------
    pandas.Series
        ψ aligned to *kt*.
    """
    # Narrowed at the boundary, the way the climatology exporter narrows its own
    # index: the arithmetic below is only defined on a DatetimeIndex.
    index = pd.DatetimeIndex(kt.index)
    hour = pd.Timedelta(hours=1)
    previous = kt.reindex(index - hour).set_axis(index)
    following = kt.reindex(index + hour).set_axis(index)
    both = pd.concat([previous, following], axis=1)
    # mean(skipna) already falls back to the one neighbour that exists; an hour
    # with neither keeps its own value rather than becoming NaN.
    return both.mean(axis=1).fillna(kt)


def marques_filho_2016(kt: NDArray) -> NDArray:
    """Diffuse fraction of Marques Filho et al. (2016), a function of Kt alone.

    Formula
    -------
    ``Kd = 0.13 + 0.86 / (1 + exp(-6.29 + 12.26 Kt))``

    Parameters
    ----------
    kt:
        Hourly clearness index, ``(N,)``.

    Returns
    -------
    numpy.ndarray
        Modelled diffuse fraction, ``(N,)``.
    """
    values = np.asarray(kt, dtype=np.float64)
    return 0.13 + 0.86 / (1.0 + np.exp(-6.29 + 12.26 * values))


def _logistic_brl(
    kt: NDArray,
    apparent_solar_time: NDArray,
    elevation_deg: NDArray,
    daily_kt: NDArray,
    persistence: NDArray,
    coefficients: tuple[float, float, float, float, float, float],
) -> NDArray:
    """The BRL logistic shared by both multi-predictor models, its coefficients apart."""
    intercept, b_kt, b_ast, b_elev, b_daily, b_psi = coefficients
    exponent = (
        intercept
        + b_kt * np.asarray(kt, dtype=np.float64)
        + b_ast * np.asarray(apparent_solar_time, dtype=np.float64)
        + b_elev * np.asarray(elevation_deg, dtype=np.float64)
        + b_daily * np.asarray(daily_kt, dtype=np.float64)
        + b_psi * np.asarray(persistence, dtype=np.float64)
    )
    return 1.0 / (1.0 + np.exp(exponent))


def lemos_2017(
    kt: NDArray,
    apparent_solar_time: NDArray,
    elevation_deg: NDArray,
    daily_kt: NDArray,
    persistence: NDArray,
) -> NDArray:
    """BRL diffuse fraction with the Brazilian coefficients of Lemos et al. (2017)."""
    # Lemos et al. (2017), the BRL form refitted to Brazilian stations.
    return _logistic_brl(
        kt,
        apparent_solar_time,
        elevation_deg,
        daily_kt,
        persistence,
        (-4.41, 7.87, -0.088, -0.0049, 1.47, 1.10),
    )


def ridley_brl_2010(
    kt: NDArray,
    apparent_solar_time: NDArray,
    elevation_deg: NDArray,
    daily_kt: NDArray,
    persistence: NDArray,
) -> NDArray:
    """BRL diffuse fraction with the original coefficients of Ridley et al. (2010)."""
    # Ridley, Boland & Lauret (2010), the generic all-site BRL fit.
    return _logistic_brl(
        kt,
        apparent_solar_time,
        elevation_deg,
        daily_kt,
        persistence,
        (-5.38, 6.63, -0.006, -0.007, 1.75, 1.31),
    )


def density_2d(kt: NDArray, kd: NDArray, kt_edges: NDArray, kd_edges: NDArray) -> dict[str, Any]:
    """Two-dimensional histogram of the Kt-Kd plane, in the page's orientation.

    Returns
    -------
    dict
        ``counts`` with **rows = Kd bands, columns = Kt bands**, so
        ``counts[i][j]`` covers ``kd_edges[i]..kd_edges[i+1]`` by
        ``kt_edges[j]..kt_edges[j+1]``, plus ``max_count``.  The renderer refuses
        a matrix whose height does not match ``kd_edges``, because a transposed
        one would still draw — mirrored about the diagonal, in silence.
    """
    finite = np.isfinite(kt) & np.isfinite(kd)
    counts, _, _ = np.histogram2d(
        np.asarray(kd, dtype=np.float64)[finite],
        np.asarray(kt, dtype=np.float64)[finite],
        bins=[np.asarray(kd_edges, dtype=np.float64), np.asarray(kt_edges, dtype=np.float64)],
    )
    integer_counts = counts.astype(np.int64)
    return {
        "kt_edges": [float(edge) for edge in kt_edges],
        "kd_edges": [float(edge) for edge in kd_edges],
        "counts": [[int(value) for value in row] for row in integer_counts],
        "max_count": int(integer_counts.max()) if integer_counts.size else 0,
        "color_scale_hint": "log",
    }


def model_band(
    kt: NDArray, predicted: NDArray, kt_edges: NDArray, *, min_samples_per_bin: int
) -> dict[str, Any]:
    """Median and p10-p90 of a multi-predictor model, per Kt bin.

    A bin holding fewer than *min_samples_per_bin* observations publishes
    ``None`` for its three quantiles rather than a summary of two points.  Its
    ``n_per_bin`` still reports the real count, so a consumer must decide on
    ``median is None`` and never on a zero count: a suppressed bin almost always
    holds a non-zero number of samples.
    """
    kt_values = np.asarray(kt, dtype=np.float64)
    predicted_values = np.asarray(predicted, dtype=np.float64)
    edges = np.asarray(kt_edges, dtype=np.float64)
    finite = np.isfinite(kt_values) & np.isfinite(predicted_values)
    index = np.digitize(kt_values[finite], edges) - 1

    centres: list[float] = []
    medians: list[float | None] = []
    p10: list[float | None] = []
    p90: list[float | None] = []
    per_bin: list[int] = []
    for bin_index in range(len(edges) - 1):
        selected = predicted_values[finite][index == bin_index]
        centres.append(float((edges[bin_index] + edges[bin_index + 1]) / 2.0))
        per_bin.append(int(selected.size))
        if selected.size < min_samples_per_bin:
            medians.append(None)
            p10.append(None)
            p90.append(None)
            continue
        low, middle, high = np.percentile(selected, [10.0, 50.0, 90.0])
        medians.append(float(middle))
        p10.append(float(low))
        p90.append(float(high))

    return {
        "kind": "band",
        "kt": centres,
        "median": medians,
        "p10": p10,
        "p90": p90,
        "n_per_bin": per_bin,
        "min_samples_per_bin": int(min_samples_per_bin),
    }


def regression_scores(observed: NDArray, predicted: NDArray) -> dict[str, Any]:
    """RMSE, MBE, MAE and the pair count, over the rows where both are finite."""
    observed_values = np.asarray(observed, dtype=np.float64)
    predicted_values = np.asarray(predicted, dtype=np.float64)
    paired = np.isfinite(observed_values) & np.isfinite(predicted_values)
    if not paired.any():
        return {"rmse": None, "mbe": None, "mae": None, "n": 0}
    residual = predicted_values[paired] - observed_values[paired]
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mbe": float(np.mean(residual)),
        "mae": float(np.mean(np.abs(residual))),
        "n": int(paired.sum()),
    }


#: Ratios above this are instrument transients, not sky states: near sunrise the
#: two pyranometers' time responses disagree and the ratio overshoots. The legacy
#: producer used the same ceiling on Kd.
MAX_RATIO = 1.2

#: Global irradiance below this leaves Kd dominated by sensor offset rather than
#: by the sky, and its denominator near zero.
MIN_GLOBAL_WM2 = 50.0


def prepare_clearness(
    hourly: pd.DataFrame,
    *,
    site: Any,
    utc_offset_hours: float,
    min_elevation_deg: float = 10.0,
    global_column: str = "Sw_dw",
) -> pd.Series:
    """Gate the record down to the hours whose clearness index means something.

    Deliberately does NOT require a diffuse measurement, unlike
    :func:`prepare_ktkd`: the clearness record must describe every hour the
    global pyranometer measured, not only the subset where the shaded sensor
    happened to be up. Using the Kt-Kd gate here would publish F(Kt) conditioned
    on the diffuse channel's availability — a different population wearing the
    same name.

    Returns
    -------
    pandas.Series
        Kt on the surviving hours, with the extraterrestrial denominator
        evaluated at the averaging window's midpoint (BSRN).
    """
    from allsky.solar import extraterrestrial_ghi

    index = pd.DatetimeIndex(hourly.index)
    midpoint = index + pd.Timedelta(minutes=30)
    extraterrestrial = pd.Series(
        extraterrestrial_ghi(midpoint, site, utc_offset_hours), index=index
    )
    # The sun must clear the floor over the WHOLE hour the row averages, which is
    # the gate the climatology page applies and the one this artifact's text
    # claims. Testing the midpoint alone admits the hours the sun rises inside.
    lowest, _highest = elevation_bounds(index, site, utc_offset_hours)
    global_flux = hourly[global_column]
    usable = (extraterrestrial > 0) & (lowest > min_elevation_deg) & global_flux.notna()
    # No ratio ceiling here, unlike the Kt-Kd gate: a Kt above the axis is a real
    # measurement, and the histogram counts it in `above` so the reader sees the
    # axis is clipped. Dropping it would delete data and put this artifact's row
    # count out of step with the clearness histogram it claims to integrate.
    return (global_flux / extraterrestrial).where(usable).dropna()


def prepare_ktkd(
    hourly: pd.DataFrame,
    *,
    site: Any,
    utc_offset_hours: float,
    min_elevation_deg: float = 10.0,
    global_column: str = "Sw_dw",
    diffuse_column: str = "Sw_dif",
) -> dict[str, Any]:
    """Gate the hourly record down to usable Kt-Kd pairs and their predictors.

    The extraterrestrial denominator is evaluated at the averaging window's
    MIDPOINT, as the BSRN recommends for interval-averaged data and as the
    climatology exporter already does — so Kt here is the same quantity the
    clearness histogram publishes, not a second definition.

    Returns
    -------
    dict
        ``kt`` and ``kd`` Series on the surviving index, the model predictor
        arrays (``ast``, ``elevation``, ``daily_kt``, ``psi``), and ``filters``:
        the gate descriptions the artifact publishes so a reader knows what the
        cloud of points was conditioned on.
    """
    from allsky.solar import extraterrestrial_ghi

    index = pd.DatetimeIndex(hourly.index)
    midpoint = index + pd.Timedelta(minutes=30)
    extraterrestrial = pd.Series(
        extraterrestrial_ghi(midpoint, site, utc_offset_hours), index=index
    )
    lowest, _highest = elevation_bounds(index, site, utc_offset_hours)

    global_flux = hourly[global_column]
    diffuse_flux = hourly[diffuse_column]
    usable = (
        (global_flux > MIN_GLOBAL_WM2)
        & (extraterrestrial > 0)
        & diffuse_flux.notna()
        & (lowest > min_elevation_deg)
    )
    kt_all = (global_flux / extraterrestrial).where(usable)
    kd_all = (diffuse_flux / global_flux).where(usable)
    keep = usable & kt_all.between(0.0, MAX_RATIO) & kd_all.between(0.0, MAX_RATIO)

    kt = kt_all[keep]
    kd = kd_all[keep]
    kept_index = pd.DatetimeIndex(kt.index)
    kept_midpoint = kept_index + pd.Timedelta(minutes=30)
    return {
        "kt": kt,
        "kd": kd,
        "ast": apparent_solar_time_hours(kept_midpoint, site.longitude, utc_offset_hours),
        "elevation": solar_elevation_for(
            kept_midpoint, site.latitude, site.longitude, utc_offset_hours
        ),
        "daily_kt": daily_clearness_index(global_flux, extraterrestrial)[kt.index].to_numpy(),
        "psi": persistence_index(kt_all)[kt.index].to_numpy(),
        "filters": [
            f"irradiancia global acima de {MIN_GLOBAL_WM2:.0f} W/m2",
            f"elevacao solar acima de {min_elevation_deg:.0f} graus em toda a hora",
            f"Kt e Kd dentro de [0, {MAX_RATIO}]",
            "denominador extraterrestre no ponto medio da janela horaria (BSRN)",
        ],
    }


def build_ktkd_payload(
    kt: pd.Series,
    kd: pd.Series,
    *,
    models: dict[str, NDArray],
    sky_conditions: dict[str, Any],
    kt_edges: NDArray,
    kd_edges: NDArray,
    station: dict[str, Any],
    sources: list[str],
    filters: list[str],
    caveats: list[str],
    version: str,
    min_samples_per_bin: int = 30,
    include_points: bool = True,
) -> dict[str, Any]:
    """Assemble the published ``labmim-ktkd-v1`` payload.

    Parameters
    ----------
    kt, kd:
        Aligned hourly clearness index and diffuse fraction, already gated.
    models:
        Model id -> its prediction on the same index. ``marques_filho_2016`` is
        published as a curve because it is a function of Kt alone; every other
        model is summarised as a band, since at one Kt it predicts a spread.
    sky_conditions:
        :func:`micrometeorology.stats.sky_condition.sky_condition_summary` over
        the same Kt, so the page reads the bounds from the payload.

    Returns
    -------
    dict
        JSON-ready payload; every non-finite number is already ``None``.
    """
    curve_kt = np.asarray(kt_edges, dtype=np.float64)
    curve_kt = (curve_kt[:-1] + curve_kt[1:]) / 2.0

    published_models = []
    for model_id, predicted in models.items():
        scores = regression_scores(kd.to_numpy(), predicted)
        if model_id == "marques_filho_2016":
            entry: dict[str, Any] = {
                "kind": "curve",
                "kt": [float(value) for value in curve_kt],
                "kd": [float(value) for value in marques_filho_2016(curve_kt)],
            }
        else:
            entry = model_band(
                kt.to_numpy(), predicted, kt_edges, min_samples_per_bin=min_samples_per_bin
            )
        published_models.append(
            {
                "id": model_id,
                "label": MODEL_LABELS[model_id],
                "reference": MODEL_REFERENCES[model_id],
                **entry,
                **scores,
            }
        )

    payload: dict[str, Any] = {
        "schema": KTKD_SCHEMA,
        "version": version,
        "station": station,
        "period": {
            "start": kt.index.min().isoformat(),
            "end": kt.index.max().isoformat(),
            "n": len(kt),
        },
        "timescale": "hourly",
        "sources": sources,
        "filters": filters,
        "density": density_2d(kt.to_numpy(), kd.to_numpy(), kt_edges, kd_edges),
        "models": published_models,
        "sky_conditions": sky_conditions,
        "caveats": caveats,
    }
    if include_points:
        # Positional [kt, kd, t], which the reader accepts and which drops three
        # repeated keys per observation: the production host serves this JSON
        # uncompressed, so those bytes are bytes on the wire. Three decimals is
        # finer than the 0.02 bins the density is drawn on; `t` stays because the
        # page's CSV export reads it.
        payload["points"] = [
            [round(float(x), 3), round(float(y), 3), stamp.isoformat()]
            for stamp, x, y in zip(kt.index, kt.to_numpy(), kd.to_numpy(), strict=True)
        ]
        payload["points_format"] = ["kt", "kd", "t"]
    return payload


def solar_elevation_for(
    timestamps: pd.DatetimeIndex, latitude: float, longitude: float, utc_offset_hours: float
) -> NDArray:
    """Solar elevation in degrees for the model predictors, from :mod:`allsky.solar`."""
    from allsky.config import SiteConfig

    site = SiteConfig(latitude=latitude, longitude=longitude)
    return np.asarray(solar_elevation_deg(timestamps, site, utc_offset_hours), dtype=np.float64)
