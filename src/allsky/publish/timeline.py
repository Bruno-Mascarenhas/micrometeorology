"""``timeline.json``: the watch's block predictions of the last days on one regular axis.

Timestamps here are **naive station-local** on the camera's clock, as in
:mod:`allsky.watch`, whose ``blocks/`` records this reads: a block is named by
its end on that clock, and the page labels every stamp with the site's fixed
offset. The station export is read on the same contract as
:func:`allsky.snapshot._sensor_row_near` — the logger's clock is naive local,
an export that carries an offset is converted into the station's zone — and
its diffuse column is screened through the archive's own sentinel table and
declared range gates before a single value is published as measured.

The axis is the monitoring page's convention: one regular grid of block ends
and parallel arrays with ``null`` where a block has no record, so a night or a
capture outage is a hole and never a segment drawn across it. Solar elevation
and the clear-sky reference are evaluated at every block's centroid
(``t - step/2``), the instant :func:`allsky.snapshot.predict_block` scores a
block at, whether or not the watch scored it.

Torch-free: the document is assembled from the records on disk and the
station export alone.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from allsky.clearsky import haurwitz_ghi_from_cos_zenith
from allsky.publish.encoding import (
    CONDITION_IDS,
    DAY_STAMP_FORMAT,
    DEFAULT_KINDEX_KIND,
    ELEVATION_DECIMALS,
    INDEX_DECIMALS,
    IRRADIANCE_DECIMALS,
    NAIVE_LOCAL_FORMAT,
    SKIPPED_REASON_LABELS_PT,
    SOURCE_LABELS_PT,
    TIMELINE_SCHEMA,
    PublishStamp,
    class_share,
    condition_of,
    document_header,
    kindex_kind_of,
    rounded_or_none,
    sky_conditions_block,
    targets_glossary,
)
from allsky.serving import ServingConfig
from allsky.snapshot import (
    StationExport,
    clearsky_dhi_series,
    read_station_export,
    shipped_sensor_limits,
)
from allsky.watch import (
    BLOCK_STEM_FORMAT,
    BLOCKS_SUBDIR,
    FRAMES_SUBDIR,
    PREDICTION_SUFFIX,
    SKIPPED_SUFFIX,
    SOURCE_BLOCK_MODEL,
    SOURCE_FRAME_AGGREGATE,
)
from labmim_core.atomic import JsonObjectError, read_json_object
from labmim_core.site import SiteConfig
from labmim_core.sky import SKY_CLASS_COUNT, SKY_CLASS_NAMES
from labmim_core.solar import cos_zenith

if TYPE_CHECKING:
    from micrometeorology.common.config import SensorRangeLimit

logger = logging.getLogger(__name__)

__all__ = ["BlockRecord", "TimelineError", "build_timeline"]

MEASURED_COLUMN = "PSP_Wm2_Avg"
MEASURED_SCREENING = "sentinels + sensor_limits"
MEASURED_SOURCE_LABEL = "Piranômetro de difusa PSP da estação LabMiM, média de 5 minutos"
MEASURED_OK = "ok"
MEASURED_NO_EXPORT = "no_export"
MEASURED_EXPORT_STALE = "export_stale"
MEASURED_NO_VALID_ROWS = "no_valid_rows"
MEASURED_INTERVAL_MISMATCH = "interval_mismatch"

STATUS_SCORED = "scored"
STATUS_SKIPPED = "skipped"
KNOWN_SOURCES = (SOURCE_FRAME_AGGREGATE, SOURCE_BLOCK_MODEL)


class TimelineError(ValueError):
    """The watch directory, a block record or the station export cannot feed the timeline."""


@dataclass(frozen=True, slots=True)
class BlockRecord:
    """One ``blocks/<stem>.prediction.json`` or ``.skipped.json``, reduced to what the page reads.

    Attributes
    ----------
    end:
        Naive station-local block end.
    status:
        ``scored`` for a prediction record, ``skipped`` for a skip record.
    probabilities:
        One entry per class in :data:`labmim_core.sky.SKY_CLASS_NAMES` order,
        ``None`` where the record carries none.
    """

    end: pd.Timestamp
    status: str
    source: str | None
    reason: str | None
    n_frames: int | None
    dhi: float | None
    kindex: float | None
    class_index: int | None
    probabilities: tuple[float | None, ...]
    kindex_kind: str | None


@dataclass(frozen=True, slots=True)
class _SolarEnvelope:
    elevation_deg: np.ndarray
    clearsky_dhi: np.ndarray
    clearsky_ghi: np.ndarray


@dataclass(frozen=True, slots=True)
class _Export:
    screened: pd.Series
    last_row_at: pd.Timestamp | None
    interval: pd.Timedelta | None


@dataclass(slots=True)
class _DayTally:
    scored: int = 0
    skipped: int = 0
    frames: int = 0
    classes: list[int] = field(default_factory=list)


def _stamp(timestamp: pd.Timestamp) -> str:
    return timestamp.strftime(NAIVE_LOCAL_FORMAT)


def _finite_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _class_index_of(name: object, path: Path) -> int | None:
    if name is None:
        return None
    if name not in SKY_CLASS_NAMES:
        raise TimelineError(
            f"{path}: sky class {name!r} is none of {SKY_CLASS_NAMES}; the record was written by "
            "a model with other classes than the page publishes"
        )
    return SKY_CLASS_NAMES.index(str(name))


def _probabilities_of(payload: object) -> tuple[float | None, ...]:
    if not isinstance(payload, Mapping):
        return (None,) * SKY_CLASS_COUNT
    return tuple(_finite_or_none(payload.get(name)) for name in SKY_CLASS_NAMES)


def _block_record(payload: Mapping[str, Any], status: str, path: Path) -> BlockRecord:
    try:
        end = pd.Timestamp(str(payload["block_end"]))
    except (KeyError, ValueError) as exc:
        raise TimelineError(f"{path}: no readable block_end") from exc
    n_frames = payload.get("n_frames")
    frames = int(n_frames) if n_frames is not None else None
    if status == STATUS_SKIPPED:
        reason = payload.get("reason")
        return BlockRecord(
            end=end,
            status=status,
            source=None,
            reason=str(reason) if reason is not None else None,
            n_frames=frames,
            dhi=None,
            kindex=None,
            class_index=None,
            probabilities=(None,) * SKY_CLASS_COUNT,
            kindex_kind=None,
        )
    source = payload.get("source")
    if source not in KNOWN_SOURCES:
        raise TimelineError(f"{path}: source {source!r} is none of {KNOWN_SOURCES}")
    predictions = payload.get("predictions") or {}
    return BlockRecord(
        end=end,
        status=status,
        source=str(source),
        reason=None,
        n_frames=frames,
        dhi=_finite_or_none(predictions.get("dhi")),
        kindex=_finite_or_none(predictions.get("kindex")),
        class_index=_class_index_of(predictions.get("sky_class"), path),
        probabilities=_probabilities_of(predictions.get("sky_probabilities")),
        kindex_kind=kindex_kind_of(payload),
    )


def _read_payload(path: Path) -> dict[str, Any] | None:
    try:
        return read_json_object(path)
    except JsonObjectError as exc:
        logger.warning("skipping block record: %s", exc)
        return None


def _read_block_records(
    blocks_dir: Path, *, since: pd.Timestamp
) -> dict[pd.Timestamp, BlockRecord]:
    records: dict[pd.Timestamp, BlockRecord] = {}
    if not blocks_dir.is_dir():
        return records
    since_stem = since.strftime(BLOCK_STEM_FORMAT)
    for path in sorted(blocks_dir.glob("*.json")):
        if path.name.endswith(PREDICTION_SUFFIX):
            status, stem = STATUS_SCORED, path.name[: -len(PREDICTION_SUFFIX)]
        elif path.name.endswith(SKIPPED_SUFFIX):
            status, stem = STATUS_SKIPPED, path.name[: -len(SKIPPED_SUFFIX)]
        else:
            continue
        if stem < since_stem:
            continue
        payload = _read_payload(path)
        if payload is None:
            continue
        record = _block_record(payload, status, path)
        records[record.end] = record
    return records


def _kindex_kind(records: Mapping[pd.Timestamp, BlockRecord], frames_dir: Path) -> str:
    for end in sorted(records, reverse=True):
        kind = records[end].kindex_kind
        if kind is not None:
            return kind
    newest = (
        max(frames_dir.glob(f"*{PREDICTION_SUFFIX}"), default=None) if frames_dir.is_dir() else None
    )
    if newest is not None:
        payload = _read_payload(newest)
        kind = kindex_kind_of(payload) if payload is not None else None
        if kind is not None:
            return kind
    logger.warning(
        "no block or frame record names the served kindex kind; publishing the glossary for %s",
        DEFAULT_KINDEX_KIND,
    )
    return DEFAULT_KINDEX_KIND


def _axis(now_local: pd.Timestamp, days: int, block_minutes: float) -> pd.DatetimeIndex:
    step = f"{block_minutes:g}min"
    window_start = now_local.normalize() - pd.Timedelta(days=days - 1)
    return pd.date_range(window_start.ceil(step), now_local.floor(step), freq=step)


def _solar_envelope(centroids: pd.DatetimeIndex, site: SiteConfig) -> _SolarEnvelope:
    cos_zenith_values = cos_zenith(centroids, site, site.utc_offset_hours)
    return _SolarEnvelope(
        elevation_deg=np.rad2deg(np.arcsin(cos_zenith_values)),
        clearsky_dhi=clearsky_dhi_series(centroids, site),
        clearsky_ghi=haurwitz_ghi_from_cos_zenith(cos_zenith_values),
    )


def _station_export(sensor_csv: str | Path | StationExport | None) -> StationExport | None:
    if sensor_csv is None or isinstance(sensor_csv, StationExport):
        return sensor_csv
    try:
        return read_station_export(sensor_csv)
    except (OSError, ValueError) as exc:
        raise TimelineError(f"cannot read the station export {sensor_csv}: {exc}") from exc


def _screened_export(export: StationExport, limits: list[SensorRangeLimit]) -> _Export | None:
    from micrometeorology.sensors.archive import mask_sentinels
    from micrometeorology.sensors.ingestion import apply_physical_limits

    frame = export.rows
    if MEASURED_COLUMN not in frame.columns:
        raise TimelineError(
            f"{export.path} has no {MEASURED_COLUMN} column; the timeline publishes the shaded "
            "pyranometer's 5-minute mean and nothing else as measured"
        )
    if not any(limit.column == MEASURED_COLUMN for limit in limits):
        logger.warning(
            "sensor_limits declares no range gate for %s; the export is not published as measured "
            "until configs/micromet/default.yaml declares one",
            MEASURED_COLUMN,
        )
        return None
    frame = frame.loc[~frame.index.duplicated(keep="first")]
    screened, _removed = mask_sentinels(frame)
    gated = apply_physical_limits(screened.loc[:, [MEASURED_COLUMN]].copy(), limits)
    values = gated[MEASURED_COLUMN].astype(np.float64)
    finite = values[np.isfinite(values.to_numpy())]
    return _Export(
        screened=values,
        last_row_at=pd.Timestamp(finite.index.max()) if not finite.empty else None,
        interval=_modal_spacing(pd.DatetimeIndex(values.index)),
    )


def _modal_spacing(index: pd.DatetimeIndex) -> pd.Timedelta | None:
    """The spacing most rows of *index* share, or ``None`` when no spacing repeats.

    An export of hourly means has a 60-minute mode; a five-minute export with
    holes still has a five-minute mode. Too few rows to repeat any spacing
    cannot be judged, and are not.
    """
    spacing = pd.Series(index).diff().dropna()
    if spacing.empty:
        return None
    counts = spacing.value_counts()
    if int(counts.iloc[0]) < 2:
        return None
    return pd.Timedelta(counts.index[0])


def _matched_values(export: _Export, ends: pd.DatetimeIndex, tolerance: pd.Timedelta) -> np.ndarray:
    matched = np.full(len(ends), np.nan)
    if export.screened.empty or len(ends) == 0:
        return matched
    positions = export.screened.index.get_indexer(ends, method="nearest", tolerance=tolerance)
    hit = positions >= 0
    matched[hit] = export.screened.to_numpy(dtype=np.float64)[positions[hit]]
    return matched


def _series(
    axis: pd.DatetimeIndex,
    records: Mapping[pd.Timestamp, BlockRecord],
    envelope: _SolarEnvelope,
    measured: np.ndarray | None,
    train_max_elevation_deg: float | None,
) -> tuple[dict[str, list[Any]], list[str | None], list[dict[str, Any]]]:
    count = len(axis)
    dhi: list[float | None] = [None] * count
    kindex: list[float | None] = [None] * count
    condition: list[int | None] = [None] * count
    probabilities: list[list[float | None]] = [[None] * count for _ in CONDITION_IDS]
    n_frames: list[int | None] = [None] * count
    source: list[str | None] = [None] * count
    skipped: list[dict[str, Any]] = []
    for position, end in enumerate(axis):
        record = records.get(end)
        if record is None:
            continue
        n_frames[position] = record.n_frames
        if record.status == STATUS_SKIPPED:
            skipped.append({"t": _stamp(end), "reason": record.reason})
            continue
        source[position] = record.source
        dhi[position] = rounded_or_none(record.dhi, IRRADIANCE_DECIMALS)
        kindex[position] = rounded_or_none(record.kindex, INDEX_DECIMALS)
        if record.class_index is not None:
            condition[position] = int(condition_of(record.class_index)["condition"])
        for index, probability in enumerate(record.probabilities):
            probabilities[index][position] = rounded_or_none(probability, INDEX_DECIMALS)
    elevation = [
        rounded_or_none(value, ELEVATION_DECIMALS) for value in envelope.elevation_deg.tolist()
    ]
    extrapolation: list[bool | None] = (
        [None] * count
        if train_max_elevation_deg is None
        else [
            bool(value > train_max_elevation_deg) if np.isfinite(value) else None
            for value in envelope.elevation_deg.tolist()
        ]
    )
    series: dict[str, list[Any]] = {
        "dhi_w_m2": dhi,
        "kindex": kindex,
        "condition": condition,
        **{f"p_{name}": values for name, values in zip(CONDITION_IDS, probabilities, strict=True)},
        "n_frames": n_frames,
        "solar_elevation_deg": elevation,
        "clearsky_dhi_w_m2": [
            rounded_or_none(value, IRRADIANCE_DECIMALS) for value in envelope.clearsky_dhi.tolist()
        ],
        "clearsky_ghi_w_m2": [
            rounded_or_none(value, IRRADIANCE_DECIMALS) for value in envelope.clearsky_ghi.tolist()
        ],
        "extrapolation": extrapolation,
        "measured_dhi_w_m2": (
            [None] * count
            if measured is None
            else [rounded_or_none(value, IRRADIANCE_DECIMALS) for value in measured.tolist()]
        ),
    }
    return series, source, skipped


def _latest(axis: pd.DatetimeIndex, records: Mapping[pd.Timestamp, BlockRecord]) -> dict[str, Any]:
    within = [records[end] for end in axis if end in records]
    scored_ends = [record.end for record in within if record.status == STATUS_SCORED]
    last = within[-1] if within else None
    return {
        "last_scored_block": _stamp(max(scored_ends)) if scored_ends else None,
        "last_block_status": last.status if last is not None else None,
        "reason": last.reason if last is not None else None,
    }


def _days(
    axis: pd.DatetimeIndex, records: Mapping[pd.Timestamp, BlockRecord]
) -> list[dict[str, Any]]:
    tallies: dict[pd.Timestamp, _DayTally] = {}
    for end in axis:
        tally = tallies.setdefault(end.normalize(), _DayTally())
        record = records.get(end)
        if record is None:
            continue
        tally.frames += record.n_frames or 0
        if record.status == STATUS_SCORED:
            tally.scored += 1
            if record.class_index is not None:
                tally.classes.append(record.class_index)
        else:
            tally.skipped += 1
    return [
        {
            "date": day.strftime(DAY_STAMP_FORMAT),
            "blocks_scored": tally.scored,
            "blocks_skipped": tally.skipped,
            "frames": tally.frames,
            "condition_share": class_share(np.asarray(tally.classes, dtype=np.int64)),
        }
        for day, tally in tallies.items()
    ]


def _measured_status(
    export: _Export | None,
    *,
    sensor_csv: Path | None,
    window_start: pd.Timestamp,
    step: pd.Timedelta,
) -> dict[str, Any]:
    if export is None:
        return {
            "available": False,
            "reason": MEASURED_NO_EXPORT,
            "last_row_at": None,
            "source_label": MEASURED_SOURCE_LABEL,
        }
    if export.last_row_at is None:
        logger.warning(
            "no row of %s survived the screening; nothing is published as measured", sensor_csv
        )
        return {
            "available": False,
            "reason": MEASURED_NO_VALID_ROWS,
            "last_row_at": None,
            "source_label": MEASURED_SOURCE_LABEL,
        }
    if export.interval is not None and export.interval != step:
        logger.warning(
            "the station export %s is spaced %s, not the %s block; its rows are not the block "
            "means the timeline compares against",
            sensor_csv,
            export.interval,
            step,
        )
        return {
            "available": False,
            "reason": MEASURED_INTERVAL_MISMATCH,
            "last_row_at": _stamp(export.last_row_at),
            "source_label": MEASURED_SOURCE_LABEL,
        }
    fresh = export.last_row_at >= window_start
    if not fresh:
        logger.warning(
            "the station export %s ends at %s, before the window start %s; the measured series "
            "is empty in this window",
            sensor_csv,
            export.last_row_at,
            window_start,
        )
    return {
        "available": fresh,
        "reason": MEASURED_OK if fresh else MEASURED_EXPORT_STALE,
        "last_row_at": _stamp(export.last_row_at),
        "source_label": MEASURED_SOURCE_LABEL,
    }


def _live_comparison(
    records: Mapping[pd.Timestamp, BlockRecord],
    export: _Export | None,
    *,
    since: pd.Timestamp,
    tolerance: pd.Timedelta,
) -> dict[str, Any] | None:
    if export is None:
        return None
    scored = [
        records[end]
        for end in sorted(records)
        if records[end].status == STATUS_SCORED and end >= since and records[end].dhi is not None
    ]
    ends = pd.DatetimeIndex([record.end for record in scored])
    predicted = np.asarray([record.dhi for record in scored], dtype=np.float64)
    measured = _matched_values(export, ends, tolerance)
    paired = np.isfinite(measured) & np.isfinite(predicted)
    n_blocks = int(paired.sum())
    error = predicted[paired] - measured[paired]
    dhi: dict[str, float | None] = (
        {
            "rmse": rounded_or_none(float(np.sqrt(np.mean(error**2))), IRRADIANCE_DECIMALS),
            "mae": rounded_or_none(float(np.mean(np.abs(error))), IRRADIANCE_DECIMALS),
            "mbe": rounded_or_none(float(np.mean(error)), IRRADIANCE_DECIMALS),
        }
        if n_blocks
        else {"rmse": None, "mae": None, "mbe": None}
    )
    return {
        "since": since.strftime(DAY_STAMP_FORMAT),
        "n_days": len({end.normalize() for end, hit in zip(ends, paired, strict=True) if hit}),
        "n_blocks": n_blocks,
        "dhi": dhi,
        "kindex": {"mae": None},
        "sky": {"balanced_accuracy": None},
    }


def _caveats(
    *,
    pin: ServingConfig,
    block_minutes: float,
    train_max_elevation_deg: float | None,
    measured_status: Mapping[str, Any],
) -> list[str]:
    half_step = block_minutes / 2
    extrapolation = (
        f"blocos com elevação acima do máximo do treino ({train_max_elevation_deg:.1f}°) são "
        "marcados como extrapolação."
        if train_max_elevation_deg is not None
        else "a marca de extrapolação não foi calculada porque o máximo de elevação do treino não "
        "foi fornecido."
    )
    if measured_status["reason"] == MEASURED_OK:
        measured = (
            f"A difusa medida é a média de {block_minutes:g} minutos do piranômetro sombreado "
            f"({MEASURED_COLUMN}), triada pelos sentinelas e pelos limites físicos do acervo e "
            "casada diretamente com o bloco que termina no mesmo carimbo, sem deslocamento. É o "
            "único holdout limpo do modelo: dias que nunca entraram em decisão alguma."
        )
    elif measured_status["reason"] == MEASURED_EXPORT_STALE:
        measured = (
            "A exportação da estação fornecida termina antes desta janela (última leitura em "
            f"{measured_status['last_row_at']}); a comparação ao vivo cobre só os blocos que ela "
            "alcança, e as métricas do split de teste não a substituem."
        )
    elif measured_status["reason"] == MEASURED_NO_VALID_ROWS:
        measured = (
            "A exportação da estação fornecida não tem nenhuma leitura de difusa que passe pelos "
            "sentinelas e pelos limites físicos do acervo; nada é publicado como medido."
        )
    elif measured_status["reason"] == MEASURED_INTERVAL_MISMATCH:
        measured = (
            f"A exportação da estação fornecida não está na grade de {block_minutes:g} minutos dos "
            "blocos; as suas linhas não são as médias que a linha do tempo compara, e nada é "
            "publicado como medido."
        )
    else:
        measured = (
            "A difusa medida não está publicada: a exportação da estação não foi fornecida nesta "
            "publicação, e a comparação ao vivo continua pendente — as métricas do split de teste "
            "não a substituem."
        )
    return [
        (
            f"Cada bloco de {block_minutes:g} minutos é a média das previsões dos quadros capturados "
            "nele (fonte frame_aggregate) ou a previsão do modelo de bloco (block_model); blocos "
            "sem registro são nulos, e um vazio no gráfico é uma noite ou uma falha de captura, "
            "nunca um segmento interpolado."
        ),
        (
            "A elevação solar e a referência de céu claro (global de Haurwitz e difusa de Erbs) "
            f"são avaliadas no centro do bloco, t - {half_step:g} min, o instante em que o modelo "
            f"pontua o bloco; {extrapolation}"
        ),
        (
            f"A rede não pontua blocos com o sol abaixo de {pin.min_elevation_deg:g}° "
            "(below_elevation_floor) nem blocos com quadros de menos (insufficient_frames); cada "
            "bloco pulado aparece em skipped com o seu motivo."
        ),
        measured,
    ]


def build_timeline(
    watch_dir: str | Path,
    *,
    pin: ServingConfig,
    stamp: PublishStamp,
    now_local: pd.Timestamp,
    days: int,
    site: SiteConfig,
    block_minutes: float = 5.0,
    sensor_csv: str | Path | StationExport | None = None,
    train_max_elevation_deg: float | None = None,
) -> dict[str, Any]:
    """Assemble ``timeline.json`` from the watch's block records.

    Parameters
    ----------
    watch_dir:
        The ``allsky watch --out`` directory holding ``blocks/`` and ``frames/``.
    pin:
        The serving pin: its elevation floor names the skip reason in the
        caveats and ``selection.decided_on`` opens the live comparison.
    stamp:
        The publish stamp shared by every document of this run.
    now_local:
        Naive station-local instant of the publish; the axis ends at the last
        block closed by then.
    days:
        Window length in calendar days ending today; the axis starts at the
        first block end on or after the first day's midnight.
    site:
        Observation site whose latitude, longitude and clock offset fix the
        sun at every block centroid.
    block_minutes:
        The logger's averaging interval and the axis step.
    sensor_csv:
        The operator's station export, as a path or already read
        (:func:`allsky.snapshot.read_station_export`, so one publish parses
        it once); ``None`` publishes ``measured`` as ``null`` and the live
        comparison as pending.
    train_max_elevation_deg:
        Highest solar elevation the served checkpoints trained on; blocks
        above it are flagged ``extrapolation``. ``None`` publishes the flag
        as ``null``.

    Returns
    -------
    dict
        The ``labmim-allsky-timeline-v1`` document: ``series`` are parallel
        arrays of length ``axis.count`` with ``null`` where a block has no
        record; every number is rounded and finite; no value is a filesystem
        path.

    Raises
    ------
    TimelineError
        When *watch_dir* does not exist, *now_local* is timezone-aware,
        *days* or *block_minutes* is not positive, a record names a sky class
        or a source the page does not publish, or the export lacks a time
        column or the diffuse column.
    """
    if now_local.tzinfo is not None:
        raise TimelineError("now_local is the camera's naive local clock; pass a naive timestamp")
    if days < 1:
        raise TimelineError(f"days must be at least 1, got {days}")
    if block_minutes <= 0:
        raise TimelineError(f"block_minutes must be positive, got {block_minutes}")
    root = Path(watch_dir)
    if not root.is_dir():
        raise TimelineError(f"{root} is not a directory; point --watch-dir at the watch's --out")

    step = pd.Timedelta(minutes=block_minutes)
    axis = _axis(now_local, days, block_minutes)
    window_start = axis[0]
    since = pd.Timestamp(pin.selection.decided_on)
    records = _read_block_records(root / BLOCKS_SUBDIR, since=min(window_start, since))
    envelope = _solar_envelope(axis - step / 2, site)

    station = _station_export(sensor_csv)
    export_path = station.path if station is not None else None
    export = _screened_export(station, shipped_sensor_limits()) if station is not None else None
    measured_status = _measured_status(
        export, sensor_csv=export_path, window_start=window_start, step=step
    )
    if measured_status["reason"] in (MEASURED_NO_VALID_ROWS, MEASURED_INTERVAL_MISMATCH):
        export = None
    measured = _matched_values(export, axis, step / 2) if export is not None else None
    series, source, skipped = _series(axis, records, envelope, measured, train_max_elevation_deg)

    return {
        **document_header(TIMELINE_SCHEMA, stamp),
        "axis": {
            "start": _stamp(axis[0]),
            "step_minutes": float(block_minutes),
            "count": len(axis),
        },
        "series": series,
        "source": source,
        "source_labels_pt": dict(SOURCE_LABELS_PT),
        "skipped": skipped,
        "reason_labels_pt": dict(SKIPPED_REASON_LABELS_PT),
        "latest": _latest(axis, records),
        "measured": (
            None
            if measured is None
            else {
                "source_column": MEASURED_COLUMN,
                "screening": MEASURED_SCREENING,
                "n": int(np.isfinite(measured).sum()),
            }
        ),
        "measured_status": measured_status,
        "live": _live_comparison(records, export, since=since, tolerance=step / 2),
        "days": _days(axis, records),
        "targets": targets_glossary(_kindex_kind(records, root / FRAMES_SUBDIR)),
        "sky_conditions": sky_conditions_block(),
        "caveats": _caveats(
            pin=pin,
            block_minutes=block_minutes,
            train_max_elevation_deg=train_max_elevation_deg,
            measured_status=measured_status,
        ),
    }
