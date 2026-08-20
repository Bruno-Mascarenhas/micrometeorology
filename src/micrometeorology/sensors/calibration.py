"""Date-precise instrument calibration corrections.

Calibration records are loaded from ``configs/calibrations.yaml``.
Each record specifies a column, a date range, and a multiplicative factor.
Records are **immutable historical facts** — new calibrations must be
appended, never overwriting existing entries.
"""

import itertools
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_calibrations(config_path: str | Path) -> list[dict[str, Any]]:
    """Load calibration records from a YAML file.

    Parameters
    ----------
    config_path:
        Path to ``calibrations.yaml``.

    Returns
    -------
    list of dict
        The ``calibrations`` list as written, each record carrying ``column``,
        optional ``start_date`` / ``end_date``, ``factor`` and ``description``.
        A missing file yields an empty list and a warning. That warning is the
        only signal: nothing downstream distinguishes "no calibrations were
        declared" from "none were needed", since a column carrying no record is
        taken to need none.
    """
    path = Path(config_path)
    if not path.exists():
        logger.warning("Calibration config not found: %s", path)
        return []
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    records: list[dict[str, Any]] = data.get("calibrations", [])
    return records


def _resolve_inclusive_end(value: Any, fallback: pd.Timestamp) -> pd.Timestamp:
    """Resolve an ``end_date`` config value to an *inclusive* upper bound.

    ``end_date`` is the *last day* the record applies, but a date-only value such
    as ``"2018-12-31"`` parses to midnight, so ``<= end`` would drop the boundary
    day's post-``00:00`` samples. A midnight-resolution timestamp is therefore
    extended by ``1 day - 1 ns``; an explicit non-midnight time keeps exact
    ``<= end``, and a falsy value returns *fallback* ("end of the dataset").
    """
    if not value:
        return fallback
    end = pd.Timestamp(value)
    if end == end.normalize():
        return end + pd.Timedelta(days=1) - pd.Timedelta(1, "ns")
    return end


# A resolved date range for one record, tagged with a human-readable label used
# only in overlap-error messages: ``(label, inclusive_start, inclusive_end)``.
_ResolvedRange = tuple[str, pd.Timestamp, pd.Timestamp]


def _describe_record(record: dict[str, Any]) -> str:
    """Return a record's identity — column, date range, description — for error messages."""
    start = record.get("start_date") or "dataset-start"
    end = record.get("end_date") or "dataset-end"
    parts = [str(record["column"]), f"{start} -> {end}"]
    description = record.get("description")
    if description:
        parts.append(str(description))
    return " | ".join(parts)


def _resolve_record_range(
    record: dict[str, Any], df: pd.DataFrame
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Resolve a record's inclusive ``[start, end]`` bounds against *df*'s index.

    Both application paths call this, so an absent ``start_date`` defaults to the
    first index timestamp and ``end_date`` goes through
    :func:`_resolve_inclusive_end` identically wherever a record is applied.
    """
    start = (
        pd.Timestamp(record["start_date"])
        if record.get("start_date")
        else pd.Timestamp(df.index.min())
    )
    end = _resolve_inclusive_end(record.get("end_date"), df.index.max())
    return start, end


def _find_overlapping_pair(
    ranges: list[_ResolvedRange],
) -> tuple[_ResolvedRange, _ResolvedRange, pd.Timestamp, pd.Timestamp] | None:
    """Return the first overlapping pair of inclusive ``(label, start, end)`` ranges.

    Sorting by ``start`` makes a single adjacent-pair scan sufficient: if any two
    ranges overlap, two consecutive ones do.

    Ranges with ``start > end`` are EMPTY and dropped first. An open-ended record
    inherits the dataset's own first timestamp, so one that closes before the data
    begins inverts and simply does not apply here (the application path agrees —
    its mask is empty); counting it would make every later record "overlap" it.
    """
    ordered = sorted((item for item in ranges if item[1] <= item[2]), key=lambda item: item[1])
    for earlier, later in itertools.pairwise(ordered):
        # Sorted by start, so the pair overlaps iff later's start falls on or
        # before earlier's inclusive end.
        if later[1] <= earlier[2]:
            overlap_start = later[1]
            overlap_end = min(earlier[2], later[2])
            return earlier, later, overlap_start, overlap_end
    return None


def _overlap_error(
    kind: str,
    group: str,
    earlier: _ResolvedRange,
    later: _ResolvedRange,
    overlap_start: pd.Timestamp,
    overlap_end: pd.Timestamp,
) -> ValueError:
    """Build a clear ``ValueError`` for an overlapping-range configuration error."""
    day_lo = overlap_start.date()
    day_hi = overlap_end.date()
    day = str(day_lo) if day_lo == day_hi else f"{day_lo}..{day_hi}"
    return ValueError(
        f"Overlapping {kind} for {group!r} on {day}: [{earlier[0]}] and "
        f"[{later[0]}] both cover {day}. Date ranges are inclusive of the whole "
        f"end day, so consecutive records must abut on the NEXT day "
        f"(e.g. end_date 2018-12-31 then start_date 2019-01-01), not the same day."
    )


def apply_calibrations(
    df: pd.DataFrame,
    calibrations: list[dict[str, Any]],
) -> pd.DataFrame:
    """Apply calibration corrections to a DataFrame in-place.

    Parameters
    ----------
    df:
        DataFrame with a naive station-local ``DatetimeIndex``.
    calibrations:
        List of calibration records, each with keys:
        ``column``, ``start_date``, ``end_date``, ``factor``, ``description``.
        ``factor`` multiplies the raw column, so it carries the unit ratio the
        instrument's sensitivity revision implies; a null ``factor`` instead
        declares the data invalid over the period and blanks it.

    Returns
    -------
    pd.DataFrame
        The same DataFrame, mutated in place, with each record applied to its
        own window. A record matching no populated sample is logged as a
        warning rather than raising: it is a configuration drift, not a data
        fault.

    Raises
    ------
    ValueError
        If two records for one column cover an overlapping date range, which
        would make the published factor depend on record order.

    Notes
    -----
    Both dates are **inclusive at day granularity**: a date-only ``end_date``
    covers every sub-daily sample of the boundary day, while one carrying an
    explicit time keeps exact ``<= end`` (see :func:`_resolve_inclusive_end`).
    """
    ranges_by_column: dict[str, list[_ResolvedRange]] = {}
    for record in calibrations:
        column = record["column"]
        if column not in df.columns:
            continue
        start, end = _resolve_record_range(record, df)
        ranges_by_column.setdefault(column, []).append((_describe_record(record), start, end))
    for column, ranges in ranges_by_column.items():
        overlap = _find_overlapping_pair(ranges)
        if overlap is not None:
            raise _overlap_error("calibrations", column, *overlap)

    for record in calibrations:
        column = record["column"]
        if column not in df.columns:
            logger.debug("Skipping calibration for missing column: %s", column)
            continue

        start, end = _resolve_record_range(record, df)
        factor = record.get("factor")
        description = record.get("description", "")

        mask = (df.index >= start) & (df.index <= end)

        # "Declared" and "applied" must stay distinguishable: a record selecting
        # no populated row corrects nothing yet looks like one that worked. Live
        # case — the Eppley PSP's post-2019 sensitivity correction named only the
        # pre-rename column spelling, publishing the diffuse sensor ~8.5% low.
        if not bool((mask & df[column].notna()).any()):
            logger.warning(
                "calibration for %r [%s -> %s] matched no populated sample (%s)",
                column,
                start.date(),
                end.date(),
                description,
            )

        if factor is None:
            df.loc[mask, column] = np.nan
            logger.info(
                "  %s [%s -> %s]: set to NaN (%s)", column, start.date(), end.date(), description
            )
        else:
            df.loc[mask, column] *= factor
            logger.info(
                "  %s [%s -> %s]: x %.10f (%s)",
                column,
                start.date(),
                end.date(),
                factor,
                description,
            )

    return df


def load_sensor_switches(config_path: str | Path) -> list[dict[str, Any]]:
    """Load sensor-switch definitions from the calibrations YAML.

    Sensor switches define which raw column maps to a unified variable
    during specific date ranges (e.g. PSP1 → CM3Up after 2018-11).

    Parameters
    ----------
    config_path:
        Path to ``calibrations.yaml``, the same file
        :func:`load_calibrations` reads.

    Returns
    -------
    list of dict
        The ``sensor_switches`` list as written, each entry carrying a
        ``unified_name`` and its ``mappings``. A missing file yields an empty
        list, silently: the unification step then simply creates no unified
        column, which the caller sees directly.
    """
    path = Path(config_path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    switches: list[dict[str, Any]] = data.get("sensor_switches", [])
    return switches


def resolve_mapping_windows(
    df: pd.DataFrame,
    switches: list[dict[str, Any]],
    unified_names: Sequence[str],
) -> dict[str, list[tuple[str, pd.Timestamp, pd.Timestamp]]]:
    """Which raw column fed each unified name, and over which inclusive window.

    :func:`unify_sensor_columns` *copies* the source into the alias, so both
    survive: anything that judges a unified channel unusable must reach its
    source too, or the same number stays published under the other name.

    Parameters
    ----------
    df:
        The frame the windows are resolved against; an open-ended mapping
        inherits its first or last timestamp, so the same switch yields
        different bounds on a different frame.
    switches:
        Sensor-switch definitions, as :func:`load_sensor_switches` returns them.
    unified_names:
        Which unified channels to report on. Names absent here are skipped
        entirely, which is what keeps the caller from having to know about
        channels it does not mask.

    Returns
    -------
    dict
        ``{unified_name: [(source column, start, end), ...]}`` for mappings
        whose column is in *df*, with inclusive naive station-local bounds
        resolved exactly as :func:`unify_sensor_columns` resolves them.
    """
    wanted = set(unified_names)
    windows: dict[str, list[tuple[str, pd.Timestamp, pd.Timestamp]]] = {}
    for switch in switches:
        unified_name = switch["unified_name"]
        if unified_name not in wanted:
            continue
        for mapping in switch["mappings"]:
            column = mapping["column"]
            if column not in df.columns:
                continue
            start, end = _resolve_record_range(mapping, df)
            windows.setdefault(unified_name, []).append((column, start, end))
    return windows


def uncalibrated_mapping_windows(
    df: pd.DataFrame,
    calibrations: list[dict[str, Any]],
    switches: list[dict[str, Any]],
) -> list[tuple[str, str, pd.Timestamp, pd.Timestamp]]:
    """Windows where a column feeds a unified series with no calibration covering it.

    "Declared" and "applied" are not the same thing. Two live faults:

    - The Eppley PSP's post-2019 sensitivity correction named only the pre-v11
      column spelling, so the diffuse sensor published ~8.5% low.
    - No ``PSP1_Wm2_Avg`` record covers 2016-09-29..2017-12-31, so the published
      global shortwave steps 6.2% at 2018-01-01 — a file handover, not a physical
      event.

    Only columns carrying at least one calibration record are considered: one
    with none needs none, its calibration being the logger's own programmed
    multiplier (temperature, humidity, tipping bucket). The reported shape is a
    column calibrated over PART of the window it feeds, which puts a step in the
    published series at the boundary and nowhere in the instrument record.

    Returns
    -------
    list
        ``(unified name, source column, gap start, gap end)`` per uncovered
        stretch, oldest first, with inclusive naive station-local bounds. A
        ``factor: null`` record counts as coverage: declaring a period invalid
        is a decision about it, not an omission.

    Notes
    -----
    Each mapping window is walked with a cursor that consumes the calibrated
    stretches in date order; whatever the cursor steps over is what gets
    reported.
    """
    covered: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for record in calibrations:
        column = record["column"]
        if column not in df.columns:
            continue
        start, end = _resolve_record_range(record, df)
        if start <= end:
            covered.setdefault(column, []).append((start, end))

    gaps: list[tuple[str, str, pd.Timestamp, pd.Timestamp]] = []
    for switch in switches:
        unified_name = switch["unified_name"]
        for mapping in switch["mappings"]:
            column = mapping["column"]
            if column not in df.columns or column not in covered:
                continue
            start, end = _resolve_record_range(mapping, df)
            if start > end:
                continue
            cursor = start
            for cal_start, cal_end in sorted(covered.get(column, [])):
                if cal_end < cursor:
                    continue
                if cal_start > cursor:
                    gaps.append(
                        (unified_name, column, cursor, min(end, cal_start - pd.Timedelta(1, "ns")))
                    )
                cursor = max(cursor, cal_end + pd.Timedelta(1, "ns"))
                if cursor > end:
                    break
            if cursor <= end:
                gaps.append((unified_name, column, cursor, end))
    return sorted(gaps, key=lambda item: item[2])


def unify_sensor_columns(
    df: pd.DataFrame,
    switches: list[dict[str, Any]],
) -> pd.DataFrame:
    """Create unified columns from sensor-switch definitions.

    For each switch, creates a new column with the ``unified_name`` that
    concatenates data from different raw columns based on date ranges.

    Parameters
    ----------
    df:
        The frame to add the unified columns to, indexed by a naive
        station-local ``DatetimeIndex``.
    switches:
        Sensor-switch definitions, as :func:`load_sensor_switches` returns them.

    Returns
    -------
    pd.DataFrame
        The same DataFrame, mutated in place, with one column added per switch
        that matched at least one present raw column. The source columns are
        COPIED, not consumed, so both spellings of a measurement stay in the
        frame -- which is why anything that invalidates a unified channel has to
        reach its sources as well (:func:`resolve_mapping_windows`).

    Raises
    ------
    ValueError
        If two mappings of one unified name cover an overlapping date range,
        which would make the concatenation order decide the published value.

    Notes
    -----
    Mapping ranges are **inclusive at day granularity**: a date-only ``end_date``
    covers the whole boundary day, so mappings abutting on a day boundary leave
    no hole; an explicit time keeps exact ``<= end`` semantics.
    """
    for switch in switches:
        unified_name = switch["unified_name"]
        mapping_ranges = [
            (_describe_record(mapping), *_resolve_record_range(mapping, df))
            for mapping in switch["mappings"]
            if mapping["column"] in df.columns
        ]
        overlap = _find_overlapping_pair(mapping_ranges)
        if overlap is not None:
            raise _overlap_error("sensor-switch mappings", unified_name, *overlap)

        series_parts: list[pd.Series] = []

        for mapping in switch["mappings"]:
            column = mapping["column"]
            if column not in df.columns:
                logger.warning("Column %s not found for unified variable %s", column, unified_name)
                continue

            start, end = _resolve_record_range(mapping, df)

            mask = (df.index >= start) & (df.index <= end)
            part = df.loc[mask, column].rename(unified_name)
            series_parts.append(part)

        if series_parts:
            df[unified_name] = pd.concat(series_parts).reindex(df.index)
            logger.info("Created unified column: %s", unified_name)

    return df


#: Arquivo de geometria solar do laboratório, em passo de cinco minutos e no mesmo
#: relógio ingênuo local do acervo. A coluna ``fc`` é o fator geométrico de
#: correção do anel de sombreamento (Drummond, 1956), que devolve à difusa medida
#: a fração do domo que o anel oculta. Sem ele o piranômetro sombreado subestima
#: a difusa: medida sobre este acervo, a fração difusa satura em 0,81 sob céu
#: encoberto (Kt < 0,10), quando a física exige que tenda a 1.
SHADE_RING_FACTOR_FILE = "teorica_2016-2030.csv"

#: Derived form of the table above, in the shape the rest of the archive is
#: written in: one ``DatetimeIndex`` in naive station-local time, ``float64``
#: columns, zstd. Preferred when present and read column-wise, which is what
#: makes it worth having — the CSV costs 1.42 s and 203 MB on disk to yield six
#: of its twenty columns, the parquet 0.02 s and 29 MB for the same answer.
#: ``float64`` and not ``float32`` deliberately: ``fc`` multiplies the measured
#: diffuse, and the 5.4e-08 relative error of single precision would move every
#: corrected value in the published archive.
SHADE_RING_FACTOR_PARQUET = "teorica_2016-2030.parquet"

_SHADE_RING_TIME_COLUMNS = ("ano_i", "mes_i", "dia_i", "hor_i", "min_i")


class MissingShadeRingFactorError(LookupError):
    """Uma amostra de difusa não encontrou fator de correção do anel."""


def solar_geometry_to_parquet(csv_path: Path | str, parquet_path: Path | str) -> Path:
    """Rewrite the solar-geometry table in the archive's own on-disk shape.

    The lab publishes it as a 203 MB CSV whose five leading integer columns spell
    out a timestamp. Reindexed on that timestamp and stored column-wise it is 29 MB
    and answers a single-column query in 0.02 s against 1.42 s, which is what the
    hourly window job pays every run.

    The CSV is left untouched: this writes a derived artifact beside it.

    Parameters
    ----------
    csv_path:
        The lab's table, as delivered.
    parquet_path:
        Destination for the derived table.

    Returns
    -------
    pathlib.Path
        The path written.
    """
    frame = pd.read_csv(csv_path, engine="pyarrow")
    stamps = pd.to_datetime(
        {
            "year": frame["ano_i"],
            "month": frame["mes_i"],
            "day": frame["dia_i"],
            "hour": frame["hor_i"],
            "minute": frame["min_i"],
        }
    )
    derived = (
        frame.drop(columns=list(_SHADE_RING_TIME_COLUMNS))
        .set_index(pd.DatetimeIndex(stamps).as_unit("us"))
        .sort_index()
    )
    derived.index.name = None
    destination = Path(parquet_path)
    derived.to_parquet(destination, compression="zstd")
    return destination


def load_shade_ring_factors(path: Path | str) -> pd.Series:
    """Fator geométrico do anel de sombreamento, por instante de cinco minutos.

    Reads :data:`SHADE_RING_FACTOR_PARQUET` when it sits beside the CSV, which is
    the same numbers in the archive's own format; the CSV remains the fallback and
    the source of truth.

    Parameters
    ----------
    path:
        Caminho do arquivo de geometria solar.

    Returns
    -------
    pandas.Series
        ``(N,)`` adimensional, índice ``DatetimeIndex`` ingênuo local, ordenado.

    Raises
    ------
    FileNotFoundError
        Se a tabela não existir sob o diretório de dados.
    ValueError
        Se o arquivo não trouxer a coluna ``fc`` ou as de tempo.
    """
    derived = Path(path).with_name(SHADE_RING_FACTOR_PARQUET)
    if derived.is_file():
        return pd.read_parquet(derived, columns=["fc"])["fc"].sort_index()
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"{path}: the shade-ring factor table is missing. The diffuse cannot be "
            "published without it — uncorrected, the diffuse fraction saturates at "
            "0.81 under an overcast sky where the physics requires 1."
        )
    frame = pd.read_csv(path, usecols=[*_SHADE_RING_TIME_COLUMNS, "fc"])
    faltando = {*_SHADE_RING_TIME_COLUMNS, "fc"} - set(frame.columns)
    if faltando:
        raise ValueError(f"{path}: faltam as colunas {sorted(faltando)}")
    instantes = pd.to_datetime(
        {
            "year": frame["ano_i"],
            "month": frame["mes_i"],
            "day": frame["dia_i"],
            "hour": frame["hor_i"],
            "minute": frame["min_i"],
        }
    )
    fatores = pd.Series(frame["fc"].to_numpy(dtype=float), index=pd.DatetimeIndex(instantes))
    return fatores.sort_index()


def apply_shade_ring_correction(
    df: pd.DataFrame,
    factors: pd.Series,
    windows: Sequence[tuple[str, pd.Timestamp, pd.Timestamp]],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Devolve à difusa medida a parcela do céu que o anel de sombreamento oculta.

    Aplicada às colunas CRUAS e dentro da janela em que cada uma alimentou
    ``Sw_dif``, pelo mesmo motivo que :func:`apply_calibrations` roda antes da
    unificação: a cópia herda a correção, e o escopo por janela é obrigatório
    porque o mesmo instrumento mede o GLOBAL fora dela — ``PSP_Wm2_Avg`` foi o
    piranômetro global até 2018-08, e corrigi-lo ali corromperia ``Sw_dw``.

    Parameters
    ----------
    df:
        Frame de cinco minutos com índice ingênuo local.
    factors:
        Fatores por instante, como :func:`load_shade_ring_factors` os devolve.
    windows:
        ``(coluna, início, fim)`` inclusivos, das janelas de mapeamento de
        ``Sw_dif``.

    Returns
    -------
    tuple
        O mesmo frame, mutado in place, e ``{coluna: amostras corrigidas}``.

    Raises
    ------
    MissingShadeRingFactorError
        Se alguma amostra de difusa dentro de uma janela não tiver fator. Um
        fator ausente que virasse NaN apagaria a medida em silêncio, e um que
        virasse 1,0 publicaria difusa sem correção sob o nome da corrigida.
    """
    alinhados = factors.reindex(pd.DatetimeIndex(df.index))
    corrigidas: dict[str, int] = {}
    for column, start, end in windows:
        if column not in df.columns:
            continue
        dentro = (df.index >= start) & (df.index <= end)
        alvo = dentro & df[column].notna().to_numpy()
        if not alvo.any():
            continue
        perdidas = int(alinhados[alvo].isna().sum())
        if perdidas:
            raise MissingShadeRingFactorError(
                f"{column}: {perdidas} amostra(s) de difusa entre {start} e {end} sem fator "
                f"de anel em {SHADE_RING_FACTOR_FILE}"
            )
        df.loc[alvo, column] = df.loc[alvo, column] * alinhados[alvo]
        corrigidas[column] = corrigidas.get(column, 0) + int(alvo.sum())
    return df, corrigidas
