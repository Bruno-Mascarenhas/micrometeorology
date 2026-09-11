"""Byte contract shared by every document the sky page reads.

Schema ids, the publish stamp every document of one run repeats, the pinned
timezone block and the writer. The writer is the site pipeline's
(:func:`micrometeorology.common.site_json.write_json`): compact separators,
UTF-8 unescaped, ``allow_nan=False``, atomic rename — the same bytes contract
the climatology and monitoring exporters honour, so the page can parse every
directory under ``site/`` with one strict ``response.json()``.
"""

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from labmim_core.site import STATION_UTC_OFFSET_HOURS
from labmim_core.sky import (
    SKY_CLASS_COUNT,
    SKY_CLASS_KT_UPPER_BOUNDS,
    SKY_CLASS_NAMES,
    SKY_CLASS_NAMES_PT,
)
from micrometeorology.common.site_json import finite, rounded, write_json

__all__ = [
    "CONDITION_IDS",
    "DEFAULT_KINDEX_KIND",
    "FRAME_SCHEMA",
    "MODEL_SCHEMA",
    "SKIPPED_REASON_LABELS_PT",
    "SOURCE_LABELS_PT",
    "STATION_TIMEZONE_NAME",
    "TIMELINE_SCHEMA",
    "PublishStamp",
    "condition_of",
    "document_header",
    "finite",
    "publish_stamp",
    "rounded",
    "rounded_or_none",
    "rounded_rows",
    "sky_conditions_block",
    "targets_glossary",
    "timezone_block",
    "write_document",
]

FRAME_SCHEMA = "labmim-allsky-frame-v2"
TIMELINE_SCHEMA = "labmim-allsky-timeline-v1"
MODEL_SCHEMA = "labmim-allsky-model-v1"

#: The camera and the datalogger stamp on this zone's fixed offset; the page
#: labels every naive timestamp with it.
STATION_TIMEZONE_NAME = "America/Bahia"
VERSION_STAMP_FORMAT = "%Y%m%dT%H%M%SZ"
NAIVE_LOCAL_FORMAT = "%Y-%m-%dT%H:%M:%S"
#: What a watch record that names no ``kindex_kind`` is taken to serve; every
#: checkpoint of the served family fits k*, and the fallback is logged wherever
#: it is applied.
DEFAULT_KINDEX_KIND = "kstar"

#: The page reads a condition by ``id`` (``i``..``iv``) or by ``condition``
#: (1..4), never by the exporter's 0-based class index.
CONDITION_IDS = ("i", "ii", "iii", "iv")
SKY_CONDITIONS_REFERENCE = (
    "Escobedo, Gomes, Oliveira & Soares (2009), Applied Energy 86(3):299-309, sec. 3.1; "
    "Portuguese nomenclature after Teramoto & Escobedo (2012), RBEAA 16(9):985-992"
)

SKIPPED_REASON_LABELS_PT: dict[str, str] = {
    "insufficient_frames": "quadros de menos no bloco (ou lacuna de captura)",
    "below_elevation_floor": "sol abaixo do piso de elevação do modelo",
    "no_frame_predictions": "nenhum quadro do bloco foi pontuado",
    "prediction_failed": "o modelo falhou ao pontuar o bloco",
}
SOURCE_LABELS_PT: dict[str, str] = {
    "frame_aggregate": "média das previsões dos quadros do bloco",
    "block_model": "modelo de bloco sobre os quadros do bloco",
}


@dataclass(frozen=True, slots=True)
class PublishStamp:
    """The ``version`` and ``generated_utc`` one publish writes into every document.

    Both derive from one aware UTC instant through :meth:`at`, so no document
    can carry a version from one publish and a generation time from another.
    """

    version: str
    generated_utc: str

    @classmethod
    def at(cls, now_utc: dt.datetime) -> PublishStamp:
        """The stamp of a publish happening at the aware UTC instant *now_utc*."""
        if now_utc.tzinfo is None or now_utc.utcoffset() != dt.timedelta(0):
            raise ValueError("the publish stamp is taken in UTC; pass an aware UTC datetime")
        return cls(
            version=now_utc.strftime(VERSION_STAMP_FORMAT),
            generated_utc=now_utc.strftime(VERSION_STAMP_FORMAT),
        )


def publish_stamp(now_utc: dt.datetime | None = None) -> PublishStamp:
    """The stamp of a publish happening at *now_utc* (default: now)."""
    return PublishStamp.at(now_utc or dt.datetime.now(tz=dt.UTC))


def timezone_block() -> dict[str, Any]:
    """The ``timezone`` block every document repeats for its naive local stamps."""
    return {"name": STATION_TIMEZONE_NAME, "utc_offset_hours": STATION_UTC_OFFSET_HOURS}


def document_header(schema: str, stamp: PublishStamp) -> dict[str, Any]:
    """The four keys every document opens with, in the order the page expects."""
    return {
        "schema": schema,
        "version": stamp.version,
        "generated_utc": stamp.generated_utc,
        "timezone": timezone_block(),
    }


def condition_of(class_index: int) -> dict[str, Any]:
    """The published names of the 0-based sky class *class_index*."""
    return {
        "name": SKY_CLASS_NAMES[class_index],
        "condition": class_index + 1,
        "id": CONDITION_IDS[class_index],
        "name_pt": SKY_CLASS_NAMES_PT[class_index],
    }


def sky_conditions_block() -> dict[str, Any]:
    """The four conditions in the shape ``ktkd.json`` already publishes them."""
    lower: list[float | None] = [None, *SKY_CLASS_KT_UPPER_BOUNDS]
    upper: list[float | None] = [*SKY_CLASS_KT_UPPER_BOUNDS, None]
    return {
        "kt_upper_bounds": list(SKY_CLASS_KT_UPPER_BOUNDS),
        "reference": SKY_CONDITIONS_REFERENCE,
        "ground_truth": "kt_bands_of_ghi",
        "conditions": [
            {**condition_of(index), "kt_range": [lower[index], upper[index]]}
            for index in range(SKY_CLASS_COUNT)
        ],
    }


def targets_glossary(kindex_kind: str) -> dict[str, Any]:
    """What the served indices are, so the page never labels k* as Kt."""
    if kindex_kind == "kstar":
        kindex = {
            "kind": "kstar",
            "symbol": "k*",
            "label_pt": "índice de céu claro",
            "definition_pt": "razão entre a irradiância global medida e a global de céu claro de Haurwitz no mesmo instante; passa de 1 sob realce por nuvens",
            "reference": "haurwitz",
        }
    else:
        kindex = {
            "kind": "kt",
            "symbol": "Kt",
            "label_pt": "índice de claridade",
            "definition_pt": "razão entre a irradiância global medida e a extraterrestre no plano horizontal",
            "reference": "escobedo",
        }
    return {
        "kindex": kindex,
        "dhi": {
            "symbol": "DHI",
            "unit": "W/m²",
            "label_pt": "irradiância difusa horizontal",
            "definition_pt": "média de 5 minutos do piranômetro sombreado da estação, a referência contra a qual o modelo é avaliado",
        },
        "sky": {
            "label_pt": "condição de céu",
            "definition_pt": "as quatro faixas de Kt de Escobedo et al. (2009) aplicadas à média de 5 minutos da global; o modelo aprende a reproduzi-las a partir da imagem",
            "reference": "escobedo",
        },
    }


def rounded_or_none(value: object, decimals: int) -> float | None:
    """:func:`rounded` over anything a report may hold: ``None``, a string, a NumPy scalar."""
    if value is None:
        return None
    try:
        return rounded(float(value), decimals)  # type: ignore[arg-type]
    except TypeError, ValueError:
        return None


def rounded_rows(
    rows: Sequence[Sequence[object]], decimals: Sequence[int | None]
) -> list[list[Any]]:
    """Round positional rows column by column; ``None`` in *decimals* leaves a column as is."""
    return [
        [
            value if digits is None else rounded_or_none(value, digits)
            for value, digits in zip(row, decimals, strict=True)
        ]
        for row in rows
    ]


def write_document(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write one document in the site encoding, atomically."""
    return write_json(path, payload)
