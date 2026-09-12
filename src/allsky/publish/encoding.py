"""Byte contract shared by every document the sky page reads.

Schema ids, the publish stamp every document of one run repeats, the pinned
timezone block and the writer. The writer is the site pipeline's
(:func:`micrometeorology.common.site_json.write_json`): compact separators,
UTF-8 unescaped, ``allow_nan=False``, atomic rename — the same bytes contract
the climatology and monitoring exporters honour, so the page can parse every
directory under ``site/`` with one strict ``response.json()``.
"""

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from allsky.config import SITE_TZ_NAME
from labmim_core.site import STATION_UTC_OFFSET_HOURS
from labmim_core.sky import (
    SKY_CLASS_COUNT,
    SKY_CLASS_KT_UPPER_BOUNDS,
    SKY_CLASS_NAMES,
    SKY_CLASS_NAMES_PT,
    SKY_CLASS_REFERENCE,
)
from micrometeorology.common.site_json import finite, rounded, write_json

__all__ = [
    "CONDITION_IDS",
    "DAY_STAMP_FORMAT",
    "DEFAULT_KINDEX_KIND",
    "ELEVATION_DECIMALS",
    "FRAME_SCHEMA",
    "INDEX_DECIMALS",
    "IRRADIANCE_DECIMALS",
    "MODEL_SCHEMA",
    "REFERENCES",
    "SHARE_DECIMALS",
    "SKIPPED_REASON_LABELS_PT",
    "SOURCE_LABELS_PT",
    "TIMELINE_SCHEMA",
    "PublishStamp",
    "class_share",
    "condition_of",
    "document_header",
    "finite",
    "kindex_kind_of",
    "publish_stamp",
    "rounded",
    "rounded_or_none",
    "sky_conditions_block",
    "targets_glossary",
    "timezone_block",
    "write_document",
]

FRAME_SCHEMA = "labmim-allsky-frame-v2"
TIMELINE_SCHEMA = "labmim-allsky-timeline-v1"
MODEL_SCHEMA = "labmim-allsky-model-v1"

VERSION_STAMP_FORMAT = "%Y%m%dT%H%M%SZ"
NAIVE_LOCAL_FORMAT = "%Y-%m-%dT%H:%M:%S"
DAY_STAMP_FORMAT = "%Y-%m-%dT00:00:00"
IRRADIANCE_DECIMALS = 2
INDEX_DECIMALS = 4
ELEVATION_DECIMALS = 2
SHARE_DECIMALS = 3
#: What a watch record that names no ``kindex_kind`` is taken to serve; every
#: checkpoint of the served family fits k*, and the fallback is logged wherever
#: it is applied.
DEFAULT_KINDEX_KIND = "kstar"

#: The page reads a condition by ``id`` (``i``..``iv``) or by ``condition``
#: (1..4), never by the exporter's 0-based class index.
CONDITION_IDS = ("i", "ii", "iii", "iv")

REFERENCES: dict[str, dict[str, str]] = {
    "escobedo": {
        "short": "Escobedo et al. (2009)",
        "citation": "Escobedo, J. F.; Gomes, E. N.; Oliveira, A. P.; Soares, J. (2009). Modeling hourly and daily fractions of UV, PAR and NIR to global solar radiation under various sky conditions at Botucatu, Brazil. Applied Energy, 86(3), 299-309.",
        "url": "https://doi.org/10.1016/j.apenergy.2008.04.013",
    },
    "teramoto": {
        "short": "Teramoto & Escobedo (2012)",
        "citation": "Teramoto, É. T.; Escobedo, J. F. (2012). Análise da frequência anual das condições de céu em Botucatu, São Paulo. Revista Brasileira de Engenharia Agrícola e Ambiental, 16(9), 985-992.",
        "url": "https://doi.org/10.1590/S1415-43662012000900009",
    },
    "haurwitz": {
        "short": "Haurwitz (1945)",
        "citation": "Haurwitz, B. (1945). Insolation in relation to cloudiness and cloud density. Journal of Meteorology, 2(3), 154-166.",
        "url": "https://doi.org/10.1175/1520-0469(1945)002%3C0154:IIRTCA%3E2.0.CO;2",
    },
    "erbs": {
        "short": "Erbs et al. (1982)",
        "citation": "Erbs, D. G.; Klein, S. A.; Duffie, J. A. (1982). Estimation of the diffuse radiation fraction for hourly, daily and monthly-average global radiation. Solar Energy, 28(4), 293-302.",
        "url": "https://doi.org/10.1016/0038-092X(82)90302-4",
    },
    "dinov3": {
        "short": "Siméoni et al. (2025)",
        "citation": "Siméoni, O. et al. (2025). DINOv3. arXiv:2508.10104.",
        "url": "https://arxiv.org/abs/2508.10104",
    },
    "cmixup": {
        "short": "Yao et al. (2022)",
        "citation": "Yao, H.; Wang, Y.; Zhang, L.; Zou, J.; Finn, C. (2022). C-Mixup: Improving Generalization in Regression. Advances in Neural Information Processing Systems 35.",
        "url": "https://arxiv.org/abs/2210.05775",
    },
}

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

    Both derive from one aware UTC instant, so no document can carry a version
    from one publish and a generation time from another.
    """

    at_utc: dt.datetime

    @classmethod
    def at(cls, now_utc: dt.datetime) -> PublishStamp:
        """The stamp of a publish happening at the aware UTC instant *now_utc*."""
        if now_utc.tzinfo is None or now_utc.utcoffset() != dt.timedelta(0):
            raise ValueError("the publish stamp is taken in UTC; pass an aware UTC datetime")
        return cls(at_utc=now_utc)

    @property
    def version(self) -> str:
        """The publish instant as the ``version`` string."""
        return self.at_utc.strftime(VERSION_STAMP_FORMAT)

    @property
    def generated_utc(self) -> str:
        """The publish instant as the ``generated_utc`` string; the same text as ``version``."""
        return self.version


def publish_stamp(now_utc: dt.datetime | None = None) -> PublishStamp:
    """The stamp of a publish happening at *now_utc* (default: now)."""
    return PublishStamp.at(now_utc or dt.datetime.now(tz=dt.UTC))


def timezone_block() -> dict[str, Any]:
    """The ``timezone`` block every document repeats for its naive local stamps."""
    return {"name": SITE_TZ_NAME, "utc_offset_hours": STATION_UTC_OFFSET_HOURS}


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
        "reference": SKY_CLASS_REFERENCE,
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


def class_share(labels: np.ndarray) -> dict[str, float | None]:
    """Share of each published condition among the valid labels of *labels* ``(N,)`` int64."""
    valid = labels[(labels >= 0) & (labels < SKY_CLASS_COUNT)]
    if valid.size == 0:
        return dict.fromkeys(CONDITION_IDS)
    counts = np.bincount(valid, minlength=SKY_CLASS_COUNT)
    return {
        condition_of(index)["id"]: rounded_or_none(counts[index] / valid.size, SHARE_DECIMALS)
        for index in range(SKY_CLASS_COUNT)
    }


def kindex_kind_of(payload: Mapping[str, Any]) -> str | None:
    """The ``kindex_kind`` a watch record's model block names, or ``None`` when none does.

    Read from the entries of ``models`` and then from ``model``, on the record
    itself and then on its ``block_model``, so a single-member record, an
    ensemble record and a block record closed by a block model all answer the
    same way.
    """
    candidates: list[Any] = []
    for holder in (payload, payload.get("block_model") or {}):
        models = holder.get("models")
        if isinstance(models, list):
            candidates.extend(models)
        candidates.append(holder.get("model"))
    for candidate in candidates:
        if isinstance(candidate, Mapping) and candidate.get("kindex_kind"):
            return str(candidate["kindex_kind"])
    return None


def write_document(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write one document in the site encoding, atomically."""
    return write_json(path, payload)
