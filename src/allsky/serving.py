"""The serving pin: which checkpoints the sky page is served from, declared once.

The best network is declared, never discovered. A versioned YAML names the
checkpoints, their SHA-256, the roles their heads play in the ensemble, the
evaluation reports the model card is built from and the sentence that says why
they were chosen. ``allsky watch --serving`` and ``allsky publish-site`` both
read it, so the process scoring the live frame and the document describing the
model cannot name different weights.

Paths in the pin are relative to the working directory, as every other config
under ``configs/allsky/`` resolves its ``output/...`` paths.

Torch-free: the pin is validated and the checkpoints are fingerprinted without
loading them.
"""

import datetime as dt
import hashlib
import logging
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError, model_validator

from labmim_core.atomic import JsonObjectError, atomic_write_strict_json, read_json_object

logger = logging.getLogger(__name__)

__all__ = [
    "Control",
    "Controls",
    "HeadRoles",
    "Member",
    "PinVerificationError",
    "PinnedCheckpoint",
    "RoleSelector",
    "Selection",
    "ServingConfig",
    "ServingConfigError",
    "ServingReports",
    "load_serving_config",
    "sha256_of_file",
    "verify_pinned_checkpoints",
]


class RoleSelector(StrEnum):
    """Which members of an ensemble a head group is read from, by checkpoint stem.

    ``best`` reads the ``best.ckpt`` members, ``last`` the ``last.ckpt`` ones
    and ``all`` every member, whatever its stem.
    """

    best = "best"
    last = "last"
    all = "all"


class HeadRoles(BaseModel):
    """Whose sky heads and whose regression heads an ensemble averages."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sky: RoleSelector = RoleSelector.all
    dhi: RoleSelector = RoleSelector.all


#: Every member's heads averaged, the default of both ensembles.
ALL_ROLES = HeadRoles()

AttributionTarget = Literal["kindex", "dhi"]

SHA256_HEX_PATTERN = r"^[0-9a-f]{64}$"
SHA256_CACHE_FILENAME = "checkpoint-sha256.json"


class PinVerificationError(ValueError):
    """A pinned checkpoint is missing or its bytes are not the ones the pin names."""


class ServingConfigError(ValueError):
    """The pin file cannot be read as a serving pin."""


def _expanded(path: Path) -> Path:
    """``~`` expanded, so a tracked pin names the operator's home without hard-coding it."""
    return Path(path).expanduser()


ExpandedPath = Annotated[Path, AfterValidator(_expanded)]


class PinnedCheckpoint(BaseModel):
    """One served checkpoint and the fingerprint of the file it must be."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: ExpandedPath
    sha256: str = Field(pattern=SHA256_HEX_PATTERN)


class Member(PinnedCheckpoint):
    """A served frame checkpoint and the ``eval-test`` report the card reads it from."""

    report: ExpandedPath


class Control(BaseModel):
    """A control trained on the served split: its test report and the checkpoint served live."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoint: PinnedCheckpoint
    report: ExpandedPath


class Controls(BaseModel):
    """The two controls trained on the served manifest, split and targets.

    Attributes
    ----------
    sensor_only:
        The scalars-only network; its checkpoint answers the live "no image"
        counterfactual.
    climatology:
        The train-mean control; its checkpoint holds the pinned means.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    sensor_only: Control
    climatology: Control


class ServingReports(BaseModel):
    """Where the model card reads its numbers from.

    Attributes
    ----------
    dataset:
        The prepared dataset directory (``manifest.parquet``, its sidecar and
        ``splits.json``) the served checkpoints trained on; verified against
        their ``manifest_sha256`` and ``split_id`` before it feeds the card.
    training_history:
        The ``metrics.csv`` of the member whose training curve the card draws.
    domain_check:
        The pinned first-day check of ``allsky-operacional.md`` once it has
        been run; ``None`` publishes the field as not measured.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: ExpandedPath
    training_history: ExpandedPath
    domain_check: ExpandedPath | None = None


SelectionSplit = Literal["train", "val", "test"]


class Selection(BaseModel):
    """Why these checkpoints, in the operator's own words, and when it was decided."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    criterion: str = Field(min_length=1)
    selection_split: SelectionSplit = "val"
    decided_on: dt.date


class ServingConfig(BaseModel):
    """The pin, as ``configs/allsky/serving/<id>.yaml`` declares it.

    Attributes
    ----------
    frame_checkpoints:
        Single-frame checkpoints served together, each with its test report;
        more than one is averaged by :func:`allsky.watch.ensemble_prediction`.
    frame_sky_role, frame_dhi_role:
        Which members' heads feed the sky class and the regression outputs,
        by checkpoint stem (``best`` / ``last`` / ``all``).
    attribution_checkpoint:
        Index into ``frame_checkpoints`` of the member whose sensitivity map
        and counterfactual the publisher computes.
    min_elevation_deg:
        Solar-elevation floor the checkpoints' manifest was built with; the
        watch scores nothing below it.
    controls:
        The scalars-only and train-mean controls: their test reports and, for
        the scalars-only one, the checkpoint the live counterfactual is scored
        with.
    reports:
        The dataset directory, the served member's ``metrics.csv`` and the
        pinned domain check when there is one.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    serving: Literal[True]
    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    frame_checkpoints: list[Member] = Field(min_length=1)
    frame_sky_role: RoleSelector = RoleSelector.all
    frame_dhi_role: RoleSelector = RoleSelector.all
    attribution_checkpoint: int = Field(default=0, ge=0)
    attribution_target: AttributionTarget = "kindex"
    min_elevation_deg: float = Field(gt=0.0, lt=90.0)
    controls: Controls
    reports: ServingReports
    selection: Selection

    @model_validator(mode="after")
    def _attribution_names_a_member(self) -> ServingConfig:
        if self.attribution_checkpoint >= len(self.frame_checkpoints):
            raise ValueError(
                f"attribution_checkpoint {self.attribution_checkpoint} names no member: the pin "
                f"lists {len(self.frame_checkpoints)} frame checkpoint(s)"
            )
        return self

    @property
    def frame_roles(self) -> HeadRoles:
        """The role selectors of the frame ensemble, as :func:`allsky.watch.run_watch` takes them."""
        return HeadRoles(sky=self.frame_sky_role, dhi=self.frame_dhi_role)

    @property
    def attribution_member(self) -> Member:
        """The member the sensitivity map and the overlay counterfactual are computed from."""
        return self.frame_checkpoints[self.attribution_checkpoint]

    @property
    def verified_checkpoints(self) -> list[PinnedCheckpoint]:
        """Every checkpoint the pin fingerprints: the members, then the two controls."""
        return [
            *self.frame_checkpoints,
            self.controls.sensor_only.checkpoint,
            self.controls.climatology.checkpoint,
        ]


def load_serving_config(path: str | Path) -> ServingConfig:
    """Load and validate a serving pin.

    Raises
    ------
    ServingConfigError
        If the file cannot be read, is not a mapping, or declares anything
        the schema forbids; the message names the offending field.
    """
    try:
        raw: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ServingConfigError(f"{path}: cannot read the serving pin: {exc}") from exc
    if not isinstance(raw, dict):
        raise ServingConfigError(f"{path}: a serving pin is a mapping, got {type(raw).__name__}")
    try:
        return ServingConfig.model_validate(raw)
    except ValidationError as exc:
        fields = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        raise ServingConfigError(f"{path}: invalid serving pin — {fields}") from exc


def _file_signature(path: Path) -> list[int]:
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns]


def _read_cache(cache_file: Path) -> dict[str, Any]:
    if not cache_file.is_file():
        return {}
    try:
        return read_json_object(cache_file)
    except JsonObjectError as exc:
        logger.warning("ignoring unreadable checkpoint hash cache: %s", exc)
        return {}


def sha256_of_file(path: str | Path, *, cache_dir: str | Path | None = None) -> str:
    """SHA-256 of a file's bytes, streamed.

    With *cache_dir* the digest is remembered beside the file's size and
    modification time, so a 350 MB checkpoint verified every ten minutes is
    hashed once and re-hashed only when its bytes could have changed.
    """
    file = Path(path).resolve()
    cache_file = Path(cache_dir) / SHA256_CACHE_FILENAME if cache_dir is not None else None
    signature = _file_signature(file)
    cache = _read_cache(cache_file) if cache_file is not None else {}
    cached = cache.get(str(file))
    if isinstance(cached, dict) and cached.get("signature") == signature:
        return str(cached["sha256"])
    with file.open("rb") as handle:
        fingerprint = hashlib.file_digest(handle, "sha256").hexdigest()
    if cache_file is not None:
        cache[str(file)] = {"signature": signature, "sha256": fingerprint}
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_strict_json(cache_file, cache)
    return fingerprint


def verify_pinned_checkpoints(
    pin: ServingConfig, *, cache_dir: str | Path | None = None
) -> list[str]:
    """Check every pinned checkpoint exists and hashes to the digest the pin names.

    Returns
    -------
    list of str
        The verified digests, in :attr:`ServingConfig.verified_checkpoints`
        order: the frame members first, then the two controls.

    Raises
    ------
    PinVerificationError
        Naming the first checkpoint that is missing or whose bytes differ.
    """
    digests: list[str] = []
    for member in pin.verified_checkpoints:
        if not member.path.is_file():
            raise PinVerificationError(f"pinned checkpoint {member.path} does not exist")
        actual = sha256_of_file(member.path, cache_dir=cache_dir)
        if actual != member.sha256:
            raise PinVerificationError(
                f"{member.path} hashes to {actual}, but the pin names {member.sha256}: the file "
                "was rewritten since the pin was decided; re-pin it or restore the file"
            )
        digests.append(actual)
    return digests
