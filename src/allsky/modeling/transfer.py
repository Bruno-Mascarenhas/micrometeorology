"""Starting a run from weights another run learned, on other data.

This is not ``--resume``: only the **weights** move, and the fresh run brings
its own optimizer, schedule and normalizers. Nothing is skipped without being
counted and named, and a mismatch inside the backbone is an error rather than a
line in a report.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from torch import nn

__all__ = [
    "TransferMismatchError",
    "TransferReport",
    "load_transferable_weights",
]

logger = logging.getLogger(__name__)

#: Parameter-name fragment marking a module that is fresh by construction and
#: must never block a transfer: the geometry adapter is zero-initialised for the
#: channels of THIS experiment, and a source with a different channel count has
#: nothing meaningful to give it.
FRESH_BY_CONSTRUCTION = ("extra_proj",)

#: Prefix of the parameters that describe the ARCHITECTURE rather than the task.
#: A shape mismatch here is a different network, and loading around it would
#: leave most of the backbone at its initialisation while the log said
#: "transferred".
ARCHITECTURE_PREFIX = "visual_encoder.backbone."


class TransferMismatchError(ValueError):
    """The source checkpoint does not describe the same backbone as the model."""


@dataclass(frozen=True, slots=True)
class TransferReport:
    """What a transfer actually moved, tensor by tensor.

    Attributes
    ----------
    loaded:
        Parameter names copied from the source.
    missing:
        Model parameters the source did not carry, left at their initialisation.
    unexpected:
        Source parameters the model has no slot for.
    reshaped:
        Names whose shapes differ, mapped to ``(model_shape, source_shape)``;
        these are left at their initialisation. Only names outside
        :data:`ARCHITECTURE_PREFIX`, or fresh by construction, can appear here —
        anything else raises instead.
    """

    loaded: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    unexpected: tuple[str, ...] = ()
    reshaped: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = field(default_factory=dict)

    @property
    def moved_anything(self) -> bool:
        """Whether any tensor was actually copied."""
        return bool(self.loaded)

    def describe(self) -> str:
        """One-line human summary, for the run log."""
        return (
            f"transferred {len(self.loaded)} tensor(s); "
            f"{len(self.missing)} left at initialisation, "
            f"{len(self.reshaped)} shape-mismatched, "
            f"{len(self.unexpected)} unused from the source"
        )


def load_transferable_weights(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    trust_pickle: bool = False,
) -> TransferReport:
    """Copy what *checkpoint_path* can give *model*, and say exactly what moved.

    Parameters
    ----------
    model:
        The freshly built model, already shaped by THIS experiment's config.
    checkpoint_path:
        A checkpoint written by :func:`allsky.training.checkpointing.save_checkpoint`.
    trust_pickle:
        Forwarded to the checkpoint reader; leave ``False`` for a file that
        travelled through Colab or a shared drive.

    Returns
    -------
    TransferReport
        What was copied and what was not. The report is also logged at INFO, and
        anything skipped is logged by name at WARNING, because a transfer that
        moved less than the operator believes is indistinguishable from one that
        worked until the metrics come back.

    Raises
    ------
    TransferMismatchError
        If a backbone parameter exists on both sides with different shapes, or if
        any backbone parameter went unfilled. A model with no backbone at all —
        ``climatology``, ``sensor_only`` — has nothing for this to check, and a
        transfer that moved nothing there is reported in the log rather than
        raised.

    Notes
    -----
    Normalizers are deliberately NOT read from the source. ``TargetNormalizer``
    and ``FeatureNormalizer`` are fitted on this station's training split, and
    adopting another dataset's statistics would standardise our targets by a
    distribution they were never drawn from.
    """
    from allsky.training.checkpointing import load_checkpoint

    payload = load_checkpoint(checkpoint_path, map_location="cpu", trust_pickle=trust_pickle)
    source: dict[str, Any] = payload.get("model_state", {})
    if not source:
        raise TransferMismatchError(
            f"{checkpoint_path} carries no 'model_state'; it is not a checkpoint this project wrote"
        )

    # state_dict() hands back detached views that share storage with the
    # parameters, so copying into them updates the model; no_grad keeps that true
    # for any caller that asked for graph-carrying vars.
    import torch

    target = model.state_dict()
    loaded: list[str] = []
    reshaped: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    hard_mismatch: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    filled: set[int] = set()

    for name, tensor in target.items():
        source_name = _source_name(name, source)
        if source_name is None:
            continue
        incoming = source[source_name]
        if tuple(incoming.shape) != tuple(tensor.shape):
            shapes = (tuple(tensor.shape), tuple(incoming.shape))
            if _is_architecture(name):
                hard_mismatch[name] = shapes
            else:
                reshaped[name] = shapes
            continue
        with torch.no_grad():
            tensor.copy_(incoming)
        loaded.append(name)
        filled.add(tensor.data_ptr())

    if hard_mismatch:
        raise TransferMismatchError(
            f"{checkpoint_path} describes a different backbone: "
            + ", ".join(f"{n} is {m} here and {s} there" for n, (m, s) in hard_mismatch.items())
            + ". Transferring around this would leave most of the backbone at its "
            "initialisation while the log said it transferred"
        )

    # A tensor registered under two names — the geometry adapter is both a child
    # of the backbone and an attribute of the encoder — is filled once and is not
    # missing under its second name.
    missing = tuple(name for name, tensor in target.items() if tensor.data_ptr() not in filled)
    report = TransferReport(
        loaded=tuple(loaded),
        missing=missing,
        unexpected=tuple(n for n in source if n not in target),
        reshaped=reshaped,
    )
    _refuse_a_half_transferred_backbone(checkpoint_path, missing, target)

    logger.info("transfer from %s: %s", checkpoint_path, report.describe())
    for name in report.missing:
        logger.warning("transfer: %s had no source and keeps its initialisation", name)
    for name, (mine, theirs) in report.reshaped.items():
        logger.warning(
            "transfer: %s is %s here and %s in the source, so it keeps its initialisation",
            name,
            mine,
            theirs,
        )
    return report


def _source_name(name: str, source: dict[str, Any]) -> str | None:
    """The source key holding *name*'s weights, or ``None``.

    Usually the same name. The exception is a convolution the geometry adapter
    wrapped: the model calls it ``...proj.pretrained.weight`` while a source that
    trained without extra channels calls it ``...proj.weight``. Without this
    alias a transfer into a geometry arm would silently leave the patch
    embedding — the one layer every pixel passes through — at its initialisation.
    """
    if name in source:
        return name
    unwrapped = name.replace(".pretrained.", ".")
    return unwrapped if unwrapped != name and unwrapped in source else None


def _refuse_a_half_transferred_backbone(
    checkpoint_path: str | Path, missing: tuple[str, ...], target: dict[str, Any]
) -> None:
    """Raise unless the whole backbone arrived.

    Transferring a backbone means the backbone arrives whole. A source that
    shares only the head and the trunk — a ResNet run pointed at a ViT
    checkpoint, say — matches a handful of tensors and leaves the network
    essentially untrained, which is indistinguishable from a working transfer
    until the metrics come back days later.

    Raises
    ------
    TransferMismatchError
        If the model has architecture parameters and any of them went unfilled.
    """
    architecture = [name for name in target if _is_architecture(name)]
    absent = [name for name in missing if _is_architecture(name)]
    if not architecture or not absent:
        return
    shown = ", ".join(absent[:4]) + (f", and {len(absent) - 4} more" if len(absent) > 4 else "")
    raise TransferMismatchError(
        f"{checkpoint_path} filled only {len(architecture) - len(absent)} of "
        f"{len(architecture)} backbone tensors; {shown} found no source. A backbone "
        "transfers whole or not at all — a partial one trains a nearly-random network "
        "while the log says it transferred"
    )


def _is_architecture(name: str) -> bool:
    """Whether *name* describes the backbone rather than the task head."""
    if any(fragment in name for fragment in FRESH_BY_CONSTRUCTION):
        return False
    return name.startswith(ARCHITECTURE_PREFIX)
