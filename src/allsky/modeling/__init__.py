"""Multimodal model zoo for the all-sky DHI stack.

.. rubric:: Torch-free import contract

Every submodule here (except :mod:`~allsky.modeling.contracts`) defines
``nn.Module`` subclasses and therefore imports torch at module scope.  To keep
``import allsky`` — and ``import allsky.modeling`` itself — torch-free, this
package exposes its public names through :pep:`562` lazy ``__getattr__``: torch
is pulled only when a torch-bearing name is first accessed.
:mod:`~allsky.modeling.contracts` (``ModelOutputs``, ``MultimodalModel``,
``group_slices``) is torch-free and safe to touch eagerly.

Public surface:

- :class:`~allsky.modeling.contracts.ModelOutputs`,
  :class:`~allsky.modeling.contracts.MultimodalModel`,
  :func:`~allsky.modeling.contracts.group_slices`.
- :class:`~allsky.modeling.sensor_encoder.SensorEncoder`,
  :class:`~allsky.modeling.visual_encoder.PrecomputedEmbedding` /
  :class:`~allsky.modeling.visual_encoder.ImageEncoder`.
- :class:`~allsky.modeling.fusion.ConcatFusion` /
  :class:`~allsky.modeling.fusion.FiLMFusion` /
  :class:`~allsky.modeling.fusion.CrossAttentionFusion`.
- :class:`~allsky.modeling.heads.Trunk` / :class:`~allsky.modeling.heads.Heads`.
- :class:`~allsky.modeling.baselines.ClimatologyModel` /
  :class:`~allsky.modeling.baselines.SensorOnlyModel` /
  :class:`~allsky.modeling.baselines.ImageOnlyModel`.
- :class:`~allsky.modeling.multimodal.MultimodalNet`.
- :data:`~allsky.modeling.registry.MODEL_BUILDERS` /
  :func:`~allsky.modeling.registry.build_model`.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "MODEL_BUILDERS",
    "ClimatologyModel",
    "ConcatFusion",
    "CrossAttentionFusion",
    "FiLMFusion",
    "Heads",
    "ImageEncoder",
    "ImageOnlyModel",
    "ModelOutputs",
    "MultimodalModel",
    "MultimodalNet",
    "PrecomputedEmbedding",
    "SensorEncoder",
    "SensorOnlyModel",
    "Trunk",
    "build_fusion",
    "build_model",
    "build_visual_encoder",
    "group_slices",
]

#: Public name -> defining submodule, resolved lazily to keep imports torch-free.
_EXPORTS = {
    "ModelOutputs": "allsky.modeling.contracts",
    "MultimodalModel": "allsky.modeling.contracts",
    "group_slices": "allsky.modeling.contracts",
    "SensorEncoder": "allsky.modeling.sensor_encoder",
    "PrecomputedEmbedding": "allsky.modeling.visual_encoder",
    "ImageEncoder": "allsky.modeling.visual_encoder",
    "build_visual_encoder": "allsky.modeling.visual_encoder",
    "ConcatFusion": "allsky.modeling.fusion",
    "FiLMFusion": "allsky.modeling.fusion",
    "CrossAttentionFusion": "allsky.modeling.fusion",
    "build_fusion": "allsky.modeling.fusion",
    "Trunk": "allsky.modeling.heads",
    "Heads": "allsky.modeling.heads",
    "ClimatologyModel": "allsky.modeling.baselines",
    "SensorOnlyModel": "allsky.modeling.baselines",
    "ImageOnlyModel": "allsky.modeling.baselines",
    "MultimodalNet": "allsky.modeling.multimodal",
    "MODEL_BUILDERS": "allsky.modeling.registry",
    "build_model": "allsky.modeling.registry",
}


def __getattr__(name: str) -> Any:
    """Lazily import a public name from its submodule (PEP 562)."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted(__all__)
