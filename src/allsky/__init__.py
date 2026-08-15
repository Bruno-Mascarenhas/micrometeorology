"""All-sky camera + radiation-sensor fusion (multimodal v2 stack).

Pairs one-day all-sky timelapse videos (one frame per minute) with Campbell
radiation-sensor records into a portable v2 dataset (manifest + frames +
precomputed visual embeddings), then trains multimodal experiments that predict
diffuse horizontal irradiance (and optionally a clear-sky index and one of the
four Escobedo sky conditions) from the sky image plus engineered sensor
features.

Diffuse targets come from a measured pyranometer column by default
(:attr:`allsky.config.PrepareTargetsConfig.diffuse_column`, ``PSP_Wm2_Avg``).
Setting that column to null falls back to Erbs-decomposition pseudo-targets
derived from global horizontal irradiance; every dataset row carries its
``target_source`` (``measured`` or ``erbs_pseudo``) so the two are never mixed
up in a metrics table.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("labmim-micrometeorology")
except PackageNotFoundError:  # Support direct source-tree imports before installation.
    __version__ = "0+unknown"
