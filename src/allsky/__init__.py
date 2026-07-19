"""All-sky camera + radiation-sensor fusion (multimodal v2 stack).

Pairs one-day all-sky timelapse videos (one frame per minute) with Campbell
radiation-sensor records into a portable v2 dataset (manifest + frames +
precomputed visual embeddings), then trains multimodal experiments that predict
diffuse horizontal irradiance (and optionally a clear-sky index and a
clear / partially-cloudy / overcast sky class) from the sky image plus
engineered sensor features.

Until a shaded pyranometer is installed, diffuse targets are Erbs-decomposition
pseudo-targets derived from global horizontal irradiance — every dataset row
carries its ``target_source`` so real measurements can replace them later.
"""

__version__ = "0.1.0"
