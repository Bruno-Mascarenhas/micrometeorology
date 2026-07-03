"""All-sky camera + radiation-sensor fusion.

Pairs one-day all-sky timelapse videos (one frame per minute) with Campbell
radiation-sensor records, and trains a multi-task DNN that classifies cloud
condition (clear / partial / overcast) and predicts diffuse radiation from
the sky image plus sensor features.

Until a shaded pyranometer is installed, diffuse targets are Erbs-decomposition
pseudo-targets derived from global horizontal irradiance — every dataset row
carries its ``target_source`` so real measurements can replace them later.
"""

__version__ = "0.1.0"
