"""Configuration models for the all-sky pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pathlib import Path


class VideoConfig(BaseModel):
    """How all-sky videos map to wall-clock time.

    The camera produces one-day timelapse files named by date; each frame
    covers ``minutes_per_frame`` minutes of real time starting at
    ``start_time`` local time.
    """

    pattern: str = "data/all-sky/allsky-*.mp4"
    filename_date_format: str = "allsky-%Y%m%d"
    start_time: str = "06:00"
    minutes_per_frame: float = 1.0


class SensorConfig(BaseModel):
    """Radiation-sensor sources and column selection."""

    paths: list[str] = Field(default_factory=lambda: ["data/LBM_lenta_2025.dat"])
    ghi_column: str = "CM3Up_Wm2_Avg"
    # CMP21 is the station's primary diffuse pyranometer, but its W/m2 channel
    # is currently zero-filled by the logger program (only the raw CMP21_Avg mV
    # channel is live) — PSP is the working diffuse measurement. Switch to
    # "CMP21_Wm2_Avg" once the CR5000 program conversion is fixed; None falls
    # back to Erbs pseudo-targets (target_source="erbs_pseudo").
    diffuse_column: str | None = "PSP_Wm2_Avg"
    feature_columns: list[str] = Field(
        default_factory=lambda: [
            "CM3Up_Wm2_Avg",
            "CG3Up_Wm2_Avg",
            "CM3Dn_Wm2_Avg",
            "Net_Wm2_Avg",
            "CUV5_Wm2_Avg",
            "PAR_Wm2_Avg",
        ]
    )
    tolerance_minutes: float = 5.0


class SiteConfig(BaseModel):
    """Observation site (LabMiM/UFBA, Salvador-BA by default)."""

    latitude: float = -13.00
    longitude: float = -38.51


class LabelConfig(BaseModel):
    """Weak cloud-condition labels from the clearness index kt."""

    kt_clear: float = 0.65
    kt_overcast: float = 0.35
    min_solar_elevation_deg: float = 10.0
    # QC guard: kt above this is a sensor artifact (GHI spikes far beyond
    # clear-sky), not weather — such rows are dropped from the dataset.
    max_kt: float = 1.2


class ModelConfig(BaseModel):
    """SkyFusionNet architecture parameters."""

    image_size: int = 224
    backbone: str = "small"  # "small" (built-in conv net) or "resnet18"
    embed_dim: int = 128
    hidden_dim: int = 256
    n_classes: int = 3


class TrainConfig(BaseModel):
    """Training run parameters."""

    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    device: str = "auto"  # auto -> cuda | mps | cpu
    # Automatic mixed precision on CUDA (Colab T4/L4/A100): ~2x throughput.
    amp: bool = True
    out_dir: str = "output/allsky"
    seed: int = 42
    cls_loss_weight: float = 1.0
    reg_loss_weight: float = 1.0


class AllSkyConfig(BaseModel):
    """Root configuration for the all-sky pipeline."""

    video: VideoConfig = Field(default_factory=VideoConfig)
    sensor: SensorConfig = Field(default_factory=SensorConfig)
    site: SiteConfig = Field(default_factory=SiteConfig)
    labels: LabelConfig = Field(default_factory=LabelConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)


def load_config(path: str | Path | None = None) -> AllSkyConfig:
    """Load configuration from YAML, falling back to defaults when *path* is None."""
    if path is None:
        return AllSkyConfig()
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return AllSkyConfig.model_validate(raw)
