"""Tests for fitting DHI as a ratio to the clear-sky reference.

The point of the parameterization is that the network stops spending capacity on
the deterministic solar-geometry envelope, which on this station's chronological
splits DRIFTS: the DHI-vs-elevation slope moves 20 % from train to test. What has
to hold for that to be safe is narrow and testable — the raw path must be
untouched, the ratio must invert exactly back to W/m2, and a manifest that cannot
support the reference must be refused rather than divided by something wrong.
"""

from pathlib import Path
from typing import cast

import imageio.v3 as iio
import numpy as np
import pandas as pd
import pytest
import torch

from allsky.clearsky import clearsky_diffuse
from allsky.config import DHIParameterization
from allsky.data.datasets import MultimodalImageDataset
from allsky.data.manifest import build_manifest
from allsky.features import resolve_feature_set
from labmim_core import solar
from labmim_core.site import SiteConfig

FRAME_PX = 16


def _manifest(tmp_path: Path, n: int = 6) -> tuple[pd.DataFrame, Path]:
    site = SiteConfig()
    times = pd.date_range("2025-03-21 09:00", periods=n, freq="45min")
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    rows = []
    for i, ts in enumerate(times):
        path = frames_dir / f"allsky-{ts:%Y%m%d-%H%M}.jpg"
        iio.imwrite(path, rng.integers(0, 256, (FRAME_PX, FRAME_PX, 3)).astype(np.uint8))
        rows.append({"frame_path": str(path), "timestamp": ts, "video": "v.mp4", "index": i})
    index = pd.date_range("2025-03-21 06:00", "2025-03-21 18:00", freq="5min")
    e0h = solar.extraterrestrial_ghi(index, site)
    sensor = pd.DataFrame(
        {
            "AirT1_C_Avg": 25.0,
            "DP1_C_Avg": 15.0,
            "RH1": 70.0,
            "BP1_mbar_Avg": 1010.0,
            "WS_ms": 2.0,
            "WindDir": 90.0,
            "CM3Up_Wm2_Avg": 0.7 * e0h,
            "PSP_Wm2_Avg": 0.2 * e0h,
        },
        index=index,
    )
    manifest, _ = build_manifest(pd.DataFrame(rows), sensor, site=site, data_root=tmp_path)
    return manifest, tmp_path


def _dataset(
    manifest: pd.DataFrame, root: Path, parameterization: DHIParameterization
) -> MultimodalImageDataset:
    return MultimodalImageDataset(
        manifest,
        resolve_feature_set("bare"),
        data_root=root,
        image_size=FRAME_PX,
        train=True,
        dhi_parameterization=parameterization,
    )


class TestClearSkyIndexTarget:
    def test_the_raw_path_keeps_a_scale_of_exactly_one(self, tmp_path: Path):
        manifest, root = _manifest(tmp_path)

        item = _dataset(manifest, root, "raw")[0]

        assert float(item["dhi_scale"]) == 1.0
        assert float(item["dhi"]) == pytest.approx(float(manifest["target_dhi"].iloc[0]))

    def test_the_ratio_inverts_back_to_the_measured_irradiance(self, tmp_path: Path):
        manifest, root = _manifest(tmp_path)
        dataset = _dataset(manifest, root, "clearsky_index")

        for idx in range(len(dataset)):
            item = dataset[idx]
            recovered = float(item["dhi"]) * float(item["dhi_scale"])
            assert recovered == pytest.approx(float(manifest["target_dhi"].iloc[idx]), rel=1e-6)

    def test_the_scale_is_the_reference_the_evaluator_scores_against(self, tmp_path: Path):
        manifest, root = _manifest(tmp_path)
        expected = clearsky_diffuse(
            manifest["solar_zenith"], pd.to_datetime(manifest["timestamp_utc"], utc=True)
        )

        dataset = _dataset(manifest, root, "clearsky_index")
        scales = np.array([float(dataset[i]["dhi_scale"]) for i in range(len(dataset))])

        assert scales == pytest.approx(expected.astype(np.float32), rel=1e-6)

    def test_the_ratio_is_dimensionless_and_near_one_under_a_clear_sky(self, tmp_path: Path):
        manifest, root = _manifest(tmp_path)
        dataset = _dataset(manifest, root, "clearsky_index")

        ratios = torch.stack([dataset[i]["dhi"] for i in range(len(dataset))])

        assert bool(torch.isfinite(ratios).all())
        assert 0.0 < float(ratios.min()) < 10.0

    def test_a_manifest_without_the_solar_columns_is_refused(self, tmp_path: Path):
        manifest, root = _manifest(tmp_path)

        with pytest.raises(ValueError, match="clear-sky-index DHI target needs"):
            MultimodalImageDataset(
                manifest.drop(columns=["solar_zenith"]),
                [c for c in resolve_feature_set("bare") if c != "solar_zenith"],
                data_root=root,
                image_size=FRAME_PX,
                train=True,
                dhi_parameterization="clearsky_index",
            )

    def test_a_non_positive_reference_is_refused_rather_than_divided_by(self, tmp_path: Path):
        """The night filter normally keeps the reference far from zero — 61.5 W/m2
        at the 10-degree elevation floor — so a non-positive one means the
        manifest is not what this parameterization assumes."""
        manifest, root = _manifest(tmp_path)
        night = manifest.copy()
        night["solar_zenith"] = 95.0

        with pytest.raises(ValueError, match="non-positive or non-finite"):
            _dataset(night, root, "clearsky_index")


def test_an_unknown_parameterization_is_refused_not_served_as_raw(tmp_path: Path) -> None:
    """Any spelling other than ``clearsky_index`` fell through to a scale of one, so
    a misspelt arm trained on raw W/m2 while every metric stayed self-consistent."""
    manifest, root = _manifest(tmp_path)

    with pytest.raises(ValueError, match="dhi_parameterization"):
        _dataset(manifest, root, cast(DHIParameterization, "clearsky-index"))
