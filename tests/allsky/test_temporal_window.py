"""Tests for feeding the image model a window of frames instead of one.

Cloud motion is what drives the diffuse component's variability, and a single
frame cannot show it. The window costs one backbone forward per frame, so what
has to hold is that the cost buys something correct: the pooled window must equal
the masked mean of the frames encoded individually, and a row whose window is
short must not have its padding averaged in as if it were sky.
"""

from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from allsky.data.datasets import MultimodalImageDataset, resolve_time_windows
from allsky.data.manifest import build_manifest
from allsky.features import resolve_feature_set
from allsky.modeling.visual_encoder import ImageEncoder
from labmim_core import solar
from labmim_core.site import SiteConfig

FRAME_PX = 16
PATCH = 8


def _manifest(tmp_path: Path, periods: int = 12) -> tuple[pd.DataFrame, Path]:
    site = SiteConfig()
    times = pd.date_range("2025-03-21 09:00", periods=periods, freq="1min")
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
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


class _Stub(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.dim = dim
        self.patch_embed: Any = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, dim, kernel_size=PATCH, stride=PATCH)
        self.blocks = nn.ModuleList(nn.Linear(dim, dim) for _ in range(2))
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, frame: Any) -> Any:
        hidden = self.pool(self.patch_embed.proj(frame)).flatten(1)
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


class TestWindowResolution:
    def test_a_window_never_crosses_a_day_boundary(self, tmp_path: Path):
        """The night gap between the last frame of one day and the first of the
        next is not a neighbourhood."""
        first, _root = _manifest(tmp_path / "a", periods=4)
        second, _ = _manifest(tmp_path / "b", periods=4)
        second["day_id"] = "other-day"
        joined = pd.concat([first, second], ignore_index=True)

        windows = resolve_time_windows(joined, 60.0)

        days = joined["day_id"].to_numpy()
        assert all(len(set(days[members])) == 1 for members in windows)

    def test_the_cap_keeps_the_ends_and_thins_the_middle(self, tmp_path: Path):
        manifest, _ = _manifest(tmp_path, periods=12)

        uncapped = resolve_time_windows(manifest, 12.0)
        capped = resolve_time_windows(manifest, 12.0, max_frames=3)

        widest = max(range(len(uncapped)), key=lambda i: len(uncapped[i]))
        assert len(capped[widest]) <= 3
        assert capped[widest][0] == uncapped[widest][0]
        assert capped[widest][-1] == uncapped[widest][-1]

    def test_a_row_is_always_in_its_own_window(self, tmp_path: Path):
        manifest, _ = _manifest(tmp_path)

        windows = resolve_time_windows(manifest, 4.0)

        assert all(row in members for row, members in enumerate(windows))


class TestWindowedImageDataset:
    def test_the_centre_strategy_still_serves_one_frame(self, tmp_path: Path):
        manifest, root = _manifest(tmp_path)
        dataset = MultimodalImageDataset(
            manifest, resolve_feature_set("bare"), data_root=root, image_size=FRAME_PX
        )

        item = dataset[0]

        assert item["image"].shape == (3, FRAME_PX, FRAME_PX)
        assert "image_seq" not in item

    def test_a_windowed_strategy_serves_a_padded_stack_and_its_mask(self, tmp_path: Path):
        manifest, root = _manifest(tmp_path)
        dataset = MultimodalImageDataset(
            manifest,
            resolve_feature_set("bare"),
            data_root=root,
            image_size=FRAME_PX,
            window="mean_embedding",
            window_minutes=6.0,
            window_max_frames=4,
        )

        item = dataset[len(dataset) // 2]

        assert item["image_seq"].shape == (4, 3, FRAME_PX, FRAME_PX)
        assert item["frame_mask"].shape == (4,)
        assert bool(item["frame_mask"][0])

    def test_the_padding_slots_are_marked_false(self, tmp_path: Path):
        """The first row of a day has no earlier neighbour, so its window is
        short and the empty slots must not read as sky."""
        manifest, root = _manifest(tmp_path)
        dataset = MultimodalImageDataset(
            manifest,
            resolve_feature_set("bare"),
            data_root=root,
            image_size=FRAME_PX,
            window="mean_embedding",
            window_minutes=2.0,
            window_max_frames=5,
        )

        item = dataset[0]
        mask = item["frame_mask"]

        assert not bool(mask.all())
        assert torch.equal(item["image_seq"][~mask], torch.zeros_like(item["image_seq"][~mask]))

    def test_a_window_of_zero_frames_is_refused(self, tmp_path: Path):
        manifest, root = _manifest(tmp_path)

        with pytest.raises(ValueError, match="window_max_frames must be at least 1"):
            MultimodalImageDataset(
                manifest,
                resolve_feature_set("bare"),
                data_root=root,
                image_size=FRAME_PX,
                window="mean_embedding",
                window_max_frames=0,
            )


class TestWindowedEncoder:
    def test_the_pooled_window_equals_the_mean_of_its_valid_frames(self):
        """The whole point of folding the window into the batch is speed, not a
        different answer."""
        encoder = ImageEncoder(_Stub()).eval()
        torch.manual_seed(0)
        window = torch.randn(3, 4, 3, 32, 32)
        mask = torch.tensor(
            [[True, True, True, True], [True, True, False, False], [True, False, False, False]]
        )

        with torch.no_grad():
            pooled = encoder({"image_seq": window, "frame_mask": mask})
            expected = torch.stack(
                [
                    torch.cat(
                        [encoder({"image": window[b, t : t + 1]}) for t in range(4) if mask[b, t]]
                    ).mean(dim=0)
                    for b in range(3)
                ]
            )

        assert torch.allclose(pooled, expected, atol=1e-6)

    def test_padding_does_not_leak_into_the_pooled_embedding(self):
        """A row with one real frame must equal that frame encoded alone,
        whatever the padded slots contain."""
        encoder = ImageEncoder(_Stub()).eval()
        torch.manual_seed(1)
        window = torch.randn(1, 4, 3, 32, 32)
        mask = torch.tensor([[True, False, False, False]])

        with torch.no_grad():
            pooled = encoder({"image_seq": window, "frame_mask": mask})
            alone = encoder({"image": window[0, 0:1]})

        assert torch.allclose(pooled, alone, atol=1e-6)
