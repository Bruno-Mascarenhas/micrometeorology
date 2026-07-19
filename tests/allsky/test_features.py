"""Tests for the allsky.features package (policy, engineering, normalization)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from allsky.config import SiteConfig
from allsky.features import (
    EXTENDED_FEATURES,
    SAFE_FEATURES,
    FeatureNormalizer,
    ForbiddenFeatureError,
    TargetNormalizer,
    active_feature_groups,
    build_feature_frame,
    fit_target_normalizers,
    resolve_feature_set,
    validate_features,
)


@pytest.fixture
def site() -> SiteConfig:
    """Default site: LabMiM/UFBA, Salvador-BA (lat -13.00, lon -38.51)."""
    return SiteConfig()


@pytest.fixture
def sensor_frame() -> pd.DataFrame:
    """Synthetic daytime met frame carrying every safe logger column."""
    index = pd.date_range("2025-03-21 08:00", periods=12, freq="30min")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "AirT1_C_Avg": rng.uniform(20.0, 30.0, len(index)),
            "DP1_C_Avg": rng.uniform(10.0, 20.0, len(index)),
            "RH1": rng.uniform(50.0, 90.0, len(index)),
            "BP1_mbar_Avg": rng.uniform(1005.0, 1015.0, len(index)),
            "WS_ms": rng.uniform(0.0, 8.0, len(index)),
            "WindDir": rng.uniform(0.0, 360.0, len(index)),
        },
        index=index,
    )


# ---------------------------------------------------------------------------
# policy.py
# ---------------------------------------------------------------------------


class TestFeaturePolicy:
    def test_resolve_safe_is_safe_features_in_order(self):
        assert resolve_feature_set("safe") == list(SAFE_FEATURES)

    def test_resolve_extended_appends_radiometry(self):
        resolved = resolve_feature_set("extended")
        assert resolved == [*SAFE_FEATURES, *EXTENDED_FEATURES]

    def test_extended_never_in_safe(self):
        safe = set(resolve_feature_set("safe"))
        assert not (safe & set(EXTENDED_FEATURES))

    def test_resolve_unknown_set_raises(self):
        with pytest.raises(ValueError, match="unknown feature set"):
            resolve_feature_set("radiometry")

    def test_resolve_extra_appended_deduplicated(self):
        resolved = resolve_feature_set("safe", extra=["air_temp_c", "custom_x"])
        # air_temp_c already present -> not duplicated; custom_x appended last.
        assert resolved[-1] == "custom_x"
        assert resolved.count("air_temp_c") == 1

    @pytest.mark.parametrize("offender", ["CM3Up_Wm2_Avg", "PSP_Wm2_Avg", "kt", "kstar", "dhi"])
    def test_forbidden_feature_raises_naming_it(self, offender: str):
        with pytest.raises(ForbiddenFeatureError) as exc:
            validate_features(["air_temp_c", offender])
        assert offender in str(exc.value)
        assert exc.value.feature == offender

    def test_target_prefix_is_forbidden(self):
        with pytest.raises(ForbiddenFeatureError, match="target_dhi"):
            validate_features(["target_dhi"])

    def test_configured_target_column_forbidden(self):
        with pytest.raises(ForbiddenFeatureError, match="my_target"):
            validate_features(["my_target"], target_columns=["my_target"])

    def test_safe_features_pass_validation(self):
        validate_features(resolve_feature_set("safe"))  # must not raise

    def test_groups_cover_exactly_resolved_safe(self):
        groups = active_feature_groups("safe")
        covered = [f for members in groups.values() for f in members]
        assert covered == resolve_feature_set("safe")
        assert "radiometry_aux" not in groups

    def test_groups_cover_exactly_resolved_extended(self):
        groups = active_feature_groups("extended")
        covered = {f for members in groups.values() for f in members}
        assert covered == set(resolve_feature_set("extended"))
        assert "radiometry_aux" in groups


# ---------------------------------------------------------------------------
# engineering.py
# ---------------------------------------------------------------------------


class TestFeatureEngineering:
    def test_deterministic_column_order(self, sensor_frame: pd.DataFrame, site: SiteConfig):
        frame = build_feature_frame(
            sensor_frame, pd.DatetimeIndex(sensor_frame.index), site, "safe"
        )
        assert list(frame.columns) == resolve_feature_set("safe")
        assert len(frame) == len(sensor_frame)
        assert frame.index.equals(sensor_frame.index)

    def test_extended_includes_radiometry_columns(self, site: SiteConfig):
        index = pd.date_range("2025-03-21 09:00", periods=3, freq="1h")
        extended_frame = pd.DataFrame(
            dict.fromkeys(EXTENDED_FEATURES.values(), 1.0)
            | {col: 1.0 for col in SAFE_FEATURES.values() if col is not None},
            index=index,
        )
        frame = build_feature_frame(extended_frame, index, site, "extended")
        assert list(frame.columns) == resolve_feature_set("extended")

    def test_missing_column_raises(self, sensor_frame: pd.DataFrame, site: SiteConfig):
        with pytest.raises(KeyError, match="WindDir"):
            build_feature_frame(
                sensor_frame.drop(columns=["WindDir"]),
                pd.DatetimeIndex(sensor_frame.index),
                site,
            )

    def test_geometry_columns_match_solar_module(
        self, sensor_frame: pd.DataFrame, site: SiteConfig
    ):
        from allsky import solar

        index = pd.DatetimeIndex(sensor_frame.index)
        frame = build_feature_frame(sensor_frame, index, site, "safe")
        elevation = solar.solar_elevation(index, site)
        azimuth = np.deg2rad(solar.solar_azimuth(index, site))
        np.testing.assert_allclose(frame["solar_elevation"].to_numpy(), elevation)
        np.testing.assert_allclose(frame["solar_zenith"].to_numpy(), 90.0 - elevation)
        np.testing.assert_allclose(frame["azimuth_sin"].to_numpy(), np.sin(azimuth))
        np.testing.assert_allclose(frame["azimuth_cos"].to_numpy(), np.cos(azimuth))
        # sin^2 + cos^2 == 1 for every cyclic pair.
        np.testing.assert_allclose(
            frame["azimuth_sin"] ** 2 + frame["azimuth_cos"] ** 2, 1.0, atol=1e-12
        )

    def test_wind_direction_cyclic_continuity(self, site: SiteConfig):
        index = pd.date_range("2025-03-21 09:00", periods=2, freq="1h")
        base = {col: 1.0 for col in SAFE_FEATURES.values() if col is not None}
        near_zero = pd.DataFrame(base | {"WindDir": [1.0, 359.0]}, index=index)
        frame = build_feature_frame(near_zero, index, site, "safe")
        # 1 deg and 359 deg straddle the wrap point -> nearly identical encoding.
        assert abs(frame["wind_dir_sin"].iloc[0] - frame["wind_dir_sin"].iloc[1]) < 0.05
        assert abs(frame["wind_dir_cos"].iloc[0] - frame["wind_dir_cos"].iloc[1]) < 0.05
        np.testing.assert_allclose(
            frame["wind_dir_sin"] ** 2 + frame["wind_dir_cos"] ** 2, 1.0, atol=1e-12
        )

    def test_doy_encoding_periodic_and_continuous(self, site: SiteConfig):
        # Same calendar day one year apart -> identical encoding (uses dayofyear).
        idx = pd.DatetimeIndex(["2025-06-01 12:00", "2026-06-01 12:00"])
        base = {col: 1.0 for col in SAFE_FEATURES.values() if col is not None}
        frame = build_feature_frame(pd.DataFrame(base, index=idx), idx, site, "safe")
        np.testing.assert_allclose(
            frame["doy_sin"].to_numpy()[0], frame["doy_sin"].to_numpy()[1], atol=1e-6
        )
        np.testing.assert_allclose(
            frame["doy_cos"].to_numpy()[0], frame["doy_cos"].to_numpy()[1], atol=1e-6
        )
        # Year boundary is continuous: Dec 31 ~ Jan 1.
        edge = pd.DatetimeIndex(["2025-01-01 12:00", "2025-12-31 12:00"])
        edge_frame = build_feature_frame(pd.DataFrame(base, index=edge), edge, site, "safe")
        assert abs(edge_frame["doy_sin"].iloc[0] - edge_frame["doy_sin"].iloc[1]) < 0.05
        assert abs(edge_frame["doy_cos"].iloc[0] - edge_frame["doy_cos"].iloc[1]) < 0.05


# ---------------------------------------------------------------------------
# normalization.py
# ---------------------------------------------------------------------------


class TestFeatureNormalizer:
    def test_fit_transform_zero_mean_unit_std(self, sensor_frame: pd.DataFrame, site: SiteConfig):
        frame = build_feature_frame(
            sensor_frame, pd.DatetimeIndex(sensor_frame.index), site, "safe"
        )
        norm = FeatureNormalizer.fit(frame)
        out = norm.transform(frame)
        assert out.shape == (len(frame), len(frame.columns))
        assert out.dtype == np.float32
        # Columns with real input variance standardize to ~0 mean / ~1 std
        # (float32 standardization of a ~1010 mbar column leaves a ~1e-4 mean
        # residual).  doy_sin/cos are constant within a single day -> clamped.
        raw_std = frame.to_numpy(dtype=np.float64).std(axis=0)
        varying = raw_std > 1e-3
        np.testing.assert_allclose(out.mean(axis=0)[varying], 0.0, atol=1e-4)
        np.testing.assert_allclose(out.std(axis=0)[varying], 1.0, atol=1e-4)

    def test_constant_column_clamped(self):
        frame = pd.DataFrame({"a": [5.0, 5.0, 5.0], "b": [1.0, 2.0, 3.0]})
        norm = FeatureNormalizer.fit(frame)
        out = norm.transform(frame)
        # Constant column -> std clamped to 1, output all zeros (no divide blow-up).
        np.testing.assert_allclose(out[:, 0], 0.0)
        assert np.isfinite(out).all()

    def test_json_roundtrip(self, sensor_frame: pd.DataFrame, site: SiteConfig):
        import json

        frame = build_feature_frame(
            sensor_frame, pd.DatetimeIndex(sensor_frame.index), site, "safe"
        )
        norm = FeatureNormalizer.fit(frame)
        restored = FeatureNormalizer.from_dict(json.loads(json.dumps(norm.to_dict())))
        assert restored.columns == norm.columns
        np.testing.assert_allclose(restored.mean, norm.mean)
        np.testing.assert_allclose(restored.std, norm.std)
        np.testing.assert_allclose(restored.transform(frame), norm.transform(frame))

    def test_transform_uses_fit_column_order(self):
        frame = pd.DataFrame({"a": [1.0, 2.0], "b": [10.0, 20.0]})
        norm = FeatureNormalizer.fit(frame, columns=["a", "b"])
        # Reordered input columns -> transform still reads them in fit order.
        reordered = frame[["b", "a"]]
        np.testing.assert_allclose(norm.transform(reordered), norm.transform(frame))


class TestTargetNormalizer:
    def test_normalize_denormalize_roundtrip(self):
        values = np.array([100.0, 200.0, 300.0, 400.0])
        norm = TargetNormalizer.fit(values)
        np.testing.assert_allclose(norm.denormalize(norm.normalize(values)), values, atol=1e-9)

    def test_fit_ignores_nan(self):
        norm = TargetNormalizer.fit(np.array([100.0, np.nan, 300.0]))
        assert norm.mean == pytest.approx(200.0)
        assert np.isfinite(norm.std)

    def test_json_roundtrip(self):
        import json

        norm = TargetNormalizer.fit(np.array([1.0, 2.0, 3.0]))
        restored = TargetNormalizer.from_dict(json.loads(json.dumps(norm.to_dict())))
        assert restored.mean == pytest.approx(norm.mean)
        assert restored.std == pytest.approx(norm.std)

    def test_fit_target_normalizers_per_column(self):
        frame = pd.DataFrame({"target_dhi": [10.0, 20.0, 30.0], "target_kindex": [0.1, 0.5, 0.9]})
        norms = fit_target_normalizers(frame, ["target_dhi", "target_kindex"])
        assert set(norms) == {"target_dhi", "target_kindex"}
        assert norms["target_dhi"].mean == pytest.approx(20.0)
