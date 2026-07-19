"""Feature policy: the anti-leakage contract for the multimodal stack.

Diffuse (DHI) is estimated from a sky image plus *non-radiometric* sensor
context.  The station's broadband radiometers are exactly the quantities the
targets are derived from (GHI drives ``kt``/``k*``; the diffuse pyranometer is
the label itself), so admitting them as features would let the model read the
answer off the inputs.  This module pins three tiers:

- :data:`SAFE_FEATURES` — solar geometry + standard-met channels, the default
  feature set; no radiometry.
- :data:`EXTENDED_FEATURES` — auxiliary radiometric channels (UV, PAR,
  longwave) admissible only for explicit ablation studies, never by default.
- :data:`FORBIDDEN_FEATURES` — the GHI/diffuse/net broadband channels and the
  derived target names; requesting any of them raises
  :class:`ForbiddenFeatureError`.

Each entry maps an *engineered* feature name to its source logger column, or
``None`` when the feature is computed from timestamps/geometry rather than read
from a column (see :mod:`allsky.features.engineering`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

__all__ = [
    "EXTENDED_FEATURES",
    "FEATURE_GROUPS",
    "FORBIDDEN_FEATURES",
    "SAFE_FEATURES",
    "FeatureSet",
    "ForbiddenFeatureError",
    "active_feature_groups",
    "resolve_feature_set",
    "source_column",
    "validate_features",
]

#: The two named feature tiers a config may request.
FeatureSet = Literal["safe", "extended"]

#: Engineered feature name -> source logger column (``None`` = computed from
#: timestamps/solar geometry).  Insertion order is the canonical feature order.
SAFE_FEATURES: Mapping[str, str | None] = {
    "solar_elevation": None,
    "solar_zenith": None,
    "azimuth_sin": None,
    "azimuth_cos": None,
    "doy_sin": None,
    "doy_cos": None,
    "air_temp_c": "AirT1_C_Avg",
    "dew_point_c": "DP1_C_Avg",
    "rel_humidity": "RH1",
    "pressure_mbar": "BP1_mbar_Avg",
    "wind_speed_ms": "WS_ms",
    "wind_dir_sin": "WindDir",
    "wind_dir_cos": "WindDir",
}

#: Auxiliary radiometric channels — ablation only, never in the default set.
EXTENDED_FEATURES: Mapping[str, str | None] = {
    "uv_wm2": "CUV5_Wm2_Avg",
    "par_wm2": "PAR_Wm2_Avg",
    "longwave_up_wm2": "CG3Dn_Wm2Cr_Avg",
    "longwave_down_wm2": "CG3Up_Wm2Cr_Avg",
}

#: Broadband radiometry and derived-target columns that must never be features.
#: Any column beginning with ``target_`` is forbidden as well (see
#: :func:`validate_features`), as is any configured target column.
FORBIDDEN_FEATURES: frozenset[str] = frozenset(
    {
        "CM3Up_Wm2_Avg",  # GHI — kt/k* are derived from it
        "CM3Dn_Wm2_Avg",  # reflected shortwave
        "Net_Wm2_Avg",  # net radiation
        "PSP_Wm2_Avg",  # live diffuse pyranometer — the label itself
        "CMP21_Wm2_Avg",  # (dead channel) diffuse pyranometer
        "CMP21_Avg",  # raw mV diffuse channel
        "PSP_Avg",  # raw mV diffuse channel
        "kt",  # clearness index target
        "kstar",  # clear-sky index target
        "dhi",  # diffuse target
        "diffuse",  # diffuse target (legacy column name)
    }
)

#: Sensor-token groups for cross-attention fusion.  ``radiometry_aux`` is only
#: populated when the extended set is enabled (see :func:`active_feature_groups`).
FEATURE_GROUPS: Mapping[str, list[str]] = {
    "solar": [
        "solar_elevation",
        "solar_zenith",
        "azimuth_sin",
        "azimuth_cos",
        "doy_sin",
        "doy_cos",
    ],
    "temperature": ["air_temp_c", "dew_point_c"],
    "humidity": ["rel_humidity"],
    "pressure": ["pressure_mbar"],
    "wind": ["wind_speed_ms", "wind_dir_sin", "wind_dir_cos"],
    "radiometry_aux": list(EXTENDED_FEATURES),
}


class ForbiddenFeatureError(ValueError):
    """Raised when a leakage-prone column is requested as a model feature.

    Subclasses :class:`ValueError` so existing ``except ValueError`` config
    handlers still catch it.  The offending name is available as
    :attr:`feature` and is always quoted in the message.
    """

    def __init__(self, feature: str, *, reason: str = "radiometric/target leakage") -> None:
        self.feature = feature
        super().__init__(
            f"forbidden feature {feature!r} ({reason}): it encodes the quantity the "
            "target is derived from and must not be used as a model input"
        )


def source_column(name: str) -> str | None:
    """Source logger column for an engineered feature name.

    Returns ``None`` for geometry/timestamp-derived features (``solar_*``,
    ``azimuth_*``, ``doy_*``).  Raises :class:`KeyError` for unknown names.
    """
    if name in SAFE_FEATURES:
        return SAFE_FEATURES[name]
    if name in EXTENDED_FEATURES:
        return EXTENDED_FEATURES[name]
    raise KeyError(f"unknown engineered feature {name!r}")


def resolve_feature_set(name: FeatureSet | str, extra: Iterable[str] = ()) -> list[str]:
    """Resolve a feature-set name to its ordered engineered-feature list.

    ``"safe"`` returns the safe features; ``"extended"`` returns the safe
    features followed by the auxiliary radiometric ones.  *extra* names are
    appended verbatim (deduplicated, order preserved) for bespoke ablations.
    The result preserves policy declaration order, which is the canonical
    feature-column order for the whole stack.

    Raises
    ------
    ValueError
        If *name* is neither ``"safe"`` nor ``"extended"``.
    """
    if name == "safe":
        resolved = list(SAFE_FEATURES)
    elif name == "extended":
        resolved = [*SAFE_FEATURES, *EXTENDED_FEATURES]
    else:
        raise ValueError(f"unknown feature set {name!r}; expected 'safe' or 'extended'")
    for feature in extra:
        if feature not in resolved:
            resolved.append(feature)
    return resolved


def active_feature_groups(name: FeatureSet | str) -> dict[str, list[str]]:
    """Sensor-token groups active for a feature set.

    Drops the ``radiometry_aux`` group for the safe set (nothing populates it),
    so the union of the returned groups is exactly
    :func:`resolve_feature_set` for that set.
    """
    groups = {group: list(members) for group, members in FEATURE_GROUPS.items()}
    if name != "extended":
        groups.pop("radiometry_aux", None)
    return groups


def validate_features(names: Iterable[str], *, target_columns: Iterable[str] = ()) -> None:
    """Assert that no requested feature is leakage-prone.

    A name fails if it is in :data:`FORBIDDEN_FEATURES`, is one of the
    configured *target_columns*, or begins with ``target_``.

    Raises
    ------
    ForbiddenFeatureError
        Naming the first offending feature.
    """
    forbidden = FORBIDDEN_FEATURES | set(target_columns)
    for name in names:
        if name in forbidden:
            reason = (
                "configured target column"
                if name in set(target_columns)
                else "radiometric/target leakage"
            )
            raise ForbiddenFeatureError(name, reason=reason)
        if name.startswith("target_"):
            raise ForbiddenFeatureError(name, reason="target column")
