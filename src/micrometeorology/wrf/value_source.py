"""One per-variable dispatcher shared by every WRF publication path.

Both the values JSON (:mod:`micrometeorology.wrf.jobs`) and the map figures
(:mod:`micrometeorology.cli.render_wrf_maps`) take a variable's 2-D frame reader
and colour-scale bounds from here, so the two products cannot disagree about
which frames exist or how they are scaled.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

from micrometeorology.common.types import WRFVariable
from micrometeorology.wrf import variables
from micrometeorology.wrf.reader import WRFDataset

# Solar radiation is only published for the local hours the sun is up; every
# other variable publishes all 24. Both products gate SWDOWN on this window.
FIRST_DAYLIGHT_HOUR = 6
LAST_DAYLIGHT_HOUR = 18

# Variables that are identically zero (or undefined) after dark. Longwave and
# net-radiation terms stay meaningful all night, so they are NOT in here.
# Membership is by variable id; both products gate on this single definition.
DAYLIGHT_ONLY_VARIABLES: frozenset[str] = frozenset(
    {
        WRFVariable.SWDOWN,
        WRFVariable.SWUP,
        WRFVariable.SWNET,
        WRFVariable.CLEARNESS_INDEX,
    }
)


# Derived surface-radiation fields, each mapped to its extractor and the wrfout
# fields it reads. All share one shape — eager whole-file 3-D array plus
# percentile scale bounds, sliced per step — so no bespoke branch is needed.
# Formulas, validation against WRF's own LWUPB/SWUPB and limitations live on the
# extractors in :mod:`micrometeorology.wrf.variables`.
DERIVED_RADIATION_SOURCES: dict[
    str, tuple[Callable[[WRFDataset], tuple[NDArray, float, float]], tuple[str, ...]]
] = {
    WRFVariable.LWUP: (variables.extract_upwelling_longwave, ("EMISS", "TSK", "GLW")),
    WRFVariable.SWUP: (variables.extract_upwelling_shortwave, ("ALBEDO", "SWDOWN")),
    WRFVariable.SWNET: (variables.extract_net_shortwave, ("ALBEDO", "SWDOWN")),
    WRFVariable.LWNET: (variables.extract_net_longwave, ("EMISS", "TSK", "GLW")),
    WRFVariable.RNET: (
        variables.extract_net_radiation,
        ("SWDOWN", "ALBEDO", "EMISS", "TSK", "GLW"),
    ),
    WRFVariable.SKY_EMISSIVITY: (variables.extract_sky_emissivity, ("GLW", "T2")),
    WRFVariable.CLEARNESS_INDEX: (variables.extract_clearness_index, ("SWDOWN", "COSZEN")),
}

# The wrfout fields each hand-written branch of build_value_frame_source reads.
# ``get_variable`` raises ``KeyError`` rather than returning ``None``, so the
# presence check has to happen before the branch runs. Keep in step with the
# ``ds.get_variable`` calls in :mod:`micrometeorology.wrf.variables`.
BESPOKE_VARIABLE_INPUTS: dict[str, tuple[str, ...]] = {
    WRFVariable.TEMPERATURE: ("T2",),
    WRFVariable.SKIN_TEMPERATURE: ("TSK",),
    WRFVariable.PRESSURE: ("PSFC",),
    WRFVariable.VAPOR: ("Q2",),
    WRFVariable.RELATIVE_HUMIDITY: ("Q2", "T2", "PSFC"),
    WRFVariable.RAIN: ("RAINC", "RAINNC"),
    WRFVariable.WIND: ("U10", "V10"),
    WRFVariable.WIND_POWER_DENSITY_10M: ("U10", "V10", "T2", "PSFC", "Q2"),
}


@dataclass(frozen=True, slots=True)
class ValueFrameSource:
    """Values and color-scale bounds for one exported WRF variable.

    Attributes
    ----------
    frame_for_step:
        Maps a time-step index to that step's ``(ny, nx)`` frame, in the
        variable's published unit.
    scale_min, scale_max:
        Colour-scale bounds in the same unit, shared by the map figures and the
        values JSON so the two products cannot render one field on two scales.
    vector_for_step:
        Set only when the frame is a vector magnitude, mapping a time-step index
        to that step's ``(u, v)`` components: the figure path draws them as a
        quiver over the frame, and re-deriving them would materialize every step
        twice.
    """

    frame_for_step: Callable[[int], NDArray]
    scale_min: float
    scale_max: float
    vector_for_step: Callable[[int], tuple[NDArray, NDArray]] | None = None


def is_daylight_step(step_metadata: Mapping[str, Any]) -> bool:
    """Whether a time step's local hour falls inside the daylight window.

    Only :data:`DAYLIGHT_ONLY_VARIABLES` drop their night steps, so callers gate
    on the variable too, not on this alone.
    """
    local_datetime: datetime = step_metadata["datetime_local"]
    return FIRST_DAYLIGHT_HOUR <= local_datetime.hour <= LAST_DAYLIGHT_HOUR


def publishes_step(variable_name: str, step_metadata: Mapping[str, Any]) -> bool:
    """Whether *variable_name* publishes the given time step at all.

    The one place the daylight rule is applied, so the two products can never
    disagree about which frames exist.
    """
    if variable_name not in DAYLIGHT_ONLY_VARIABLES:
        return True
    return is_daylight_step(step_metadata)


def build_value_frame_source(
    dataset: WRFDataset,
    variable_name: str,
) -> ValueFrameSource | None:
    """Build a frame reader and color-scale bounds for one exported variable.

    Parameters
    ----------
    dataset:
        Open wrfout the frames are read from.
    variable_name:
        Exported variable id, as the CLI and the site name it — not the NetCDF
        field name, which several of these are derived from rather than read.

    Returns
    -------
    ValueFrameSource | None
        ``None`` when the file carries none of the fields the variable needs,
        which every caller reports as a skipped variable rather than failing the
        run. The bounds are computed eagerly over the whole file; the frames
        themselves are sliced per step.
    """
    # ``WRFDataset.get_variable`` raises ``KeyError`` on an absent field, so a
    # wrfout missing one input would abort the whole run instead of skipping one
    # variable. Check the declared inputs before dispatching.
    required = BESPOKE_VARIABLE_INPUTS.get(variable_name)
    if required and not all(dataset.has_variable(field) for field in required):
        return None

    if variable_name in (WRFVariable.TEMPERATURE, WRFVariable.SKIN_TEMPERATURE):
        # Both arrive in Kelvin and share extract_temperature_step's conversion;
        # only the field they read differs.
        extract_kelvin = (
            variables.extract_temperature
            if variable_name == WRFVariable.TEMPERATURE
            else variables.extract_skin_temperature
        )
        kelvin, scale_min, scale_max = extract_kelvin(dataset)
        return ValueFrameSource(
            frame_for_step=lambda time_step_index: variables.materialize_2d(
                variables.extract_temperature_step(
                    kelvin[time_step_index : time_step_index + 1, :, :]
                )
            ),
            scale_min=scale_min,
            scale_max=scale_max,
        )
    if variable_name == WRFVariable.RAIN:
        cumulative_rain, scale_min, scale_max = variables.extract_rain(dataset)
        return ValueFrameSource(
            frame_for_step=lambda time_step_index: variables.materialize_2d(
                variables.extract_rain_step(cumulative_rain, time_step_index)
            ),
            scale_min=scale_min,
            scale_max=scale_max,
        )
    if variable_name == WRFVariable.WIND:
        u10_values, v10_values, scale_min, scale_max = variables.extract_wind(dataset)

        def wind_components_for_step(time_step_index: int) -> tuple[NDArray, NDArray]:
            return (
                variables.materialize_2d(u10_values[time_step_index : time_step_index + 1]),
                variables.materialize_2d(v10_values[time_step_index : time_step_index + 1]),
            )

        def wind_speed_for_step(time_step_index: int) -> NDArray:
            u10_step, v10_step = wind_components_for_step(time_step_index)
            wind_speed: NDArray = np.hypot(u10_step, v10_step)
            return wind_speed

        return ValueFrameSource(
            frame_for_step=wind_speed_for_step,
            scale_min=scale_min,
            scale_max=scale_max,
            vector_for_step=wind_components_for_step,
        )
    if variable_name == WRFVariable.PRESSURE:
        scalar_values, scale_min, scale_max = variables.extract_pressure(dataset)
    elif variable_name == WRFVariable.VAPOR:
        scalar_values, scale_min, scale_max = variables.extract_vapor(dataset)
    elif variable_name == WRFVariable.RELATIVE_HUMIDITY:
        scalar_values, scale_min, scale_max = variables.extract_relative_humidity(dataset)
    elif variable_name == WRFVariable.WIND_POWER_DENSITY_10M:
        scalar_values, scale_min, scale_max = variables.extract_wind_power_density_10m(dataset)
    elif variable_name in DERIVED_RADIATION_SOURCES:
        extractor, required_fields = DERIVED_RADIATION_SOURCES[variable_name]
        # Checked up front so a missing input is reported as the ordinary
        # "variable not found — skipped" case, not a KeyError from the extractor.
        if not all(dataset.has_variable(field) for field in required_fields):
            return None
        scalar_values, scale_min, scale_max = extractor(dataset)
    else:
        netcdf_variable_name = variable_name.upper()
        if not dataset.has_variable(netcdf_variable_name):
            return None
        scalar_values, scale_min, scale_max = variables.extract_scalar(
            dataset, netcdf_variable_name
        )
    return ValueFrameSource(
        frame_for_step=lambda time_step_index: variables.materialize_2d(
            scalar_values[time_step_index : time_step_index + 1, :, :]
        ),
        scale_min=scale_min,
        scale_max=scale_max,
    )
