"""One per-variable dispatcher shared by every WRF publication path.

Both published products — the values JSON written by
:mod:`micrometeorology.wrf.jobs` and the map figures built by
:mod:`micrometeorology.cli.render_wrf_maps` — need the same things for a
variable: how to read one time step's 2-D frame, and the colour-scale bounds
that frame is drawn against. Declaring a variable here reaches both, so the two
products cannot disagree about which frames exist or how they are scaled.
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

# Every variable that is identically zero (or undefined) after dark, so the
# night steps carry no information worth publishing. Longwave and net-radiation
# terms are NOT in here: they stay meaningful all night, which is exactly when
# the surface is losing energy. Membership is by variable id, and this is the
# single definition both products gate on.
DAYLIGHT_ONLY_VARIABLES: frozenset[str] = frozenset(
    {
        WRFVariable.SWDOWN,
        WRFVariable.SWUP,
        WRFVariable.SWNET,
        WRFVariable.CLEARNESS_INDEX,
    }
)


# The derived surface-radiation fields, each mapped to its extractor and the
# wrfout fields that extractor reads. They share one shape — an eager
# whole-file 3-D array plus percentile scale bounds, sliced per step — so they
# need no bespoke branch beyond declaring their inputs. Formulas, validation
# against WRF's own LWUPB/SWUPB, and limitations live on the extractors in
# :mod:`micrometeorology.wrf.variables`.
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
# Declared here because the extractors call ``get_variable``, which raises
# ``KeyError`` rather than returning ``None``, so the presence check has to
# happen before the branch runs. Keep in step with the ``ds.get_variable`` calls
# in :mod:`micrometeorology.wrf.variables`.
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
    """Values and color-scale bounds for one exported WRF variable."""

    frame_for_step: Callable[[int], NDArray]
    scale_min: float
    scale_max: float
    # Set only for variables whose frame is the magnitude of a vector field.
    # The figure path draws the components as a quiver over that magnitude;
    # re-deriving them from the dataset would materialize every step twice.
    vector_for_step: Callable[[int], tuple[NDArray, NDArray]] | None = None


def is_daylight_step(step_metadata: Mapping[str, Any]) -> bool:
    """Whether a time step's local hour falls inside the daylight window.

    Callers gate on the variable, not on this alone: only the variables in
    :data:`DAYLIGHT_ONLY_VARIABLES` drop their night steps.
    """
    local_datetime: datetime = step_metadata["datetime_local"]
    return FIRST_DAYLIGHT_HOUR <= local_datetime.hour <= LAST_DAYLIGHT_HOUR


def publishes_step(variable_name: str, step_metadata: Mapping[str, Any]) -> bool:
    """Whether *variable_name* publishes the given time step at all.

    The one place the daylight rule is applied, so the values JSON and the map
    figures can never disagree about which frames exist.
    """
    if variable_name not in DAYLIGHT_ONLY_VARIABLES:
        return True
    return is_daylight_step(step_metadata)


def build_value_frame_source(
    dataset: WRFDataset,
    variable_name: str,
) -> ValueFrameSource | None:
    """Build a frame reader and color-scale bounds for one exported variable.

    Returns ``None`` when the variable is absent from the file, which every
    caller reports as a skipped variable rather than failing the run.
    """
    # The bespoke branches reach ``WRFDataset.get_variable``, a bare
    # ``variables[name]`` lookup that raises ``KeyError`` on an absent field, so
    # a wrfout missing one input would abort the whole run instead of skipping
    # one variable. Check the declared inputs before dispatching.
    required = BESPOKE_VARIABLE_INPUTS.get(variable_name)
    if required and not all(dataset.has_variable(field) for field in required):
        return None

    if variable_name == WRFVariable.TEMPERATURE:
        temperature_kelvin, scale_min, scale_max = variables.extract_temperature(dataset)
        return ValueFrameSource(
            frame_for_step=lambda time_step_index: variables.materialize_2d(
                variables.extract_temperature_step(
                    temperature_kelvin[time_step_index : time_step_index + 1, :, :]
                )
            ),
            scale_min=scale_min,
            scale_max=scale_max,
        )
    if variable_name == WRFVariable.SKIN_TEMPERATURE:
        skin_temperature_kelvin, scale_min, scale_max = variables.extract_skin_temperature(dataset)
        return ValueFrameSource(
            frame_for_step=lambda time_step_index: variables.materialize_2d(
                variables.extract_temperature_step(
                    skin_temperature_kelvin[time_step_index : time_step_index + 1, :, :]
                )
            ),
            scale_min=scale_min,
            scale_max=scale_max,
        )
    if variable_name == WRFVariable.RELATIVE_HUMIDITY:
        relative_humidity, scale_min, scale_max = variables.extract_relative_humidity(dataset)
        return ValueFrameSource(
            frame_for_step=lambda time_step_index: variables.materialize_2d(
                relative_humidity[time_step_index : time_step_index + 1, :, :]
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
    if variable_name == WRFVariable.WIND_POWER_DENSITY_10M:
        power_density, scale_min, scale_max = variables.extract_wind_power_density_10m(dataset)
        return ValueFrameSource(
            frame_for_step=lambda time_step_index: variables.materialize_2d(
                power_density[time_step_index : time_step_index + 1, :, :]
            ),
            scale_min=scale_min,
            scale_max=scale_max,
        )
    if variable_name == WRFVariable.PRESSURE:
        scalar_values, scale_min, scale_max = variables.extract_pressure(dataset)
    elif variable_name == WRFVariable.VAPOR:
        scalar_values, scale_min, scale_max = variables.extract_vapor(dataset)
    elif variable_name in DERIVED_RADIATION_SOURCES:
        extractor, required_fields = DERIVED_RADIATION_SOURCES[variable_name]
        # Checked up front so a wrfout missing one input is reported as the
        # ordinary "variable not found — skipped" case instead of raising
        # KeyError from inside the extractor.
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
