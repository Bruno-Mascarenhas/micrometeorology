"""Physical constants and elementwise conversions shared across the packages.

Declared once here for the same reason the station's coordinates are: a constant
that lives in two modules is two things that can drift, and a value corrected in
one copy and not the other changes a published quantity in one pipeline while
leaving the other alone, with nothing failing.

The solar constant is not redeclared here — :mod:`allsky.solar` owns it, and
this package already reads the rest of its solar geometry from there.
"""

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "KELVIN_AT_ZERO_CELSIUS",
    "MOLAR_MASS_RATIO",
    "PASCAL_PER_HECTOPASCAL",
    "STEFAN_BOLTZMANN",
    "saturation_vapor_pressure",
    "vapor_pressure",
]

#: Stefan-Boltzmann constant, W m-2 K-4 (CODATA 2018).
#:
#: Exact by definition since the 2019 SI redefinition: sigma = 2 pi^5 k^4 /
#: (15 h^3 c^2), and k, h and c are all exact, so CODATA publishes it with zero
#: uncertainty. The literal is that exact value truncated to ten significant
#: figures, which is 3.3e-11 relative — far below any measurement it enters.
STEFAN_BOLTZMANN = 5.670374419e-8

#: Ratio of the molar masses of water vapour and dry air, M_w / M_d.
#:
#: 18.015 / 28.964 from the CIPM-2007 composition of dry air; the value is the
#: conventional 0.622 every formulation below is written against.
MOLAR_MASS_RATIO = 0.622

# Bolton (1980), eq. 10: saturation vapour pressure over liquid water from
# temperature in Celsius. Accurate to 0.3% over -35..35 C.
_BOLTON_E0_PA = 611.2
_BOLTON_A = 17.67
_BOLTON_B = 243.5

#: Zero Celsius on the Kelvin scale.
KELVIN_AT_ZERO_CELSIUS = 273.15

#: Pascals in one hectopascal.
PASCAL_PER_HECTOPASCAL = 100.0

#: What these conversions accept: they are elementwise, and the callers range
#: from a whole time series to one cell of a row being repaired.
type Elementwise = NDArray | float


def vapor_pressure(mixing_ratio: Elementwise, pressure: Elementwise) -> Elementwise:
    """Water-vapour partial pressure from a MIXING ratio and total pressure.

    ``e = w*p / (epsilon + w)``. WRF's ``Q2`` is ``QV at 2 M``, a mixing ratio,
    which is why the denominator is not the ``epsilon + (1-epsilon)*w`` of the
    specific-humidity form -- applying that one to a mixing ratio is the defect
    :func:`~micrometeorology.wrf.operational_record.migrate_to_v2` repairs
    across the whole operational record.

    Parameters
    ----------
    mixing_ratio:
        ``(N,)`` kg/kg, or a scalar.
    pressure:
        ``(N,)`` total pressure, or a scalar, broadcastable against
        *mixing_ratio*.

    Returns
    -------
    Elementwise
        Partial pressure in the unit *pressure* was given in: the conversion is
        linear in it, so hPa in gives hPa out and Pa in gives Pa out.
    """
    partial: Elementwise = mixing_ratio * pressure / (MOLAR_MASS_RATIO + mixing_ratio)
    return partial


def saturation_vapor_pressure(temperature_c: Elementwise) -> Elementwise:
    """Saturation vapour pressure over liquid water, Pa, from °C.

    Bolton (1980), eq. 10.

    Parameters
    ----------
    temperature_c:
        ``(N,)`` degrees Celsius, or a scalar.

    Returns
    -------
    Elementwise
        ``(N,)`` Pa.
    """
    saturation: Elementwise = _BOLTON_E0_PA * np.exp(
        _BOLTON_A * temperature_c / (temperature_c + _BOLTON_B)
    )
    return saturation
