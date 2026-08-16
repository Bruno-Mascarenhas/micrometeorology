"""Physical constants shared across the micrometeorology packages.

Declared once here for the same reason the station's coordinates are: a constant
that lives in two modules is two things that can drift, and a value corrected in
one copy and not the other changes a published quantity in one pipeline while
leaving the other alone, with nothing failing.

The solar constant is not redeclared here — :mod:`allsky.solar` owns it, and
this package already reads the rest of its solar geometry from there.
"""

__all__ = ["STEFAN_BOLTZMANN"]

#: Stefan-Boltzmann constant, W m-2 K-4 (CODATA 2018).
#:
#: Exact by definition since the 2019 SI redefinition: sigma = 2 pi^5 k^4 /
#: (15 h^3 c^2), and k, h and c are all exact, so CODATA publishes it with zero
#: uncertainty. The literal is that exact value truncated to ten significant
#: figures, which is 3.3e-11 relative — far below any measurement it enters.
STEFAN_BOLTZMANN = 5.670374419e-8
