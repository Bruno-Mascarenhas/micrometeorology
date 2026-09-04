"""Primitives the three packages share, and that none of them owns.

``allsky``, ``micrometeorology`` and ``solrad_correction`` used to import each
other for these: solar geometry, the station's coordinates, the atomic writer,
the sky partition, the seeding helper. The import graph closed into a cycle —
``micrometeorology`` -> ``allsky`` -> ``solrad_correction`` ->
``micrometeorology`` — which is why moving a constant between two of them could
not be done without touching the third.

Nothing here imports any of the three. That is the whole rule of this package,
and it is what makes the graph a DAG: the three depend on this, this depends on
nobody.

``__version__`` is ``0+unknown`` when the package is imported from the source
tree before installation.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("labmim-micrometeorology")
except PackageNotFoundError:
    __version__ = "0+unknown"
