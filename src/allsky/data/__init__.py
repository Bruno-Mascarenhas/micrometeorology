"""Data layer for the multimodal all-sky stack (manifest v2).

Public surface:

- :mod:`~allsky.data.contracts` — the manifest column registry, :class:`QCFlag`,
  sky-class constants and portable-path helpers.
- :mod:`~allsky.data.alignment` — image<->sensor alignment strategies
  (:class:`CenterFrame` at build time; windowed poolers at dataset level).
- :mod:`~allsky.data.manifest` — :func:`build_manifest` +
  :func:`write_manifest_parquet`.
- :mod:`~allsky.data.validation` — :func:`validate_manifest` /
  :class:`ValidationReport`.
- :mod:`~allsky.data.splits` — :func:`create_day_splits` and the persisted
  split artifact.
- :mod:`~allsky.data.datasets` — the torch datasets (torch imported lazily;
  importing this package never pulls torch).
"""
