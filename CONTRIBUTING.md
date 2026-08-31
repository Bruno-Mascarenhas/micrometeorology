# Contributing

This is a research codebase: it backs measurements that end up in papers and on a
public site. Traceability of a result matters more than brevity, and a number
without provenance is a defect. What follows is what that costs a contributor.

## Environment

Python **3.14+**, `uv` for dependencies, inside a conda environment:

```bash
make install-dev      # or: make install-cuda, for a GPU torch build
```

## Before you open a pull request

```bash
make fix              # ruff format + ruff check --fix
make check            # lockfile sync + lint + typecheck + full suite
```

`pre-commit` is the lint source of truth and is what CI runs — `ruff check` alone
is **not** equivalent, since the hooks also cover formatting, Markdown code blocks
and the `pyproject.toml` schema. CI additionally builds the wheel, smoke-tests it
and audits dependencies for known vulnerabilities.

The tree is kept free of checker-silencing workarounds. See
[Code standards](README.md#code-standards) in the README for the rules on
`# noqa`, `# type: ignore`, `from __future__ import annotations` and
`if TYPE_CHECKING:` — all four have measured reasons behind them.

## Commits and pull requests

- **Conventional Commits, one line**: `type: description`, imperative, no body,
  no footer, ~72 characters. Types: `feat`, `fix`, `docs`, `style`, `refactor`,
  `test`, `build`, `ci`, `chore`, `revert`.
- **One commit per logical change.** A refactor does not travel with a feature.
- A pull request says what changes, why, and how to test it.

## Tests

- The test name describes the behaviour, not the method. That is where the
  context lives that does not fit in a function name.
- One behaviour per test; arrange / act / assert visible without section comments.
- `pytest` functions, never `unittest` classes. `pytest.mark.parametrize` for
  data variation.
- **A fixed bug earns a test that fails before the fix.**
- Mock only boundaries you do not control — network, clock, filesystem.

## Scientific conventions

These are not style preferences; each one has cost time when broken.

- **Units in the name** when ambiguity is possible: `ghi_w_m2`, `theta_z_rad`,
  `temp_c`, `dt_s`. Without a unit in the name, SI is assumed.
- **Angles in radians internally.** Degrees only at the input and output
  boundary, and the name says which: `azimuth_deg` vs `azimuth_rad`.
- **Time is timezone-aware UTC internally**, with one documented exception: where
  data is stamped by an instrument's own clock (Campbell datalogger, all-sky
  camera overlay) the pipeline stays naive local end to end, and the manifest
  layer is the UTC boundary. A module using that exception declares it in its
  docstring.
- **`NaN` never passes silently.** Either an explicit mask or validation at the
  boundary. Never substitute zero — in irradiance, zero is a valid measurement.
- **No magic numbers** in the middle of a calculation. Named constant at the top
  of the module, with its source in a comment.
- **Float comparison with an explicit tolerance**, never `==`.
- Array shapes declared in the docstring in symbolic notation: `(N,)`,
  `(B, C, H, W)`, `(T, F)`.

## Machine learning

- Seed fixed and recorded in the experiment config — Python, NumPy and torch.
- **Temporal splits, always.** Shuffling an irradiance time series leaks: the
  frame at t and the one at t+10s are nearly the same sample. Split by day or by
  contiguous block, with a gap between train and test.
- Normalization statistics computed on the training split only and persisted with
  the checkpoint.
- Every metric reported against a baseline — persistence and clear-sky at
  minimum. Report the skill score, not just the absolute error.
- Irradiance metrics come as a set: RMSE, MAE and MBE together. MBE alone hides
  dispersion; RMSE alone hides systematic bias.

## Data

- **Nothing acquired is versioned**: images, `.h5`, `.nc`, acquisition `.csv`,
  datalogger `.dat`, `.pt`, `.ckpt`, `.npy`. Only minimal test fixtures, and even
  those stay small.
- **Raw data is immutable.** A transformation writes a new artifact at a new
  path; it never overwrites an acquisition.
- Data paths come from config or environment, never an absolute machine path in
  the code.
- Each instrument format has its own parser module, with the format specification
  cited in its docstring.

## Documentation

Everything under `docs/` is in **English**, including file names. A document that
records a measurement keeps the number, its uncertainty and where it came from —
that is the point of the file. Notebooks in `notebooks/` are for exploration; as
soon as a snippet is used twice it becomes a function in an importable module.
