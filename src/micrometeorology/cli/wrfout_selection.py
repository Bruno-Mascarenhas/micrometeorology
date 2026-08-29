"""The one wrfout glob the WRF commands share, with usage errors as usage errors.

Four commands select wrfout files, and each wrote its own wrapper around
:func:`micrometeorology.wrf.reader.resolve_wrfout_paths` — so a mistyped
``--date`` was translated into a :class:`typer.BadParameter` four times, and
``run_wrf_pipeline`` reached into ``render_wrf_maps`` for a *private* name to
avoid writing it a fifth. A private import between sibling commands is the sign
that the shared piece has no address of its own.

Only the glob and the error translation live here. What each command does around
them stays with the command, because those tails are deliberately different: one
has a batch mode with no ``--date``, one refuses when nothing matches, one only
warns. Folding those together would change what an operator sees.

Imports typer, since ``BadParameter`` is interface vocabulary, plus the reader —
and no command, so nothing here can close a cycle back into ``cli``.
"""

from __future__ import annotations

from pathlib import Path

import typer

from micrometeorology.wrf.reader import resolve_wrfout_paths

__all__ = ["glob_wrfout_day"]


def glob_wrfout_day(wrf_dir: Path | str, date: str, domains: tuple[int, ...] = ()) -> list[Path]:
    """Glob one day of wrfout files, reporting a mistyped *date* as a usage error.

    Parameters
    ----------
    wrf_dir:
        Directory holding the wrfout files.
    date:
        Day to select, as the reader spells it.
    domains:
        Domain numbers to keep; empty means every domain present.

    Returns
    -------
    list of pathlib.Path
        Matching files, possibly empty — an empty day is not an error here, and
        each command decides whether it is one for them.

    Raises
    ------
    typer.BadParameter
        If *date* is not a date the reader can parse. It is an operator typo,
        not an internal fault, so it surfaces as a usage error rather than a
        traceback.
    """
    try:
        return resolve_wrfout_paths(wrf_dir, date, domains or None)
    except ValueError as invalid_date:
        raise typer.BadParameter(str(invalid_date)) from invalid_date
