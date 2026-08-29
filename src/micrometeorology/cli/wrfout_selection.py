"""The one wrfout glob the WRF commands share, with usage errors as usage errors.

Four commands select wrfout files and all of them glob through here, so a
mistyped ``--date`` becomes the same :class:`typer.BadParameter` in every one.

Only the glob and the error translation live here. What each command does around
them stays with the command, because those tails are deliberately different: one
has a batch mode with no ``--date``, one refuses when nothing matches, one only
warns. Folding those together would change what an operator sees.

Imports typer, since ``BadParameter`` is interface vocabulary, plus the reader —
and no command, so nothing here can close a cycle back into ``cli``.
"""

from collections.abc import Sequence
from pathlib import Path

import typer

from micrometeorology.wrf.jobs import output_id_owner
from micrometeorology.wrf.reader import resolve_wrfout_paths

__all__ = ["glob_wrfout_day", "reject_output_id_variables"]


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


def reject_output_id_variables(var_list: Sequence[str]) -> None:
    """Refuse a ``-v`` token that names an OUTPUT file id instead of an input variable.

    The three WRF commands share the refusal so an operator gets the same
    message and the same exit code from all of them; which ids these are is
    :data:`~micrometeorology.wrf.jobs.OUTPUT_ID_OWNERS`, and this is only the
    translation of that fact into usage vocabulary.

    Raises
    ------
    typer.BadParameter
        On the first offending token, naming the variable to pass instead.
        These ids cannot produce correct site bytes, so they fail loudly rather
        than silently mislabelling the map.
    """
    for variable in var_list:
        owner = output_id_owner(variable)
        if owner:
            raise typer.BadParameter(
                f"{variable} is the output file id of {owner}; pass -v {owner}"
            )
