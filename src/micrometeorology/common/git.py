"""Best-effort git interrogation, shared by every package's provenance stamp.

Both :mod:`allsky.provenance` and :mod:`solrad_correction.utils.metadata` need
the same thing -- "tell me the commit, and say nothing if you cannot" -- so the
subprocess call lives here once instead of being reimplemented per package.

That matters beyond deduplication: ruff's two subprocess rules are mutually
exclusive for this call. S607 (partial executable path) demands the binary be
resolved rather than found on PATH, while S603 stays silent only when *every*
argv element is a string literal -- which a resolved path never is. Resolving is
the safer half of the trade (it also turns "git is not installed" into a plain
``None`` instead of an exec failure), so this module takes S603, documents it,
and is the only place in the tree that has to.

Pure stdlib: importing this never pulls pandas, torch or any package internals.
"""

import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

__all__ = ["run_git"]

#: Seconds before a hung git invocation is abandoned. Provenance is optional;
#: no metadata probe may stall a training run.
_TIMEOUT_SECONDS = 5.0


def run_git(args: Sequence[str], *, cwd: Path | None = None) -> str | None:
    """Run ``git *args`` and return its stripped stdout, or None if unavailable.

    Every failure mode collapses to ``None`` -- git absent, non-zero exit, a
    hang, or undecodable output -- because callers stamp provenance and must
    never fail the run they are describing. Note the distinction a caller may
    care about: a *successful* command with empty output returns ``""``, not
    ``None``, so "ran and said nothing" stays separable from "could not run".

    Parameters
    ----------
    args:
        Arguments after the executable, e.g. ``["rev-parse", "HEAD"]``.
    cwd:
        Directory to run in. ``None`` inherits the process working directory.
    """
    # Resolved up front: an absent git is the common case (a source tarball, a
    # container without the client) and means "no commit info" -- the same
    # answer the exec failure below would produce, just without the exception.
    git_executable = shutil.which("git")
    if git_executable is None:
        return None
    try:
        # S603: see the module docstring. Ruff exempts only an inline list of
        # string literals, which here would mean hardcoding an absolute path to
        # git rather than resolving the one on this machine.
        result = subprocess.run(  # noqa: S603
            [git_executable, *args],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
        )
    # A vanished or unexecutable binary (OSError), a hung call (TimeoutExpired,
    # a SubprocessError) and output that is not valid text (UnicodeDecodeError,
    # raised while decoding under ``text=True``) all mean "no commit info".
    except OSError, subprocess.SubprocessError, UnicodeDecodeError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()
