"""Uploads all-sky artifacts to Google Drive through the ``rclone`` binary.

Why rclone rather than the Drive REST API, and the one-time setup, are in
``docs/allsky-archive.md``.
"""

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "DriveTarget",
    "RcloneError",
    "RcloneUploader",
]

DEFAULT_TIMEOUT = 3600.0


class RcloneError(RuntimeError):
    """An ``rclone`` invocation failed, or the binary is missing."""


@dataclass(frozen=True)
class DriveTarget:
    """An rclone remote plus the folder prefix uploads land under."""

    remote: str
    root: str = "LabMiM/allsky"

    def path(self, *parts: str) -> str:
        """Join *parts* under the root into an ``remote:folder/sub`` rclone spec."""
        segments = [segment.strip("/") for segment in (self.root, *parts) if segment.strip("/")]
        return f"{self.remote}:{'/'.join(segments)}"


@dataclass
class RcloneUploader:
    """Idempotent wrapper over ``rclone``, one process per file or per directory.

    Every transfer runs with ``--checksum``, so re-uploading an unchanged file
    moves no bytes and a partial sync can simply be rerun. With *dry_run* set,
    rclone reports what it would do and transfers nothing.
    """

    target: DriveTarget
    binary: str = "rclone"
    extra_args: tuple[str, ...] = ()
    dry_run: bool = False
    timeout: float = DEFAULT_TIMEOUT
    _resolved_binary: str | None = field(default=None, init=False, repr=False)

    def executable(self) -> str:
        """Absolute path to the rclone binary, resolved once and cached.

        Raises
        ------
        RcloneError
            If the binary is not on PATH; the message carries the one-time
            install and remote-configuration steps.
        """
        if self._resolved_binary is None:
            found = shutil.which(self.binary)
            if found is None:
                raise RcloneError(
                    f"{self.binary!r} not found on PATH. Install it and configure the remote "
                    "once:\n"
                    "  sudo apt install rclone\n"
                    "  rclone config      # n) new remote, type: drive\n"
                    f"  rclone lsd {self.target.remote}:"
                )
            self._resolved_binary = found
        return self._resolved_binary

    def _run(self, args: list[str]) -> str:
        command = [self.executable(), *args, *self.extra_args]
        if self.dry_run:
            command.append("--dry-run")
        logger.debug("rclone: %s", " ".join(command))
        try:
            # S603: the resolved-executable case documented in
            # micrometeorology.common.git.run_git — ruff exempts only an inline list of string
            # literals, which would mean hardcoding an absolute path to rclone. Every element is
            # that resolved path, a literal flag, or a path this package built; no shell is used.
            completed = subprocess.run(  # noqa: S603
                command, check=False, capture_output=True, text=True, timeout=self.timeout
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RcloneError(f"rclone {args[0]} failed to run: {exc}") from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise RcloneError(f"rclone {args[0]} exited {completed.returncode}: {detail}")
        return completed.stdout

    def check(self) -> None:
        """Fail fast when the remote name is wrong or unconfigured."""
        self._run(["lsd", f"{self.target.remote}:", "--max-depth", "1"])
        logger.info("rclone remote %s: reachable", self.target.remote)

    def upload_file(self, local: str | Path, *remote_parts: str) -> str:
        """Copy one file to ``target.path(*remote_parts)`` and return that destination.

        Raises
        ------
        RcloneError
            If *local* is not a file, or rclone exits non-zero.
        """
        source = Path(local)
        if not source.is_file():
            raise RcloneError(f"nothing to upload: {source} is not a file")
        destination = self.target.path(*remote_parts)
        self._run(["copyto", str(source), destination, "--checksum"])
        logger.info("uploaded %s -> %s", source.name, destination)
        return destination

    def upload_dir(self, local_dir: str | Path, *remote_parts: str, pattern: str = "*") -> str:
        """Copy a whole directory in one invocation, not one process per file.

        *pattern* narrows the copy to matching names via ``--include``.

        Raises
        ------
        RcloneError
            If *local_dir* is not a directory, or rclone exits non-zero.
        """
        source = Path(local_dir)
        if not source.is_dir():
            raise RcloneError(f"nothing to upload: {source} is not a directory")
        destination = self.target.path(*remote_parts)
        args = ["copy", str(source), destination, "--checksum"]
        if pattern != "*":
            args += ["--include", pattern]
        self._run(args)
        logger.info("uploaded %s/%s -> %s", source.name, pattern, destination)
        return destination

    def list_remote(self, *remote_parts: str) -> tuple[str, ...]:
        """Names directly under a remote folder, or an empty tuple if it cannot be listed.

        A missing folder and an unreachable remote both come back empty: this is
        an informational listing, and a caller that needs the remote to be
        configured should use :meth:`check`.
        """
        destination = self.target.path(*remote_parts)
        try:
            output = self._run(["lsf", destination])
        except RcloneError as exc:
            logger.debug("remote listing of %s unavailable: %s", destination, exc)
            return ()
        return tuple(line.strip("/") for line in output.splitlines() if line.strip())
