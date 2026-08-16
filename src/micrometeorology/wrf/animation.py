"""GIF and WebM video generation from WRF map image sequences.

Supports direct PNG → WebM conversion (no GIF intermediary) for
production use, and GIF for quick previews.
"""

import logging
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from micrometeorology.common.paths import ensure_dir

logger = logging.getLogger(__name__)


def create_webm_from_images(
    image_paths: Sequence[str | Path],
    output_path: str | Path,
    fps: int = 2,
) -> Path:
    """Create a WebM video directly from a list of PNG files.

    Uses ``moviepy`` — no GIF intermediary, no ffmpeg CLI dependency.
    Requires the ``video`` optional dependency (``uv sync --extra video``).

    Parameters
    ----------
    image_paths:
        Ordered list of image file paths.
    output_path:
        Path for the output ``.webm`` file.
    fps:
        Frames per second.

    Returns
    -------
    Path
        The WebM written. When *image_paths* is empty, the path is returned with
        nothing written and a warning logged.

    Raises
    ------
    ImportError
        When the ``video`` extra is not installed.
    """
    try:
        from moviepy import ImageSequenceClip
    except ImportError as exc:
        raise ImportError(
            "moviepy is required for WebM creation.  "
            "Install with: uv sync --extra video (see the video extra in pyproject: "
            "moviepy caps pillow below the CVE floor this project pins, so a bare "
            "pip install of the extra cannot resolve)."
        ) from exc

    if not image_paths:
        logger.warning("No images to create WebM")
        return Path(output_path)

    out = Path(output_path)
    ensure_dir(out.parent)

    str_paths = [str(p) for p in image_paths]
    clip = ImageSequenceClip(str_paths, fps=fps)
    try:
        clip.write_videofile(str(out), audio=False, threads=1, logger=None)
    finally:
        clip.close()

    logger.info("Created WebM: %s (%d frames, %d fps)", out, len(image_paths), fps)
    return out


def _batch_single_webm(args: tuple[str, list[str], str, int]) -> str | None:
    """Worker: create one WebM from a group of PNGs."""
    name, paths, output_dir, fps = args
    if not paths:
        return None
    out = Path(output_dir) / f"{name}.webm"
    try:
        create_webm_from_images(paths, out, fps=fps)
        return str(out)
    except Exception:
        logger.exception("Failed to create WebM: %s", name)
        return None


def batch_create_webm(
    grouped_images: dict[str, list[str]],
    output_dir: str | Path,
    fps: int = 2,
    workers: int | None = None,
) -> list[str]:
    """Create WebM videos for multiple groups of images in parallel.

    Parameters
    ----------
    grouped_images:
        Mapping of ``{video_name: [path1.png, path2.png, ...]}``
        where images are in chronological order.
    output_dir:
        Directory for output WebM files.
    fps:
        Frames per second for each video.
    workers:
        Number of parallel workers.  Defaults to ``min(cpu_count - 4, num_groups)``.

    Returns
    -------
    list[str]
        Paths of the videos written, in COMPLETION order. A group whose encode
        failed is logged and left out, so the list may be shorter than the input.
    """
    n_workers = workers or max(1, (os.cpu_count() or 4) - 4)
    n_workers = min(n_workers, len(grouped_images)) if grouped_images else 1

    out_dir = str(Path(output_dir))
    tasks = [(name, paths, out_dir, fps) for name, paths in grouped_images.items()]

    logger.info("Creating %d WebM videos with %d workers", len(tasks), n_workers)

    results: list[str] = []
    max_tasks_per_child = 1 if n_workers > 1 else None
    with ProcessPoolExecutor(
        max_workers=n_workers,
        max_tasks_per_child=max_tasks_per_child,
    ) as pool:
        futures = {pool.submit(_batch_single_webm, t): t[0] for t in tasks}
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    logger.info("✓ Created %d WebM videos", len(results))
    return results
