"""GIF and WebM video generation from WRF map image sequences.

Supports direct PNG → WebM conversion (no GIF intermediary) for
production use, and GIF for quick previews.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from labmim_micrometeorology.common.paths import ensure_dir

logger = logging.getLogger(__name__)


def create_gif(
    image_dir: str | Path,
    output_path: str | Path,
    pattern: str = "*.png",
    duration: float = 0.5,
) -> Path:
    """Create an animated GIF from a directory of images.

    Parameters
    ----------
    image_dir:
        Directory containing the source images.
    output_path:
        Path for the output GIF file.
    pattern:
        Glob pattern to select images.
    duration:
        Duration of each frame in seconds.
    """
    import imageio.v3 as iio

    images_dir = Path(image_dir)
    files = sorted(images_dir.glob(pattern))
    if not files:
        logger.warning("No images found in %s matching %s", images_dir, pattern)
        return Path(output_path)

    out = Path(output_path)
    ensure_dir(out.parent)

    frames = [iio.imread(str(f)) for f in files]
    iio.imwrite(str(out), frames, duration=int(duration * 1000), loop=0)
    logger.info("Created GIF: %s (%d frames)", out, len(frames))
    return out


def create_webm_from_images(
    image_paths: list[str | Path],
    output_path: str | Path,
    fps: int = 2,
) -> Path:
    """Create a WebM video directly from a list of PNG files.

    Uses ``moviepy`` — no GIF intermediary, no ffmpeg CLI dependency.
    Requires ``pip install labmim-micrometeorology[video]``.

    Parameters
    ----------
    image_paths:
        Ordered list of image file paths.
    output_path:
        Path for the output ``.webm`` file.
    fps:
        Frames per second.
    """
    try:
        from moviepy.editor import ImageSequenceClip
    except ImportError as exc:
        raise ImportError(
            "moviepy is required for WebM creation.  "
            "Install with: pip install labmim-micrometeorology[video]"
        ) from exc

    if not image_paths:
        logger.warning("No images to create WebM")
        return Path(output_path)

    out = Path(output_path)
    ensure_dir(out.parent)

    str_paths = [str(p) for p in image_paths]
    clip = ImageSequenceClip(str_paths, fps=fps)
    clip.write_videofile(str(out), audio=False, threads=1, logger=None)
    clip.close()

    logger.info("Created WebM: %s (%d frames, %d fps)", out, len(image_paths), fps)
    return out


def gif_to_webm(
    gif_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Convert a GIF to WebM video using moviepy.

    Requires the ``video`` optional dependency (``pip install labmim-micrometeorology[video]``).
    """
    try:
        from moviepy.editor import VideoFileClip
    except ImportError as exc:
        raise ImportError(
            "moviepy is required for video conversion.  "
            "Install with: pip install labmim-micrometeorology[video]"
        ) from exc

    gif = Path(gif_path)
    out = Path(output_path) if output_path else gif.with_suffix(".webm")
    ensure_dir(out.parent)

    clip = VideoFileClip(str(gif))
    clip.write_videofile(str(out), audio=False, threads=1, logger=None)
    clip.close()
    logger.info("Created WebM: %s", out)
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
    """
    n_workers = workers or max(1, (os.cpu_count() or 4) - 4)
    n_workers = min(n_workers, len(grouped_images)) if grouped_images else 1

    out_dir = str(Path(output_dir))
    tasks = [(name, paths, out_dir, fps) for name, paths in grouped_images.items()]

    logger.info("Creating %d WebM videos with %d workers", len(tasks), n_workers)

    results: list[str] = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_batch_single_webm, t): t[0] for t in tasks}
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    logger.info("✓ Created %d WebM videos", len(results))
    return results
