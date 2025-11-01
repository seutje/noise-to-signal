"""
Utilities for piping frames to FFmpeg and generating preview assets.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

try:  # pragma: no cover - optional for environments without imageio
    import imageio.v3 as imageio
except Exception:  # pragma: no cover
    imageio = None  # type: ignore[assignment]

LOG = logging.getLogger("renderer.frame_writer")


class FFMpegError(RuntimeError):
    """Raised when FFmpeg exits with a non-zero status."""


@dataclass
class PreviewBundle:
    still: Optional[Path]
    animated: Optional[Path]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _frame_to_bytes(frame: np.ndarray) -> bytes:
    uint8_frame = np.clip(frame * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return uint8_frame.tobytes()


class FFMpegWriter:
    def __init__(
        self,
        *,
        output_path: Path,
        frame_rate: int,
        resolution: Sequence[int],
        audio_path: Path,
        ffmpeg_path: str = "ffmpeg",
        trim_start: Optional[float] = None,
        trim_end: Optional[float] = None,
        keep_frames: bool = False,
        preview_dir: Optional[Path] = None,
        video_codec: str = "libx264",
        audio_codec: str = "aac",
        crf: int = 18,
    ) -> None:
        self.output_path = output_path
        self.frame_rate = frame_rate
        self.resolution = (int(resolution[0]), int(resolution[1]))
        self.audio_path = audio_path
        self.ffmpeg_path = ffmpeg_path
        self.trim_start = trim_start
        self.trim_end = trim_end
        self.keep_frames = keep_frames
        self.preview_dir = preview_dir
        self.video_codec = video_codec
        self.audio_codec = audio_codec
        self.crf = crf

        self.frames_written = 0
        self._start_time = time.perf_counter()
        self._frame_dir: Optional[Path] = None
        self._preview_paths: list[Path] = []
        self._preview_frames: list[np.ndarray] = []
        self._process = self._spawn_ffmpeg()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def _spawn_ffmpeg(self) -> subprocess.Popen:
        width, height = self.resolution
        _ensure_parent(self.output_path)

        video_input = [
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(self.frame_rate),
            "-i",
            "-",
        ]

        audio_input: list[str] = []
        if self.trim_start is not None:
            audio_input += ["-ss", f"{self.trim_start:.3f}"]
        if self.trim_end is not None and self.trim_start is not None:
            duration = max(0.0, self.trim_end - self.trim_start)
            audio_input += ["-t", f"{duration:.3f}"]
        elif self.trim_end is not None:
            audio_input += ["-to", f"{self.trim_end:.3f}"]
        audio_input += ["-i", str(self.audio_path)]

        output_args = [
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            self.video_codec,
            "-preset",
            "slow",
            "-crf",
            str(self.crf),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            self.audio_codec,
            "-b:a",
            "320k",
            "-shortest",
            str(self.output_path),
        ]

        cmd = [self.ffmpeg_path, "-y", "-loglevel", "error", *video_input, *audio_input, *output_args]
        LOG.debug("FFmpeg command: %s", " ".join(cmd))
        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:  # pragma: no cover - depends on system
            raise FFMpegError(f"ffmpeg not found: {exc}") from exc

        if self.keep_frames:
            self._frame_dir = self.output_path.parent / "frames"
            self._frame_dir.mkdir(parents=True, exist_ok=True)
        return process

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #

    def write_batch(self, frames: np.ndarray) -> None:
        if self._process.stdin is None:
            raise FFMpegError("FFmpeg process has no stdin.")
        for frame in frames:
            self._write_frame(frame)

    def _write_frame(self, frame: np.ndarray) -> None:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Frame must have shape (height, width, 3).")
        if self._process.stdin is None:
            raise FFMpegError("FFmpeg process has terminated unexpectedly.")

        self._process.stdin.write(_frame_to_bytes(frame))
        if self.keep_frames:
            self._persist_frame(frame)
        if self.preview_dir and self.frames_written % 90 == 0:
            self._save_preview_frame(frame)
        self.frames_written += 1

    def _persist_frame(self, frame: np.ndarray) -> None:
        if self._frame_dir is None:
            return
        index = f"{self.frames_written:06d}.png"
        path = self._frame_dir / index
        if imageio is None:
            return
        imageio.imwrite(path, np.clip(frame * 255.0, 0, 255).astype(np.uint8))

    def _save_preview_frame(self, frame: np.ndarray) -> None:
        if imageio is None:
            return
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        still = self.preview_dir / f"frame_{self.frames_written:06d}.png"
        array = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        imageio.imwrite(still, array)
        self._preview_paths.append(still)
        self._preview_frames.append(array)

    # ------------------------------------------------------------------ #
    # Close & metadata
    # ------------------------------------------------------------------ #

    def close(self) -> PreviewBundle:
        if self._process.stdin and not self._process.stdin.closed:
            try:
                self._process.stdin.flush()
            except ValueError:
                pass
            self._process.stdin.close()
        stdout, stderr = self._process.communicate()
        if self._process.returncode != 0:
            LOG.error("FFmpeg stderr: %s", stderr.decode("utf-8", errors="ignore"))
            raise FFMpegError(f"FFmpeg exited with status {self._process.returncode}")

        still_path = None
        animated_path = None
        if self.preview_dir and self._preview_frames and imageio is not None:
            still_path = self.preview_dir / "preview.png"
            animated_path = self.preview_dir / "preview.gif"
            key_frame = self._preview_frames[len(self._preview_frames) // 2]
            imageio.imwrite(still_path, key_frame)
            sample = np.stack(
                self._preview_frames[: min(24, len(self._preview_frames))],
                axis=0,
            )
            if sample.size > 0:
                imageio.imwrite(
                    animated_path,
                    sample,
                    loop=0,
                    duration=0.08,
                )
        return PreviewBundle(still=still_path, animated=animated_path)

    @property
    def duration(self) -> float:
        return self.frames_written / max(self.frame_rate, 1)

    def to_metadata(self) -> dict:
        return {
            "output": str(self.output_path),
            "frames": self.frames_written,
            "frame_rate": self.frame_rate,
            "duration_seconds": self.duration,
            "resolution": list(self.resolution),
        }


def concat_videos(
    video_paths: Iterable[Path],
    *,
    output_path: Path,
    ffmpeg_path: str = "ffmpeg",
) -> None:
    paths = list(video_paths)
    if not paths:
        raise ValueError("No video paths provided for concatenation.")
    _ensure_parent(output_path)
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as tmp:
        for path in paths:
            tmp.write(f"file '{path}'\n")
        list_path = Path(tmp.name)

    cmd = [
        ffmpeg_path,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-c",
        "copy",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    list_path.unlink(missing_ok=True)
    if result.returncode != 0:
        LOG.error("Album concatenation failed: %s", result.stderr)
        raise FFMpegError("FFmpeg concat failed")


def compute_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "FFMpegWriter",
    "FFMpegError",
    "PreviewBundle",
    "concat_videos",
    "compute_sha256",
]
