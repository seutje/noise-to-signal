"""
Image post-processing utilities for the offline renderer.

Operations are implemented with NumPy so they work in CPU-only environments.
The `PostFXProcessor` class maintains state for temporal effects such as
motion trails and grain RNG continuity.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Tuple

import numpy as np

from .config_schema import PostFXConfig

LOG = logging.getLogger("renderer.postfx")


def _log_debug(message: str, **payload: object) -> None:
    if LOG.isEnabledFor(logging.DEBUG):
        LOG.debug("%s | %s", message, payload)


def _resize_bilinear(frames: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    target_width, target_height = int(size[0]), int(size[1])
    batch, height, width, channels = frames.shape
    if width == target_width and height == target_height:
        return frames

    y_coords = np.linspace(0, height - 1, target_height, dtype=np.float32)
    y0 = np.floor(y_coords).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, height - 1)
    y_alpha = (y_coords - y0).astype(np.float32)

    top = frames[:, y0, :, :]
    bottom = frames[:, y1, :, :]
    vertical = top * (1.0 - y_alpha)[None, :, None, None] + bottom * y_alpha[None, :, None, None]

    x_coords = np.linspace(0, width - 1, target_width, dtype=np.float32)
    x0 = np.floor(x_coords).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, width - 1)
    x_alpha = (x_coords - x0).astype(np.float32)

    left = vertical[:, :, x0, :]
    right = vertical[:, :, x1, :]
    resized = left * (1.0 - x_alpha)[None, None, :, None] + right * x_alpha[None, None, :, None]

    return np.clip(resized, 0.0, 1.0)


def _tone_curve(frames: np.ndarray, curve: str) -> np.ndarray:
    curve = curve.lower()
    if curve == "linear":
        return frames
    if curve == "filmlog":
        return np.log1p(frames * 5.0) / np.log1p(5.0)
    if curve == "punch":
        return np.power(frames, 0.85) * 0.95 + 0.05 * frames
    if curve == "pastel":
        return np.power(frames, 1.25) * 0.9 + 0.1
    return frames


def _build_vignette(width: int, height: int, strength: float) -> np.ndarray:
    xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)
    radius = np.sqrt(xv**2 + yv**2)
    mask = 1.0 - np.clip(radius, 0.0, 1.0)
    mask = mask**(strength * 4.0 + 1e-3)
    return mask.astype(np.float32)


def _apply_vignette(frames: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return frames * mask[None, :, :, None]


def _apply_chroma_shift(frames: np.ndarray, offset_ratio: float) -> np.ndarray:
    if abs(offset_ratio) < 1e-6:
        return frames
    batch, height, width, _ = frames.shape
    pixels = max(1, int(abs(offset_ratio) * min(width, height)))
    direction = 1 if offset_ratio >= 0 else -1
    shifted = frames.copy()
    shifted[..., 0] = np.roll(frames[..., 0], shift=direction * pixels, axis=2)
    shifted[..., 2] = np.roll(frames[..., 2], shift=-direction * pixels, axis=1)
    return shifted


@dataclass
class PostFXProcessor:
    config: PostFXConfig
    resolution: Tuple[int, int]
    seed: int

    def __post_init__(self) -> None:
        width, height = self.resolution
        self._rng = np.random.default_rng(self.seed)
        self._vignette = (
            _build_vignette(width, height, self.config.vignette_strength)
            if self.config.vignette_strength > 0
            else None
        )
        self._trail_state: np.ndarray | None = None
        self._frames_processed = 0

    def process(self, frames: np.ndarray) -> np.ndarray:
        """
        Apply configured post-processing to a batch of frames (NHWC, float32).
        """
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError("Frames must have shape (batch, height, width, 3).")

        width, height = self.resolution
        frames = np.asarray(frames, dtype=np.float32)
        frames = np.clip(frames, 0.0, 1.0)
        frames = _resize_bilinear(frames, (width, height))
        frames = _tone_curve(frames, self.config.tone_curve)

        if self._vignette is not None:
            frames = _apply_vignette(frames, self._vignette)

        frames = _apply_chroma_shift(frames, self.config.chroma_shift)

        if self.config.motion_trails:
            frames = self._apply_motion_trails(frames)

        if self.config.grain_intensity > 0:
            frames = self._apply_grain(frames)

        frames = np.clip(frames, 0.0, 1.0)
        self._frames_processed += frames.shape[0]
        return frames

    # ------------------------------------------------------------------ #
    # Effect implementations
    # ------------------------------------------------------------------ #

    def _apply_motion_trails(self, frames: np.ndarray) -> np.ndarray:
        alpha = 0.82
        if self._trail_state is None:
            self._trail_state = frames[0].copy()
        output = np.empty_like(frames)
        for idx, frame in enumerate(frames):
            self._trail_state = alpha * self._trail_state + (1.0 - alpha) * frame
            mix = 0.7 * frame + 0.3 * self._trail_state
            output[idx] = np.clip(mix, 0.0, 1.0)
        return output

    def _apply_grain(self, frames: np.ndarray) -> np.ndarray:
        batch, height, width, _ = frames.shape
        noise = self._rng.standard_normal((batch, height, width, 1)).astype(np.float32)
        intensity = float(self.config.grain_intensity)
        # Use a mild blur by averaging neighbouring noise samples to avoid harsh speckles.
        noise = (noise + np.roll(noise, 1, axis=1) + np.roll(noise, 1, axis=2)) / 3.0
        output = frames + noise * intensity * 0.15
        return output


def apply_postfx(
    frames: np.ndarray,
    *,
    processor: PostFXProcessor,
) -> np.ndarray:
    """
    Convenience wrapper mirroring the Phase 3 API.
    """
    return processor.process(frames)


__all__ = ["PostFXProcessor", "apply_postfx"]

