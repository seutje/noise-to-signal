from __future__ import annotations

import numpy as np
import pytest

from renderer.postfx import PostFXProcessor
from renderer.config_schema import PostFXConfig


def _make_frames(batch: int = 4, size: int = 32) -> np.ndarray:
    coords = np.linspace(0.0, 1.0, size * size * 3, dtype=np.float32)
    frames = coords.reshape(size, size, 3)
    stacked = np.stack([frames for _ in range(batch)], axis=0)
    return stacked


def test_postfx_processor_deterministic_seed() -> None:
    config = PostFXConfig(
        tone_curve="filmlog",
        grain_intensity=0.2,
        chroma_shift=0.01,
        vignette_strength=0.4,
        motion_trails=True,
    )
    frames = _make_frames()

    proc_a = PostFXProcessor(config=config, resolution=(32, 32), seed=123)
    proc_b = PostFXProcessor(config=config, resolution=(32, 32), seed=123)

    out_a = proc_a.process(frames)
    out_b = proc_b.process(frames)

    assert out_a.shape == frames.shape
    assert np.allclose(out_a, out_b)
    assert np.all(out_a >= 0.0)
    assert np.all(out_a <= 1.0)


def test_postfx_rejects_invalid_shape() -> None:
    processor = PostFXProcessor(config=PostFXConfig(), resolution=(16, 16), seed=1)
    with pytest.raises(ValueError):
        processor.process(np.zeros((1, 16, 16), dtype=np.float32))
