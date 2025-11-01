from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from pathlib import Path

import numpy as np

import pytest

from renderer import audio_features as af


@dataclass
class DummyLibrosa:
    frame_count: int = 6
    sample_rate: int = 48_000

    def __post_init__(self) -> None:
        self.load_calls = 0
        self.captured_inputs: list[np.ndarray] = []
        self.feature = SimpleNamespace(
            rms=self._make_feature(offset=0.0),
            spectral_centroid=self._make_feature(offset=0.1),
            spectral_flatness=self._make_feature(offset=0.2),
            mfcc=self._mfcc,
        )
        self.onset = SimpleNamespace(onset_strength=self._onset)

    def load(self, path: Path, sr: int, mono: bool) -> tuple[np.ndarray, int]:
        self.load_calls += 1
        assert mono is True
        assert sr == self.sample_rate
        signal = np.linspace(-0.5, 0.5, self.frame_count * 32, dtype=np.float32)
        return signal, sr

    def frames_to_time(self, frames: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
        step = hop_length / sr
        return (frames.astype(np.float32) * step).astype(np.float32)

    def _register_input(self, array: np.ndarray) -> None:
        self.captured_inputs.append(array.astype(np.float32, copy=True))

    def _make_feature(self, offset: float):
        def _feature(*, y: np.ndarray, **_: object) -> np.ndarray:
            self._register_input(y)
            values = np.linspace(0.1, 0.9, self.frame_count, dtype=np.float32)
            return np.expand_dims(values + offset, axis=0)

        return _feature

    def _mfcc(
        self,
        *,
        y: np.ndarray,
        n_mfcc: int,
        **_: object,
    ) -> np.ndarray:
        self._register_input(y)
        base = np.linspace(-1.0, 1.0, self.frame_count, dtype=np.float32)
        return np.vstack(
            [base + (i * 0.01) for i in range(n_mfcc)]
        ).astype(np.float32)

    def _onset(
        self,
        *,
        y: np.ndarray,
        **_: object,
    ) -> np.ndarray:
        self._register_input(y)
        return np.linspace(0.0, 1.0, self.frame_count, dtype=np.float32)


@pytest.mark.parametrize("normalization", [None, -12.0])
def test_compute_features_caches_results(tmp_path: Path, normalization: float | None, monkeypatch: pytest.MonkeyPatch) -> None:
    audio_path = tmp_path / "fixture.wav"
    audio_path.write_bytes(b"stub audio")

    cache_root = tmp_path / "features"
    dummy = DummyLibrosa()

    monkeypatch.setattr(af, "librosa", dummy)

    result_first = af.compute_features(
        audio_path=audio_path,
        cache_root=cache_root,
        track_id="track-1",
        sample_rate=dummy.sample_rate,
        normalization=normalization,
    )

    assert result_first.timeline.frame_count == dummy.frame_count
    assert result_first.cache_path.exists()
    assert dummy.load_calls == 1
    assert len(dummy.captured_inputs) >= 4  # feature functions observed the signal
    assert np.max(np.abs(dummy.captured_inputs[0])) <= 1.0

    def _fail(*args: object, **kwargs: object) -> None:  # pragma: no cover - sanity guard
        raise AssertionError("Cache path should short-circuit feature computation")

    dummy.feature.rms = _fail  # type: ignore[assignment]

    result_cached = af.compute_features(
        audio_path=audio_path,
        cache_root=cache_root,
        track_id="track-1",
        sample_rate=dummy.sample_rate,
        normalization=normalization,
    )

    assert result_cached.timeline.frame_count == result_first.timeline.frame_count
    assert np.allclose(result_cached.timeline.times, result_first.timeline.times)
    assert result_cached.cache_path == result_first.cache_path
    assert result_cached.checksum == result_first.checksum
