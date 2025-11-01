"""
Audio feature extraction and caching utilities.

Computes deterministic per-track features (RMS, spectral centroid, spectral
flatness, MFCC-lite, onset strength) and caches results under
`cache/features/<track>.npz` with checksum validation to avoid expensive
recomputation.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Dict, Optional

import json
import logging

try:
    import librosa  # type: ignore
except ImportError:  # pragma: no cover - exercised when dependency missing
    librosa = None  # type: ignore[assignment]
import numpy as np

LOG = logging.getLogger("renderer.audio")

HOP_LENGTH = 512
FRAME_LENGTH = 2048
MFCC_COMPONENTS = 13


def _log(event: str, **payload: object) -> None:
    message = {"event": event, **payload}
    LOG.info(json.dumps(message, sort_keys=True))


@dataclass(slots=True)
class FeatureLayout:
    sample_rate: int
    hop_length: int = HOP_LENGTH
    frame_length: int = FRAME_LENGTH
    mfcc_components: int = MFCC_COMPONENTS

    def to_dict(self) -> Dict[str, int]:
        return {
            "sample_rate": self.sample_rate,
            "hop_length": self.hop_length,
            "frame_length": self.frame_length,
            "mfcc_components": self.mfcc_components,
        }


@dataclass(slots=True)
class FeatureTimeline:
    times: np.ndarray
    rms: np.ndarray
    centroid: np.ndarray
    flatness: np.ndarray
    mfcc: np.ndarray
    onset: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.times.shape[0])

    def to_cache_payload(self) -> Dict[str, np.ndarray]:
        return {
            "times": self.times.astype(np.float32),
            "rms": self.rms.astype(np.float32),
            "centroid": self.centroid.astype(np.float32),
            "flatness": self.flatness.astype(np.float32),
            "mfcc": self.mfcc.astype(np.float32),
            "onset": self.onset.astype(np.float32),
        }


@dataclass(slots=True)
class FeatureResult:
    timeline: FeatureTimeline
    layout: FeatureLayout
    cache_path: Path
    checksum: str
    source_audio: Path


def _checksum_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_audio(y: np.ndarray, target_dbfs: float) -> np.ndarray:
    eps = 1e-8
    rms = np.sqrt(np.mean(np.square(y)) + eps)
    current_db = 20 * np.log10(max(rms, eps))
    gain = 10 ** ((target_dbfs - current_db) / 20)
    return np.clip(y * gain, -1.0, 1.0)


def _compute_features_array(y: np.ndarray, sr: int) -> FeatureTimeline:
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[
        0
    ]
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=HOP_LENGTH
    )[0]
    flatness = librosa.feature.spectral_flatness(
        y=y, hop_length=HOP_LENGTH, n_fft=FRAME_LENGTH
    )[0]
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=MFCC_COMPONENTS,
        hop_length=HOP_LENGTH,
        n_fft=FRAME_LENGTH,
    )
    onset_env = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=HOP_LENGTH,
        aggregate=np.mean,
    )

    frames = np.arange(rms.shape[0])
    times = librosa.frames_to_time(frames, sr=sr, hop_length=HOP_LENGTH)

    # Align lengths (onset strength can differ by one frame).
    min_len = min(
        times.shape[0],
        rms.shape[0],
        centroid.shape[0],
        flatness.shape[0],
        mfcc.shape[1],
        onset_env.shape[0],
    )

    return FeatureTimeline(
        times=times[:min_len],
        rms=rms[:min_len],
        centroid=centroid[:min_len],
        flatness=flatness[:min_len],
        mfcc=mfcc[:, :min_len],
        onset=onset_env[:min_len],
    )


def _cache_metadata(
    checksum: str,
    layout: FeatureLayout,
    source_path: Path,
) -> Dict[str, object]:
    return {
        "checksum": checksum,
        "layout": layout.to_dict(),
        "source": str(source_path),
    }


def _load_cached_result(cache_path: Path, checksum: str) -> Optional[FeatureResult]:
    if not cache_path.exists():
        return None

    with np.load(cache_path, allow_pickle=False) as data:
        metadata_json = str(data["metadata"])
        metadata = json.loads(metadata_json)
        if metadata.get("checksum") != checksum:
            return None

        timeline = FeatureTimeline(
            times=data["times"],
            rms=data["rms"],
            centroid=data["centroid"],
            flatness=data["flatness"],
            mfcc=data["mfcc"],
            onset=data["onset"],
        )
        layout_dict = metadata["layout"]
        layout = FeatureLayout(
            sample_rate=int(layout_dict["sample_rate"]),
            hop_length=int(layout_dict["hop_length"]),
            frame_length=int(layout_dict["frame_length"]),
            mfcc_components=int(layout_dict["mfcc_components"]),
        )

    _log(
        "features.cache_hit",
        cache_path=str(cache_path),
        checksum=checksum,
        frames=timeline.frame_count,
    )
    return FeatureResult(
        timeline=timeline,
        layout=layout,
        cache_path=cache_path,
        checksum=checksum,
        source_audio=Path(metadata["source"]),
    )


def _save_cache(
    cache_path: Path,
    feature_result: FeatureResult,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = feature_result.timeline.to_cache_payload()
    metadata = _cache_metadata(
        checksum=feature_result.checksum,
        layout=feature_result.layout,
        source_path=feature_result.source_audio,
    )
    np.savez_compressed(cache_path, **payload, metadata=json.dumps(metadata))


def compute_features(
    *,
    audio_path: Path,
    cache_root: Path,
    track_id: str,
    sample_rate: int,
    normalization: Optional[float],
) -> FeatureResult:
    """
    Compute deterministic audio features, using cache when available.
    """
    if librosa is None:
        raise ImportError(
            "librosa is required for feature extraction. "
            "Install dependencies listed in renderer/requirements.txt."
        )
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio source not found: {audio_path}")

    cache_path = cache_root / f"{track_id}.npz"
    checksum = _checksum_file(audio_path)
    cached = _load_cached_result(cache_path, checksum)
    if cached:
        return cached

    _log(
        "features.compute",
        track_id=track_id,
        cache_path=str(cache_path),
        sample_rate=sample_rate,
    )

    y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    if normalization is not None:
        y = _normalise_audio(y, normalization)

    timeline = _compute_features_array(y, sr)
    layout = FeatureLayout(sample_rate=sr)
    result = FeatureResult(
        timeline=timeline,
        layout=layout,
        cache_path=cache_path,
        checksum=checksum,
        source_audio=audio_path,
    )
    _save_cache(cache_path, result)
    _log(
        "features.cache_write",
        track_id=track_id,
        cache_path=str(cache_path),
        frames=timeline.frame_count,
    )
    return FeatureResult(
        timeline=timeline,
        layout=layout,
        cache_path=cache_path,
        checksum=checksum,
        source_audio=audio_path,
    )
