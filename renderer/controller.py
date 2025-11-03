"""Latent controller implementation for the noise-to-signal renderer."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np

from . import PACKAGE_ROOT
from .config_schema import RenderConfig, TrackConfig

if TYPE_CHECKING:  # pragma: no cover
    from .audio_features import FeatureLayout, FeatureTimeline

LOG = logging.getLogger("renderer.controller")

MODEL_ROOT = PACKAGE_ROOT.parent / "models"
META_PATH = MODEL_ROOT / "meta.json"

_META_CACHE: Optional[Dict[str, object]] = None


def _log(event: str, **payload: object) -> None:
    message = {"event": event, **payload}
    LOG.info(json.dumps(message, sort_keys=True))


def _load_meta() -> Dict[str, object]:
    global _META_CACHE
    if _META_CACHE is None:
        content = META_PATH.read_text()
        _META_CACHE = json.loads(content)
    return _META_CACHE


@dataclass(slots=True)
class SeedConfig:
    controller_seed: int
    track_seed: Optional[int] = None

    def merged_seed(self) -> int:
        if self.track_seed is None:
            return self.controller_seed
        return np.int64(self.controller_seed) ^ np.int64(self.track_seed)


@dataclass(slots=True)
class LatentResult:
    latents: np.ndarray  # shape (frames, C, H, W)
    frame_times: np.ndarray
    weights: np.ndarray  # shape (frames, anchor_count)
    anchor_name: str
    layout: "FeatureLayout"
    frame_rate: int

    @property
    def frame_count(self) -> int:
        return int(self.latents.shape[0])


class AnchorSet:
    def __init__(self, name: str, meta: Dict[str, object]) -> None:
        anchor_sets = meta.get("anchor_sets", {})
        if name not in anchor_sets:
            available = ", ".join(sorted(anchor_sets))
            raise KeyError(
                f"Anchor set '{name}' not defined in models/meta.json. "
                f"Available: {available}"
            )
        info = anchor_sets[name]
        file_path = MODEL_ROOT.parent / info["file"]  # type: ignore[index]
        if not file_path.exists():
            raise FileNotFoundError(
                f"Anchor file for '{name}' not found at {file_path}"
            )
        data = np.load(file_path, allow_pickle=False)
        anchors = data["anchors"]
        if anchors.ndim != 4:
            raise ValueError("Anchor array must have shape (count, C, H, W)")
        self.name = name
        self.anchors = anchors.astype(np.float32)
        self.anchor_count = anchors.shape[0]
        self.seed = int(info.get("seed", 0))
        self.description = info.get("description", "")


class LatentController:
    def __init__(
        self,
        *,
        config: RenderConfig,
        track: TrackConfig,
        anchor_set: AnchorSet,
        layout: "FeatureLayout",
        frame_rate: int,
    ) -> None:
        meta = _load_meta()
        latent_shape = tuple(meta.get("latent_shape", []))
        if len(latent_shape) != 3:
            raise ValueError("models/meta.json must define latent_shape [C, H, W].")

        self.config = config
        self.track = track
        self.anchor_set = anchor_set
        self.layout = layout
        self.frame_rate = frame_rate
        self.latent_shape = (int(latent_shape[0]), int(latent_shape[1]), int(latent_shape[2]))
        self.latent_size = int(np.prod(self.latent_shape))

        self.base_wander = self._resolve_wander_amount()
        self.smoothing = float(config.controller.smoothing_alpha)
        self.tempo_sync = config.controller.tempo_sync

        seed = np.int64(anchor_set.seed) ^ np.int64(config.controller.wander_seed)
        if track.seed is not None:
            seed ^= np.int64(track.seed)
        self.rng = np.random.default_rng(int(seed))
        self._projection = None
        self._wander_offsets = self.rng.random(self.latent_size).astype(np.float32) * np.pi * 2
        self._wander_speeds = self.rng.uniform(0.05, 0.25, self.latent_size).astype(np.float32)

    @classmethod
    def from_config(
        cls,
        *,
        config: RenderConfig,
        track: TrackConfig,
        feature_layout: FeatureLayout,
    ) -> "LatentController":
        anchor_name = track.anchors or config.controller.anchor_set or config.controller.preset
        anchor_set = AnchorSet(anchor_name, _load_meta())
        return cls(
            config=config,
            track=track,
            anchor_set=anchor_set,
            layout=feature_layout,
            frame_rate=config.frame_rate,
        )

    def _resolve_wander_amount(self) -> float:
        preset = (self.track.preset or self.config.controller.preset or "default").lower()
        wander_map = {
            "pulse": 0.18,
            "surge": 0.18,
            "drift": 0.28,
            "nocturne": 0.3,
            "fracture": 0.24,
            "lumen": 0.2,
        }
        return wander_map.get(preset, 0.22)

    def _build_projection(self, feature_dim: int) -> np.ndarray:
        if self._projection is not None and self._projection.shape[1] == feature_dim:
            return self._projection
        projection = self.rng.normal(
            loc=0.0,
            scale=0.5,
            size=(self.anchor_set.anchor_count, feature_dim),
        ).astype(np.float32)
        self._projection = projection
        return projection

    def generate(
        self,
        *,
        features: "FeatureTimeline",
        seeds: SeedConfig,
    ) -> LatentResult:
        frame_times, feature_matrix = self._resample_features(features)
        projection = self._build_projection(feature_matrix.shape[1])
        anchors_flat = self.anchor_set.anchors.reshape(self.anchor_set.anchor_count, -1)

        _log(
            "controller.generate_start",
            track_id=self.track.id,
            anchor_set=self.anchor_set.name,
            frames=int(frame_times.shape[0]),
            feature_dim=int(feature_matrix.shape[1]),
        )

        frame_total = frame_times.shape[0]
        latents = np.zeros((frame_total, *self.latent_shape), dtype=np.float32)
        weights = np.zeros((frame_total, self.anchor_set.anchor_count), dtype=np.float32)
        blended = np.zeros(self.latent_size, dtype=np.float32)
        ema_state = np.zeros_like(blended)
        wander_phase = 0.0

        seed_value = seeds.merged_seed()
        rng = np.random.default_rng(int(seed_value))
        wander_jitter = rng.uniform(0.8, 1.2)

        onset_column = feature_matrix[:, 3] if feature_matrix.shape[1] > 3 else np.zeros(frame_total, dtype=np.float32)
        if onset_column.size and onset_column.max() > 1e-4:
            upper = float(np.quantile(onset_column, 0.75))
            median = float(np.quantile(onset_column, 0.5))
            onset_threshold = float(np.clip(median + (upper - median) * 0.5, 0.2, 0.75))
        else:
            onset_threshold = 0.6

        tempo_counter = 0
        tempo_interval = max(
            1,
            int(self.frame_rate * max(self.config.controller.tempo_sync.subdivision, 0.125)),
        )

        for idx, (time_sec, feature_vec) in enumerate(zip(frame_times, feature_matrix)):
            scores = projection @ feature_vec
            # Temperature scales with RMS to produce more dynamic changes.
            temperature = 1.0 + float(feature_vec[0]) * 0.5
            weights[idx] = _softmax(scores, temperature)
            blended[:] = weights[idx] @ anchors_flat

            # Wander modulation from RMS + centroid.
            rms_norm = float(feature_vec[0])
            centroid_norm = float(feature_vec[1])
            wander_speed = (0.4 + centroid_norm * 0.5) * wander_jitter
            wander_phase += wander_speed

            onset_power = float(feature_vec[3]) if feature_vec.shape[0] > 3 else 0.0
            if self.tempo_sync.enabled:
                tempo_counter += 1
                if onset_power > onset_threshold or tempo_counter >= tempo_interval:
                    tempo_counter = 0
                    wander_phase += np.pi * 0.5

            wander_strength = self.base_wander * (0.25 + rms_norm * 1.35 + onset_power * 0.3)

            noise = np.sin(self._wander_offsets + wander_phase * self._wander_speeds)
            latent = blended + noise * wander_strength

            dynamic_smoothing = np.clip(
                self.smoothing - (rms_norm * 0.35 + onset_power * 0.25),
                0.35,
                self.smoothing,
            )
            ema_state = dynamic_smoothing * ema_state + (1.0 - dynamic_smoothing) * latent
            latents[idx] = ema_state.reshape(self.latent_shape)

        result = LatentResult(
            latents=latents,
            frame_times=frame_times,
            weights=weights,
            anchor_name=self.anchor_set.name,
            layout=self.layout,
            frame_rate=self.frame_rate,
        )
        _log(
            "controller.generate_complete",
            track_id=self.track.id,
            anchor_set=self.anchor_set.name,
            frames=result.frame_count,
            wander=self.base_wander,
        )
        return result

    def _resample_features(
        self, timeline: FeatureTimeline
    ) -> Tuple[np.ndarray, np.ndarray]:
        duration = float(timeline.times[-1]) if timeline.times.size else 0.0
        frame_count = max(1, int(np.ceil(duration * self.frame_rate)))
        frame_times = np.arange(frame_count, dtype=np.float32) / float(self.frame_rate)
        frame_times = np.clip(frame_times, 0.0, duration if duration > 0 else 0.0)

        def interp(values: np.ndarray) -> np.ndarray:
            return np.interp(
                frame_times,
                timeline.times,
                values,
                left=float(values[0]),
                right=float(values[-1]),
            )

        rms = interp(timeline.rms)
        centroid = interp(timeline.centroid) / (self.layout.sample_rate / 2.0)
        centroid = np.clip(centroid, 0.0, 1.0)
        flatness = interp(timeline.flatness)

        onset = interp(timeline.onset)
        if onset.max() > 0:
            onset = onset / onset.max()

        mfcc = np.vstack(
            [
                interp(timeline.mfcc[i])
                for i in range(min(timeline.mfcc.shape[0], 8))
            ]
        )
        mfcc = np.tanh(mfcc / 20.0)

        feature_matrix = np.concatenate(
            [
                rms[np.newaxis, :],
                centroid[np.newaxis, :],
                flatness[np.newaxis, :],
                onset[np.newaxis, :],
                mfcc,
            ],
            axis=0,
        ).T.astype(np.float32)

        return frame_times.astype(np.float32), feature_matrix


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    exps = np.exp((values - values.max()) / (temperature + 1e-6))
    denom = exps.sum()
    if denom == 0:
        return np.full_like(values, 1.0 / values.size)
    return exps / denom


def save_latent_result(result: LatentResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "frame_rate": result.frame_rate,
        "audio_sample_rate": result.layout.sample_rate,
        "anchor_set": result.anchor_name,
        "frames": result.frame_count,
        "latent_shape": list(result.latents.shape[1:]),
    }
    np.savez_compressed(
        path,
        latents=result.latents.astype(np.float32),
        frame_times=result.frame_times.astype(np.float32),
        weights=result.weights.astype(np.float32),
        metadata=json.dumps(metadata),
    )
    LOG.info(
        "Latent result saved | %s",
        {"path": str(path), "frames": result.frame_count, "anchor": result.anchor_name},
    )


def get_anchor_sets() -> Dict[str, Dict[str, object]]:
    """Return metadata for available anchor sets from models/meta.json."""
    meta = _load_meta()
    anchor_sets = meta.get("anchor_sets", {})
    result: Dict[str, Dict[str, object]] = {}
    for name, info in anchor_sets.items():
        result[name] = {
            "anchor_count": info.get("anchor_count"),
            "file": info.get("file"),
            "description": info.get("description", ""),
        }
    return result
