from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from renderer import render_track
from renderer import audio_features
from renderer.audio_features import FeatureLayout, FeatureTimeline, FeatureResult
from renderer.config_schema import (
    AudioConfig,
    ControllerConfig,
    DecoderConfig,
    PostFXConfig,
    RenderConfig,
    TrackConfig,
)
from renderer.controller import LatentResult, SeedConfig, LatentController


class FakeDecoder:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], int]] = []

    @property
    def current_provider(self) -> str:
        return "CPUExecutionProvider"

    @property
    def precision(self) -> str:
        return "int8"

    def decode(self, latents: np.ndarray, *, batch_size: int, validate: bool) -> np.ndarray:
        self.calls.append((latents.shape, batch_size))
        frames = np.full((latents.shape[0], 16, 16, 3), 0.5, dtype=np.float32)
        return frames


class FakeFFMpegWriter:
    def __init__(
        self,
        *,
        output_path: Path,
        frame_rate: int,
        resolution: list[int] | tuple[int, int],
        audio_path: Path,
        **_: object,
    ) -> None:
        self.output_path = output_path
        self.frame_rate = frame_rate
        self.duration = 0.0
        self._frames_written = 0

    def write_batch(self, frames: np.ndarray) -> None:
        self._frames_written += int(frames.shape[0])
        self.duration += frames.shape[0] / self.frame_rate

    def close(self) -> render_track.PreviewBundle:
        self.output_path.write_text("video")
        return render_track.PreviewBundle(still=None, animated=None)


def test_render_track_pipeline(tmp_path: Path, monkeypatch) -> None:
    audio_path = tmp_path / "track.wav"
    audio_path.write_bytes(b"audio")

    track = TrackConfig(id="fixture", src=audio_path, preset="drift", seed=7)
    checkpoint = tmp_path / "vae-best.ckpt"
    checkpoint.write_text("checkpoint")
    config = RenderConfig(
        output_root=tmp_path / "renders",
        frame_rate=30,
        resolution=[1920, 1088],
        audio=AudioConfig(sample_rate=48_000, normalization=-14.0),
        controller=ControllerConfig(preset="default", wander_seed=42),
        decoder=DecoderConfig(batch_size=4, execution_provider="cpu", checkpoint=checkpoint),
        postfx=PostFXConfig(grain_intensity=0.0, motion_trails=False),
        tracks=[track],
    )
    config.validate()

    layout = FeatureLayout(sample_rate=config.audio.sample_rate)
    frames = 4
    timeline = FeatureTimeline(
        times=np.linspace(0.0, 1.0, frames, dtype=np.float32),
        rms=np.linspace(0.1, 0.5, frames, dtype=np.float32),
        centroid=np.linspace(100.0, 300.0, frames, dtype=np.float32),
        flatness=np.linspace(0.2, 0.4, frames, dtype=np.float32),
        mfcc=np.tile(np.linspace(-1.0, 1.0, frames, dtype=np.float32), (13, 1)),
        onset=np.linspace(0.0, 1.0, frames, dtype=np.float32),
    )
    feature_result = FeatureResult(
        timeline=timeline,
        layout=layout,
        cache_path=tmp_path / "cache" / "features" / "fixture.npz",
        checksum="deadbeef",
        source_audio=audio_path,
    )

    latent_result = LatentResult(
        latents=np.ones((frames, 8, 16, 16), dtype=np.float32),
        frame_times=np.linspace(0.0, 1.0, frames, dtype=np.float32),
        weights=np.full((frames, 2), 0.5, dtype=np.float32),
        anchor_name="baseline",
        layout=layout,
        frame_rate=config.frame_rate,
    )

    monkeypatch.setattr(render_track, "FFMpegWriter", FakeFFMpegWriter)

    def fake_compute_features(**_: object) -> FeatureResult:
        return feature_result

    monkeypatch.setattr(audio_features, "compute_features", fake_compute_features)

    def fake_save_latents(result: LatentResult, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "frames": int(result.frame_count),
            "shape": list(result.latents.shape[1:]),
        }
        path.write_text(json.dumps(payload))

    monkeypatch.setattr("renderer.controller.save_latent_result", fake_save_latents)

    def fake_from_config(cls, *, config: RenderConfig, track: TrackConfig, feature_layout: FeatureLayout):
        assert config.controller.preset == "drift"
        assert config.controller.anchor_set == "drift"
        assert not config.controller.tempo_sync.enabled
        assert config.postfx.tone_curve == "pastel"

        class _FakeController:
            def generate(self, *, features: FeatureTimeline, seeds: SeedConfig) -> LatentResult:
                assert features.frame_count == latent_result.frame_count
                assert isinstance(seeds, SeedConfig)
                return latent_result

        return _FakeController()

    monkeypatch.setattr(LatentController, "from_config", classmethod(fake_from_config))

    def fake_checksum(path: Path) -> str:
        assert path.exists()
        return "sha256:deadbeef"

    monkeypatch.setattr(render_track, "compute_sha256", fake_checksum)

    decoder = FakeDecoder()
    summary = render_track.render_track(
        track=track,
        config=config,
        output_dir=tmp_path / "renders" / track.id,
        decoder=decoder,
        preview=False,
    )

    assert summary is not None
    assert summary.track_id == track.id
    assert summary.frames == latent_result.frame_count
    assert summary.video_path.exists()
    assert summary.latents_path.exists()
    assert summary.checksum == "sha256:deadbeef"
    assert "render_sec" in summary.timings
    assert decoder.calls == [((frames, 8, 16, 16), config.decoder.batch_size)]
    assert summary.applied_preset == "drift"
    assert summary.preset_metadata is not None
