from __future__ import annotations

import numpy as np
import pytest

from renderer.audio_features import FeatureLayout, FeatureTimeline
from renderer.config_schema import (
    AudioConfig,
    ControllerConfig,
    DecoderConfig,
    PostFXConfig,
    RenderConfig,
    TrackConfig,
)
from renderer.controller import LatentController, SeedConfig


def _make_config(tmp_path) -> tuple[RenderConfig, TrackConfig]:
    track = TrackConfig(
        id="fixture",
        src=tmp_path / "fixture.wav",
        preset="default",
    )
    checkpoint = tmp_path / "vae-best.ckpt"
    checkpoint.write_text("checkpoint")
    config = RenderConfig(
        output_root=tmp_path / "renders",
        frame_rate=60,
        resolution=[1920, 1088],
        audio=AudioConfig(sample_rate=48_000, normalization=-14.0),
        controller=ControllerConfig(preset="default", smoothing_alpha=0.9, wander_seed=42),
        decoder=DecoderConfig(batch_size=4, execution_provider="cpu", checkpoint=checkpoint),
        postfx=PostFXConfig(),
        tracks=[track],
    )
    config.validate()
    return config, track


def _make_timeline() -> FeatureTimeline:
    frames = 120
    times = np.linspace(0, 2.0, frames, dtype=np.float32)
    rms = np.linspace(0.1, 0.9, frames, dtype=np.float32)
    centroid = np.linspace(200.0, 5000.0, frames, dtype=np.float32)
    flatness = np.linspace(0.1, 0.8, frames, dtype=np.float32)
    mfcc = np.vstack(
        [
            np.sin(times * (i + 1)) * 5 for i in range(13)
        ]
    ).astype(np.float32)
    onset = np.maximum(0.0, np.sin(times * np.pi * 2))  # repeating pulses
    return FeatureTimeline(
        times=times,
        rms=rms,
        centroid=centroid,
        flatness=flatness,
        mfcc=mfcc,
        onset=onset,
    )


def test_latent_controller_deterministic(tmp_path):
    config, track = _make_config(tmp_path)
    layout = FeatureLayout(sample_rate=config.audio.sample_rate)
    timeline = _make_timeline()

    controller = LatentController.from_config(
        config=config,
        track=track,
        feature_layout=layout,
    )
    seeds = SeedConfig(controller_seed=1337, track_seed=7)

    result_a = controller.generate(features=timeline, seeds=seeds)
    result_b = controller.generate(features=timeline, seeds=seeds)

    assert np.allclose(result_a.latents, result_b.latents)
    assert np.allclose(result_a.weights.sum(axis=1), 1.0, atol=1e-4)


@pytest.mark.parametrize(
    "seed_pair",
    [
        (42, 1),
        (43, 1),
    ],
)
def test_latent_controller_seed_variation(tmp_path, seed_pair):
    controller_seed, track_seed = seed_pair
    config, track = _make_config(tmp_path)
    layout = FeatureLayout(sample_rate=config.audio.sample_rate)
    timeline = _make_timeline()

    controller = LatentController.from_config(
        config=config,
        track=track,
        feature_layout=layout,
    )

    result_one = controller.generate(
        features=timeline,
        seeds=SeedConfig(controller_seed=controller_seed, track_seed=track_seed),
    )
    result_two = controller.generate(
        features=timeline,
        seeds=SeedConfig(controller_seed=controller_seed + 1, track_seed=track_seed),
    )

    assert not np.allclose(result_one.latents, result_two.latents)
