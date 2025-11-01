"""
Track-level orchestration for the noise-to-signal renderer.

The `render_track` function covers feature extraction, latent trajectory
generation, and (in later phases) frame decoding + post-processing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .config_schema import RenderConfig, TrackConfig

LOG = logging.getLogger("renderer.track")


@dataclass(slots=True)
class TrackRenderSummary:
    track_id: str
    output_dir: Path
    latents_path: Path
    frames: int
    anchor_set: str
    feature_cache: Path
    feature_checksum: str


def render_track(
    track: TrackConfig,
    config: RenderConfig,
    output_dir: Path,
    *,
    dry_run: bool = False,
) -> TrackRenderSummary | None:
    """
    Render a single track according to the supplied configuration.

    For Phase 3 this covers:
    - Audio feature extraction with caching.
    - Latent trajectory generation and persistence for downstream decoding.
    """
    if dry_run:
        LOG.info("[dry-run] Track %s → %s", track.id, output_dir)
        return None

    from . import audio_features
    from . import controller

    output_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Extracting audio features for track '%s'", track.id)
    feature_result = audio_features.compute_features(
        audio_path=track.src,
        cache_root=Path("cache") / "features",
        track_id=track.id,
        sample_rate=config.audio.sample_rate,
        normalization=config.audio.normalization,
    )

    LOG.info(
        "Generating latent trajectory | track=%s preset=%s",
        track.id,
        track.preset or config.controller.preset,
    )
    controller_instance = controller.LatentController.from_config(
        config=config,
        track=track,
        feature_layout=feature_result.layout,
    )
    latent_result = controller_instance.generate(
        features=feature_result.timeline,
        seeds=controller.SeedConfig(
            controller_seed=config.controller.wander_seed,
            track_seed=track.seed,
        ),
    )

    latent_path = output_dir / "latents.npz"
    controller.save_latent_result(latent_result, latent_path)
    summary = TrackRenderSummary(
        track_id=track.id,
        output_dir=output_dir,
        latents_path=latent_path,
        frames=latent_result.frame_count,
        anchor_set=latent_result.anchor_name,
        feature_cache=feature_result.cache_path,
        feature_checksum=feature_result.checksum,
    )
    LOG.info(
        "Latent trajectory stored",
        extra={
            "track_id": track.id,
            "frames": latent_result.frame_count,
            "anchor": latent_result.anchor_name,
            "latents_path": str(latent_path),
        },
    )
    return summary
