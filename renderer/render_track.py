"""
Track-level orchestration for the noise-to-signal renderer.

The `render_track` function covers feature extraction, latent trajectory
generation, decoder execution, post-processing, and FFmpeg packaging.
"""

from __future__ import annotations
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config_schema import RenderConfig, TrackConfig, resolve_track_config
from .decoder import DecoderSession
from .frame_writer import FFMpegWriter, PreviewBundle, compute_sha256
from .postfx import PostFXProcessor, apply_postfx

LOG = logging.getLogger("renderer.track")


@dataclass(slots=True)
class TrackRenderSummary:
    track_id: str
    output_dir: Path
    latents_path: Path
    video_path: Path
    frames: int
    duration: float
    anchor_set: str
    feature_cache: Path
    feature_checksum: str
    preview_still: Optional[Path]
    preview_anim: Optional[Path]
    checksum: str
    decode_provider: str
    decode_precision: str
    timings: dict[str, float]
    applied_preset: Optional[str] = None
    preset_metadata: Optional[dict[str, object]] = None


def render_track(
    track: TrackConfig,
    config: RenderConfig,
    output_dir: Path,
    *,
    decoder: DecoderSession,
    ffmpeg_path: str = "ffmpeg",
    keep_frames: bool = False,
    preview: bool = True,
    dry_run: bool = False,
) -> TrackRenderSummary | None:
    """
    Render a single track according to the supplied configuration.

    Pipeline steps:
    - Audio feature extraction with caching.
    - Latent trajectory generation and persistence for downstream decoding.
    - Batched decoder execution with PostFX + FFmpeg streaming.
    """
    if dry_run:
        LOG.info("[dry-run] Track %s → %s", track.id, output_dir)
        return None

    from . import audio_features
    from . import controller

    output_dir.mkdir(parents=True, exist_ok=True)

    active_config, preset_metadata = resolve_track_config(config, track)

    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    LOG.info("Extracting audio features for track '%s'", track.id)
    feature_result = audio_features.compute_features(
        audio_path=track.src,
        cache_root=Path("cache") / "features",
        track_id=track.id,
        sample_rate=active_config.audio.sample_rate,
        normalization=active_config.audio.normalization,
    )
    timings["features_sec"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    LOG.info(
        "Generating latent trajectory | track=%s preset=%s",
        track.id,
        track.preset or active_config.controller.preset,
    )
    controller_instance = controller.LatentController.from_config(
        config=active_config,
        track=track,
        feature_layout=feature_result.layout,
    )
    latent_result = controller_instance.generate(
        features=feature_result.timeline,
        seeds=controller.SeedConfig(
            controller_seed=active_config.controller.wander_seed,
            track_seed=track.seed,
        ),
    )
    timings["controller_sec"] = time.perf_counter() - t1

    latent_path = output_dir / "latents.npz"
    controller.save_latent_result(latent_result, latent_path)

    postfx_seed = active_config.controller.wander_seed ^ (track.seed or 0)
    processor = PostFXProcessor(
        config=active_config.postfx,
        resolution=(active_config.resolution[0], active_config.resolution[1]),
        seed=int(postfx_seed),
    )

    video_path = output_dir / "video.mp4"
    preview_dir = output_dir / "previews" if preview else None
    writer = FFMpegWriter(
        output_path=video_path,
        frame_rate=active_config.frame_rate,
        resolution=active_config.resolution,
        audio_path=track.src,
        ffmpeg_path=ffmpeg_path,
        trim_start=track.trim.start,
        trim_end=track.trim.end,
        keep_frames=keep_frames,
        preview_dir=preview_dir,
    )

    render_start = time.perf_counter()
    frames_total = latent_result.frame_count
    batch_size = max(1, active_config.decoder.batch_size)
    for start in range(0, frames_total, batch_size):
        end = min(frames_total, start + batch_size)
        latents_chunk = latent_result.latents[start:end]
        decoded = decoder.decode(
            latents_chunk,
            batch_size=active_config.decoder.batch_size,
            validate=True,
        )
        graded = apply_postfx(decoded, processor=processor)
        writer.write_batch(graded)

    preview_bundle = writer.close()
    timings["render_sec"] = time.perf_counter() - render_start

    checksum = compute_sha256(video_path)

    track_metadata = {
        "track_id": track.id,
        "frames": latent_result.frame_count,
        "duration_sec": writer.duration,
        "video": str(video_path),
        "preview_still": str(preview_bundle.still) if preview_bundle.still else None,
        "preview_anim": str(preview_bundle.animated) if preview_bundle.animated else None,
        "feature_cache": str(feature_result.cache_path),
        "feature_checksum": feature_result.checksum,
        "latents_path": str(latent_path),
        "decoder_provider": decoder.current_provider,
        "decoder_precision": decoder.precision,
        "timings": timings,
        "checksum_sha256": checksum,
        "preset": track.preset,
        "controller": {
            "preset": active_config.controller.preset,
            "smoothing_alpha": active_config.controller.smoothing_alpha,
            "wander_seed": active_config.controller.wander_seed,
            "anchor_set": active_config.controller.anchor_set,
            "tempo_sync": {
                "enabled": active_config.controller.tempo_sync.enabled,
                "subdivision": active_config.controller.tempo_sync.subdivision,
            },
        },
        "postfx": {
            "tone_curve": active_config.postfx.tone_curve,
            "grain_intensity": active_config.postfx.grain_intensity,
            "chroma_shift": active_config.postfx.chroma_shift,
            "vignette_strength": active_config.postfx.vignette_strength,
            "motion_trails": active_config.postfx.motion_trails,
        },
        "preset_metadata": preset_metadata or {},
    }
    (output_dir / "summary.json").write_text(json.dumps(track_metadata, indent=2))

    summary = TrackRenderSummary(
        track_id=track.id,
        output_dir=output_dir,
        latents_path=latent_path,
        video_path=video_path,
        frames=latent_result.frame_count,
        duration=writer.duration,
        anchor_set=latent_result.anchor_name,
        feature_cache=feature_result.cache_path,
        feature_checksum=feature_result.checksum,
        preview_still=preview_bundle.still,
        preview_anim=preview_bundle.animated,
        checksum=checksum,
        decode_provider=decoder.current_provider,
        decode_precision=decoder.precision,
        timings=timings,
        applied_preset=track.preset,
        preset_metadata=preset_metadata or None,
    )
    LOG.info(
        "Track render completed",
        extra={
            "track_id": track.id,
            "frames": latent_result.frame_count,
            "anchor": latent_result.anchor_name,
            "latents_path": str(latent_path),
            "video_path": str(video_path),
            "provider": decoder.current_provider,
        },
    )
    return summary
