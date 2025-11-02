"""
Album-level orchestration entry point for the noise-to-signal renderer.

Usage:
    python -m renderer.render_album --config render.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Optional

from datetime import datetime, timezone

from .config_schema import RenderConfig, load_render_config
from . import PACKAGE_ROOT
from .decoder import DecoderSession, DecoderUnavailableError
from .frame_writer import FFMpegError, concat_videos
from .render_track import render_track, TrackRenderSummary
from .controller import get_anchor_sets


LOG = logging.getLogger("renderer.album")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _parse_set_overrides(values: Optional[List[str]]) -> dict[str, object]:
    if not values:
        return {}
    overrides: dict[str, object] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must use KEY=VALUE format.")
        key, raw_value = item.split("=", 1)
        try:
            # Allow YAML parsing for convenience (numbers, booleans, lists)
            value = load_yaml_value(raw_value)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Failed to parse override '{item}': {exc}") from exc
        overrides[key.strip()] = value
    return overrides


def load_yaml_value(text: str) -> object:
    """Parse a single YAML value (used for CLI overrides)."""
    import yaml

    return yaml.safe_load(text)


def _filter_tracks(config: RenderConfig, include: Optional[Iterable[str]]) -> RenderConfig:
    if not include:
        return config
    include_set = {track_id.strip() for track_id in include}
    filtered_tracks = [track for track in config.tracks if track.id in include_set]
    if not filtered_tracks:
        raise ValueError(
            f"No matching tracks for filter {sorted(include_set)}. "
            f"Available tracks: {[t.id for t in config.tracks]}"
        )
    config.tracks = filtered_tracks
    return config


def render_album(
    config: RenderConfig,
    *,
    dry_run: bool = False,
    verbose: bool = False,
    keep_frames: bool = False,
    ffmpeg_path: str = "ffmpeg",
    skip_previews: bool = False,
) -> None:
    """Render each track defined in the configuration."""
    _setup_logging(verbose)
    LOG.info(
        "Starting album render | %s",
        config.describe(),
    )

    if dry_run:
        LOG.info("Dry run enabled; no renders will be executed.")
        for track in config.tracks:
            LOG.info(
                "Track plan | id=%s src=%s preset=%s trim=%s",
                track.id,
                track.src,
                track.preset or config.controller.preset,
                track.trim.to_dict(),
            )
        return

    config.output_root.mkdir(parents=True, exist_ok=True)

    track_summaries: List[TrackRenderSummary] = []
    track_videos: List[Path] = []

    try:
        decoder = DecoderSession(
            execution_provider=config.decoder.execution_provider,
            batch_size=config.decoder.batch_size,
            checkpoint_path=config.decoder.checkpoint,
            use_ema=config.decoder.use_ema,
        )
    except DecoderUnavailableError as exc:
        LOG.error("Decoder initialisation failed: %s", exc)
        raise

    for track in config.tracks:
        LOG.info("Rendering track '%s' from %s", track.id, track.src)
        track_output = config.output_root / track.id
        track_output.mkdir(parents=True, exist_ok=True)
        summary = render_track(
            track,
            config,
            track_output,
            decoder=decoder,
            ffmpeg_path=ffmpeg_path,
            keep_frames=keep_frames,
            preview=not skip_previews,
        )
        if summary:
            track_summaries.append(summary)
            track_videos.append(summary.video_path)

    if len(track_videos) > 1:
        album_path = config.output_root / "album.mp4"
        try:
            concat_videos(track_videos, output_path=album_path, ffmpeg_path=ffmpeg_path)
            LOG.info("Album concatenation completed → %s", album_path)
        except FFMpegError as exc:
            LOG.warning("Album concatenation failed: %s", exc)

    _write_run_metadata(config, track_summaries)

    LOG.info("Album render completed. Outputs stored in %s", config.output_root)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Noise-to-signal album renderer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("render.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--preset",
        action="append",
        dest="presets",
        help="Apply a named preset overlay (can be specified multiple times).",
    )
    parser.add_argument(
        "--set",
        action="append",
        dest="overrides",
        metavar="KEY=VALUE",
        help="Apply inline overrides using dotted paths, e.g., controller.preset=pulse.",
    )
    parser.add_argument(
        "--track",
        action="append",
        dest="tracks",
        help="Restrict render to specific track ids (repeatable).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        dest="output_root",
        help="Override output root directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without rendering.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Persist intermediate decoded frames to disk.",
    )
    parser.add_argument(
        "--ffmpeg",
        type=str,
        default="ffmpeg",
        help="Path to the ffmpeg executable.",
    )
    parser.add_argument(
        "--no-previews",
        action="store_true",
        help="Disable preview PNG/GIF generation.",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available preset overlays and exit.",
    )
    return parser


def list_available_presets() -> None:
    presets_dir = PACKAGE_ROOT / "presets"
    print("Preset overlays:")
    for path in sorted(presets_dir.glob("*.yaml")):
        print(f"  - {path.stem}")
    print("\nAnchor sets:")
    for name, info in sorted(get_anchor_sets().items()):
        desc = info.get("description", "")
        print(f"  - {name}: {desc}")


def _write_run_metadata(
    config: RenderConfig,
    track_summaries: List[TrackRenderSummary],
) -> None:
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    metadata = {
        "run_id": config.output_root.name,
        "created_at": timestamp,
        "output_root": str(config.output_root),
        "config": config.describe(),
        "controller_seed": config.controller.wander_seed,
        "tracks": [
            {
                "id": summary.track_id,
                "anchor_set": summary.anchor_set,
                "frames": summary.frames,
                "duration_seconds": summary.duration,
                "latents": str(summary.latents_path),
                "video": str(summary.video_path),
                "preview_still": str(summary.preview_still) if summary.preview_still else None,
                "preview_anim": str(summary.preview_anim) if summary.preview_anim else None,
                "features_cache": str(summary.feature_cache),
                "feature_checksum": summary.feature_checksum,
                "video_checksum": summary.checksum,
                "decoder_provider": summary.decode_provider,
                "decoder_precision": summary.decode_precision,
                "timings": summary.timings,
            }
            for summary in track_summaries
        ],
    }
    track_seeds = {
        track.id: track.seed
        for track in config.tracks
        if track.seed is not None
    }
    if track_seeds:
        metadata["track_seeds"] = track_seeds
    album_video = config.output_root / "album.mp4"
    if album_video.exists():
        metadata["album_video"] = str(album_video)
    run_path = config.output_root / "run.json"
    run_path.write_text(json.dumps(metadata, indent=2))
    LOG.info("Run metadata written to %s", run_path)


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.list_presets:
        list_available_presets()
        return

    overrides = _parse_set_overrides(args.overrides)
    if args.output_root:
        overrides["output_root"] = str(args.output_root)

    config = load_render_config(
        args.config,
        extra_presets=args.presets,
        overrides=overrides,
    )

    config = _filter_tracks(config, args.tracks)

    render_album(
        config,
        dry_run=args.dry_run,
        verbose=args.verbose,
        keep_frames=args.keep_frames,
        ffmpeg_path=args.ffmpeg,
        skip_previews=args.no_previews,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
