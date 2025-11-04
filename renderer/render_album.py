"""
Album-level orchestration entry point for the noise-to-signal renderer.

Usage:
    python -m renderer.render_album --config render.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed, Future
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Iterable, List, Optional

from . import PACKAGE_ROOT
from .config_schema import RenderConfig, TrackConfig, load_render_config
from .controller import get_anchor_sets
from .decoder import DecoderSession, DecoderUnavailableError
from .frame_writer import FFMpegError, concat_videos
from .render_track import TrackRenderSummary, render_track


LOG = logging.getLogger("renderer.album")
_WORKER_DECODER: DecoderSession | None = None
_WORKER_LOGGING_SETUP: bool = False
WorkerPayload = tuple[
    RenderConfig,
    TrackConfig,
    Path,
    dict[str, object],
    str,
    bool,
    bool,
    int,
]


def _select_mp_context(execution_provider: str):
    """
    Choose an appropriate multiprocessing context for worker pools.

    CPU pools can use fork/forkserver on POSIX systems to share decoder memory.
    CUDA runs stick to spawn to avoid inheriting GPU contexts.
    """
    provider = execution_provider.lower()
    candidates: list[str] = []
    if provider == "cpu" and os.name != "nt":
        # Prefer fork for copy-on-write sharing; fall back gracefully if unavailable.
        candidates.extend(["fork", "forkserver"])
    candidates.append("spawn")

    last_error: Exception | None = None
    for method in candidates:
        try:
            return get_context(method)
        except ValueError as exc:  # pragma: no cover - depends on platform
            last_error = exc
            continue
    if last_error is not None:
        LOG.debug("Falling back to default multiprocessing context due to: %s", last_error)
    return get_context()


def _get_worker_decoder(
    decoder_kwargs: dict[str, object],
    *,
    log_level: int,
) -> DecoderSession:
    """
    Lazily instantiate a decoder per worker process.
    """
    global _WORKER_DECODER, _WORKER_LOGGING_SETUP
    if not _WORKER_LOGGING_SETUP:
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
        _WORKER_LOGGING_SETUP = True
    if _WORKER_DECODER is None:
        _WORKER_DECODER = DecoderSession(
            execution_provider=str(decoder_kwargs["execution_provider"]),
            batch_size=int(decoder_kwargs["batch_size"]),
            checkpoint_path=Path(decoder_kwargs["checkpoint_path"]),
            use_ema=bool(decoder_kwargs["use_ema"]),
        )
    return _WORKER_DECODER


def _render_track_worker(args: WorkerPayload) -> TrackRenderSummary:
    (
        base_config,
        track,
        output_dir,
        decoder_kwargs,
        ffmpeg_path,
        keep_frames,
        preview,
        log_level,
    ) = args
    decoder = _get_worker_decoder(decoder_kwargs, log_level=log_level)
    # Each process receives its own dataclass copies, so safe to mutate locally.
    return render_track(
        track,
        base_config,
        output_dir,
        decoder=decoder,
        ffmpeg_path=ffmpeg_path,
        keep_frames=keep_frames,
        preview=preview,
    )


def _dispatch_parallel_tracks(
    config: RenderConfig,
    decoder_kwargs: dict[str, object],
    *,
    workers: int,
    ffmpeg_path: str,
    keep_frames: bool,
    preview: bool,
    log_level: int,
) -> tuple[List[TrackRenderSummary], List[Path]]:
    """
    Render all tracks in parallel worker processes.
    """
    results: dict[str, TrackRenderSummary] = {}
    execution_provider = str(decoder_kwargs.get("execution_provider", "cpu"))
    ctx = _select_mp_context(execution_provider)
    LOG.debug("Using multiprocessing start method '%s'", ctx.get_start_method())
    payloads: List[WorkerPayload] = []
    for track in config.tracks:
        track_output = config.output_root / track.id
        track_output.mkdir(parents=True, exist_ok=True)
        payloads.append(
            (
                config,
                track,
                track_output,
                decoder_kwargs,
                ffmpeg_path,
                keep_frames,
                preview,
                log_level,
            )
        )

    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
            futures: List[Future[TrackRenderSummary]] = [
                executor.submit(_render_track_worker, payload) for payload in payloads
            ]
            try:
                for future in as_completed(futures):
                    summary = future.result()
                    if summary:
                        results[summary.track_id] = summary
            except Exception:
                for pending in futures:
                    pending.cancel()
                raise
    except BrokenProcessPool as exc:
        raise RuntimeError(
            "Parallel track rendering failed because a worker exited unexpectedly. "
            "Reduce 'runtime.workers', lower 'decoder.batch_size', or inspect system logs for OOM/segfault details."
        ) from exc

    summaries: List[TrackRenderSummary] = []
    videos: List[Path] = []
    for track in config.tracks:
        summary = results.get(track.id)
        if summary:
            summaries.append(summary)
            videos.append(summary.video_path)
    return summaries, videos


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
    workers: Optional[int] = None,
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
    requested_workers = workers if workers is not None else config.runtime.workers
    try:
        worker_count = int(requested_workers)
    except (TypeError, ValueError):
        LOG.warning(
            "Invalid worker count '%s'; falling back to a single worker.",
            requested_workers,
        )
        worker_count = 1
    if worker_count < 1:
        LOG.warning("Worker count %s is below 1; using 1.", worker_count)
        worker_count = 1
    max_workers = os.cpu_count() or 1
    if worker_count > max_workers:
        LOG.warning(
            "Requested %d workers exceeds available CPU cores (%d); capping to %d.",
            worker_count,
            max_workers,
            max_workers,
        )
        worker_count = max_workers
    if worker_count > len(config.tracks):
        worker_count = len(config.tracks)
        LOG.debug(
            "Capping workers to number of tracks (%d).",
            worker_count,
        )
    decoder_provider = config.decoder.execution_provider.lower()
    if worker_count > 1 and decoder_provider == "cuda":
        LOG.warning(
            "Parallel track rendering is disabled for CUDA execution; "
            "falling back to a single worker."
        )
        worker_count = 1

    decoder_kwargs = {
        "execution_provider": config.decoder.execution_provider,
        "batch_size": config.decoder.batch_size,
        "checkpoint_path": config.decoder.checkpoint,
        "use_ema": config.decoder.use_ema,
    }

    if worker_count > 1 and len(config.tracks) > 1:
        LOG.info("Parallel rendering enabled with %d workers.", worker_count)
        log_level = logging.DEBUG if verbose else logging.INFO
        summaries, videos = _dispatch_parallel_tracks(
            config,
            decoder_kwargs,
            workers=worker_count,
            ffmpeg_path=ffmpeg_path,
            keep_frames=keep_frames,
            preview=not skip_previews,
            log_level=log_level,
        )
        track_summaries.extend(summaries)
        track_videos.extend(videos)
    else:
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
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes for track rendering.",
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
                "preset": summary.applied_preset,
                "preset_metadata": summary.preset_metadata,
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
        workers=args.workers,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
