# noise-to-signal Renderer Guide

## 1. Overview
The renderer translates offline audio tracks into latent trajectories for the β-VAE decoder. It performs three primary steps:

1. **Feature Extraction** – Deterministic metrics from source audio (`renderer/audio_features.py`).
2. **Latent Control** – Anchor blending, smoothing, and wander layers (`renderer/controller.py`).
3. **Decoding & PostFX** – ONNX Runtime decoder batches latent tensors and applies tone mapping / vignette / grain (`renderer/decoder.py`, `renderer/postfx.py`).
4. **Encoding & Packaging** – `renderer/frame_writer.py` streams frames to FFmpeg, emits previews, and `renderer/render_album.py` orchestrates album concatenation.

## 2. Configuration Files

- `render.yaml`: Entry configuration (copy of `renderer/templates/render.yaml`).
- `renderer/presets/*.yaml`: Partial overlays merged via `--preset`.
- `models/meta.json`: Contains latent shape and anchor set references (`models/anchors/*.npz`).
- `runtime.workers`: Optional track-level parallelism (set >1 for CPU-only renders).

Use `python -m renderer.render_album --list-presets` to print preset overlays and anchor sets.

## 3. Audio Feature Schema

The feature cache saved at `cache/features/<track>.npz` contains the following arrays (all `float32`):

| Key        | Shape           | Description |
|------------|-----------------|-------------|
| `times`    | `(F,)`          | Frame timestamps in seconds (hop=512, frame=2048).
| `rms`      | `(F,)`          | Root-mean-square energy, normalized to `[0, 1]`.
| `centroid` | `(F,)`          | Spectral centroid (Hz).
| `flatness` | `(F,)`          | Spectral flatness metric `[0, 1]`.
| `mfcc`     | `(13, F)`       | MFCC-lite representation (first 13 coefficients).
| `onset`    | `(F,)`          | Normalized onset strength envelope `[0, 1]`.
| `metadata` | JSON string     | SHA-256 checksum of the source audio, sample rate, and layout metadata.

Constants:
- Sample rate: configurable via `render.yaml` (`audio.sample_rate`, default 48000).
- Frame length: 2048 samples.
- Hop length: 512 samples.

Caches are invalidated automatically when the audio checksum changes.

## 4. Latent Controller

`renderer/controller.py` consumes cached features and outputs latent tensors shaped `(frames, 8, 16, 16)`:

- Anchors are loaded from `models/anchors/<name>.npz` as referenced in `models/meta.json` (`anchor_sets`).
- Anchor blending uses a softmax projection seeded by the controller seed (`controller.wander_seed`) and optional per-track `seed`.
- Wander noise is deterministic and driven by RMS + centroid metrics.
- Tempo sync nudges wander phase on onset spikes or after the configured subdivision interval.
- Outputs are persisted to `<run>/track-id/latents.npz` containing `latents`, `weights`, `frame_times`, and metadata.

Determinism is validated via `tests/test_controller.py`.

## 5. Run Metadata

Invoking `python -m renderer.render_album` creates `<output_root>/run.json` with:

```json
{
  "run_id": "album-demo",
  "created_at": "2025-11-01T12:34:56Z",
  "output_root": "renders/album-demo",
  "config": { ... },
  "controller_seed": 42,
  "tracks": [
    {
      "id": "opening",
      "anchor_set": "drift",
      "frames": 7215,
      "duration_seconds": 120.25,
      "latents": "renders/album-demo/opening/latents.npz",
      "video": "renders/album-demo/opening/video.mp4",
      "preview_still": "renders/album-demo/opening/previews/preview.png",
      "features_cache": "cache/features/opening.npz",
      "feature_checksum": "...",
      "video_checksum": "...",
      "decoder_provider": "CUDAExecutionProvider",
      "decoder_precision": "int8",
      "timings": {
        "features_sec": 8.23,
        "controller_sec": 2.11,
        "render_sec": 64.5
      }
    }
  ]
}
```

This metadata now captures decoder/provider selections, timing metrics, and checksums so renders remain reproducible across reruns and audits.

## 6. CLI Usage

```
python -m renderer.render_album --config render.yaml --dry-run
python -m renderer.render_album --config render.yaml --preset pulse --set controller.wander_seed=99
python -m renderer.render_album --config render.yaml --track opening --verbose --keep-frames
python -m renderer.render_album --config render.yaml --ffmpeg /usr/local/bin/ffmpeg --no-previews
python -m renderer.render_album --config render.yaml --workers 3  # CPU-only parallel tracks
```

Dry runs emit the planned actions without touching caches. Real runs populate feature caches, latent trajectories, per-track MP4s, preview assets, and `run.json`. Use `--keep-frames` to retain intermediate PNG sequences and `--no-previews` to skip GIF/PNG generation for batch jobs.
When setting `runtime.workers` (or `--workers`), parallel track rendering is enabled for CPU execution. CUDA runs remain single-process to keep decoder ownership exclusive to one GPU context.
Large worker counts multiply decoder memory usage (each worker loads its own PyTorch checkpoint). If the OS kills a worker, reduce `runtime.workers` or lower `decoder.batch_size` and rerun.

## 7. Next Steps (Phases 5-6)

- Expand tests to cover decoder batching, postFX determinism, and render orchestration.
- Wire CI for linting, pytest coverage, and artifact upload (`PLAN.md` §5).
- Finalise packaging/licensing documentation ahead of release (`PLAN.md` §6).
