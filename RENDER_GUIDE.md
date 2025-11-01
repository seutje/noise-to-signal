# noise-to-signal Renderer Guide

## 1. Overview
The renderer translates offline audio tracks into latent trajectories for the β-VAE decoder. It performs three primary steps:

1. **Feature Extraction** – Deterministic metrics from source audio (`renderer/audio_features.py`).
2. **Latent Control** – Anchor blending, smoothing, and wander layers (`renderer/controller.py`).
3. **Rendering Orchestration** – CLI entry points that manage presets, run metadata, and downstream decoding (`renderer/render_album.py`, `renderer/render_track.py`).

Future phases will attach ONNX decoding, post-processing, and video encoding.

## 2. Configuration Files

- `render.yaml`: Entry configuration (copy of `renderer/templates/render.yaml`).
- `renderer/presets/*.yaml`: Partial overlays merged via `--preset`.
- `models/meta.json`: Contains latent shape and anchor set references (`models/anchors/*.npz`).

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
      "latents": "renders/album-demo/opening/latents.npz",
      "features_cache": "cache/features/opening.npz",
      "feature_checksum": "..."
    }
  ]
}
```

This metadata acts as the provenance record for downstream phases (decoder, postFX, encoding).

## 6. CLI Usage

```
python -m renderer.render_album --config render.yaml --dry-run
python -m renderer.render_album --config render.yaml --preset pulse --set controller.wander_seed=99
python -m renderer.render_album --config render.yaml --track opening --verbose
```

Dry runs emit the planned actions without touching caches. Real runs populate feature caches, latent trajectories, and `run.json`.

## 7. Next Steps (Phases 4-6)

- Integrate ONNX decoder and FFmpeg frame writer.
- Expand tests to cover decoder batching and render orchestration.
- Update documentation (`README.md`, `PLAN.md`) after human verification of each phase.

