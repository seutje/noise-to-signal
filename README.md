# noise-to-signal

Offline GAN-driven album visualizer. This project renders a fixed album into precomputed videos by
mapping audio features to a compact latent space and decoding frames with a trained generator.
The workflow is deterministic: given the same audio, config, and seeds, the render output is
reproducible.

## What it does

- Generates abstract video for each track and an optional stitched album cut.
- Extracts deterministic audio features and drives latent trajectories.
- Decodes frames with a PyTorch Lightning GAN checkpoint (GPU optional).
- Applies offline post-processing and streams frames to FFmpeg for encoding.
- Logs metadata, checksums, and timings for auditability.

## Repository layout

- `renderer/` Python rendering pipeline (CLI, controller, decoder, postfx, FFmpeg writer).
- `training/` dataset generation and GAN training tools.
- `album/` source audio (Git LFS).
- `data/` generated training images (Git LFS).
- `models/` checkpoints, anchors, and metadata (Git LFS).
- `renders/` output videos, previews, and run metadata.
- `cache/` cached audio features and intermediate artifacts.
- `tests/` pytest coverage for renderer modules.
- `logs/` phase logs and issue tracking.

## Requirements

- Python 3.10+
- FFmpeg in PATH
- Git LFS for large assets (audio, images, checkpoints)
- Optional CUDA GPU for faster decoding

After cloning, fetch LFS assets:

```bash
git lfs pull
```

Install renderer dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r renderer/requirements.txt
```

For training and full rendering (includes torch, diffusers, Lightning, etc.):

```bash
pip install -r training/requirements.txt -r renderer/requirements.txt
```

## Quick start (rendering)

1. Ensure album audio exists under `album/` and the generator checkpoint is available
   (default: `training/outputs/checkpoints/gan-best.ckpt`).
2. Copy or edit the root `render.yaml` to match your track list and presets.
3. Dry-run to validate configuration:

```bash
python -m renderer.render_album --config render.yaml --dry-run
```

4. Render the full album:

```bash
python -m renderer.render_album --config render.yaml
```

Render a single track or override a config value:

```bash
python -m renderer.render_album --config render.yaml --track null-hypothesis
python -m renderer.render_album --config render.yaml --set controller.wander_seed=99
```

Outputs land in `renders/<run-id>/` with `run.json`, previews, and per-track videos.
Feature caches are stored in `cache/features/`.

## Configuration

- Base configuration: `render.yaml`
- Presets: `renderer/presets/*.yaml`
- Model metadata and anchors: `models/meta.json`

See `RENDER_GUIDE.md` for the full feature schema, CLI flags, and metadata format.

## Training pipeline (optional)

Generate dataset images (Stable Diffusion) and train the GAN locally:

```bash
python training/sd_generate.py \
  --prompts training/prompts.yaml \
  --output-root data/sd15_abstract \
  --images-per-bucket 1000 \
  --model runwayml/stable-diffusion-v1-5 \
  --precision fp16

python training/train_gan.py --data-root data/sd15_abstract
```

Export the generator to ONNX (optional):

```bash
python -m training.export_onnx --checkpoint training/outputs/checkpoints/gan-best.ckpt
```

Training details, architecture, and design intent live in `DESIGN.md`.

## Testing and linting

```bash
pytest
ruff check renderer tests
black --check renderer tests
```

## Project docs

- `DESIGN.md`: architecture and pipeline details
- `PLAN.md`: phased delivery plan and verification status
- `RENDER_GUIDE.md`: renderer usage details
- `AGENTS.md`: collaboration rules and logging protocol
- `DEVLOG.md`: append-only development log
