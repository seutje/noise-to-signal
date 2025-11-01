# noise-to-signal — Multi-Phase Development Plan (for AI Agents)

**Purpose:**  
Structured roadmap for AI agents (and human supervisors) to execute and verify the full development of the *noise-to-signal* project — a Python-based offline renderer that prerenders album visuals using a VAE decoder driven by audio analysis.

---

## Phase 1 — Environment & Dataset Preparation

### Objectives
- Set up local development and training environments.
- Generate abstract dataset using Stable Diffusion 1.5 (SD 1.5).
- Store outputs and metadata in the repository under Git LFS.

### Steps
1. **Environment Setup**
   - [x] Install Python ≥ 3.10, CUDA 12.x, PyTorch ≥ 2.2, diffusers, transformers, accelerate.
   - [x] Verify GPU availability (RTX 4070 recommended) and fp16 support.
   - [x] Initialize Git LFS and confirm `.gitattributes` tracks `*.onnx`, `*.png`, and `*.mp3`.

2. **Prompt Configuration**
   - [x] Create `training/prompts.yaml` with 4–5 style buckets (abstract, glitch, color-field, noise, organic).
   - [x] Write negative prompts to filter text/logos/faces.

3. **Image Generation**
   - [x] Run `training/sd_generate.py` to produce 5,000 512×512 abstract images.
   - [x] Store sidecar JSON with seed, steps, CFG, and prompt text.
   - [x] Export `manifest.csv` (filename, prompt, seed, cfg, steps).

4. **Quality Filtering**
   - [x] Run CLIP-based aesthetic scoring to remove low-quality outputs.
   - [x] Check 10% random sample manually for diversity and clarity.

### Deliverable
`/data/sd15_abstract/` with 5,000 curated PNGs and complete `manifest.csv`.

**Human Verification Checklist**
- [x] Randomly inspect 50 images → confirm they are abstract and varied.  
- [x] Check total count and size.  
- [x] Confirm manifest CSV matches image filenames and JSON metadata.  
- [x] Run dataset license audit (ensure all generated).

---

## Phase 2 — Model Training (β-VAE)

### Objectives
- Train and validate a β-VAE on the generated dataset.
- Export ONNX model for offline inference.

### Steps
1. **Model Definition**
   - [x] Implement encoder/decoder in `training/models.py` (ResBlocks, GroupNorm, SiLU).
   - [x] Ensure latent dimension (8×16×16).

2. **Training Execution**
   - [x] Use `train_vae.py` (PyTorch Lightning).  
   - [x] Loss: L1 + LPIPS + β-KL (β=3.0).  
   - [x] Enable AMP and gradient checkpointing.
   - [x] Train to convergence (target LPIPS < 0.12).

3. **Validation & Visualization**
   - [x] Generate reconstructions and latent interpolations.
   - [x] Export comparison grids to `/training/outputs/val_recon.png`.

4. **Export & Quantization**
   - [x] Convert model to ONNX with `export_onnx.py` (opset ≥ 18, dynamic shapes).  
   - [x] Quantize to INT8 via `quantize_onnx.py`.  
   - [x] Validate parity between FP16 and INT8 outputs (PSNR drop < 2 dB).

### Deliverable
`/models/decoder.int8.onnx`, `/models/decoder.fp16.onnx`, and `meta.json`.

**Human Verification Checklist**
- [x] Inspect reconstructions visually for variety and smooth interpolation.  
- [x] Confirm ONNX files exist and load without errors in `onnxruntime`.  
- [x] Compare FP16 vs. INT8 results for visual parity.  
- [x] Check training log for stable loss convergence.

---

## Phase 3 — Renderer Pipeline Foundations

### Objectives
- Establish Python renderer package structure and CLI entry points.
- Implement deterministic audio feature extraction with caching.
- Build latent controller that maps features to VAE trajectories.

### Steps
1. **Renderer Scaffolding**
   - [x] Create `/renderer/` package with `render_album.py`, `render_track.py`, and `config_schema.py`.  
   - [x] Define `render.yaml` template plus preset overrides under `/renderer/presets/`.  
   - [x] Implement CLI (`python -m renderer.render_album --help`) with argument parsing.

2. **Audio Feature Extraction**
   - [x] Implement `audio_features.py` using `librosa`/`torchaudio` to compute RMS, centroid, flatness, MFCC-lite, onset strength.  
   - [x] Add caching to `/cache/features/<track>.npz` and checksum validation vs. source audio.  
   - [x] Document feature schema in `RENDER_GUIDE.md`.

3. **Latent Controller**
   - [x] Implement `controller.py` with anchor blending, EMA smoothing, wander noise, and tempo sync support.  
   - [x] Validate controller determinism under fixed seeds (unit tests).  
   - [x] Expose preset definitions referencing anchors stored in `models/meta.json`.

4. **Configuration & Logging**
   - [x] Add structured logging (JSON or text) for feature extraction and controller stages.  
   - [x] Persist run metadata seeds/config to `/renders/<run-id>/run.json`.  
   - [x] Update `DEVLOG.md` with foundation progress.

### Deliverable
`/renderer` package capable of computing cached features and generating latent trajectories from configuration files.

**Human Verification Checklist**
- [x] Run `python renderer/render_album.py --config render.yaml --dry-run` → CLI lists planned actions without errors.  
- [x] Inspect generated feature cache for one track (shape, dtype, timestamps).  
- [x] Confirm controller unit tests demonstrate deterministic outputs.  
- [x] Review `run.json` metadata for completeness (seeds, presets, timestamps).

---

## Phase 4 — Rendering & Post-Processing

### Objectives
- Decode latent trajectories into frames via ONNX Runtime.
- Pipe frames through post-processing and encode with FFmpeg.
- Generate deterministic per-track renders and stitched album video.

### Steps
1. **Decoder Integration**
   - [x] Implement `decoder.py` wrapper supporting CUDA and CPU execution providers with configurable batch size.  
   - [x] Add retry/fallback logic when GPU memory exhausted (auto-switch to CPU or reduced batch).  
   - [x] Validate decoded frames against reference latents (numerical checks).

2. **PostFX Pipeline**
   - [x] Implement `postfx.py` for tone curves, vignette, chroma shift, grain, and optional motion trails.  
   - [x] Ensure operations are vectorized (numpy/CuPy) and deterministic with seeds.  
   - [x] Provide preset-driven parameter sets stored alongside controller presets.

3. **Encoding & Previews**
   - [x] Implement `frame_writer.py` streaming raw frames to FFmpeg (pipe) with audio mux.  
   - [x] Support optional intermediate image sequence output for debugging (`--keep-frames`).  
   - [x] Generate preview PNGs and animated GIF/WebM snippets for each track.

4. **Full Track & Album Renders**
   - [x] Render at least one full-length track end-to-end (features → MP4) and record timings.  
   - [x] Implement album concatenation pipeline (`ffmpeg -f concat`) with metadata copy.  
   - [x] Store render logs, timings, and checksums under `/renders/<run-id>/`.

### Deliverable
Per-track MP4 renders plus a concatenated album video with associated metadata, previews, and logs.

**Human Verification Checklist**
- [x] Review one rendered track for audiovisual sync and artifact-free visuals.  
- [x] Confirm MP4 metadata (duration, resolution, bitrate) meets targets.  
- [x] Inspect previews directory for representative frames/GIFs.  
- [x] Validate `run.json` logs final status and timing information.

---

## Phase 5 — Testing & Quality Assurance

### Objectives
- Build automated Pytest suite covering feature extraction, controller logic, decoder batching, and rendering orchestration.
- Establish CI pipeline for linting, tests, and coverage.

### Steps
1. **Unit Tests**
   - [x] Add Pytest coverage for `audio_features.py`, `controller.py`, `decoder.py`, and `postfx.py` using deterministic fixtures.  
   - [x] Create mocks/stubs for ONNX Runtime and FFmpeg pipelines to avoid heavy dependencies in tests.  
   - [x] Ensure tests validate seed determinism and numerical tolerances.

2. **Integration Tests**
   - [x] Implement `tests/test_render_track.py` to render a 5-second fixture using mock FFmpeg and verify metadata output.  
   - [x] Add `tests/test_render_album.py` to exercise resume/eject path and concatenation manifest generation.

3. **CI/CD Setup**
   - [x] Configure GitHub Actions workflow (`.github/workflows/ci.yml`) with Python matrix (CPU).  
   - [x] Run `ruff`, `black --check`, and `pytest --cov`.  
   - [x] Upload coverage XML/HTML as build artifacts and ensure LFS checkout works.

### Deliverable
Green CI pipeline with ≥85% Pytest coverage and linting gates.

**Human Verification Checklist**
- [x] Run `pytest --cov` locally → all tests pass, coverage ≥85%.  
- [x] Inspect CI run on GitHub Actions for lint/test success.  
- [x] Review coverage report to confirm critical modules covered.  
- [x] Spot-check test fixtures to ensure no real FFmpeg/ONNX execution in CI.

---

## Phase 6 — Final Packaging & Documentation

### Objectives
- Package project for distribution and archival.  
- Document dataset, model, audio rights, and rendering workflow.

### Steps
1. **Licensing**
   - [ ] Add `DATA_LICENSE.md` and `AUDIO_LICENSE.md`.  
   - [ ] Ensure all audio and generated assets have redistributable licenses and attribution notes.

2. **Repo Cleanup**
   - [ ] Confirm all large assets tracked by Git LFS.  
   - [ ] Remove temporary caches/logs or move them under `/archive/`.

3. **Documentation**
   - [ ] Finalize `DESIGN.md`, `PLAN.md`, and `AGENTS.md` to reflect completed work.  
   - [ ] Update `README.md` with environment setup, render instructions, and troubleshooting.  
   - [ ] Create/refresh `RENDER_GUIDE.md` detailing presets, config options, and FFmpeg requirements.

### Deliverable
A fully documented Git repository ready for public release.

**Human Verification Checklist**
- [ ] Open `README.md` → confirm setup and usage instructions.  
- [ ] Spot-check LICENSE files for accuracy.  
- [ ] Ensure repo clone succeeds with partial LFS checkout.  
- [ ] Execute sample render command to verify offline pipeline works end-to-end.

---

## Phase 7 — Optional Extensions

### Potential Enhancements
- Add beat-detected camera moves or spline-based latent choreography.  
- Integrate super-resolution upscaling pass (e.g., latent SR) for 4K renders.  
- Export per-track latent timelines and feature logs for archival/analysis.  
- Implement procedural dataset generator to replace SD 1.5.

**Human Verification Checklist**
- [ ] Each extension isolated in its own branch.  
- [ ] Visual output remains performant and aesthetic.  
- [ ] Regression tests remain green.

---

*End of PLAN.md.*
