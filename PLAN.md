# noise-to-signal — Multi-Phase Development Plan (for AI Agents)

**Purpose:**  
Structured roadmap for AI agents (and human supervisors) to execute and verify the full development of the *noise-to-signal* project — a static, album-based latent-space visualizer using a VAE decoder and real-time audio features.

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
- Export ONNX model for browser inference.

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

## Phase 3 — Web Visualization (Canvas + Three.js)

### Objectives
- Build static frontend that loads ONNX decoder, computes audio features, and visualizes results.
- Integrate Three.js (CDN) for optional WebGL rendering.

### Steps
1. **App Structure**
   - [x] Create `/site/` folder with `index.html`, `app.js`, and supporting modules.
   - [x] Add controls for quality, renderer mode, and recording.

2. **Canvas Renderer**
   - [x] Implement baseline renderer (`viz.canvas.js`) that draws to `<canvas>` via `ImageData`.

3. **Three.js Renderer**
   - [x] Implement WebGL variant (`viz.three.js`) that updates a DataTexture on a quad.
   - [x] Include minimal post-fx (bloom-lite, afterimage).

4. **ONNX Integration**
   - [x] Load decoder via CDN (`onnxruntime-web`) and test WebGPU/WebGL/WASM backends.
   - [ ] Ensure decoding at ≥30 fps (256–384 px).

5. **Recording System**
   - [x] Add `recorder.js` to capture WebM output using MediaRecorder API.

### Deliverable
`/site/` folder containing a self-contained static app (runnable by double-clicking `index.html`).

**Human Verification Checklist**
- [ ] Open `index.html` locally → confirm visuals appear and move with audio.  
- [ ] Switch between Canvas and Three.js renderers.  
- [ ] Confirm controls (play/pause, record, quality, renderer) all work.  
- [ ] Record and play back a short WebM clip successfully.

---

## Phase 4 — Album Playback Integration

### Objectives
- Integrate fixed MP3 playlist (`tracklist.json`).
- Visualize album playback without microphone input.

### Steps
1. **Playlist Setup**
   - [ ] Add `/album/` directory with MP3 files (LFS).  
   - [ ] Create `tracklist.json` with title, src, artist, art, bpm.

2. **Playback UI**
   - [ ] Implement play/pause/prev/next buttons, seek slider, time display.  
   - [ ] Use `<audio>` element bound to WebAudio `AnalyserNode`.

3. **Feature Mapping**
   - [ ] Extract RMS, spectral centroid, flatness, and MFCC-lite from audio buffer.  
   - [ ] Smooth features via EMA (α=0.8–0.95).

4. **Latent Controller**
   - [ ] Map features → latent deltas using small JS MLP or linear transform.  
   - [ ] Enable latent “wander” modulation by loudness.

### Deliverable
Album playback with synchronized visual response.

**Human Verification Checklist**
- [ ] Open app → verify album loads, first track auto-plays.  
- [ ] Switch tracks; visuals change accordingly.  
- [ ] Observe consistent frame rate across songs.  
- [ ] Playback position resumes on refresh (localStorage check).

---

## Phase 5 — Testing & Quality Assurance

### Objectives
- Create automated Jest test suite.
- Validate components (audio, controller, viz, playlist).

### Steps
1. **Unit Tests**
   - [ ] Write tests for `features.js`, `controller.js`, `viz.canvas.js`, and `viz.three.js`.
   - [ ] Mock ORT, Three.js, and Audio elements.

2. **Integration Tests**
   - [ ] Verify end-to-end playlist flow (load → play → next → resume).  
   - [ ] Ensure rendering and decoding pipelines execute without exceptions.

3. **CI/CD Setup**
   - [ ] Configure GitHub Actions workflow (`.github/workflows/ci.yml`).  
   - [ ] Include lint, test, and coverage artifact upload.

### Deliverable
Green CI pipeline with ≥90% Jest coverage.

**Human Verification Checklist**
- [ ] Run `npm test` → all tests pass.  
- [ ] Inspect coverage report (≥90%).  
- [ ] Review CI logs on GitHub Actions.  
- [ ] Manually open app post-build to confirm functional parity.

---

## Phase 6 — Final Packaging & Documentation

### Objectives
- Package project for distribution and archival.  
- Document dataset, model, and audio rights.

### Steps
1. **Licensing**
   - [ ] Add `DATA_LICENSE.md` and `AUDIO_LICENSE.md`.  
   - [ ] Ensure all MP3s and generated data have redistributable licenses.

2. **Repo Cleanup**
   - [ ] Confirm all large assets tracked by Git LFS.  
   - [ ] Remove temporary files and logs.

3. **Documentation**
   - [ ] Finalize `DESIGN.md` and this `PLAN.md`.  
   - [ ] Add `README.md` with usage, installation, and credits.

### Deliverable
A fully documented Git repository ready for public release.

**Human Verification Checklist**
- [ ] Open `README.md` → confirm setup and usage instructions.  
- [ ] Spot-check LICENSE files for accuracy.  
- [ ] Ensure repo clone succeeds with partial LFS checkout.  
- [ ] Confirm app runs standalone (no build, no network beyond CDN).

---

## Phase 7 — Optional Extensions

### Potential Enhancements
- Add beat detection or BPM-synced latent oscillation.  
- Introduce additional post-fx in Three.js (glitch, lens distortion).  
- Export latent evolution as JSON for each track.  
- Implement procedural dataset generator to replace SD 1.5.

**Human Verification Checklist**
- [ ] Each extension isolated in its own branch.  
- [ ] Visual output remains performant and aesthetic.  
- [ ] Regression tests remain green.

---

*End of PLAN.md.*
