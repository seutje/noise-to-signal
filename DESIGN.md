# noise-to-signal — VAE Album Visualizer (Design Doc)

**Author:**  
**Date:** October 31, 2025  
**Status:** Draft v2.0 (album playlist edition; Three.js via CDN; static-only runtime)

---

## 1) Product Goal

A single-page, static web app (no server, no build step) that visualizes a **predefined album** (a fixed set of MP3 tracks bundled with the project). It renders abstract, “dream-like” visuals that respond to the currently playing track by decoding and interpolating within a compact image **VAE** latent space trained on abstract imagery.

**Project name:** `noise-to-signal`

**Key outcomes**
- **Runs anywhere** as plain static files (e.g., GitHub Pages, local file).  
- **Album-first experience:** A bundled playlist of MP3s plays in sequence with minimal UI, creating a cohesive visual album.  
- **Real-time (30–60 fps)** on mid-tier consumer hardware at 256–384 px; optional 512 px.  
- **Minimal runtime deps (CDN):** `onnxruntime-web` (required) and **optionally** `three` (Three.js) from a CDN.  
- **No React/TypeScript; no build step** to run.  
- **All assets in repo:** training **data images** and final **model (ONNX)** are committed via **Git LFS** (plus the album MP3s you own the rights to).  
- **Automated tests:** **Jest** (node + jsdom + mocks) with ≥90% coverage.  
- **Dataset generation:** Local script using **Stable Diffusion 1.5** (SD 1.5) to synthesize training images.

> **Non-goals:** Back-end servers, runtime package managers. Development-time tooling (PyTorch, diffusers, Jest, Node-based CLIs) is allowed.

---

## 2) System Overview

```
[ Prompts + Seeds ] --(SD 1.5 / diffusers)--> [ 5k Training Images (+ JSON meta) ]

[ 5k Images ] --(augment)--> [ VAE Training (PyTorch/Lightning) ]
                              | losses: L1 + LPIPS + β-KL (β-VAE)
                              v
                       [ Decoder.onnx (fp16/int8) + meta.json ]

[ Predefined Album ]
  /album/*.mp3 + tracklist.json (titles, order, start times, art)

[ Static Web App (no build, CDN deps) ]
  - Load ONNX Runtime Web (WebGPU/WebGL)
  - Optionally load Three.js (CDN)
  - WebAudio features from HTMLAudioElement → latent controller → ONNX decode → render
  - Record to WebM, presets, export latent-control JSON
```

---

## 3) Requirements

### 3.1 Functional
- **Album playback:** Play a predefined ordered set of MP3s (loop entire album).  
- Playback UI: **Play/Pause**, **Previous/Next track**, **seek bar**, **elapsed/remaining time**, track title display.  
- **Autoplay sequencing:** auto-advance to next track; remember last position within session (localStorage).  
- **Visualization:** Latent-space decoding modulated by per-frame features from the audio element.  
- **Presets:** “moods” (latent axes weights), sensitivity, smoothing, resolution toggle.  
- **Recording:** Export visual to WebM; export a **latent-control JSON** for deterministic replays.

### 3.2 Non-functional
- **Static-only runtime**; **no build step**; plain ES modules.  
- **Runtime deps via CDN** (or vendored copies): `onnxruntime-web` (required), `three` (optional).  
- **Performance:** 30–60 fps @ 256–384 px on mid-tier GPUs; ≤100 ms latency.  
- **Size:** decoder.onnx ≤ 30–60 MB; dataset ~1–3 GB (LFS); album MP3s as provided.  
- **Tests:** Jest coverage ≥90% statements/branches.

---

## 4) Dataset Generation (Local) — Stable Diffusion 1.5

> **Licensing:** Generated images are generally redistributable; do **not** commit SD 1.5 weights unless permitted. Commit the generated dataset and scripts; fetch weights locally during generation. You must own the rights to the album MP3s or use permissive licenses; document them in `AUDIO_LICENSE.md`.

### 4.1 Environment
Python ≥ 3.10, CUDA 12.x, PyTorch ≥ 2.2, `diffusers`, `transformers`, `safetensors`, `accelerate` (optional `xformers`).

### 4.2 Prompts
Buckets covering abstract, glitch, color-field, reaction–diffusion, monochrome noise; negative prompts exclude text/logos/people/objects. Steps 20–35, CFG 6.5–9.5, 512×512.

### 4.3 Script
`/training/sd_generate.py` produces `PNG` + sidecar JSON + `manifest.csv`. Optional CLIP/aesthetic culling of the worst 10–20%.

### 4.4 Augmentation
Dataloader-time: random crop/resize, color jitter, mild blur/JPEG, small cutout.

---

## 5) Model & Training (β-VAE)

### 5.1 Architecture
- Train at **256** first; optional **512** pass.  
- Latent **z** shape: **(8, 16, 16)**.  
- Encoder/Decoder: ResBlocks, GroupNorm, SiLU, stride-2 downs/ups; decoder `tanh`.

### 5.2 Loss & Optim
- Recon: L1 (λ=10) + LPIPS (λ=1).  
- KL (β=3.0; warm-up over first 10–20% epochs).  
- AdamW (3e-4), EMA of decoder (0.999).  
- 4070 settings: AMP bf16/fp16, batch 32 @ 256; 8–12 @ 512; 50–100 epochs, early stop.

### 5.3 Validation
Recon grids; latent traversals; qualitative checks (LPIPS trend).

---

## 6) Export, Quantization & Metadata

- Export **decoder** to ONNX (`/training/export_onnx.py`, opset ≥ 18, dynamic shapes).  
- Quantize to **INT8** (`/training/quantize_onnx.py`); keep FP16 reference.  
- `models/meta.json` stores latent shape, normalize params, version, checksums, precomputed **anchor latents**, and optional **PCA axes**.

---

## 7) Web App (Static, No Build)

### 7.1 File Layout
```
/album
  tracklist.json          # [{ "src":"01.mp3","title":"…","artist":"…","art":"…"}, …]
  01.mp3 … NN.mp3         # your album files (LFS)
  cover.png               # optional album art
/site
  index.html
  styles.css
  app.js                  # bootstrap + UI wiring + playlist
  audio.js                # HTMLAudioElement + WebAudio analyser
  features.js             # RMS/centroid/flatness/MFCC-lite from analyser
  controller.js           # anchors, EMA, latent walk, tiny MLP
  viz.canvas.js           # 2D Canvas decode & post-fx
  viz.three.js            # Three.js (CDN) texture-plane + shader post-fx
  recorder.js             # MediaRecorder → WebM
  utils.js                # RNG, PCA load, perf, storage
  assets/ui/*             # icons, LUTs
  vendor/                 # optional pinned copies (offline)
    ort.min.js
    three.min.js
/models
  decoder.int8.onnx       # default runtime (LFS)
  decoder.fp16.onnx       # reference (LFS)
  meta.json
/training
  sd_generate.py
  prompts.yaml
  train_vae.py
  models.py
  export_onnx.py
  quantize_onnx.py
/data
  sd15_abstract/*.png     # generated images (LFS)
  manifest.csv
/tests
.github/workflows/ci.yml
AUDIO_LICENSE.md
DATA_LICENSE.md
```

### 7.2 Playlist & Player

**tracklist.json (example):**
```json
[
  { "src": "01-opening.mp3", "title": "Opening", "artist": "noise-to-signal", "art": "cover.png" },
  { "src": "02-interference.mp3", "title": "Interference", "artist": "noise-to-signal" },
  { "src": "03-signal.mp3", "title": "Signal", "artist": "noise-to-signal" }
]
```

**Playback behavior**
- Load `tracklist.json` at startup; build playlist UI (no framework).  
- Create a single hidden `<audio>` element with `crossOrigin="anonymous"` and hook it to a WebAudio `AnalyserNode`.  
- Preload next track’s metadata; optional small buffer warmup.  
- Autoplay sequences track → track; loop album at end; persist `{trackIndex, currentTime}` to `localStorage` every few seconds.  
- User controls: play/pause, prev/next, seek slider, track title/artist display, timecode, optional volume.

**CORS note:** When hosting remotely, the server must allow range requests for MP3 and permit CORS if the origin differs. GitHub Pages serves static with correct headers; local `file://` is fine but MediaRecorder blob save is used for recording, not uploads.

### 7.3 index.html
```html
<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>noise-to-signal</title>
<link rel="stylesheet" href="./styles.css">
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.164.1/build/three.min.js"></script>
<body>
  <header>
    <h1>noise-to-signal</h1>
    <div id="player">
      <button id="prevBtn">⟸</button>
      <button id="playBtn">►</button>
      <button id="nextBtn">⟹</button>
      <input id="seek" type="range" min="0" max="1000" step="1" value="0">
      <span id="time">0:00 / 0:00</span>
      <label>Quality
        <select id="quality">
          <option value="256">256</option>
          <option value="384" selected>384</option>
          <option value="512">512</option>
        </select>
      </label>
      <label>Renderer
        <select id="renderer">
          <option value="canvas" selected>Canvas 2D</option>
          <option value="three">Three.js</option>
        </select>
      </label>
      <label>Sensitivity <input id="sensitivity" type="range" min="0" max="1" step="0.01" value="0.5"></label>
      <label>Smoothing <input id="smoothing" type="range" min="0" max="0.99" step="0.01" value="0.9"></label>
      <button id="recordBtn">● Record</button>
      <span id="fps"></span>
    </div>
  </header>

  <div id="nowPlaying"><img id="art" src="./../album/cover.png" alt="" /><span id="title"></span></div>

  <canvas id="stage2d" width="768" height="768" style="display:block;"></canvas>
  <div id="three-root" style="display:none;"></div>

  <audio id="albumAudio" crossorigin="anonymous"></audio>

  <script type="module" src="./app.js"></script>
</body>
</html>
```

### 7.4 Runtime Flow
1. **Bootstrap**: load `meta.json` and `album/tracklist.json`; init ORT (prefer WebGPU, fallback WebGL/WASM).  
2. **Audio element**: set `src` to current track, `AudioContext.createMediaElementSource(audio)` → `AnalyserNode`.  
3. **Features**: compute RMS, spectral centroid, flatness, rolloff, MFCC-lite from `AnalyserNode` FFT data at ~25–50 Hz; EMA smoothing.  
4. **Controller**: softmax-over-anchors + wander → latent `z` (1×8×16×16).  
5. **Decode**: `viz.canvas.js` (baseline) or `viz.three.js` (optional) updates pixels/texture each frame.  
6. **Player**: update seek/time; handle `ended` → next track; persist position.  
7. **Recorder**: MediaRecorder → WebM save (no upload).

### 7.5 Performance Notes
- Decode at 256–384 internal; upscale via CSS/Three.js quad.  
- INT8 model default; FP16 toggle if fast.  
- Preallocate typed arrays; reuse `ImageData` / `DataTexture`.  
- In Three.js, reuse `WebGLRenderTarget`s; avoid per-frame allocations.

---

## 8) Automated Tests (Jest)

> GPU/Three.js/ORT/Audio are **mocked**; tests run deterministically.

### 8.1 Node-side tests
- `prompts.spec.js` — prompt bucket balance.  
- `sd_params.spec.js` — steps/CFG/seed bounds & determinism.  
- `augment.spec.js` — transforms on synthetic arrays.

### 8.2 Browser-logic tests (JSDOM + mocks)
- `playlist.spec.js` — loads `tracklist.json`, orders tracks, auto-advance, loop, prev/next logic; persistence to localStorage; seek math.  
- `audio.spec.js` — HTMLAudioElement mock wiring; `AnalyserNode` data flow.  
- `features.spec.js` — FFT/MFCC-lite correctness; EMA smoothing; step/impulse tests.  
- `controller.spec.js` — weights sum≈1; z bounds; deterministic under fixed features/seed.  
- `viz.canvas.spec.js` — ORT mock decoding → pixel mapping correct; quality switch updates tensor shapes.  
- `viz.three.spec.js` — Three.js mock initializes once; `DataTexture.needsUpdate` toggles with buffer writes; shader uniforms updated.  
- `app.spec.js` — UI wiring; renderer toggle; play/pause/next/prev; time/seek updates; record start/stop.  
- `interop.spec.js` — meta/model checksum verification; backend fallback chain (WebGPU→WebGL→WASM).

### 8.3 Mocks
- **ORT mock**: `InferenceSession.create` returns object with `run` → deterministic Float32 outputs.  
- **Three.js mock**: constructors for `WebGLRenderer`, `Scene`, `OrthographicCamera`, `PlaneGeometry`, `DataTexture`, `ShaderMaterial`, `WebGLRenderTarget` with call tracking.  
- **WebAudio mock**: HTMLAudioElement events; `AnalyserNode.getByteFrequencyData`/`getFloatTimeDomainData` fed from synthetic buffers.

### 8.4 Coverage & CI
- Target **≥90%** coverage.  
- GitHub Actions workflow runs lint + tests + coverage artifact. LFS checkout enabled for models/data/audio.

---

## 9) Repository & Size Management

- Use **Git LFS** for `*.onnx`, `data/*.png`, `album/*.mp3`, sample `*.webm`.  
- `AUDIO_LICENSE.md` documents rights to the included MP3s (creator, license, attribution if required).  
- `DATA_LICENSE.md` documents generated-image licensing.  
- Provide `CONTRIBUTING.md` with guidance for partial LFS clone to reduce bandwidth.

---

## 10) Security & Privacy

- No microphone required (album-first).  
- All processing local; no network calls after page load (if vendor copies used); otherwise only CDN for ORT/Three.js.  
- No analytics; suggest strict CSP if host allows.

---

## 11) Risk Log & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| CDN drift/outages | App fails to load deps | Pin versions; vendor copies in `/site/vendor/` as offline fallback. |
| WebGPU/GL variance | Perf/compatibility issues | Fallback chain; default Canvas 2D; auto-quality downshift. |
| Three.js post-fx cost | FPS drops | Toggle effects; low-res targets; cap blur passes. |
| VAE blur | Aesthetic | Strong LPIPS; optional small PatchGAN; consider VQ-VAE swap. |
| Large repo | Slow clones | LFS + partial clone docs; cap dataset to 5k @ 512; consider WEBP for previews. |
| Audio rights | Legal | Only include tracks you own or with permissive licenses; document in `AUDIO_LICENSE.md`. |

---

## 12) Milestones

1. **Week 1** — SD dataset (5k), VAE@256, ONNX export, Canvas baseline, playlist loader.  
2. **Week 2** — Audio features from HTMLAudioElement, latent controller, INT8 quant, quality toggle.  
3. **Week 3** — Three.js renderer + minimal shader post-fx; recording; tests ≥90%; CI.  
4. **Week 4** — 512 px model; polish; docs and acceptance checks.

---

## 13) Appendices

### A) `tracklist.json` (full example)
```json
[
  { "src": "01-opening.mp3", "title": "Opening", "artist": "noise-to-signal", "art": "cover.png", "bpm": 120 },
  { "src": "02-interference.mp3", "title": "Interference", "artist": "noise-to-signal", "bpm": 98 },
  { "src": "03-signal.mp3", "title": "Signal", "artist": "noise-to-signal", "bpm": 132 }
]
```

### B) Export & Quantization (CLI)
```bash
# Export decoder to ONNX
python training/export_onnx.py --checkpoint ckpt.pt --out models/decoder.fp16.onnx --opset 18
# Quantize to INT8
python training/quantize_onnx.py --in models/decoder.fp16.onnx --out models/decoder.int8.onnx
```

### C) `app.js` (playlist sketch)
```javascript
import { createFeatureExtractor } from './features.js';
import { createController } from './controller.js';
import { createCanvasViz } from './viz.canvas.js';
import { createThreeViz } from './viz.three.js';

const meta = await (await fetch('./../models/meta.json')).json();
const tracks = await (await fetch('./../album/tracklist.json')).json();

const audio = document.getElementById('albumAudio');
const seek = document.getElementById('seek');
const playBtn = document.getElementById('playBtn');
const prevBtn = document.getElementById('prevBtn');
const nextBtn = document.getElementById('nextBtn');
const title = document.getElementById('title');
const art = document.getElementById('art');

let idx = parseInt(localStorage.getItem('nts_idx') || '0', 10);
let t0 = parseFloat(localStorage.getItem('nts_t') || '0');

function load(i) {
  idx = (i + tracks.length) % tracks.length;
  const tr = tracks[idx];
  audio.src = `./../album/${tr.src}`;
  title.textContent = `${tr.title} — ${tr.artist || 'noise-to-signal'}`;
  if (tr.art) art.src = `./../album/${tr.art}`;
  audio.currentTime = idx === i ? t0 : 0;
  localStorage.setItem('nts_idx', String(idx));
}

playBtn.onclick = () => audio.paused ? audio.play() : audio.pause();
prevBtn.onclick = () => { load(idx - 1); audio.play(); };
nextBtn.onclick = () => { load(idx + 1); audio.play(); };
audio.onended = () => { load(idx + 1); audio.play(); };

audio.ontimeupdate = () => {
  localStorage.setItem('nts_t', audio.currentTime.toFixed(2));
  if (audio.duration) seek.value = Math.floor(1000 * audio.currentTime / audio.duration);
};
seek.oninput = () => {
  if (audio.duration) audio.currentTime = (seek.value / 1000) * audio.duration;
};

// WebAudio analyser
const ctx = new (window.AudioContext || window.webkitAudioContext)();
const src = ctx.createMediaElementSource(audio);
const analyser = ctx.createAnalyser();
analyser.fftSize = 2048;
src.connect(analyser);
analyser.connect(ctx.destination);

const feat = createFeatureExtractor({ analyser });
const ctrl = createController(meta);

const qualitySel = document.getElementById('quality');
const rendererSel = document.getElementById('renderer');
let viz = null;

async function initViz() {
  viz?.dispose?.();
  const size = parseInt(qualitySel.value, 10);
  if (rendererSel.value === 'three' && window.THREE) {
    viz = await createThreeViz(meta, { size });
    document.getElementById('three-root').style.display = 'block';
    document.getElementById('stage2d').style.display = 'none';
  } else {
    viz = await createCanvasViz(meta, { size });
    document.getElementById('three-root').style.display = 'none';
    document.getElementById('stage2d').style.display = 'block';
  }
}
await initViz();
qualitySel.onchange = initViz;
rendererSel.onchange = initViz;

function loop() {
  const f = feat.nextFrame();      // 32-dim-ish features (smoothed)
  const z = ctrl.update(f);        // 1x8x16x16 latent
  viz.render(z);                   // decode & draw
  requestAnimationFrame(loop);
}
load(idx);
loop();
```

### D) Test Mocks
- **ORT**: mock `InferenceSession.create` and `run` to return stable tensors.  
- **Three.js**: stub constructors with call tracking; no GL context needed.  
- **Audio**: mock `HTMLAudioElement` events (`play`, `pause`, `ended`, `timeupdate`) and `AnalyserNode` outputs.

---

## 14) Acceptance Criteria

- `site/index.html` loads and runs **without a build step**.  
- Album playlist loads from `/album/tracklist.json`; tracks play in order with auto-advance and resume-on-refresh.  
- Visualizer responds to playback features; ≥30 fps at 384 px (Canvas and Three.js).  
- `npm test` passes with ≥90% coverage (mocks for ORT/Three/Audio).  
- Repo includes `/data` (5k SD images), `/models` (ONNX + meta), and `/album` (MP3s + tracklist) via **Git LFS**.  
- LICENSE, `DATA_LICENSE.md`, and `AUDIO_LICENSE.md` included.  
- No server, no build; dependencies from CDN or vendored; minimal runtime deps.

---

*End of DESIGN.md.*
