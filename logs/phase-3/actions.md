# Phase 3 — Actions

- 2025-11-01: Scaffolded static site shell (`site/index.html`, `site/styles.css`) with playback controls, renderer toggles, and recording UI per DESIGN.md §7.
- 2025-11-01: Implemented web runtime modules (`site/app.js`, `audio.js`, `features.js`, `controller.js`, `utils.js`) covering ONNX session init, audio analysis, latent control, and loop orchestration.
- 2025-11-01: Added dual visualization backends (`site/viz.canvas.js`, `site/viz.three.js`) with CDN Three.js integration, post-processing, and quality scaling.
- 2025-11-01: Wired MediaRecorder capture (`site/recorder.js`) and manual audio ingest helpers (file loader + internal test tone) to unblock validation ahead of playlist delivery (PLAN.md §3 & §4).
- 2025-11-01: Created Phase 3 logging placeholders (`logs/phase-3/*`) and updated DEVLOG with execution summary.
- 2025-11-01: Generated float32 decoder fallback (`models/decoder.fp32.onnx` + `meta.json`) and updated loader to prioritize INT8→FP32→FP16 for broader browser support.
- 2025-11-01: Narrowed runtime provider selection to WebGPU→WASM (skipping WebGL) to avoid InstanceNormalization kernel shape constraints in `onnxruntime-web` (`site/utils.js`).
- 2025-11-01: Added WASM tuning, decode throttling, and secure-context guidance to highlight performance expectations (`site/utils.js`, `site/app.js`).
- 2025-11-01: Patched viz renderers to avoid reusing detached WASM buffers by passing per-frame Float32Array copies directly into ORT tensors (`site/viz.canvas.js`, `site/viz.three.js`).
- 2025-11-01: Vendored `onnxruntime-web` + `three` into `/site/vendor`, updated `index.html` to load local copies and expose THREE globally for legacy scripts.
- 2025-11-01: Added COOP/COEP meta fallbacks and WebGPU adapter diagnostics (manual request + skip if unavailable) to improve backend selection messaging (`site/index.html`, `site/app.js`, `site/utils.js`).
- 2025-11-01: Enabled WebGL provider fallback when WebGPU is unavailable and guarded WebGPU adapter usage against missing `requestAdapterInfo` support (`site/utils.js`).
