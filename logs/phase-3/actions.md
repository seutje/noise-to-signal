# Phase 3 — Actions

- 2025-11-01: Scaffolded static site shell (`site/index.html`, `site/styles.css`) with playback controls, renderer toggles, and recording UI per DESIGN.md §7.
- 2025-11-01: Implemented web runtime modules (`site/app.js`, `audio.js`, `features.js`, `controller.js`, `utils.js`) covering ONNX session init, audio analysis, latent control, and loop orchestration.
- 2025-11-01: Added dual visualization backends (`site/viz.canvas.js`, `site/viz.three.js`) with CDN Three.js integration, post-processing, and quality scaling.
- 2025-11-01: Wired MediaRecorder capture (`site/recorder.js`) and manual audio ingest helpers (file loader + internal test tone) to unblock validation ahead of playlist delivery (PLAN.md §3 & §4).
- 2025-11-01: Created Phase 3 logging placeholders (`logs/phase-3/*`) and updated DEVLOG with execution summary.
- 2025-11-01: Generated float32 decoder fallback (`models/decoder.fp32.onnx` + `meta.json`) and updated loader to prioritize INT8→FP32→FP16 for broader browser support.
- 2025-11-01: Narrowed runtime provider selection to WebGPU→WASM (skipping WebGL) to avoid InstanceNormalization kernel shape constraints in `onnxruntime-web` (`site/utils.js`).
