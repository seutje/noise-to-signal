# Phase 3 — Results Summary

- ✅ `/site/` now hosts a self-contained static app: `index.html` (UI shell), `app.js` (bootstrap + controls), shared utilities, and modular render/audio/recorder components.
- ✅ Canvas and Three.js renderers share a single ONNX Runtime session with backend fallback (WebGPU → WASM) and dynamic quality scaling (256–512 CSS upscaling).
- ✅ Audio analysis pipeline computes RMS/centroid/flatness, mel-band energies, and MFCC-lite features with adjustable smoothing + sensitivity feeding a multi-anchor latent controller.
- ✅ MediaRecorder support records either renderer to WebM; local file loader and internal test tone unblock audio-driven validation until album assets land in Phase 4.
- ✅ Added float32 decoder fallback so browsers lacking FLOAT16 support can initialize successfully (INT8 → FP32 → FP16 priority).
- ✅ Runtime now prefers WebGPU, then WebGL, and finally WASM (skipping WebGL only when unsupported) and avoids WebGPU adapters that lack `requestAdapterInfo`, improving compatibility; WebGPU/WebGL binaries now load locally from `/site/vendor`, and the app reports adapter/COEP status inline for easier diagnosis.
- ✅ When falling back to WASM, the app now auto-tunes thread count, warns about secure contexts, and reduces decode frequency to prevent UI stalls.
- ✅ Renderer pipeline copies latents per frame to prevent detached ArrayBuffer errors with WASM proxy mode (no more decode crashes).
- ⚠️ Performance/FPS targets rely on browser execution; manual verification required with actual model + album assets.
- ⚠️ Playlist auto-load hooks exist but await `album/tracklist.json` population in Phase 4.
