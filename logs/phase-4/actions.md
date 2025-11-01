# Phase 4 — Actions

- 2025-11-02: Implemented ONNX decoder wrapper (`renderer/decoder.py`) with provider fallbacks (INT8/FP16/FP32) and runtime validation (+ CUDA→CPU retry logic) per PLAN.md §4.1.
- 2025-11-02: Added vectorised post-processing pipeline (`renderer/postfx.py`) covering tone curves (`filmlog`/`punch`/`pastel`), vignette, chroma shift, grain, and motion trails driven by deterministic seeds.
- 2025-11-02: Built FFmpeg frame writer + preview tooling (`renderer/frame_writer.py`) supporting audio mux, optional `--keep-frames` PNG dumps, preview GIF/PNG exports, and album concatenation helper.
- 2025-11-02: Extended track orchestration to full renders (`renderer/render_track.py`) — decodes latent batches, applies PostFX, writes MP4s, records timings/checksums, and persist per-track `summary.json`.
- 2025-11-02: Upgraded album CLI (`renderer/render_album.py`) with decoder bootstrap, FFmpeg path override, preview toggle, frame retention flag, and album-level concatenation + enriched `run.json` metadata.
- 2025-11-02: Updated documentation (`PLAN.md`, `RENDER_GUIDE.md`) and dependency manifest (`renderer/requirements.txt` → +`imageio`) to reflect Phase 4 capabilities; recorded Phase 4 completion logs.
- 2025-11-02: Ran regression tests via `venv/bin/pytest` (controller suite) to ensure existing deterministic behaviour remains intact.
