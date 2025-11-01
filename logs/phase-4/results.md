# Phase 4 — Results Summary

- ✅ Renderer now produces per-track MP4s with deterministic PostFX and SHA-256 checksums recorded in `<run>/<track>/summary.json` plus enriched album-level `run.json` metadata.
- ✅ CLI exposes `--keep-frames`, `--ffmpeg`, and `--no-previews`, enabling frame dumps, custom FFmpeg binaries, and headless runs while defaulting to album concatenation output (`album.mp4`).
- ✅ Decoder session negotiates CUDA/CPU providers automatically, reports the active backend, and validates outputs (finite values, proper ranges) before streaming to FFmpeg.
- ✅ FFmpeg writer shutdown now tolerates early-closed stdin handles, preventing post-render `flush` errors observed on long CPU-bound runs.
- ✅ Preview pipeline generates PNG keyframes and animated GIF snippets when `imageio` is available; gracefully degrades when the dependency is absent.
- ✅ `RENDER_GUIDE.md` documents the end-to-end flow and new CLI affordances; `PLAN.md` Phase 4 checkboxes marked complete.
- ✅ `venv/bin/pytest` -> green (3 tests) confirming no regressions in controller determinism.
- ⚠️ Human verification still required: run an actual track render to confirm audiovisual sync, inspect generated previews, and validate album concatenation output + metadata against design targets (see `logs/phase-4/verify.md`).
