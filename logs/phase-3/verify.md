# Phase 3 — Verification Checklist

- [ ] Run `python renderer/render_album.py --config render.yaml --dry-run` and confirm CLI enumerates tracks without errors.
- [ ] Execute a short render on CPU to verify PyTorch checkpoint loading (`renderer/decoder.py`).
- [ ] Inspect generated metadata (`renders/<run>/run.json`) for seeds, presets, and decoder provider info.
- [ ] Confirm docs (DESIGN.md, PLAN.md, RENDER_GUIDE.md) reflect GAN + Python runtime scope.

> Human supervisor: complete the checks above once offline renderer validation passes. Update `PLAN.md` Phase 3 checklist accordingly.
