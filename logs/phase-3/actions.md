# Phase 3 — Actions

- 2025-11-01: Re-scoped runtime plan from browser/WebGPU to offline Python renderer in line with DESIGN.md §7 updates.
- 2025-11-01: Removed legacy web assets (`site/*`, vendored ORT/Three.js, npm manifests) to simplify repository footprint.
- 2025-11-01: Replaced ONNX Runtime loader with direct PyTorch Lightning checkpoint loader (`renderer/decoder.py`).
- 2025-11-01: Updated render configuration defaults to point at GAN checkpoints and PyTorch execution (`render.yaml`).
- 2025-11-01: Adjusted documentation (DESIGN.md, PLAN.md, RENDER_GUIDE.md) to reflect the GAN + Python runtime focus.
- 2025-11-01: Created verification notes clarifying new fp32-only export path and removal of quantization tooling.
