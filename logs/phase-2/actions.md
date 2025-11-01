# Phase 2 — Actions

- 2025-10-31: Implemented β-VAE architecture and Lightning training harness (`training/models.py`, `training/train_vae.py`).
- Added ONNX export and quantization utilities (`training/export_onnx.py`, `training/quantize_onnx.py`) plus updated dependency manifest.
- Created phase log stubs and metadata hooks for decoder exports.
- Added manifest filtering and auto-detection of nested `images/` directory so training skips/locates assets without crashing (`training/train_vae.py`).
- Tuned training defaults for ≤10 GB GPUs: smaller default batch size, automatic Tensor Core precision hint, and optional gradient checkpointing wired through the VAE modules (`training/train_vae.py`, `training/models.py`).
- Hardened EMA handling and LPIPS device placement to prevent CPU/GPU mismatches during AMP training (`training/models.py`, `training/train_vae.py`).
- Updated ONNX export script to disable gradient checkpointing when loading checkpoints, preventing export-time tracing failures (`training/export_onnx.py`).
- Reworked INT8 quantization to convert FP16 → FP32, apply static QDQ quantization with random latent calibration, and update metadata/reporting (`training/quantize_onnx.py`).

> Pending: install training dependencies and execute full GPU training run to produce checkpoints and recon assets.
