# Phase 2 — Actions

- 2025-10-31: Implemented GAN generator/discriminator architecture plus Lightning training harness (`training/models.py`, `training/train_gan.py`).
- Added manifest filtering and auto-detection of nested `images/` directory so training skips/locates assets without crashing (`training/train_gan.py`).
- Tuned training defaults for ≤10 GB GPUs: smaller default batch size, gradient checkpointing toggle for the generator, and explicit fp32 precision for stability (`training/train_gan.py`, `training/models.py`).
- Hardened EMA handling and LPIPS device placement to prevent CPU/GPU mismatches during evaluation (`training/models.py`, `training/train_gan.py`).
- Updated ONNX export script to target the generator directly in fp32, removing the old quantization flow (`training/export_onnx.py`).

> Pending: install training dependencies and execute full GPU training run to produce checkpoints and sample grids.
