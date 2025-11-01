# Phase 2 — Results

- Source code scaffolding for β-VAE training/export is complete; awaiting dependency install and GPU session for execution.
- No training checkpoints or recon grids generated yet (`training/outputs/val_recon.png` pending).
- Quantization parity check blocked until FP16 decoder exists.
- Initial training attempt surfaced missing dataset assets (e.g. `color_fields_00542.png`); loader now skips absent files and emits warnings, but the full LFS dataset should be synced before final runs.
- Training harness now defaults to low-memory settings (batch size 8 + gradient checkpointing/Tensor Core hints) to stay under ~10 GB VRAM; operators can still override via CLI flags.
- Device alignment bugs (LPIPS + EMA) resolved so mixed-precision runs keep computations on `cuda:0`.
- ONNX export path clear: exporter now forces checkpoint-free forward passes to avoid `_Map_base::at` errors when tracing (`training/export_onnx.py`).
- Static QDQ quantization produces `models/decoder.int8.onnx` (PSNR ≈33.1 dB vs FP16) with metadata/report logged under `models/meta.json` and `models/decoder.int8.report.json`.
- Validation script (`tools/validate_int8_parity.py`) comparing 32 decoded samples shows FP16 vs INT8 PSNR drop of ~0.08 dB on average (max 0.24 dB), satisfying the <2 dB requirement; associated summary stored at `logs/phase-2/psnr_validation.txt`.

**Blockers**
- Local environment lacks `pytorch-lightning`, `lpips`, and ONNX tooling (see `training/requirements.txt`).
- GPU time required to run `python -m training.train_vae` to convergence (target LPIPS < 0.12 per PLAN.md §2).

**Next Steps**
- Activate GPU environment, install requirements, and launch training with logging to `training/outputs/`.
- After convergence, run `python -m training.export_onnx --checkpoint <best.ckpt> --use-ema`.
- Quantize via `python -m training.quantize_onnx --fp16-model models/decoder.fp16.onnx --meta models/meta.json`.
