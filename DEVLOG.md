2025-10-31 06:38 UTC — BuildAgent
Phase 1 scaffolding completed: added Stable Diffusion generation script (`training/sd_generate.py`), prompt buckets (`training/prompts.yaml`), dependency manifest, and Git LFS configuration. Logs recorded under `logs/phase-1/`. Dataset synthesis deferred to GPU environment per PLAN.md §1.
2025-10-31 18:57 UTC — TrainingAgent
Executed PLAN.md §2 scaffolding: implemented β-VAE model + Lightning harness (`training/models.py`, `training/train_vae.py`), ONNX export/quantization utilities, and phase logging. Awaiting GPU-enabled environment to run training, export checkpoints, and complete verification tasks.
2025-10-31 19:08 UTC — TrainingAgent
First GPU run surfaced missing dataset files (`data/sd15_abstract/color_fields_00542.png`). Patched loader to skip absent entries with warnings (`training/train_vae.py`) and noted in `logs/phase-2/results.md`. Recommend syncing full Git LFS dataset before next training attempt. Cross-ref: PLAN.md §2.
2025-10-31 19:15 UTC — TrainingAgent
Addressed CUDA OOM on 10 GB-class GPUs by reducing default batch size to 8, enabling optional gradient checkpointing inside the β-VAE blocks, and nudging Tensor Core precision (`training/models.py`, `training/train_vae.py`). Documented memory-safe defaults in `logs/phase-2/results.md`. Cross-ref: PLAN.md §2.
2025-10-31 21:03 UTC — TrainingAgent
Fixed device mismatch runtime (LPIPS + EMA shadows) by moving EMA buffers to match the decoder and ensuring LPIPS/EMA always sync to the active device. Training should now proceed without CPU/GPU mixing errors. See `training/models.py`, `training/train_vae.py`, and `logs/phase-2/results.md`. Cross-ref: PLAN.md §2.
2025-11-01 00:56 UTC — TrainingAgent
Resolved ONNX export crash by disabling gradient checkpointing when loading saved models (`training/export_onnx.py`), ensuring tracing succeeds for decoder export. Cross-ref: PLAN.md §2.
2025-11-01 00:59 UTC — TrainingAgent
Patched ONNX quantization shim to align with latest API (`optimize_model` flag removed), so INT8 export runs without TypeError. See `training/quantize_onnx.py`. Cross-ref: PLAN.md §2.
2025-11-01 01:00 UTC — TrainingAgent
Quantization now auto-converts the FP16 decoder back to FP32 prior to dynamic INT8 pass to satisfy ONNX Runtime's input requirements (`training/quantize_onnx.py`). Cross-ref: PLAN.md §2.
2025-11-01 01:19 UTC — TrainingAgent
Switched to static QDQ quantization with random latent calibration, yielding `models/decoder.int8.onnx` (PSNR≈33.1 dB vs FP16). Metadata/report updated accordingly. See `training/quantize_onnx.py`, `models/meta.json`. Cross-ref: PLAN.md §2.
2025-11-01 16:20 UTC — WebAgent
Phase 3: Delivered static web visualizer scaffold under `/site/` (Canvas + Three.js renderers, audio feature pipeline, latent controller, MediaRecorder). Added manual audio ingest (file picker + test tone) pending album playlist, and recorded outcomes in `logs/phase-3/`. Cross-ref: PLAN.md §3, DESIGN.md §7.
2025-11-01 18:05 UTC — WebAgent
Addressed browser bootstrap failure (`FLOAT16` unsupported) by generating `models/decoder.fp32.onnx`, augmenting `meta.json`, and updating the runtime loader to prefer INT8 → FP32 → FP16. Cross-ref: PLAN.md §3, logs/phase-3/actions.md.
2025-11-01 18:32 UTC — WebAgent
Mitigated ONNX Runtime WebGL `InstanceNormalization` shape errors by skipping the WebGL provider and falling back from WebGPU to WASM only (`site/utils.js`). Cross-ref: logs/phase-3/actions.md.
