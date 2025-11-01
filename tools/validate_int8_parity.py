"""Compute PSNR parity between FP16 and INT8 decoder exports.

Usage:
    python tools/validate_int8_parity.py \
        --manifest data/sd15_abstract/manifest.csv \
        --data-root data/sd15_abstract \
        --checkpoint training/outputs/checkpoints/vae-epoch=077-val_lpips=0.0000.ckpt \
        --fp16 models/decoder.fp16.onnx \
        --int8 models/decoder.int8.onnx

Outputs a summary of PSNR for FP16 vs original, INT8 vs original, and the
per-image drop (INT8 - FP16) to verify the <2 dB requirement.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torchvision import transforms

from training.train_vae import BetaVAELightning, build_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate FP16 ↔ INT8 decoder parity (PSNR drop < 2 dB)")
    parser.add_argument("--manifest", type=Path, default=Path("data/sd15_abstract/manifest.csv"))
    parser.add_argument("--data-root", type=Path, default=Path("data/sd15_abstract"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fp16", type=Path, required=True, help="Path to FP16 ONNX decoder")
    parser.add_argument("--int8", type=Path, required=True, help="Path to INT8 ONNX decoder")
    parser.add_argument("--samples", type=int, default=32, help="Number of dataset samples to evaluate")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def psnr_db(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0.0:
        return float("inf")
    return 10.0 * np.log10((2.0 ** 2) / mse)


def iter_samples(manifest: Path, data_root: Path, count: int) -> Iterable[Tuple[Path, str]]:
    images_dir = data_root / "images"
    base = images_dir if images_dir.is_dir() else data_root
    with manifest.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            if idx >= count:
                break
            yield base / row["filename"], row.get("prompt", "")


def main() -> None:
    args = parse_args()
    lightning = BetaVAELightning.load_from_checkpoint(args.checkpoint, map_location="cpu")
    if lightning.ema is not None:
        lightning.ema.copy_to(lightning.model.decoder)
    lightning.model.encoder.use_checkpoint = False
    lightning.model.decoder.use_checkpoint = False
    lightning.eval()

    fp16_decoder = lightning.model.decoder
    ort_fp16 = ort.InferenceSession(str(args.fp16), providers=["CPUExecutionProvider"])
    ort_int8 = ort.InferenceSession(str(args.int8), providers=["CPUExecutionProvider"])

    transform = build_transforms(args.resolution, train=False)

    fp16_psnr, int8_psnr, drops = [], [], []

    for image_path, _prompt in iter_samples(args.manifest, args.data_root, args.samples):
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            mean, _ = lightning.model.encoder(tensor)
            recon_fp16 = fp16_decoder(mean)
        latent = mean.cpu().numpy().astype(np.float32)
        ort_fp16_out = torch.from_numpy(ort_fp16.run(None, {"latent": latent})[0])
        ort_int8_out = torch.from_numpy(ort_int8.run(None, {"latent": latent})[0])

        fp16_psnr.append(psnr_db(recon_fp16, tensor))
        int8_psnr.append(psnr_db(ort_int8_out, tensor))
        drops.append(fp16_psnr[-1] - int8_psnr[-1])

    print(f"Samples evaluated: {len(fp16_psnr)}")
    print(f"FP16 PSNR (mean ± std): {np.mean(fp16_psnr):.3f} ± {np.std(fp16_psnr):.3f} dB")
    print(f"INT8 PSNR (mean ± std): {np.mean(int8_psnr):.3f} ± {np.std(int8_psnr):.3f} dB")
    print(f"Mean drop: {np.mean(drops):.3f} dB")
    print(f"Max drop: {np.max(drops):.3f} dB")
    print(f"Requirement satisfied (<2 dB): {np.max(drops) < 2.0}")


if __name__ == "__main__":
    main()
