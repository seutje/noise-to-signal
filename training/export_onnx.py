"""Export the trained GAN generator to ONNX."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch

from .train_gan import GANLightning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained GAN generator to ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to Lightning checkpoint (.ckpt).")
    parser.add_argument("--output-dir", type=Path, default=Path("models"), help="Directory for exported models.")
    parser.add_argument("--onnx-name", type=str, default="generator.fp32.onnx", help="Filename for ONNX generator.")
    parser.add_argument("--meta-name", type=str, default="meta.json", help="Metadata filename.")
    parser.add_argument("--use-ema", action="store_true", help="Export EMA weights if available.")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version.")
    parser.add_argument("--dynamic-batch", action="store_true", help="Enable dynamic batch dimension.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for dummy latent input.")
    return parser.parse_args()


def export_generator(
    model: GANLightning,
    output_path: Path,
    opset: int,
    dynamic_batch: bool,
    seed: int,
) -> Dict[str, Any]:
    generator = model.generator.eval()
    latent = model.latent
    torch.manual_seed(seed)
    dummy = torch.randn(1, latent.channels, latent.height, latent.width, dtype=torch.float32)

    input_names = ["latent"]
    output_names = ["image"]
    dynamic_axes = {"latent": {0: "batch"}, "image": {0: "batch"}} if dynamic_batch else None

    torch.onnx.export(
        generator.float(),
        dummy,
        output_path,
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )

    return {
        "latent_shape": [latent.channels, latent.height, latent.width],
        "output_path": str(output_path),
        "opset": opset,
        "dynamic_axes": bool(dynamic_axes),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = GANLightning.load_from_checkpoint(args.checkpoint, map_location="cpu")
    if hasattr(model.generator, "use_checkpoint"):
        model.generator.use_checkpoint = False
    if args.use_ema and model.generator_ema is not None:
        model.generator = model.generator_ema

    output_path = args.output_dir / args.onnx_name
    metadata = export_generator(model, output_path, args.opset, args.dynamic_batch, args.seed)

    payload = {
        "version": 2,
        "checkpoint": str(args.checkpoint),
        "image_resolution": model.hparams.image_resolution,
        "latent_shape": metadata["latent_shape"],
        "onnx_generator": metadata["output_path"],
        "dynamic_axes": metadata["dynamic_axes"],
        "opset": metadata["opset"],
        "ema_exported": bool(args.use_ema and model.generator_ema is not None),
        "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
    }
    meta_path = args.output_dir / args.meta_name
    meta_path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
