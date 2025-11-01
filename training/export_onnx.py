import argparse
import json
from pathlib import Path
from typing import Any, Dict

import onnx
import torch
from onnxconverter_common import float16

from .train_vae import BetaVAELightning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained β-VAE decoder to ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to Lightning checkpoint (.ckpt).")
    parser.add_argument("--output-dir", type=Path, default=Path("models"), help="Directory for exported models.")
    parser.add_argument("--fp16-name", type=str, default="decoder.fp16.onnx", help="Filename for FP16 decoder.")
    parser.add_argument("--meta-name", type=str, default="meta.json", help="Metadata filename.")
    parser.add_argument("--use-ema", action="store_true", help="Export EMA weights if available.")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version.")
    parser.add_argument("--dynamic-batch", action="store_true", help="Enable dynamic batch dimension.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for dummy latent input.")
    return parser.parse_args()


def export_decoder(
    model: BetaVAELightning,
    output_path: Path,
    opset: int,
    dynamic_batch: bool,
    seed: int,
) -> Dict[str, Any]:
    decoder = model.model.decoder.eval()
    latent = model.model.latent
    torch.manual_seed(seed)
    dummy = torch.randn(1, latent.channels, latent.height, latent.width, dtype=torch.float32)

    input_names = ["latent"]
    output_names = ["image"]
    dynamic_axes = {"latent": {0: "batch"}, "image": {0: "batch"}} if dynamic_batch else None

    tmp_path = output_path.with_suffix(".tmp.onnx")
    torch.onnx.export(
        decoder.float(),
        dummy,
        tmp_path,
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )

    onnx_model = onnx.load(tmp_path)
    onnx_model_fp16 = float16.convert_float_to_float16(onnx_model, keep_io_types=True)
    onnx.save(onnx_model_fp16, output_path)
    tmp_path.unlink(missing_ok=True)

    return {
        "latent_shape": [latent.channels, latent.height, latent.width],
        "output_path": str(output_path),
        "opset": opset,
        "dynamic_axes": bool(dynamic_axes),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = BetaVAELightning.load_from_checkpoint(args.checkpoint, map_location="cpu")
    # Disable gradient checkpointing for export to avoid autograd graph issues.
    if hasattr(model.model.encoder, "use_checkpoint"):
        model.model.encoder.use_checkpoint = False
    if hasattr(model.model.decoder, "use_checkpoint"):
        model.model.decoder.use_checkpoint = False
    if args.use_ema and model.ema is not None:
        model.ema.copy_to(model.model.decoder)

    output_path = args.output_dir / args.fp16_name
    metadata = export_decoder(model, output_path, args.opset, args.dynamic_batch, args.seed)

    payload = {
        "version": 1,
        "checkpoint": str(args.checkpoint),
        "image_resolution": model.hparams.image_resolution,
        "latent_shape": metadata["latent_shape"],
        "fp16_decoder": metadata["output_path"],
        "dynamic_axes": metadata["dynamic_axes"],
        "opset": metadata["opset"],
        "beta": model.hparams.beta,
        "kl_warmup_epochs": model.hparams.kl_warmup_epochs,
        "ema_exported": bool(args.use_ema and model.ema is not None),
        "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
    }
    meta_path = args.output_dir / args.meta_name
    meta_path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
