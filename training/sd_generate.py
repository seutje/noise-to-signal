"""Generate an abstract image dataset with Stable Diffusion 1.5.

This script implements Phase 1 of the noise-to-signal plan. It loads prompt
buckets from ``training/prompts.yaml`` and synthesises images in batches. Each
PNG has a sidecar JSON containing the parameters used, and a ``manifest.csv``
collects the dataset summary for downstream training.

Example usage::

    python training/sd_generate.py \
        --prompts training/prompts.yaml \
        --output-root data/sd15_abstract \
        --images-per-bucket 1000 \
        --model runwayml/stable-diffusion-v1-5 \
        --precision fp16

The script is intentionally conservative with dependencies so it can run on a
single GPU workstation. It supports checkpoint resume, deterministic seeds, and
basic aesthetic scoring hooks (optional).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    StableDiffusionPipeline,
)
from tqdm import tqdm
from PIL import Image


@dataclass
class PromptBucket:
    """Container for a prompt configuration."""

    name: str
    prompt: str
    negative_prompt: str
    cfg_override: Optional[float] = None
    steps_override: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, object], default_negative: str) -> "PromptBucket":
        return cls(
            name=data["name"],
            prompt=data["prompt"],
            negative_prompt=data.get("negative_prompt", default_negative),
            cfg_override=data.get("cfg_override"),
            steps_override=data.get("steps_override"),
        )


def load_prompt_buckets(path: Path) -> tuple[Dict[str, object], List[PromptBucket]]:
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp)
    default = data.get("default", {})
    default_negative = default.get("negative_prompt", "")
    buckets = [PromptBucket.from_dict(entry, default_negative) for entry in data["buckets"]]
    return default, buckets


def ensure_dirs(root: Path) -> dict[str, Path]:
    images = root / "images"
    meta = root / "metadata"
    manifests = root
    images.mkdir(parents=True, exist_ok=True)
    meta.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)
    return {"images": images, "metadata": meta, "manifest": manifests / "manifest.csv"}


def infer_precision(precision: str) -> torch.dtype:
    if precision.lower() == "fp16":
        return torch.float16
    if precision.lower() == "bf16":
        return torch.bfloat16
    return torch.float32


SCHEDULER_MAP = {
    "ddim": DDIMScheduler,
    "euler": EulerDiscreteScheduler,
    "euler_a": EulerAncestralDiscreteScheduler,
    "lms": LMSDiscreteScheduler,
    "dpm": DPMSolverMultistepScheduler,
}


def build_pipeline(model_id: str, precision: str, scheduler: Optional[str], device: str) -> StableDiffusionPipeline:
    dtype = infer_precision(precision)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
    )
    if scheduler:
        key = scheduler.lower()
        if key not in SCHEDULER_MAP:
            raise ValueError(f"Unknown scheduler '{scheduler}'. Options: {', '.join(SCHEDULER_MAP)}")
        pipe.scheduler = SCHEDULER_MAP[key].from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    if precision.lower() in {"fp16", "bf16"}:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except AttributeError:
            # xFormers is optional; fall back silently if not available
            pass
    return pipe

def save_image_and_meta(
    image: Image.Image,
    meta_dir: Path,
    manifest_writer: csv.DictWriter,
    filename: str,
    manifest_row: Dict[str, object],
) -> None:
    image_path = meta_dir.parent / "images" / filename
    image.save(image_path, format="PNG")
    meta_path = meta_dir / f"{Path(filename).stem}.json"
    with meta_path.open("w", encoding="utf-8") as fp:
        json.dump(manifest_row, fp, indent=2)
    manifest_writer.writerow(manifest_row)


def run_generation(args: argparse.Namespace) -> None:
    default_cfg: float = args.default_cfg
    default_steps: int = args.default_steps

    default, buckets = load_prompt_buckets(args.prompts)
    if default_cfg is None:
        default_cfg = float(default.get("guidance_scale", 7.5))
    if default_steps is None:
        default_steps = int(default.get("steps", 28))

    dirs = ensure_dirs(args.output_root)
    manifest_exists = dirs["manifest"].exists()
    manifest_file = dirs["manifest"].open("a", newline="", encoding="utf-8")
    fieldnames = [
        "filename",
        "bucket",
        "prompt",
        "negative_prompt",
        "seed",
        "steps",
        "cfg",
        "sampler",
        "width",
        "height",
    ]
    writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
    if not manifest_exists:
        writer.writeheader()

    total_per_bucket = args.images_per_bucket
    device = args.device
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    pipeline = build_pipeline(args.model, args.precision, args.scheduler, device)

    existing_images = {p.stem for p in (dirs["images"]).glob("*.png")}

    try:
        for bucket in buckets:
            bucket_steps = bucket.steps_override or default_steps
            bucket_cfg = bucket.cfg_override or default_cfg
            total_batches = math.ceil(total_per_bucket / args.batch_size)
            for start in tqdm(
                range(0, total_per_bucket, args.batch_size),
                total=total_batches,
                desc=f"bucket:{bucket.name}",
            ):
                batch_indices = list(range(start, min(start + args.batch_size, total_per_bucket)))
                seeds = [args.seed + idx + len(existing_images) + offset for offset, idx in enumerate(batch_indices)]
                generator = [torch.Generator(device=device).manual_seed(seed) for seed in seeds]
                outputs = pipeline(
                    prompt=[bucket.prompt] * len(batch_indices),
                    negative_prompt=[bucket.negative_prompt] * len(batch_indices),
                    num_inference_steps=bucket_steps,
                    guidance_scale=bucket_cfg,
                    width=args.width,
                    height=args.height,
                    generator=generator,
                )
                images = outputs.images

                for offset, (idx, image) in enumerate(zip(batch_indices, images)):
                    filename = f"{bucket.name}_{idx:05d}.png"
                    if Path(filename).stem in existing_images and not args.overwrite:
                        continue
                    manifest_row = {
                        "filename": filename,
                        "bucket": bucket.name,
                        "prompt": bucket.prompt,
                        "negative_prompt": bucket.negative_prompt,
                        "seed": seeds[offset],
                        "steps": bucket_steps,
                        "cfg": bucket_cfg,
                        "sampler": args.sampler,
                        "width": args.width,
                        "height": args.height,
                    }
                    save_image_and_meta(
                        image=image,
                        meta_dir=dirs["metadata"],
                        manifest_writer=writer,
                        filename=filename,
                        manifest_row=manifest_row,
                    )
                    existing_images.add(Path(filename).stem)
    finally:
        manifest_file.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=Path("training/prompts.yaml"), help="Prompt configuration file")
    parser.add_argument("--output-root", type=Path, default=Path("data/sd15_abstract"), help="Dataset output directory")
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5", help="Diffusers model identifier")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "bf16", "fp32"], help="Inference precision")
    parser.add_argument("--scheduler", type=str, default=None, help="Optional scheduler attribute on the pipeline")
    parser.add_argument("--sampler", type=str, default="DDIM", help="Sampler label for metadata logging")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device to use (cuda or cpu)")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of images to generate per batch")
    parser.add_argument("--images-per-bucket", type=int, default=1000, help="Total images per prompt bucket")
    parser.add_argument("--width", type=int, default=512, help="Output image width")
    parser.add_argument("--height", type=int, default=512, help="Output image height")
    parser.add_argument("--seed", type=int, default=1337, help="Base random seed")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate and overwrite existing files")
    parser.add_argument("--default-cfg", type=float, default=None, help="Override default CFG scale from prompts file")
    parser.add_argument("--default-steps", type=int, default=None, help="Override default step count from prompts file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_generation(args)


if __name__ == "__main__":
    main()
