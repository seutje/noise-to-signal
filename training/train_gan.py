"""Lightning training harness for the noise-to-signal GAN."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from lpips import LPIPS
from PIL import Image
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from .models import Discriminator, EMA, Generator, LatentConfig


logger = logging.getLogger(__name__)


def _to_rgb01(x: torch.Tensor) -> torch.Tensor:
    """Convert tensors from [-1, 1] to [0, 1] range."""

    return x * 0.5 + 0.5


def _mean_saturation(rgb: torch.Tensor) -> torch.Tensor:
    """Return batch-averaged HSV saturation for RGB inputs in [0, 1]."""

    c_max, _ = rgb.max(dim=1)
    c_min, _ = rgb.min(dim=1)
    denom = torch.clamp(c_max, min=1e-3)
    saturation = (c_max - c_min) / denom
    return saturation.mean()


def _chroma_components(rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert RGB in [0, 1] to approximate YUV chroma components."""

    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = 0.492 * (b - y)
    v = 0.877 * (r - y)
    return u, v


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GAN on the generated abstract dataset.")
    parser.add_argument("--data-root", type=Path, default=Path("data/sd15_abstract"), help="Root folder with PNGs.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/sd15_abstract/manifest.csv"),
        help="CSV manifest describing dataset entries.",
    )
    parser.add_argument("--output", type=Path, default=Path("training/outputs"), help="Folder for checkpoints and logs.")
    parser.add_argument("--resolution", type=int, default=256, help="Training image resolution.")
    parser.add_argument("--batch-size", type=int, default=8, help="Global batch size per device.")
    parser.add_argument("--num-workers", type=int, default=4, help="Data loader workers.")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs.")
    parser.add_argument("--val-split", type=float, default=0.05, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--generator-lr", type=float, default=1e-5, help="Generator learning rate.")
    parser.add_argument("--discriminator-lr", type=float, default=1e-5, help="Discriminator learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--latent-channels", type=int, default=8, help="Latent channel count.")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay for generator weights.")
    parser.add_argument("--no-ema", action="store_true", help="Disable EMA tracking.")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from lightning checkpoint.")
    parser.add_argument("--override-manifest-limit", type=int, default=None, help="Optional limit for debugging.")
    parser.add_argument(
        "--no-grad-checkpoint",
        action="store_true",
        help="Disable gradient checkpointing for generator blocks (uses more memory).",
    )
    parser.add_argument("--precision", type=str, default="32-true", help="Lightning precision flag.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm value.")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Save checkpoint every N epochs.")
    parser.add_argument("--devices", type=int, default=None, help="Optional override for device count.")
    parser.add_argument("--strategy", type=str, default="auto", help="Lightning training strategy override.")
    parser.add_argument("--accumulate-grad-batches", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--num-samples", type=int, default=8, help="Validation samples saved for image grid.")
    parser.add_argument("--latent-sigma", type=float, default=1.0, help="Standard deviation for latent sampling.")
    parser.add_argument("--r1-gamma", type=float, default=10.0, help="R1 gradient penalty strength.")
    parser.add_argument("--r1-interval", type=int, default=16, help="Apply R1 penalty every N steps.")
    parser.add_argument(
        "--chroma-weight",
        type=float,
        default=1.5,
        help="Weight applied to chroma statistics alignment loss.",
    )
    parser.add_argument(
        "--saturation-weight",
        type=float,
        default=0.75,
        help="Weight applied to saturation preservation penalty.",
    )
    parser.add_argument("--sample-interval", type=int, default=200, help="How often to dump generated sample grids.")
    return parser.parse_args()


def _read_manifest(manifest_path: Path, limit: Optional[int] = None) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with manifest_path.open("r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for idx, row in enumerate(reader):
            rows.append(row)
            if limit is not None and idx + 1 >= limit:
                break
    if not rows:
        raise ValueError(f"No rows read from manifest: {manifest_path}")
    return rows


class ManifestImageDataset(Dataset):
    """Dataset that pairs manifest metadata with on-disk PNG images."""

    def __init__(self, root: Path, manifest: Path, transform: transforms.Compose, limit: Optional[int] = None) -> None:
        self.root = root
        self.search_roots = [self.root]
        image_subdir = self.root / "images"
        if image_subdir.is_dir():
            self.search_roots.append(image_subdir)

        raw_entries = _read_manifest(manifest, limit)
        self.entries: List[Dict[str, Any]] = []
        self.transform = transform

        missing: List[str] = []
        for entry in raw_entries:
            resolved = self._resolve_path(entry["filename"])
            if resolved is None:
                missing.append(entry["filename"])
                continue
            entry_with_path = dict(entry)
            entry_with_path["_path"] = resolved
            self.entries.append(entry_with_path)

        if missing:
            sample = ", ".join(missing[:5])
            logger.warning(
                "ManifestImageDataset: skipped %d missing files (showing up to 5): %s", len(missing), sample
            )

        if not self.entries:
            raise FileNotFoundError(
                f"No valid images found in {manifest}. Check that Git LFS assets are pulled and paths are correct."
            )

    def _resolve_path(self, filename: str) -> Optional[Path]:
        for base in self.search_roots:
            candidate = base / filename
            if candidate.exists():
                return candidate
        return None

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # type: ignore[override]
        entry = self.entries[idx]
        image_path = entry["_path"]
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            image = self.transform(img)
        return {
            "image": image,
            "filename": entry["filename"],
            "prompt": entry.get("prompt", ""),
            "seed": entry.get("seed", ""),
        }


def build_transforms(resolution: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(resolution, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.1, 0.05, 0.05, 0.05),
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=Image.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


class ManifestDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: Path,
        manifest: Path,
        resolution: int,
        batch_size: int,
        num_workers: int,
        val_split: float,
        seed: int,
        limit: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.manifest = manifest
        self.resolution = resolution
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed
        self.limit = limit
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:  # type: ignore[override]
        full_dataset = ManifestImageDataset(
            root=self.data_root,
            manifest=self.manifest,
            transform=build_transforms(self.resolution, train=True),
            limit=self.limit,
        )
        val_dataset = ManifestImageDataset(
            root=self.data_root,
            manifest=self.manifest,
            transform=build_transforms(self.resolution, train=False),
            limit=self.limit,
        )

        total_len = len(full_dataset)
        val_len = max(1, int(total_len * self.val_split))
        train_len = total_len - val_len
        generator = torch.Generator().manual_seed(self.seed)
        permutation = torch.randperm(total_len, generator=generator).tolist()
        val_indices = permutation[:val_len]
        train_indices = permutation[val_len:]

        self.train_dataset = Subset(full_dataset, train_indices)
        self.val_dataset = Subset(val_dataset, val_indices)

    def train_dataloader(self) -> DataLoader:  # type: ignore[override]
        assert self.train_dataset is not None, "setup() must be called before requesting dataloaders."
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:  # type: ignore[override]
        assert self.val_dataset is not None, "setup() must be called before requesting dataloaders."
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )


class GANLightning(pl.LightningModule):
    def __init__(
        self,
        *,
        image_resolution: int,
        generator_lr: float,
        discriminator_lr: float,
        weight_decay: float,
        max_epochs: int,
        latent_channels: int,
        ema_decay: float,
        disable_ema: bool,
        output_dir: Path,
        num_samples: int,
        use_grad_checkpoint: bool,
        latent_sigma: float,
        chroma_weight: float,
        saturation_weight: float,
        r1_gamma: float,
        r1_interval: int,
        sample_interval: int,
        grad_accum_steps: int = 1,
    ) -> None:
        super().__init__()
        latent = LatentConfig(channels=latent_channels, height=16, width=16)
        self.generator = Generator(latent_channels=latent.channels, use_checkpoint=use_grad_checkpoint)
        self.discriminator = Discriminator()
        self.latent = latent
        self.latent_sigma = latent_sigma
        self.automatic_optimization = False
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_samples = num_samples
        self.sample_interval = sample_interval
        self.image_resolution = image_resolution
        self.max_epochs = max_epochs
        self.lpips = LPIPS(net="vgg")
        self.lpips.eval()
        self.lpips.requires_grad_(False)
        self.example_noise: Optional[torch.Tensor] = None
        self.example_real: Optional[torch.Tensor] = None
        self.best_lpips = math.inf
        self.chroma_weight = chroma_weight
        self.saturation_weight = saturation_weight
        self.r1_gamma = r1_gamma
        self.r1_interval = max(1, r1_interval)
        self.ema = None if disable_ema else EMA(self.generator, decay=ema_decay)
        self.generator_ema = None if disable_ema else deepcopy(self.generator).requires_grad_(False)
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self._pending_d_steps = 0
        self._pending_g_steps = 0
        self._non_finite_events = 0

        self.save_hyperparameters(
            {
                "image_resolution": image_resolution,
                "generator_lr": generator_lr,
                "discriminator_lr": discriminator_lr,
                "weight_decay": weight_decay,
                "max_epochs": max_epochs,
                "latent_channels": latent_channels,
                "ema_decay": ema_decay,
                "disable_ema": disable_ema,
                "output_dir": str(output_dir),
                "num_samples": num_samples,
                "use_grad_checkpoint": use_grad_checkpoint,
                "latent_sigma": latent_sigma,
                "chroma_weight": chroma_weight,
                "saturation_weight": saturation_weight,
                "r1_gamma": r1_gamma,
                "r1_interval": self.r1_interval,
                "sample_interval": sample_interval,
                "grad_accum_steps": self.grad_accum_steps,
            }
        )

    def _reset_module_weights(self, module: torch.nn.Module) -> None:
        def _init_fn(m: torch.nn.Module) -> None:
            if hasattr(m, "reset_parameters"):
                try:
                    m.reset_parameters()
                except Exception:
                    pass

        module.apply(_init_fn)

    def _restore_generator_from_ema(self) -> None:
        if self.ema is not None and self.generator_ema is not None:
            self.ema.copy_to(self.generator)

    def _handle_non_finite(
        self,
        *,
        stage: str,
        optimizer: Optional[torch.optim.Optimizer],
    ) -> None:
        self._non_finite_events += 1
        logger.warning("Detected non-finite values during %s step (event #%d).", stage, self._non_finite_events)
        if optimizer is not None:
            optimizer.zero_grad()
        self._restore_generator_from_ema()
        self._reset_module_weights(self.discriminator)

    def configure_optimizers(self) -> List[Dict[str, Any]]:  # type: ignore[override]
        g_opt = torch.optim.AdamW(
            self.generator.parameters(),
            lr=self.hparams.generator_lr,
            betas=(0.5, 0.999),
            weight_decay=self.hparams.weight_decay,
        )
        d_opt = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=self.hparams.discriminator_lr,
            betas=(0.5, 0.999),
            weight_decay=self.hparams.weight_decay,
        )
        sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(g_opt, T_max=self.hparams.max_epochs)
        sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(d_opt, T_max=self.hparams.max_epochs)
        return [
            {"optimizer": g_opt, "lr_scheduler": {"scheduler": sched_g, "interval": "epoch"}},
            {"optimizer": d_opt, "lr_scheduler": {"scheduler": sched_d, "interval": "epoch"}},
        ]

    def _sample_latent(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randn(
            batch_size,
            self.latent.channels,
            self.latent.height,
            self.latent.width,
            device=device,
        ) * self.latent_sigma

    def _current_generator(self) -> Generator:
        if self.generator_ema is not None:
            return self.generator_ema
        return self.generator

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:  # type: ignore[override]
        images = batch["image"]
        batch_size = images.size(0)
        device = images.device
        g_opt, d_opt = self.optimizers()  # type: ignore[assignment]

        # Discriminator update
        self.toggle_optimizer(d_opt)
        if self._pending_d_steps == 0:
            d_opt.zero_grad()
        images.requires_grad_(True)
        real_logits = self.discriminator(images)
        noise = self._sample_latent(batch_size, device)
        fake = self.generator(noise).detach()
        fake_logits = self.discriminator(fake)
        d_loss = F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()
        r1_penalty = torch.tensor(0.0, device=device)
        if self.r1_gamma > 0 and (self.global_step % self.r1_interval == 0):
            grad = torch.autograd.grad(real_logits.sum(), images, create_graph=True)[0]
            grad = grad.view(grad.size(0), -1)
            r1_penalty = grad.pow(2).sum(dim=1).mean()
            d_loss = d_loss + (self.r1_gamma * 0.5) * r1_penalty
        if not torch.isfinite(d_loss):
            self._handle_non_finite(stage="discriminator", optimizer=d_opt)
            self._pending_d_steps = 0
            self.untoggle_optimizer(d_opt)
            images = images.detach()
            return
        scaled_d_loss = d_loss / self.grad_accum_steps
        self.manual_backward(scaled_d_loss)
        self._pending_d_steps += 1
        if self._pending_d_steps >= self.grad_accum_steps:
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.5)
            d_opt.step()
            self._pending_d_steps = 0
        self.untoggle_optimizer(d_opt)
        images = images.detach()

        # Generator update
        self.toggle_optimizer(g_opt)
        if self._pending_g_steps == 0:
            g_opt.zero_grad()
        noise = self._sample_latent(batch_size, device)
        fake = self.generator(noise)
        fake_logits = self.discriminator(fake)
        g_adv = F.softplus(-fake_logits).mean()
        fake_rgb = _to_rgb01(fake)
        real_rgb = _to_rgb01(images)
        sat_fake = _mean_saturation(fake_rgb)
        sat_real = _mean_saturation(real_rgb)
        sat_penalty = torch.relu(sat_real.detach() - sat_fake)
        fake_u, fake_v = _chroma_components(fake_rgb)
        real_u, real_v = _chroma_components(real_rgb)
        chroma_l1 = (
            F.l1_loss(fake_u.mean(dim=0), real_u.mean(dim=0).detach())
            + F.l1_loss(fake_v.mean(dim=0), real_v.mean(dim=0).detach())
        )
        g_loss = g_adv + self.chroma_weight * chroma_l1 + self.saturation_weight * sat_penalty
        if not torch.isfinite(g_loss):
            self._handle_non_finite(stage="generator", optimizer=g_opt)
            self._pending_g_steps = 0
            self.untoggle_optimizer(g_opt)
            return
        scaled_g_loss = g_loss / self.grad_accum_steps
        self.manual_backward(scaled_g_loss)
        self._pending_g_steps += 1
        if self._pending_g_steps >= self.grad_accum_steps:
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 0.5)
            g_opt.step()
            self._pending_g_steps = 0
        self.untoggle_optimizer(g_opt)

        if self._pending_g_steps == 0 and self.ema is not None and self.generator_ema is not None:
            self.ema.update(self.generator)
            self.ema.copy_to(self.generator_ema)
            self.generator_ema = self.generator_ema.to(device)

        if self.example_noise is None or self.example_noise.size(0) != self.num_samples:
            self.example_noise = self._sample_latent(self.num_samples, device).detach().cpu()
        if self.example_real is None:
            self.example_real = images[: self.num_samples].detach().cpu()

        metrics = {
            "train/d_loss": d_loss.detach(),
            "train/g_loss": g_loss.detach(),
            "train/g_adv": g_adv.detach(),
            "train/r1": r1_penalty.detach(),
            "train/sat_fake": sat_fake.detach(),
            "train/sat_real": sat_real.detach(),
            "train/chroma": chroma_l1.detach(),
        }
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True)

        if self.global_step % self.sample_interval == 0:
            self._log_samples(device)

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        images = batch["image"]
        batch_size = images.size(0)
        device = images.device
        generator = self._current_generator().to(device)
        noise = self._sample_latent(batch_size, device)
        fake = generator(noise)
        fake_rgb = _to_rgb01(fake)
        real_rgb = _to_rgb01(images)
        sat_fake = _mean_saturation(fake_rgb)
        sat_real = _mean_saturation(real_rgb)
        fake_u, fake_v = _chroma_components(fake_rgb)
        real_u, real_v = _chroma_components(real_rgb)
        chroma_l1 = (
            F.l1_loss(fake_u.mean(dim=0), real_u.mean(dim=0))
            + F.l1_loss(fake_v.mean(dim=0), real_v.mean(dim=0))
        )
        lpips_val = self.lpips(fake, images).mean()

        self.log_dict(
            {
                "val/lpips": lpips_val,
                "val/sat_fake": sat_fake,
                "val/sat_real": sat_real,
                "val/chroma": chroma_l1,
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        if lpips_val < self.best_lpips:
            self.best_lpips = float(lpips_val)

        return {
            "lpips": lpips_val.detach(),
            "sat_fake": sat_fake.detach(),
            "sat_real": sat_real.detach(),
        }

    def _log_samples(self, device: torch.device) -> None:
        if self.example_noise is None:
            return
        generator = self._current_generator().to(device)
        noise = self.example_noise.to(device)
        with torch.no_grad():
            samples = generator(noise)
        samples = _to_rgb01(samples.cpu())
        grid = make_grid(samples, nrow=int(math.sqrt(self.num_samples)), normalize=False)
        out_path = self.output_dir / f"samples_step{self.global_step:06d}.png"
        save_image(grid, out_path)

    def on_validation_epoch_end(self) -> None:  # type: ignore[override]
        if self.example_noise is not None and self.example_real is not None:
            device = self.device if isinstance(self.device, torch.device) else torch.device("cpu")
            self._log_samples(device)

    def on_train_epoch_end(self) -> None:  # type: ignore[override]
        g_opt, d_opt = self.optimizers()  # type: ignore[assignment]
        if self._pending_d_steps > 0:
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.5)
            d_opt.step()
            d_opt.zero_grad()
            self._pending_d_steps = 0
        if self._pending_g_steps > 0:
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 0.5)
            g_opt.step()
            g_opt.zero_grad()
            self._pending_g_steps = 0


def setup_trainer(args: argparse.Namespace) -> pl.Trainer:
    strategy: Optional[object]
    if args.strategy == "ddp_find_unused_parameters_false":
        strategy = DDPStrategy(find_unused_parameters=False)
    elif args.strategy == "auto":
        strategy = "auto"
    else:
        strategy = args.strategy

    callbacks: List[pl.callbacks.Callback] = [LearningRateMonitor(logging_interval="epoch")]
    checkpoint_dir = args.output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callbacks.append(
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="gan-{epoch:03d}-{val_lpips:.4f}",
            save_top_k=3,
            monitor="val/lpips",
            mode="min",
            every_n_epochs=args.checkpoint_every,
        )
    )

    clip_val: Optional[float] = None
    if args.grad_clip not in (None, 0, 0.0):
        logger.warning(
            "Trainer gradient clipping is unsupported with manual optimization; "
            "clipping is handled inside the GAN lightning module."
        )

    trainer = pl.Trainer(
        accelerator="auto",
        devices=args.devices,
        max_epochs=args.epochs,
        precision=args.precision,
        strategy=strategy,
        default_root_dir=str(args.output),
        accumulate_grad_batches=1,
        gradient_clip_val=clip_val,
        callbacks=callbacks,
        log_every_n_steps=25,
    )
    return trainer


def main() -> None:
    pl.seed_everything(42)
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    shutil.copy2(__file__, args.output / Path(__file__).name)
    manifest_copy = args.output / "manifest.json"
    args_payload = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    manifest_copy.write_text(json.dumps({"args": args_payload}, indent=2))

    data_module = ManifestDataModule(
        data_root=args.data_root,
        manifest=args.manifest,
        resolution=args.resolution,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
        limit=args.override_manifest_limit,
    )

    model = GANLightning(
        image_resolution=args.resolution,
        generator_lr=args.generator_lr,
        discriminator_lr=args.discriminator_lr,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        latent_channels=args.latent_channels,
        ema_decay=args.ema_decay,
        disable_ema=args.no_ema,
        output_dir=args.output,
        num_samples=args.num_samples,
        use_grad_checkpoint=not args.no_grad_checkpoint,
        latent_sigma=args.latent_sigma,
        chroma_weight=args.chroma_weight,
        saturation_weight=args.saturation_weight,
        r1_gamma=args.r1_gamma,
        r1_interval=args.r1_interval,
        sample_interval=args.sample_interval,
        grad_accum_steps=args.accumulate_grad_batches,
    )

    trainer = setup_trainer(args)

    trainer.fit(model, datamodule=data_module, ckpt_path=str(args.resume) if args.resume else None)


if __name__ == "__main__":
    main()
