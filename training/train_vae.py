import argparse
import csv
import json
import logging
import math
import shutil
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

from .models import BetaVAE, EMA, LatentConfig


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
    """
    Convert RGB in [0, 1] to approximate YUV chroma components.

    Returns
    -------
    Tuple of (U, V) tensors preserving gradients for chroma-aware losses.
    """
    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = 0.492 * (b - y)
    v = 0.877 * (r - y)
    return u, v


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train β-VAE on generated abstract dataset.")
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
    parser.add_argument("--epochs", type=int, default=80, help="Max epochs.")
    parser.add_argument("--val-split", type=float, default=0.05, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--lr", type=float, default=3e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--beta", type=float, default=1.0, help="β coefficient for KL term.")
    parser.add_argument(
        "--chroma-weight",
        type=float,
        default=2.0,
        help="Weight applied to U/V chroma reconstruction loss.",
    )
    parser.add_argument(
        "--saturation-weight",
        type=float,
        default=0.5,
        help="Weight applied to saturation preservation penalty.",
    )
    parser.add_argument("--kl-warmup-epochs", type=float, default=8.0, help="KL warmup duration in epochs.")
    parser.add_argument("--precision", type=str, default="bf16-mixed", help="Lightning precision flag.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm value.")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Save checkpoint every N epochs.")
    parser.add_argument("--devices", type=int, default=None, help="Optional override for device count.")
    parser.add_argument("--strategy", type=str, default="auto", help="Lightning training strategy override.")
    parser.add_argument("--accumulate-grad-batches", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--num-samples", type=int, default=8, help="Validation samples saved for recon grid.")
    parser.add_argument("--latent-channels", type=int, default=8, help="Latent channel count (z).")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay for decoder weights.")
    parser.add_argument("--no-ema", action="store_true", help="Disable EMA tracking.")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from lightning checkpoint.")
    parser.add_argument("--override-manifest-limit", type=int, default=None, help="Optional limit for debugging.")
    parser.add_argument(
        "--no-grad-checkpoint",
        action="store_true",
        help="Disable gradient checkpointing for encoder/decoder blocks (uses more memory).",
    )
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

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
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

    def setup(self, stage: Optional[str] = None) -> None:
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

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None, "setup() must be called before requesting dataloaders."
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None, "setup() must be called before requesting dataloaders."
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )


class BetaVAELightning(pl.LightningModule):
    def __init__(
        self,
        image_resolution: int,
        lr: float,
        weight_decay: float,
        beta: float,
        kl_warmup_epochs: float,
        max_epochs: int,
        latent_channels: int,
        ema_decay: float,
        disable_ema: bool,
        output_dir: Path,
        num_samples: int,
        use_grad_checkpoint: bool,
        chroma_weight: float,
        saturation_weight: float,
    ) -> None:
        super().__init__()
        latent = LatentConfig(channels=latent_channels, height=16, width=16)
        self.model = BetaVAE(latent=latent, use_checkpoint=use_grad_checkpoint)
        self.lpips = LPIPS(net="vgg")
        self.lpips.eval()
        self.lpips.requires_grad_(False)
        self.image_resolution = image_resolution
        self.save_hyperparameters(
            {
                "image_resolution": image_resolution,
                "lr": lr,
                "weight_decay": weight_decay,
                "beta": beta,
                "kl_warmup_epochs": kl_warmup_epochs,
                "max_epochs": max_epochs,
                "latent_channels": latent_channels,
                "ema_decay": ema_decay,
                "disable_ema": disable_ema,
                "output_dir": str(output_dir),
                "num_samples": num_samples,
                "use_grad_checkpoint": use_grad_checkpoint,
                "chroma_weight": chroma_weight,
                "saturation_weight": saturation_weight,
            }
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_lpips = math.inf
        self.example_batch: Optional[torch.Tensor] = None
        self.ema = None if disable_ema else EMA(self.model.decoder, decay=ema_decay)
        self.chroma_weight = chroma_weight
        self.saturation_weight = saturation_weight

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model(x)

    def kl_weight(self) -> float:
        if self.hparams.kl_warmup_epochs <= 0:
            return self.hparams.beta
        current = min(self.current_epoch + 1, self.hparams.max_epochs)
        factor = min(1.0, current / self.hparams.kl_warmup_epochs)
        return self.hparams.beta * factor

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        images = batch["image"]
        device = images.device
        self.lpips = self.lpips.to(device)
        recon, mean, logvar, _ = self(images)
        l1 = F.l1_loss(recon, images)
        lpips_loss = self.lpips(recon, images).mean()
        kl = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        recon_rgb = _to_rgb01(recon)
        target_rgb = _to_rgb01(images)
        recon_u, recon_v = _chroma_components(recon_rgb)
        target_u, target_v = _chroma_components(target_rgb)
        chroma_l1 = F.l1_loss(recon_u, target_u) + F.l1_loss(recon_v, target_v)
        sat_recon = _mean_saturation(recon_rgb)
        sat_target = _mean_saturation(target_rgb).detach()
        sat_penalty = torch.relu(sat_target - sat_recon)
        loss = (
            10.0 * l1
            + lpips_loss
            + self.kl_weight() * kl
            + self.chroma_weight * chroma_l1
            + self.saturation_weight * sat_penalty
        )

        self.log_dict(
            {
                "train/loss": loss,
                "train/l1": l1,
                "train/lpips": lpips_loss,
                "train/kl": kl,
                "train/chroma_l1": chroma_l1,
                "train/sat_recon": sat_recon,
                "train/sat_target": sat_target,
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        return {"loss": loss}

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        images = batch["image"]
        device = images.device
        self.lpips = self.lpips.to(device)
        recon, mean, logvar, _ = self(images)
        l1 = F.l1_loss(recon, images)
        lpips_loss = self.lpips(recon, images).mean()
        kl = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        recon_rgb = _to_rgb01(recon)
        target_rgb = _to_rgb01(images)
        recon_u, recon_v = _chroma_components(recon_rgb)
        target_u, target_v = _chroma_components(target_rgb)
        chroma_l1 = F.l1_loss(recon_u, target_u) + F.l1_loss(recon_v, target_v)
        sat_recon = _mean_saturation(recon_rgb)
        sat_target = _mean_saturation(target_rgb)
        sat_penalty = torch.relu(sat_target.detach() - sat_recon)
        loss = (
            10.0 * l1
            + lpips_loss
            + self.hparams.beta * kl
            + self.chroma_weight * chroma_l1
            + self.saturation_weight * sat_penalty
        )

        if self.example_batch is None and self.global_rank == 0:
            self.example_batch = images[: self.hparams.num_samples].detach().cpu()

        metrics = {
            "val/loss": loss,
            "val/l1": l1,
            "val/lpips": lpips_loss,
            "val/kl": kl,
            "val/chroma_l1": chroma_l1,
            "val/sat_recon": sat_recon,
            "val/sat_target": sat_target,
        }
        self.log_dict(metrics, prog_bar=True, on_epoch=True, sync_dist=True)
        return metrics

    def on_validation_epoch_end(self) -> None:
        if not self.trainer or getattr(self.trainer, "sanity_checking", False):
            return
        if not self.trainer.is_global_zero:
            return

        metrics = getattr(self.trainer, "callback_metrics", {})
        lpips_value = metrics.get("val/lpips")
        if lpips_value is not None:
            lpips_scalar = float(lpips_value)
            if lpips_scalar < self.best_lpips:
                self.best_lpips = lpips_scalar

        if self.example_batch is None:
            return

        self.model.eval()
        with torch.no_grad():
            example = self.example_batch.to(self.device)
            recon, _, _, _ = self(example)
        original = example.cpu()
        reconstructed = recon.cpu()
        batch = torch.cat([original, reconstructed], dim=0)
        grid = make_grid(batch, nrow=self.hparams.num_samples, normalize=True, value_range=(-1, 1))
        save_path = self.output_dir / "val_recon.png"
        save_image(grid, save_path)

    def on_train_batch_end(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        if self.ema is not None:
            self.ema.update(self.model.decoder)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.ema is not None and "ema_state" in checkpoint:
            for name, tensor in checkpoint["ema_state"].items():
                target = self.ema.shadow[name]
                self.ema.shadow[name].data.copy_(tensor.to(target.device))

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.ema is not None:
            checkpoint["ema_state"] = {name: tensor.clone() for name, tensor in self.ema.shadow.items()}

    def on_fit_start(self) -> None:
        self.lpips = self.lpips.to(self.device)
        if self.ema is not None:
            self.ema.to(self.device)

    def on_validation_start(self) -> None:
        self.lpips = self.lpips.to(self.device)
        if self.ema is not None:
            self.ema.to(self.device)


def select_strategy(strategy_flag: str) -> Any:
    if strategy_flag == "ddp":
        return DDPStrategy(find_unused_parameters=False)
    if strategy_flag == "auto":
        return "auto"
    return strategy_flag


def main() -> None:
    args = parse_args()
    if torch.cuda.is_available():
        try:
            torch.set_float32_matmul_precision("medium")
        except AttributeError:
            pass
    pl.seed_everything(args.seed, workers=True)
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

    model = BetaVAELightning(
        image_resolution=args.resolution,
        lr=args.lr,
        weight_decay=args.weight_decay,
        beta=args.beta,
        kl_warmup_epochs=args.kl_warmup_epochs,
        max_epochs=args.epochs,
        latent_channels=args.latent_channels,
        ema_decay=args.ema_decay,
        disable_ema=args.no_ema,
        output_dir=args.output,
        num_samples=args.num_samples,
        use_grad_checkpoint=not args.no_grad_checkpoint,
        chroma_weight=args.chroma_weight,
        saturation_weight=args.saturation_weight,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output / "checkpoints",
        filename="vae-{epoch:03d}-{val_lpips:.4f}",
        save_top_k=3,
        monitor="val/lpips",
        every_n_epochs=args.checkpoint_every,
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    devices = args.devices if args.devices is not None else "auto"
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        precision=args.precision,
        devices=devices,
        accelerator="auto",
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[checkpoint_callback, lr_monitor],
        strategy=select_strategy(args.strategy),
        default_root_dir=args.output,
        gradient_clip_val=args.grad_clip,
        log_every_n_steps=25,
    )

    trainer.fit(model, datamodule=data_module, ckpt_path=args.resume)
    best_path = Path(checkpoint_callback.best_model_path or "")
    if best_path.is_file():
        target = best_path.parent / "vae-best.ckpt"
        try:
            shutil.copy2(best_path, target)
            logger.info("Copied best checkpoint %s → %s", best_path.name, target)
        except OSError as exc:
            logger.warning("Failed to copy best checkpoint to %s: %s", target, exc)

    if model.example_batch is not None:
        metadata = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "resolution": args.resolution,
            "best_lpips": float(model.best_lpips),
            "latent_channels": args.latent_channels,
            "beta": args.beta,
            "kl_warmup_epochs": args.kl_warmup_epochs,
            "ema_enabled": not args.no_ema,
            "chroma_weight": args.chroma_weight,
            "saturation_weight": args.saturation_weight,
        }
        meta_path = args.output / "training_summary.json"
        meta_path.write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
