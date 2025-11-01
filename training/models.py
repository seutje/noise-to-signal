from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential


def _group_count(channels: int) -> int:
    """Determine a valid GroupNorm group count for the given channel size."""
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ResBlock(nn.Module):
    """Residual block with GroupNorm and SiLU activation."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(self.act1(self.norm1(x)))
        x = self.conv2(self.dropout(self.act2(self.norm2(x))))
        return x + residual


class Downsample(nn.Module):
    """Downsamples by a factor of 2 using a strided convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Upsamples by a factor of 2 using nearest interpolation + conv."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.upsample(x))


class Encoder(nn.Module):
    """β-VAE encoder producing mean and log variance at 16×16 spatial resolution."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        latent_channels: int = 8,
        layers: Tuple[int, ...] = (64, 128, 256, 256),
        dropout: float = 0.0,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.stem = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        blocks = []
        prev_channels = base_channels
        for channels in layers:
            blocks.append(ResBlock(prev_channels, channels, dropout=dropout))
            blocks.append(ResBlock(channels, channels, dropout=dropout))
            blocks.append(Downsample(channels))
            prev_channels = channels
        self.blocks = nn.Sequential(*blocks)
        self.use_checkpoint = use_checkpoint
        self.checkpoint_segments = max(1, len(self.blocks) // 2)

        self.to_stats = nn.Conv2d(prev_channels, latent_channels * 2, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        if self.use_checkpoint:
            h = checkpoint_sequential(self.blocks, self.checkpoint_segments, h)
        else:
            h = self.blocks(h)
        stats = self.to_stats(h)
        mean, logvar = torch.chunk(stats, chunks=2, dim=1)
        return mean, logvar


class Decoder(nn.Module):
    """β-VAE decoder that reconstructs 3×H×W images from latent maps."""

    def __init__(
        self,
        out_channels: int = 3,
        base_channels: int = 64,
        latent_channels: int = 8,
        layers: Tuple[int, ...] = (256, 256, 128, 64),
        dropout: float = 0.0,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.initial = nn.Conv2d(latent_channels, layers[0], kernel_size=3, padding=1)

        blocks = []
        prev_channels = layers[0]
        for channels in layers:
            blocks.append(ResBlock(prev_channels, channels, dropout=dropout))
            blocks.append(ResBlock(channels, channels, dropout=dropout))
            blocks.append(Upsample(channels))
            prev_channels = channels
        self.blocks = nn.Sequential(*blocks)
        self.use_checkpoint = use_checkpoint
        self.checkpoint_segments = max(1, len(self.blocks) // 2)

        self.out_norm = nn.GroupNorm(_group_count(prev_channels), prev_channels)
        self.out_act = nn.SiLU()
        self.to_rgb = nn.Conv2d(prev_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.initial(z)
        if self.use_checkpoint:
            h = checkpoint_sequential(self.blocks, self.checkpoint_segments, h)
        else:
            h = self.blocks(h)
        h = self.to_rgb(self.out_act(self.out_norm(h)))
        return torch.tanh(h)


@dataclass
class LatentConfig:
    channels: int = 8
    height: int = 16
    width: int = 16

    @property
    def flattened(self) -> int:
        return self.channels * self.height * self.width


class BetaVAE(nn.Module):
    """Compact β-VAE with convolutional encoder/decoder pairs."""

    def __init__(
        self,
        image_channels: int = 3,
        base_channels: int = 64,
        latent: LatentConfig = LatentConfig(),
        dropout: float = 0.0,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        encoder_layers = (base_channels, base_channels * 2, base_channels * 4, base_channels * 4)
        decoder_layers = (base_channels * 4, base_channels * 4, base_channels * 2, base_channels)

        self.latent = latent
        self.encoder = Encoder(
            in_channels=image_channels,
            base_channels=base_channels,
            latent_channels=latent.channels,
            layers=encoder_layers,
            dropout=dropout,
            use_checkpoint=use_checkpoint,
        )
        self.decoder = Decoder(
            out_channels=image_channels,
            base_channels=base_channels,
            latent_channels=latent.channels,
            layers=decoder_layers,
            dropout=dropout,
            use_checkpoint=use_checkpoint,
        )

    @staticmethod
    def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        return recon, mean, logvar, z


class EMA:
    """Lightweight exponential moving average tracker for decoder weights."""

    def __init__(self, module: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {name: param.detach().clone() for name, param in module.named_parameters()}

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            shadow = self.shadow[name]
            if shadow.device != param.device:
                shadow = shadow.to(param.device)
                self.shadow[name] = shadow
            shadow.add_(param - shadow, alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, module: nn.Module) -> None:
        for name, param in module.named_parameters():
            if name in self.shadow:
                param.copy_(self.shadow[name].to(param.device))

    def to(self, device: torch.device) -> "EMA":
        for name, tensor in self.shadow.items():
            self.shadow[name] = tensor.to(device)
        return self
