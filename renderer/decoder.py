"""
PyTorch checkpoint-backed decoder for the noise-to-signal renderer.

The decoder loads the β-VAE weights from a Lightning checkpoint and performs
batched inference using PyTorch. This replaces the earlier ONNX Runtime path,
which was required for the in-browser demo but is unnecessary for the Python
prerender pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from training.train_vae import BetaVAELightning

from . import PACKAGE_ROOT

LOG = logging.getLogger("renderer.decoder")

DEFAULT_CHECKPOINT = (
    PACKAGE_ROOT.parent / "training" / "outputs" / "checkpoints" / "vae-best.ckpt"
)


class DecoderUnavailableError(RuntimeError):
    """Raised when the decoder cannot be initialised."""


class DecoderExecutionError(RuntimeError):
    """Raised when a decode operation fails."""


class DecoderSession:
    """
    Thin wrapper around a β-VAE decoder loaded from a Lightning checkpoint.

    Parameters
    ----------
    execution_provider:
        Preferred device ("cuda" or "cpu"). If CUDA is unavailable the session
        falls back to CPU automatically.
    batch_size:
        Default number of frames decoded per batch.
    checkpoint_path:
        Optional override for the checkpoint path. Defaults to
        `training/outputs/checkpoints/vae-best.ckpt`.
    use_ema:
        When True and the checkpoint includes EMA weights, they are applied
        before inference to match training-time evaluation.
    """

    def __init__(
        self,
        *,
        execution_provider: str = "cuda",
        batch_size: int = 32,
        checkpoint_path: Optional[Path] = None,
        use_ema: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self._default_batch = batch_size
        self._requested_provider = execution_provider.lower()
        self._checkpoint = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
        if not self._checkpoint.is_file():
            raise DecoderUnavailableError(
                f"Decoder checkpoint not found at {self._checkpoint}"
            )

        self._device, actual_provider = self._select_device(self._requested_provider)
        self._current_provider = actual_provider
        self._decoder, self._latent_shape, self._precision = self._load_decoder(
            self._checkpoint, use_ema
        )
        self._decoder = self._decoder.to(self._device).eval()

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        return self._latent_shape

    @property
    def current_provider(self) -> str:
        return self._current_provider

    @property
    def precision(self) -> str:
        return self._precision

    @property
    def default_batch_size(self) -> int:
        return self._default_batch

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def decode(
        self,
        latents: np.ndarray,
        *,
        batch_size: Optional[int] = None,
        validate: bool = True,
    ) -> np.ndarray:
        if latents is None:
            raise ValueError("latents cannot be None.")

        array = np.asarray(latents, dtype=np.float32)
        if array.ndim != 4:
            raise ValueError("Latents must have shape (frames, C, H, W).")
        if tuple(array.shape[1:]) != self._latent_shape:
            raise ValueError(
                f"Latent shape mismatch: expected {self._latent_shape}, got {tuple(array.shape[1:])}."
            )

        requested = batch_size if batch_size is not None else self._default_batch
        batch = max(1, requested)
        results: list[torch.Tensor] = []

        remaining = array.shape[0]
        cursor = 0

        while remaining > 0:
            take = min(batch, remaining)
            chunk = array[cursor : cursor + take]

            try:
                decoded = self._run_decoder(chunk)
            except RuntimeError as exc:  # noqa: BLE001
                if self._handle_runtime_error(exc, batch):
                    batch = self._default_batch
                    continue
                raise DecoderExecutionError(f"Decoder failed: {exc}") from exc

            results.append(decoded.cpu())
            cursor += take
            remaining -= take

        frames = torch.cat(results, dim=0)
        frames = self._postprocess(frames)
        frames_np = frames.numpy()

        if validate:
            self._validate_frames(frames_np)

        return frames_np

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _select_device(self, requested: str) -> tuple[torch.device, str]:
        if requested == "cuda" and torch.cuda.is_available():
            return torch.device("cuda"), "cuda"
        if requested == "cuda":
            LOG.warning("CUDA decoder requested but unavailable; falling back to CPU.")
        return torch.device("cpu"), "cpu"

    def _load_decoder(
        self,
        checkpoint: Path,
        use_ema: bool,
    ) -> tuple[torch.nn.Module, tuple[int, int, int], str]:
        LOG.info("Loading decoder checkpoint from %s", checkpoint)
        model = BetaVAELightning.load_from_checkpoint(str(checkpoint), map_location="cpu")

        if use_ema and getattr(model, "ema", None) is not None:
            model.ema.copy_to(model.model.decoder)
            precision = "fp32-ema"
        else:
            precision = "fp32"

        decoder = model.model.decoder.float()
        latent = model.model.latent
        latent_shape = (latent.channels, latent.height, latent.width)

        # Release Lightning module to free CPU memory.
        del model

        return decoder, latent_shape, precision

    def _run_decoder(self, chunk: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(chunk)
        if self._device.type == "cuda":
            tensor = tensor.pin_memory()
        tensor = tensor.to(self._device, non_blocking=self._device.type == "cuda")

        with torch.no_grad():
            decoded = self._decoder(tensor)
        return decoded

    def _handle_runtime_error(self, exc: RuntimeError, batch: int) -> bool:
        message = str(exc).lower()
        if "out of memory" in message and self._device.type == "cuda":
            new_batch = max(1, batch // 2)
            if new_batch == batch:
                new_batch = 1
            if new_batch == 0:
                new_batch = 1
            LOG.warning(
                "CUDA OOM encountered (batch=%d); retrying with batch=%d",
                batch,
                new_batch,
            )
            torch.cuda.empty_cache()
            self._default_batch = new_batch
            return True

        if self._device.type == "cuda":
            LOG.warning(
                "Decoder error on CUDA ('%s'); switching to CPU fallback.",
                exc,
            )
            self._fallback_to_cpu()
            return True

        return False

    def _fallback_to_cpu(self) -> None:
        if self._device.type == "cpu":
            return
        self._device = torch.device("cpu")
        self._current_provider = "cpu"
        self._decoder = self._decoder.to(self._device)
        self._default_batch = min(self._default_batch, 8)
        LOG.info("Decoder switched to CPU fallback.")

    def _postprocess(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.clamp(-1.0, 1.0)
        tensor = tensor.mul_(0.5).add_(0.5)
        tensor = tensor.clamp_(0.0, 1.0)
        return tensor.permute(0, 2, 3, 1).contiguous()

    def _validate_frames(self, frames: np.ndarray) -> None:
        if not np.all(np.isfinite(frames)):
            raise DecoderExecutionError("Decoded frames contain non-finite values.")
        if frames.size == 0:
            raise DecoderExecutionError("No frames were decoded.")
        if np.max(frames) <= 0.0:
            raise DecoderExecutionError("Decoded frames appear to be empty (all zeros).")


__all__ = [
    "DecoderSession",
    "DecoderUnavailableError",
    "DecoderExecutionError",
]
