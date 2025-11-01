"""
ONNX Runtime decoder wrapper for the noise-to-signal renderer.

This module abstracts model selection (INT8/FP16/FP32), execution provider
fallbacks, and batch decoding of latent tensors into RGB frames. The public
entry point is `DecoderSession`, which can be shared across tracks during a
render pass.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - exercised in environments without onnxruntime
    import onnxruntime as ort
    from onnxruntime.capi._pybind_state import (  # type: ignore[attr-defined]
        InvalidArgument as _ORTInvalidArgument,
        RuntimeException as _ORTRuntimeException,
    )
except ImportError:  # pragma: no cover - allows doc builds without runtime
    ort = None  # type: ignore[assignment]
    _ORTInvalidArgument = RuntimeError  # type: ignore[assignment]
    _ORTRuntimeException = RuntimeError  # type: ignore[assignment]

from . import PACKAGE_ROOT

LOG = logging.getLogger("renderer.decoder")

MODEL_ROOT = PACKAGE_ROOT.parent / "models"
META_PATH = MODEL_ROOT / "meta.json"


class DecoderUnavailableError(RuntimeError):
    """Raised when the decoder cannot be initialised."""


class DecoderExecutionError(RuntimeError):
    """Raised when a decode operation fails after all fallbacks."""


@dataclass(frozen=True)
class DecoderInfo:
    model_path: Path
    precision: str
    provider: str


def _load_meta(path: Path = META_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"models/meta.json not found at {path}")
    return json.loads(path.read_text())


def _resolve_model_paths(meta: dict, preferred_provider: str) -> List[DecoderInfo]:
    """Return model candidates ordered by suitability for the provider."""
    def _path_from_meta(key: str) -> Optional[Path]:
        value = meta.get(key)
        if not value:
            return None
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = (PACKAGE_ROOT.parent / candidate).resolve()
        return candidate

    int8 = _path_from_meta("int8_decoder")
    fp16 = _path_from_meta("fp16_decoder")
    fp32 = _path_from_meta("fp32_decoder")

    candidates: List[DecoderInfo] = []
    provider = preferred_provider.lower()

    if provider == "cuda":
        if int8:
            candidates.append(DecoderInfo(int8, "int8", "CUDAExecutionProvider"))
        if fp16:
            candidates.append(DecoderInfo(fp16, "fp16", "CUDAExecutionProvider"))
        if fp32:
            candidates.append(DecoderInfo(fp32, "fp32", "CUDAExecutionProvider"))
        # CPU fallbacks
        if int8:
            candidates.append(DecoderInfo(int8, "int8", "CPUExecutionProvider"))
        if fp32:
            candidates.append(DecoderInfo(fp32, "fp32", "CPUExecutionProvider"))
        if fp16:
            candidates.append(DecoderInfo(fp16, "fp16", "CPUExecutionProvider"))
    else:
        if int8:
            candidates.append(DecoderInfo(int8, "int8", "CPUExecutionProvider"))
        if fp32:
            candidates.append(DecoderInfo(fp32, "fp32", "CPUExecutionProvider"))
        if fp16:
            candidates.append(DecoderInfo(fp16, "fp16", "CPUExecutionProvider"))

    # Filter duplicates while preserving order.
    seen: set[Tuple[Path, str]] = set()
    ordered: List[DecoderInfo] = []
    for info in candidates:
        key = (info.model_path, info.provider)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(info)
    return ordered


def _log_event(event: str, **payload: object) -> None:
    LOG.info(json.dumps({"event": event, **payload}, sort_keys=True))


class DecoderSession:
    """
    Thin wrapper around ONNX Runtime for decoding latent tensors.

    Parameters
    ----------
    execution_provider:
        Preferred execution provider ("cuda" or "cpu"). The session will
        attempt fallbacks automatically if the preferred provider fails.
    batch_size:
        Default batch size when decoding. Can be overridden per call.
    """

    def __init__(
        self,
        *,
        execution_provider: str = "cuda",
        batch_size: int = 32,
        meta_path: Path = META_PATH,
        intra_op_threads: Optional[int] = None,
    ) -> None:
        if ort is None:
            raise DecoderUnavailableError(
                "onnxruntime is not installed. Install it to decode renders."
            )

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self._meta = _load_meta(meta_path)
        self._latent_shape = tuple(int(v) for v in self._meta.get("latent_shape", []))
        if len(self._latent_shape) != 3:
            raise ValueError("models/meta.json must define latent_shape [C, H, W].")

        norm = self._meta.get("normalization", {})
        mean = np.asarray(norm.get("mean", [0.5, 0.5, 0.5]), dtype=np.float32)
        std = np.asarray(norm.get("std", [0.5, 0.5, 0.5]), dtype=np.float32)
        self._mean = mean.reshape(1, -1, 1, 1)
        self._std = std.reshape(1, -1, 1, 1)

        self._requested_provider = execution_provider.lower()
        self._default_batch = batch_size
        self._current_provider: Optional[str] = None
        self._precision: Optional[str] = None
        self._session: Optional["ort.InferenceSession"] = None
        self._input_name: Optional[str] = None
        self._output_name: Optional[str] = None
        self._candidates = _resolve_model_paths(self._meta, self._requested_provider)

        if not self._candidates:
            raise DecoderUnavailableError(
                "No decoder model paths available in models/meta.json."
            )

        self._sess_options = ort.SessionOptions()
        if intra_op_threads is not None:
            self._sess_options.intra_op_num_threads = intra_op_threads

        self._initialise_session()

    # ------------------------------------------------------------------ #
    # Session lifecycle
    # ------------------------------------------------------------------ #

    def _initialise_session(self) -> None:
        last_error: Optional[Exception] = None
        for info in self._candidates:
            if not info.model_path.exists():
                _log_event(
                    "decoder.missing_model",
                    provider=info.provider,
                    precision=info.precision,
                    path=str(info.model_path),
                )
                continue

            providers = [info.provider]
            try:
                session = ort.InferenceSession(
                    str(info.model_path),
                    sess_options=self._sess_options,
                    providers=providers,
                )
                input_name = session.get_inputs()[0].name
                output_name = session.get_outputs()[0].name
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                _log_event(
                    "decoder.session_init_failed",
                    provider=info.provider,
                    precision=info.precision,
                    error=str(exc),
                )
                continue

            actual_provider = session.get_providers()[0] if session.get_providers() else info.provider
            self._session = session
            self._current_provider = actual_provider
            self._precision = info.precision
            self._input_name = input_name
            self._output_name = output_name
            _log_event(
                "decoder.session_ready",
                provider=info.provider,
                actual_provider=actual_provider,
                precision=info.precision,
                model=str(info.model_path),
            )
            return

        raise DecoderUnavailableError(
            f"Failed to initialise decoder session. Last error: {last_error}"
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def latent_shape(self) -> Tuple[int, int, int]:
        return self._latent_shape

    @property
    def current_provider(self) -> str:
        return self._current_provider or "unknown"

    @property
    def precision(self) -> str:
        return self._precision or "unknown"

    @property
    def default_batch_size(self) -> int:
        return self._default_batch

    def decode(
        self,
        latents: np.ndarray,
        *,
        batch_size: Optional[int] = None,
        validate: bool = True,
    ) -> np.ndarray:
        """
        Decode latent tensors into RGB frames.

        Parameters
        ----------
        latents:
            Array of shape (frames, C, H, W).
        batch_size:
            Optional override of the batch size for this invocation.
        validate:
            When True, perform basic numerical validation on the decoded frames.
        """
        if self._session is None or self._input_name is None or self._output_name is None:
            raise DecoderUnavailableError("Decoder session is not initialised.")

        array = np.asarray(latents, dtype=np.float32)
        if array.ndim != 4:
            raise ValueError("Latents must have shape (frames, C, H, W).")
        if tuple(array.shape[1:]) != self._latent_shape:
            raise ValueError(
                f"Latent shape mismatch: expected {self._latent_shape}, got {tuple(array.shape[1:])}."
            )

        remaining = array.shape[0]
        cursor = 0
        batches: List[np.ndarray] = []
        requested_batch = batch_size if batch_size is not None else self._default_batch
        batch = max(1, min(requested_batch, self._default_batch))

        while remaining > 0:
            take = min(batch, remaining)
            chunk = array[cursor : cursor + take]

            try:
                outputs = self._session.run(
                    [self._output_name],
                    {self._input_name: chunk},
                )
                decoded = outputs[0]
            except (_ORTRuntimeException, _ORTInvalidArgument) as exc:
                message = str(exc)
                if "invalid dimensions" in message.lower():
                    if batch == 1:
                        raise DecoderExecutionError(f"Decoder failed: {exc}") from exc
                    _log_event(
                        "decoder.batch_adjust",
                        error=message,
                        previous_batch=batch,
                        new_batch=1,
                    )
                    batch = 1
                    self._default_batch = 1
                    continue

                if "CUDA" in message.upper() and self.current_provider != "CPUExecutionProvider":
                    _log_event(
                        "decoder.cuda_failure",
                        error=str(exc),
                        fallback="CPUExecutionProvider",
                    )
                    # Reinitialise on CPU and retry the same chunk with a smaller batch.
                    self._candidates = _resolve_model_paths(self._meta, "cpu")
                    self._initialise_session()
                    batch = max(1, min(max(1, batch // 2), self._default_batch))
                    continue
                raise DecoderExecutionError(f"Decoder failed: {exc}") from exc

            batches.append(self._postprocess(decoded))
            remaining -= take
            cursor += take

        frames = np.concatenate(batches, axis=0)

        if validate:
            self._validate_frames(frames)

        return frames

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _postprocess(self, decoded: np.ndarray) -> np.ndarray:
        array = np.asarray(decoded, dtype=np.float32)
        if array.ndim != 4:
            raise DecoderExecutionError(
                f"Decoder output must have shape (batch, C, H, W); received {array.shape}."
            )
        if array.shape[1] != self._mean.shape[1]:
            raise DecoderExecutionError(
                f"Expected {self._mean.shape[1]} channels, got {array.shape[1]}."
            )

        array = array * self._std + self._mean
        array = np.clip(array, 0.0, 1.0)
        return np.transpose(array, (0, 2, 3, 1))  # NHWC

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
