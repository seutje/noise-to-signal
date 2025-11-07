from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch

from renderer import decoder


class _FakeLatent:
    channels = 8
    height = 16
    width = 16


class _ConstantDecoder(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        batch = x.shape[0]
        device = x.device
        return torch.ones((batch, 3, 16, 16), device=device, dtype=torch.float32)


class _ExplodingDecoder(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        raise RuntimeError("decoder failure")


class _DummyLightning:
    def __init__(self, module: torch.nn.Module) -> None:
        self.generator = module
        self.latent = _FakeLatent()
        self.generator_ema = None

    @classmethod
    def load_from_checkpoint(cls, *_args, **_kwargs) -> "_DummyLightning":  # type: ignore[override]
        raise AssertionError("load_from_checkpoint should be monkeypatched in tests")


def _install_dummy_loader(monkeypatch: pytest.MonkeyPatch, module: torch.nn.Module) -> None:
    def _loader(*_args, **_kwargs) -> _DummyLightning:
        return _DummyLightning(module)

    monkeypatch.setattr(decoder, "GANLightning", _DummyLightning)
    monkeypatch.setattr(_DummyLightning, "load_from_checkpoint", classmethod(_loader))


def test_decoder_session_decodes_batches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = tmp_path / "vae-best.ckpt"
    checkpoint.write_text("checkpoint")

    _install_dummy_loader(monkeypatch, _ConstantDecoder())

    session = decoder.DecoderSession(
        execution_provider="cpu",
        batch_size=3,
        checkpoint_path=checkpoint,
    )

    latents = np.zeros((5, 8, 16, 16), dtype=np.float32)
    frames = session.decode(latents, batch_size=4)

    assert frames.shape == (5, 16, 16, 3)
    assert np.allclose(frames, 1.0, atol=1e-6)
    assert session.current_provider == "cpu"
    assert session.precision == "fp32"


def test_decoder_session_wraps_runtime_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = tmp_path / "vae-best.ckpt"
    checkpoint.write_text("checkpoint")

    _install_dummy_loader(monkeypatch, _ExplodingDecoder())

    session = decoder.DecoderSession(
        execution_provider="cpu",
        batch_size=2,
        checkpoint_path=checkpoint,
    )

    latents = np.zeros((2, 8, 16, 16), dtype=np.float32)
    with pytest.raises(decoder.DecoderExecutionError):
        session.decode(latents)


def test_decoder_session_rejects_missing_checkpoint(tmp_path: Path) -> None:
    missing = tmp_path / "missing.ckpt"
    with pytest.raises(decoder.DecoderUnavailableError):
        decoder.DecoderSession(
            execution_provider="cpu",
            batch_size=2,
            checkpoint_path=missing,
        )
