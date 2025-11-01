from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from renderer import decoder


class FakeInvalidDims(RuntimeError):
    pass


class FakeOrtModule:
    def __init__(self) -> None:
        self.SessionOptions = SimpleNamespace  # Placeholder; attributes assigned dynamically
        self.attempts: list[tuple[str, str]] = []
        self.calls: list[int] = []

    def InferenceSession(self, path: str, sess_options: object, providers: list[str]):
        self.attempts.append((path, providers[0]))
        return FakeSession(path, providers[0], self.calls)


class FakeSession:
    def __init__(self, path: str, provider: str, calls: list[int]) -> None:
        self._path = path
        self._provider = provider
        self._calls = calls

    def get_inputs(self):
        return [SimpleNamespace(name="latent")]

    def get_outputs(self):
        return [SimpleNamespace(name="decoded")]

    def get_providers(self):
        return [self._provider]

    def run(self, outputs: list[str], feeds: dict[str, np.ndarray]):
        (array,) = feeds.values()
        batch = int(array.shape[0])
        self._calls.append(batch)
        if batch > 2:
            raise FakeInvalidDims("invalid dimensions for batch")
        # Decoder output is channels-first; three colour channels expected.
        frames = np.ones((batch, 3, 16, 16), dtype=np.float32) * 0.6
        return [frames]


def _write_meta(tmp_path: Path) -> Path:
    meta = {
        "latent_shape": [8, 16, 16],
        "int8_decoder": str(tmp_path / "decoder.int8.onnx"),
        "fp16_decoder": str(tmp_path / "decoder.fp16.onnx"),
        "fp32_decoder": str(tmp_path / "decoder.fp32.onnx"),
        "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
    }
    for key in ("decoder.int8.onnx", "decoder.fp16.onnx", "decoder.fp32.onnx"):
        (tmp_path / key).write_bytes(b"onnx")
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps(meta))
    return meta_path


def test_decoder_session_adjusts_batch_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_ort = FakeOrtModule()
    monkeypatch.setattr(decoder, "ort", fake_ort)
    monkeypatch.setattr(decoder, "_ORTInvalidArgument", FakeInvalidDims)
    monkeypatch.setattr(decoder, "_ORTRuntimeException", FakeInvalidDims)

    meta_path = _write_meta(tmp_path)

    # Ensure SessionOptions provides expected attribute for intra_op threads.
    class FakeSessionOptions:
        def __init__(self) -> None:
            self.intra_op_num_threads = None

    monkeypatch.setattr(fake_ort, "SessionOptions", FakeSessionOptions)

    session = decoder.DecoderSession(
        execution_provider="cuda",
        batch_size=4,
        meta_path=meta_path,
    )

    latents = np.zeros((5, 8, 16, 16), dtype=np.float32)
    frames = session.decode(latents, batch_size=3)

    assert frames.shape == (5, 16, 16, 3)
    assert np.all(frames >= 0.0)
    assert np.all(frames <= 1.0)
    # The failing batch (>2) forces a retry with batch size 1.
    assert session.default_batch_size == 1
    assert fake_ort.calls.count(1) == 5
    assert fake_ort.attempts[0][1] == "CUDAExecutionProvider"

    with pytest.raises(ValueError):
        session.decode(np.zeros((2, 4, 16, 16), dtype=np.float32))
