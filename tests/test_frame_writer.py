from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from renderer import frame_writer


class FakeStdin:
    def __init__(self) -> None:
        self.closed = False
        self.buffer: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.buffer.append(data)

    def flush(self) -> None:  # pragma: no cover - no-op for stub
        return

    def close(self) -> None:
        self.closed = True


class FakeProcess:
    def __init__(self, output_path: Path) -> None:
        self._output_path = output_path
        self.stdin = FakeStdin()
        self.stdout = SimpleNamespace()
        self.stderr = SimpleNamespace()
        self.returncode = 0

    def communicate(self) -> tuple[bytes, bytes]:
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._output_path.exists():
            self._output_path.write_bytes(b"video")
        return (b"", b"")


class FakeImageIO:
    def __init__(self) -> None:
        self.writes: list[Path] = []

    def imwrite(self, path: Path | str, array: np.ndarray, **_: object) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"img")
        self.writes.append(target)


def test_ffmpeg_writer_handles_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    popen_calls: list[list[str]] = []

    def fake_popen(cmd, stdin, stdout, stderr):
        popen_calls.append(cmd)
        return FakeProcess(output_path=Path(cmd[-1]))

    monkeypatch.setattr("renderer.frame_writer.subprocess.Popen", fake_popen)
    image_stub = FakeImageIO()
    monkeypatch.setattr(frame_writer, "imageio", image_stub)

    audio_path = tmp_path / "track.wav"
    audio_path.write_bytes(b"audio")
    output_path = tmp_path / "out" / "video.mp4"
    writer = frame_writer.FFMpegWriter(
        output_path=output_path,
        frame_rate=24,
        resolution=[16, 16],
        audio_path=audio_path,
        ffmpeg_path="ffmpeg",
        keep_frames=True,
        preview_dir=tmp_path / "previews",
    )

    frames = np.linspace(0.0, 1.0, 2 * 16 * 16 * 3, dtype=np.float32).reshape(2, 16, 16, 3)
    writer.write_batch(frames)
    bundle = writer.close()

    assert popen_calls
    process = popen_calls[0]
    assert "ffmpeg" in process[0]
    assert writer.frames_written == 2
    assert pytest.approx(writer.duration, rel=1e-6) == 2 / 24
    assert output_path.exists()
    assert bundle.still and bundle.still.exists()
    assert bundle.animated and bundle.animated.exists()
    assert len(image_stub.writes) >= 2  # preview + frame persistence
    assert writer.to_metadata()["frames"] == 2


def test_concat_videos_builds_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    called_commands: list[list[str]] = []

    def fake_run(cmd, capture_output: bool, text: bool):
        called_commands.append(cmd)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("renderer.frame_writer.subprocess.run", fake_run)

    video_a = tmp_path / "a.mp4"
    video_b = tmp_path / "b.mp4"
    video_a.write_bytes(b"a")
    video_b.write_bytes(b"b")
    output_path = tmp_path / "combined.mp4"

    frame_writer.concat_videos([video_a, video_b], output_path=output_path, ffmpeg_path="ffmpeg")

    assert called_commands
    assert called_commands[0][0] == "ffmpeg"
    assert output_path.exists() is False  # concat only runs ffmpeg; stub does not write file


def test_concat_videos_error_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_fail(cmd, capture_output: bool, text: bool):
        return SimpleNamespace(returncode=1, stderr="boom")

    monkeypatch.setattr("renderer.frame_writer.subprocess.run", fake_run_fail)

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"clip")

    with pytest.raises(frame_writer.FFMpegError):
        frame_writer.concat_videos([video], output_path=tmp_path / "out.mp4")


def test_compute_sha256(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    data = b"noise-to-signal"
    path.write_bytes(data)
    digest = frame_writer.compute_sha256(path)
    import hashlib  # Local import to mirror implementation

    assert digest == hashlib.sha256(data).hexdigest()
