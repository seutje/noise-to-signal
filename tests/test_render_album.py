from __future__ import annotations

import json
from pathlib import Path

from renderer.render_album import render_album, TrackRenderSummary
from renderer.config_schema import RenderConfig, TrackConfig
from renderer.config_schema import AudioConfig, ControllerConfig, DecoderConfig, PostFXConfig


class DummyDecoderSession:
    def __init__(
        self,
        *,
        execution_provider: str,
        batch_size: int,
        checkpoint_path: Path,
        use_ema: bool,
    ) -> None:
        self.execution_provider = execution_provider
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path
        self.use_ema = use_ema


def test_render_album_writes_metadata(tmp_path: Path, monkeypatch) -> None:
    audio_one = tmp_path / "one.wav"
    audio_two = tmp_path / "two.wav"
    for path in (audio_one, audio_two):
        path.write_bytes(b"a")

    track_one = TrackConfig(id="one", src=audio_one, seed=11)
    track_two = TrackConfig(id="two", src=audio_two, seed=22)

    checkpoint = tmp_path / "vae-best.ckpt"
    checkpoint.write_text("checkpoint")

    config = RenderConfig(
        output_root=tmp_path / "outputs",
        frame_rate=30,
        resolution=[1920, 1088],
        audio=AudioConfig(),
        controller=ControllerConfig(),
        decoder=DecoderConfig(
            batch_size=2,
            execution_provider="cpu",
            checkpoint=checkpoint,
            use_ema=True,
        ),
        postfx=PostFXConfig(),
        tracks=[track_one, track_two],
    )
    config.validate()

    decoder_inits: list[tuple[str, int]] = []

    def fake_decoder_session(
        *,
        execution_provider: str,
        batch_size: int,
        checkpoint_path: Path,
        use_ema: bool,
    ):
        decoder_inits.append((execution_provider, batch_size))
        return DummyDecoderSession(
            execution_provider=execution_provider,
            batch_size=batch_size,
            checkpoint_path=checkpoint_path,
            use_ema=use_ema,
        )

    monkeypatch.setattr(
        "renderer.render_album.DecoderSession",
        fake_decoder_session,
    )

    video_paths: list[Path] = []

    def fake_render_track(track, config, output_dir, **kwargs):
        video_path = output_dir / "video.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_text(track.id)
        summary = TrackRenderSummary(
            track_id=track.id,
            output_dir=output_dir,
            latents_path=output_dir / "latents.npz",
            video_path=video_path,
            frames=120,
            duration=4.0,
            anchor_set="baseline",
            feature_cache=tmp_path / "cache" / f"{track.id}.npz",
            feature_checksum=f"{track.id}-checksum",
            preview_still=None,
            preview_anim=None,
            checksum=f"sha256-{track.id}",
            decode_provider="CPUExecutionProvider",
            decode_precision="int8",
            timings={"render_sec": 1.23},
        )
        summary.latents_path.write_text("latents")
        video_paths.append(video_path)
        return summary

    monkeypatch.setattr("renderer.render_album.render_track", fake_render_track)

    concat_calls: list[tuple[list[Path], Path, str]] = []

    def fake_concat(videos, *, output_path: Path, ffmpeg_path: str):
        concat_calls.append((videos, output_path, ffmpeg_path))
        output_path.write_text("album video")

    monkeypatch.setattr("renderer.render_album.concat_videos", fake_concat)

    render_album(config=config, skip_previews=True, ffmpeg_path="ffmpeg")

    run_path = config.output_root / "run.json"
    assert run_path.exists()

    metadata = json.loads(run_path.read_text())
    assert len(metadata["tracks"]) == 2
    assert metadata["album_video"] == str(config.output_root / "album.mp4")
    assert metadata["track_seeds"] == {"one": 11, "two": 22}
    assert decoder_inits == [("cpu", 2)]

    (videos, output_path, ffmpeg) = concat_calls[0]
    assert videos == video_paths
    assert output_path == config.output_root / "album.mp4"
    assert ffmpeg == "ffmpeg"
