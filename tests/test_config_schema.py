from __future__ import annotations

import json
from pathlib import Path

import pytest

from renderer import config_schema as cs
from renderer.config_schema import (
    RenderConfig,
    TrackConfig,
    AudioConfig,
    ControllerConfig,
    DecoderConfig,
    PostFXConfig,
    TempoSyncConfig,
    TrimWindow,
)


def test_load_render_config_with_presets_and_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "render.yaml"
    config_path.write_text(
        json.dumps(
            {
                "output_root": "outputs",
                "frame_rate": 30,
                "resolution": [1920, 1088],
                "tracks": [{"id": "demo", "src": "track.wav"}],
                "metadata": {"album": "noise-to-signal"},
            }
        )
    )

    def fake_load_preset(name: str):
        assert name == "cinematic"
        return {"frame_rate": 48, "controller": {"preset": "pulse"}}

    monkeypatch.setattr(cs, "load_preset_dict", fake_load_preset)

    config = cs.load_render_config(
        config_path,
        extra_presets=["cinematic"],
        overrides={"controller.wander_seed": 99},
    )

    assert config.frame_rate == 48
    assert config.controller.preset == "pulse"
    assert config.controller.wander_seed == 99
    assert config.output_root == (tmp_path / "outputs").resolve()
    assert config.metadata["album"] == "noise-to-signal"


def test_load_render_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        cs.load_render_config(tmp_path / "missing.yaml")


def test_load_yaml_file_errors(tmp_path: Path) -> None:
    empty = tmp_path / "empty.yaml"
    empty.write_text("null")
    with pytest.raises(ValueError):
        cs._load_yaml_file(empty)

    not_mapping = tmp_path / "list.yaml"
    not_mapping.write_text("- 1\n- 2\n")
    with pytest.raises(TypeError):
        cs._load_yaml_file(not_mapping)


def test_render_config_validation_resolution(tmp_path: Path) -> None:
    track = TrackConfig(id="t", src=tmp_path / "t.wav")
    config = RenderConfig(
        output_root=tmp_path / "out",
        frame_rate=30,
        resolution=[1921, 1087],
        audio=AudioConfig(),
        controller=ControllerConfig(),
        decoder=DecoderConfig(),
        postfx=PostFXConfig(),
        tracks=[track],
    )
    with pytest.raises(ValueError):
        config.validate()


def test_tempo_sync_validation() -> None:
    tempo = TempoSyncConfig(enabled=True, subdivision=-0.5)
    with pytest.raises(ValueError):
        tempo.validate()


def test_track_trim_validation() -> None:
    trim = TrimWindow(start=5.0, end=3.0)
    with pytest.raises(ValueError):
        trim.validate("demo")


def test_render_config_describe_includes_tracks(tmp_path: Path) -> None:
    track = TrackConfig(id="demo", src=tmp_path / "track.wav", seed=5)
    config = RenderConfig(
        output_root=tmp_path / "renders",
        frame_rate=60,
        resolution=[1920, 1088],
        tracks=[track],
    )
    description = config.describe()
    assert description["tracks"][0]["id"] == "demo"
    assert description["controller"]["anchor_set"] == config.controller.anchor_set
