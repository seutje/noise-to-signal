"""
Configuration schema and loader utilities for the noise-to-signal renderer.

The schema is intentionally lightweight (dataclasses + manual validation) so
that we avoid adding heavy dependencies. Configurations are expressed as YAML
and can be combined with preset overlays stored under `renderer/presets`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import copy
import json

import yaml

from . import PACKAGE_ROOT


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TrimWindow:
    """Optional start/end trim values in seconds."""

    start: Optional[float] = None
    end: Optional[float] = None

    def validate(self, track_id: str) -> None:
        if self.start is not None and self.start < 0:
            raise ValueError(f"Track '{track_id}': trim.start must be >= 0.")
        if self.end is not None and self.end < 0:
            raise ValueError(f"Track '{track_id}': trim.end must be >= 0.")
        if (
            self.start is not None
            and self.end is not None
            and self.end <= self.start
        ):
            raise ValueError(
                f"Track '{track_id}': trim.end ({self.end}) must be greater than trim.start ({self.start})."
            )

    def to_dict(self) -> Dict[str, float]:
        data: Dict[str, float] = {}
        if self.start is not None:
            data["start"] = self.start
        if self.end is not None:
            data["end"] = self.end
        return data


@dataclass(slots=True)
class TempoSyncConfig:
    enabled: bool = False
    subdivision: float = 1.0

    def validate(self) -> None:
        if self.subdivision <= 0:
            raise ValueError("tempo_sync.subdivision must be > 0.")


@dataclass(slots=True)
class ControllerConfig:
    preset: str = "default"
    smoothing_alpha: float = 0.9
    wander_seed: int = 42
    tempo_sync: TempoSyncConfig = field(default_factory=TempoSyncConfig)
    anchor_set: str = "baseline"

    def validate(self) -> None:
        if not 0.0 <= self.smoothing_alpha <= 1.0:
            raise ValueError("controller.smoothing_alpha must be within [0, 1].")
        if not isinstance(self.wander_seed, int):
            raise TypeError("controller.wander_seed must be an integer.")
        self.tempo_sync.validate()


@dataclass(slots=True)
class AudioConfig:
    sample_rate: int = 48_000
    normalization: Optional[float] = None

    def validate(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("audio.sample_rate must be > 0.")
        if self.normalization is not None and not (-60.0 <= self.normalization <= 0.0):
            raise ValueError("audio.normalization must be between -60 and 0 dB.")


@dataclass(slots=True)
class DecoderConfig:
    batch_size: int = 32
    execution_provider: str = "cuda"
    checkpoint: Path = Path("training/outputs/checkpoints/vae-best.ckpt")
    use_ema: bool = True

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("decoder.batch_size must be > 0.")
        allowed = {"cuda", "cpu"}
        if self.execution_provider.lower() not in allowed:
            raise ValueError(
                f"decoder.execution_provider must be one of {sorted(allowed)}."
            )
        if not isinstance(self.checkpoint, Path):
            raise TypeError("decoder.checkpoint must be a filesystem path.")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"decoder.checkpoint not found at {self.checkpoint}"
            )


@dataclass(slots=True)
class PostFXConfig:
    tone_curve: str = "filmlog"
    grain_intensity: float = 0.1
    chroma_shift: float = 0.0
    vignette_strength: float = 0.0
    motion_trails: bool = False

    def validate(self) -> None:
        if self.grain_intensity < 0:
            raise ValueError("postfx.grain_intensity must be >= 0.")
        if not (0 <= self.vignette_strength <= 1):
            raise ValueError("postfx.vignette_strength must be within [0, 1].")
        if abs(self.chroma_shift) > 0.05:
            raise ValueError("postfx.chroma_shift magnitude should be <= 0.05.")


@dataclass(slots=True)
class TrackConfig:
    id: str
    src: Path
    preset: Optional[str] = None
    trim: TrimWindow = field(default_factory=TrimWindow)
    anchors: Optional[str] = None
    seed: Optional[int] = None

    def validate(self) -> None:
        if not self.id:
            raise ValueError("Track id cannot be empty.")
        if not self.src:
            raise ValueError(f"Track '{self.id}': src must be provided.")
        self.trim.validate(self.id)


@dataclass(slots=True)
class RenderConfig:
    output_root: Path
    frame_rate: int
    resolution: Sequence[int]
    audio: AudioConfig = field(default_factory=AudioConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    postfx: PostFXConfig = field(default_factory=PostFXConfig)
    tracks: List[TrackConfig] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.frame_rate < 24 or self.frame_rate > 120:
            raise ValueError("frame_rate must be between 24 and 120 FPS.")
        if len(self.resolution) != 2:
            raise ValueError("resolution must contain width and height.")
        width, height = self.resolution
        if width % 16 != 0 or height % 16 != 0:
            raise ValueError("resolution values must be divisible by 16.")
        if width <= 0 or height <= 0:
            raise ValueError("resolution values must be positive.")
        self.audio.validate()
        self.controller.validate()
        self.decoder.validate()
        self.postfx.validate()
        if not self.tracks:
            raise ValueError("At least one track entry is required.")
        for track in self.tracks:
            track.validate()

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-serializable summary (useful for logging)."""
        summary = {
            "output_root": str(self.output_root),
            "frame_rate": self.frame_rate,
            "resolution": list(self.resolution),
            "audio": {
                "sample_rate": self.audio.sample_rate,
                "normalization": self.audio.normalization,
            },
            "controller": {
                "preset": self.controller.preset,
                "smoothing_alpha": self.controller.smoothing_alpha,
                "wander_seed": self.controller.wander_seed,
                "tempo_sync": {
                    "enabled": self.controller.tempo_sync.enabled,
                    "subdivision": self.controller.tempo_sync.subdivision,
                },
                "anchor_set": self.controller.anchor_set,
            },
            "decoder": {
                "batch_size": self.decoder.batch_size,
                "execution_provider": self.decoder.execution_provider,
                "checkpoint": str(self.decoder.checkpoint),
                "use_ema": self.decoder.use_ema,
            },
            "postfx": {
                "tone_curve": self.postfx.tone_curve,
                "grain_intensity": self.postfx.grain_intensity,
                "chroma_shift": self.postfx.chroma_shift,
                "vignette_strength": self.postfx.vignette_strength,
                "motion_trails": self.postfx.motion_trails,
            },
            "tracks": [
                {
                    "id": t.id,
                    "src": str(t.src),
                    "preset": t.preset,
                    "trim": t.trim.to_dict(),
                    "anchors": t.anchors,
                    "seed": t.seed,
                }
                for t in self.tracks
            ],
            "metadata": self.metadata,
        }
        return summary

    def to_json(self) -> str:
        return json.dumps(self.describe(), indent=2)


# ---------------------------------------------------------------------------
# Loading utilities
# ---------------------------------------------------------------------------


def _load_yaml_file(path: Path) -> Mapping[str, Any]:
    raw = yaml.safe_load(path.read_text())
    if raw is None:
        raise ValueError(f"Configuration file '{path}' is empty.")
    if not isinstance(raw, Mapping):
        raise TypeError(f"Configuration '{path}' must be a mapping at top level.")
    return raw


def _deep_update(base: MutableMapping[str, Any], overlay: Mapping[str, Any]) -> None:
    for key, value in overlay.items():
        if (
            isinstance(value, Mapping)
            and key in base
            and isinstance(base[key], MutableMapping)
        ):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


def load_preset_dict(name: str) -> Mapping[str, Any]:
    """Load a preset overlay by name."""
    preset_path = PACKAGE_ROOT / "presets" / f"{name}.yaml"
    if not preset_path.exists():
        available = sorted(p.stem for p in (PACKAGE_ROOT / "presets").glob("*.yaml"))
        raise FileNotFoundError(
            f"Preset '{name}' not found. Available presets: {', '.join(available)}"
        )
    return _load_yaml_file(preset_path)


def apply_presets(
    base_config: MutableMapping[str, Any], preset_names: Iterable[str]
) -> MutableMapping[str, Any]:
    """Apply one or more preset overlays to the base config mapping."""
    for name in preset_names:
        overlay = load_preset_dict(name)
        _deep_update(base_config, overlay)
    return base_config


def resolve_track_config(
    base_config: RenderConfig,
    track: TrackConfig,
) -> tuple[RenderConfig, Mapping[str, Any]]:
    """
    Return a copy of `base_config` with any track-level preset overlay applied.

    The original config is not mutated. The accompanying mapping contains any
    metadata defined in the preset (may be empty).
    """
    if not track.preset:
        return base_config, {}

    overlay = load_preset_dict(track.preset)
    controller_overlay = overlay.get("controller", {})
    postfx_overlay = overlay.get("postfx", {})
    metadata = overlay.get("metadata", {})

    controller_dict = asdict(base_config.controller)
    postfx_dict = asdict(base_config.postfx)

    if controller_overlay:
        _deep_update(controller_dict, controller_overlay)
    if postfx_overlay:
        _deep_update(postfx_dict, postfx_overlay)

    tempo_dict = controller_dict.get("tempo_sync") or {}
    controller_cfg = ControllerConfig(
        preset=str(controller_dict.get("preset", base_config.controller.preset)),
        smoothing_alpha=float(
            controller_dict.get("smoothing_alpha", base_config.controller.smoothing_alpha)
        ),
        wander_seed=int(controller_dict.get("wander_seed", base_config.controller.wander_seed)),
        tempo_sync=TempoSyncConfig(
            enabled=bool(tempo_dict.get("enabled", base_config.controller.tempo_sync.enabled)),
            subdivision=float(
                tempo_dict.get("subdivision", base_config.controller.tempo_sync.subdivision)
            ),
        ),
        anchor_set=str(controller_dict.get("anchor_set", base_config.controller.anchor_set)),
    )
    controller_cfg.validate()

    postfx_cfg = PostFXConfig(
        tone_curve=str(postfx_dict.get("tone_curve", base_config.postfx.tone_curve)),
        grain_intensity=float(
            postfx_dict.get("grain_intensity", base_config.postfx.grain_intensity)
        ),
        chroma_shift=float(postfx_dict.get("chroma_shift", base_config.postfx.chroma_shift)),
        vignette_strength=float(
            postfx_dict.get("vignette_strength", base_config.postfx.vignette_strength)
        ),
        motion_trails=bool(postfx_dict.get("motion_trails", base_config.postfx.motion_trails)),
    )
    postfx_cfg.validate()

    merged = replace(base_config, controller=controller_cfg, postfx=postfx_cfg)
    return merged, metadata


def _parse_track(entry: Mapping[str, Any], base_dir: Path) -> TrackConfig:
    try:
        track_id = str(entry["id"])
        src = Path(entry["src"])
    except KeyError as exc:
        raise KeyError(f"Track entries must include 'id' and 'src'. Missing: {exc}")

    track = TrackConfig(
        id=track_id,
        src=(base_dir / src).resolve() if not src.is_absolute() else src,
        preset=entry.get("preset"),
        trim=TrimWindow(
            start=entry.get("trim", {}).get("start"),
            end=entry.get("trim", {}).get("end"),
        ),
        anchors=entry.get("anchors"),
        seed=entry.get("seed"),
    )
    track.validate()
    return track


def _parse_config_mapping(
    mapping: Mapping[str, Any], config_dir: Path
) -> RenderConfig:
    output_root = mapping.get("output_root")
    if not output_root:
        raise ValueError("render.yaml must include an output_root.")
    output_root_path = Path(output_root)
    if not output_root_path.is_absolute():
        output_root_path = (config_dir / output_root_path).resolve()

    frame_rate = int(mapping.get("frame_rate", 60))
    resolution = mapping.get("resolution", [1920, 1080])

    audio_cfg = mapping.get("audio", {})
    controller_cfg = mapping.get("controller", {})
    tempo_cfg = controller_cfg.get("tempo_sync", {})
    decoder_cfg = mapping.get("decoder", {})
    postfx_cfg = mapping.get("postfx", {})
    tracks_cfg = mapping.get("tracks") or []
    metadata_cfg = mapping.get("metadata", {})

    checkpoint_value = decoder_cfg.get("checkpoint")
    if checkpoint_value:
        checkpoint_path = Path(checkpoint_value)
        if not checkpoint_path.is_absolute():
            checkpoint_path = (config_dir / checkpoint_path).resolve()
    else:
        checkpoint_path = (
            PACKAGE_ROOT.parent
            / "training"
            / "outputs"
            / "checkpoints"
            / "vae-best.ckpt"
        ).resolve()

    render_cfg = RenderConfig(
        output_root=output_root_path,
        frame_rate=frame_rate,
        resolution=resolution,
        audio=AudioConfig(
            sample_rate=int(audio_cfg.get("sample_rate", 48_000)),
            normalization=audio_cfg.get("normalization"),
        ),
        controller=ControllerConfig(
            preset=str(controller_cfg.get("preset", "default")),
            smoothing_alpha=float(controller_cfg.get("smoothing_alpha", 0.9)),
            wander_seed=int(controller_cfg.get("wander_seed", 42)),
            tempo_sync=TempoSyncConfig(
                enabled=bool(tempo_cfg.get("enabled", False)),
                subdivision=float(tempo_cfg.get("subdivision", 1.0)),
            ),
            anchor_set=str(controller_cfg.get("anchor_set", "baseline")),
        ),
        decoder=DecoderConfig(
            batch_size=int(decoder_cfg.get("batch_size", 32)),
            execution_provider=str(
                decoder_cfg.get("execution_provider", "cuda")
            ).lower(),
            checkpoint=checkpoint_path,
            use_ema=bool(decoder_cfg.get("use_ema", True)),
        ),
        postfx=PostFXConfig(
            tone_curve=str(postfx_cfg.get("tone_curve", "filmlog")),
            grain_intensity=float(postfx_cfg.get("grain_intensity", 0.1)),
            chroma_shift=float(postfx_cfg.get("chroma_shift", 0.0)),
            vignette_strength=float(postfx_cfg.get("vignette_strength", 0.0)),
            motion_trails=bool(postfx_cfg.get("motion_trails", False)),
        ),
        tracks=[
            _parse_track(entry, config_dir) for entry in tracks_cfg
        ],
        metadata=dict(metadata_cfg),
    )

    render_cfg.validate()
    return render_cfg


def load_render_config(
    config_path: Path,
    *,
    extra_presets: Optional[Sequence[str]] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> RenderConfig:
    """
    Load `render.yaml`, optionally applying preset overlays and inline overrides.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.
    extra_presets:
        Optional sequence of preset names (without `.yaml`) to overlay on top
        of the base configuration.
    overrides:
        Optional mapping of key paths to override. Only a shallow subset is
        supported (see docstring below).
    """

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file '{config_path}' does not exist.")

    raw_mapping = dict(_load_yaml_file(config_path))

    if extra_presets:
        apply_presets(raw_mapping, extra_presets)

    if overrides:
        # Support limited overrides (e.g. `"frame_rate": 30`, `"controller.preset": "pulse"`)
        for dotted_key, value in overrides.items():
            parts = dotted_key.split(".")
            cursor: MutableMapping[str, Any] = raw_mapping
            for part in parts[:-1]:
                if part not in cursor or not isinstance(cursor[part], MutableMapping):
                    cursor[part] = {}
                cursor = cursor[part]  # type: ignore[assignment]
            cursor[parts[-1]] = value

    config_dir = config_path.parent.resolve()
    return _parse_config_mapping(raw_mapping, config_dir)


def write_config_template(path: Path) -> None:
    """Write a default render.yaml template to `path`."""
    template_path = PACKAGE_ROOT / "templates" / "render.yaml"
    if not template_path.exists():
        raise FileNotFoundError("Bundled render.yaml template is missing.")
    path.write_text(template_path.read_text())
