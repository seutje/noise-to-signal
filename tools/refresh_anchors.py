"""Utility to regenerate anchor sets and projection matrices from existing renders.

Steps:
1. Scan rendered latent archives under `renders/**/latents.npz`.
2. Match each latent archive to its cached audio feature timeline.
3. Sample frame-wise latent/feature pairs per anchor family.
4. Run a light-weight k-means to choose representative anchor frames.
5. Fit a linear projection (with bias + feature normalisation) that maps
   controller features to anchor logits.
6. Emit refreshed anchor packs and projection metadata under `models/anchors/`
   and update `models/meta.json` accordingly.

Run:
    python tools/refresh_anchors.py

Adjustable flags allow experimenting with sample counts and anchor sizes.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from renderer.audio_features import FeatureLayout, FeatureTimeline


@dataclass(slots=True)
class AnchorSamples:
    latents: np.ndarray  # shape (samples, latent_size)
    features: np.ndarray  # shape (samples, feature_dim)
    frame_rate: int


def load_feature_cache(cache_path: Path) -> Tuple[FeatureTimeline, FeatureLayout]:
    if not cache_path.exists():
        raise FileNotFoundError(f"Feature cache missing: {cache_path}")
    with np.load(cache_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        layout_info = metadata["layout"]
        layout = FeatureLayout(
            sample_rate=int(layout_info["sample_rate"]),
            hop_length=int(layout_info["hop_length"]),
            frame_length=int(layout_info["frame_length"]),
            mfcc_components=int(layout_info["mfcc_components"]),
        )
        timeline = FeatureTimeline(
            times=data["times"].astype(np.float32),
            rms=data["rms"].astype(np.float32),
            centroid=data["centroid"].astype(np.float32),
            flatness=data["flatness"].astype(np.float32),
            mfcc=data["mfcc"].astype(np.float32),
            onset=data["onset"].astype(np.float32),
        )
    return timeline, layout


def resample_feature_matrix(
    timeline: FeatureTimeline,
    layout: FeatureLayout,
    frame_rate: int,
) -> np.ndarray:
    duration = float(timeline.times[-1]) if timeline.times.size else 0.0
    frame_count = max(1, int(math.ceil(duration * frame_rate)))
    frame_times = np.arange(frame_count, dtype=np.float32) / float(frame_rate)
    frame_times = np.clip(frame_times, 0.0, duration if duration > 0 else 0.0)

    def interp(values: np.ndarray) -> np.ndarray:
        return np.interp(
            frame_times,
            timeline.times,
            values,
            left=float(values[0]),
            right=float(values[-1]),
        )

    rms = interp(timeline.rms)
    centroid = interp(timeline.centroid) / (layout.sample_rate / 2.0)
    centroid = np.clip(centroid, 0.0, 1.0)
    flatness = interp(timeline.flatness)

    onset = interp(timeline.onset)
    if onset.max() > 0:
        onset = onset / onset.max()

    mfcc_components = min(timeline.mfcc.shape[0], 8)
    mfcc = np.vstack(
        [interp(timeline.mfcc[i]) for i in range(mfcc_components)]
    )
    mfcc = np.tanh(mfcc / 20.0)

    feature_matrix = np.concatenate(
        [
            rms[np.newaxis, :],
            centroid[np.newaxis, :],
            flatness[np.newaxis, :],
            onset[np.newaxis, :],
            mfcc,
        ],
        axis=0,
    ).T.astype(np.float32)
    return feature_matrix


def gather_samples(
    *,
    renders_root: Path,
    cache_root: Path,
    anchor_names: Iterable[str],
    rng: np.random.Generator,
    max_samples_per_anchor: int,
) -> Dict[str, AnchorSamples]:
    anchor_names = list(anchor_names)
    buckets: Dict[str, Dict[str, List[np.ndarray]]] = {
        name: {"latents": [], "features": [], "frame_rates": []} for name in anchor_names
    }

    latent_paths = sorted(renders_root.rglob("latents.npz"))
    if not latent_paths:
        raise RuntimeError("No latent archives found under renders/.")

    for latent_path in latent_paths:
        with np.load(latent_path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            anchor_set = meta.get("anchor_set")
            if anchor_set not in buckets:
                continue
            latents = data["latents"].reshape(data["latents"].shape[0], -1).astype(np.float32)
            frame_rate = int(meta.get("frame_rate", 60))

        track_id = latent_path.parent.name
        cache_path = cache_root / f"{track_id}.npz"
        try:
            timeline, layout = load_feature_cache(cache_path)
        except FileNotFoundError:
            print(f"[warn] feature cache missing for {track_id}; skipping.")
            continue

        feature_matrix = resample_feature_matrix(timeline, layout, frame_rate)
        if feature_matrix.shape[0] != latents.shape[0]:
            min_len = min(feature_matrix.shape[0], latents.shape[0])
            feature_matrix = feature_matrix[:min_len]
            latents = latents[:min_len]

        buckets[anchor_set]["latents"].append(latents)
        buckets[anchor_set]["features"].append(feature_matrix.astype(np.float32))
        buckets[anchor_set]["frame_rates"].append(np.full(latents.shape[0], frame_rate, dtype=np.int32))

    results: Dict[str, AnchorSamples] = {}
    for name, payload in buckets.items():
        if not payload["latents"]:
            print(f"[warn] no samples collected for anchor '{name}'.")
            continue
        latents = np.concatenate(payload["latents"], axis=0)
        features = np.concatenate(payload["features"], axis=0)
        frame_rates = np.concatenate(payload["frame_rates"], axis=0)

        if latents.shape[0] > max_samples_per_anchor:
            idx = rng.choice(latents.shape[0], size=max_samples_per_anchor, replace=False)
            latents = latents[idx]
            features = features[idx]
            frame_rates = frame_rates[idx]

        majority_frame_rate = int(np.bincount(frame_rates).argmax())
        results[name] = AnchorSamples(
            latents=latents,
            features=features,
            frame_rate=majority_frame_rate,
        )
        print(
            f"[info] anchor '{name}': {latents.shape[0]} samples "
            f"(feature_dim={features.shape[1]}, frame_rate≈{majority_frame_rate})."
        )
    return results


def kmeans_select(
    data: np.ndarray,
    feature_matrix: np.ndarray,
    *,
    k: int,
    rng: np.random.Generator,
    max_iter: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    if data.shape[0] < k:
        raise ValueError(f"Not enough samples ({data.shape[0]}) for k={k}.")

    indices = rng.choice(data.shape[0], size=k, replace=False)
    centers = data[indices].copy()
    prev_inertia = None

    for iteration in range(max_iter):
        distances = ((data[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)

        inertia = float(np.sum((data - centers[labels]) ** 2))
        if prev_inertia is not None and abs(prev_inertia - inertia) < 1e-3:
            break
        prev_inertia = inertia

        for i in range(k):
            mask = labels == i
            if not np.any(mask):
                centers[i] = data[rng.integers(0, data.shape[0])]
            else:
                centers[i] = data[mask].mean(axis=0)

    anchors = np.zeros((k, data.shape[1]), dtype=np.float32)
    energies = np.zeros(k, dtype=np.float32)
    for i in range(k):
        mask = labels == i
        if not np.any(mask):
            idx = rng.integers(0, data.shape[0])
            anchors[i] = data[idx]
            energies[i] = feature_matrix[idx, 0]
            continue
        cluster_latents = data[mask]
        cluster_feats = feature_matrix[mask]
        center = centers[i]
        dist = np.sum((cluster_latents - center[None, :]) ** 2, axis=1)
        best_idx = np.argmin(dist)
        anchors[i] = cluster_latents[best_idx]
        energies[i] = float(cluster_feats[:, 0].mean())

    order = np.argsort(energies)[::-1]
    anchors = anchors[order]
    energies = energies[order]
    return anchors, energies


def compute_projection(
    *,
    anchors: np.ndarray,
    features: np.ndarray,
    latents: np.ndarray,
    lambda_reg: float = 1e-3,
) -> Dict[str, np.ndarray]:
    latent_size = anchors.shape[1]
    feature_dim = features.shape[1]
    k = anchors.shape[0]

    pseudo = np.linalg.pinv(anchors.T, rcond=1e-5)
    weights = latents @ pseudo.T
    weights = np.maximum(weights, 0.0)
    sums = weights.sum(axis=1, keepdims=True)
    weights = np.where(sums > 1e-6, weights / sums, 1.0 / k)

    logits = np.log(weights + 1e-6)

    feature_mean = features.mean(axis=0)
    feature_std = features.std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0
    features_norm = (features - feature_mean) / feature_std

    ones = np.ones((features_norm.shape[0], 1), dtype=np.float32)
    design = np.hstack([features_norm, ones])

    xtx = design.T @ design
    xtx += np.eye(xtx.shape[0], dtype=np.float32) * lambda_reg
    xty = design.T @ logits
    theta = np.linalg.solve(xtx, xty)

    matrix = theta[:-1].T.astype(np.float32)
    bias = theta[-1].astype(np.float32)

    return {
        "matrix": matrix,
        "bias": bias,
        "feature_mean": feature_mean.astype(np.float32),
        "feature_std": feature_std.astype(np.float32),
    }


def save_anchor_pack(path: Path, anchors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, anchors=anchors.astype(np.float32))


def save_projection_pack(path: Path, payload: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def refresh_anchor_sets(
    *,
    args: argparse.Namespace,
    meta: Dict[str, object],
) -> Dict[str, Dict[str, str]]:
    rng = np.random.default_rng(args.seed)

    anchor_sets = meta.get("anchor_sets", {})
    if not anchor_sets:
        raise RuntimeError("models/meta.json does not define anchor_sets.")

    samples = gather_samples(
        renders_root=args.renders_root,
        cache_root=args.cache_root,
        anchor_names=anchor_sets.keys(),
        rng=rng,
        max_samples_per_anchor=args.max_samples,
    )

    projections_meta: Dict[str, Dict[str, str]] = {}

    for name, sample in samples.items():
        latent_count = sample.latents.shape[0]
        k = min(args.anchor_count, latent_count)
        anchors_flat, energies = kmeans_select(
            sample.latents,
            sample.features,
            k=k,
            rng=rng,
        )
        anchors = anchors_flat.reshape(k, args.latent_shape[0], args.latent_shape[1], args.latent_shape[2])

        anchor_path = args.output_root / f"{name}.npz"
        save_anchor_pack(anchor_path, anchors)

        projection_payload = compute_projection(
            anchors=anchors_flat,
            features=sample.features,
            latents=sample.latents,
        )
        projection_path = args.output_root / f"{name}_projection.npz"
        save_projection_pack(projection_path, projection_payload)

        info = anchor_sets[name]
        info["file"] = str(anchor_path.as_posix())
        info["anchor_count"] = int(k)
        info["seed"] = int(args.seed)
        info["energy_hint"] = float(np.mean(energies))

        projections_meta[name] = {
            "file": str(projection_path.as_posix()),
            "normalization": "zscore",
        }

        print(
            f"[done] {name}: k={k}, anchors saved to {anchor_path}, "
            f"projection saved to {projection_path}."
        )
    return projections_meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh anchor packs using rendered latent archives.")
    parser.add_argument("--renders-root", type=Path, default=Path("renders"))
    parser.add_argument("--cache-root", type=Path, default=Path("cache/features"))
    parser.add_argument("--output-root", type=Path, default=Path("models/anchors"))
    parser.add_argument("--anchor-count", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=45000)
    parser.add_argument("--seed", type=int, default=20241103)
    parser.add_argument("--latent-shape", type=int, nargs=3, default=[8, 16, 16])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta_path = Path("models/meta.json")
    meta = json.loads(meta_path.read_text())
    args.latent_shape = tuple(args.latent_shape)  # type: ignore[assignment]

    projections_meta = refresh_anchor_sets(args=args, meta=meta)
    meta["projections"] = projections_meta
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[meta] models/meta.json updated with projection references.")


if __name__ == "__main__":
    main()
