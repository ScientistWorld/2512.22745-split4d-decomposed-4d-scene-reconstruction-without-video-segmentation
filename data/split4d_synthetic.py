"""Deterministic small-scale data for Split4D reproduction.

The original paper evaluates on large multi-view videos with fixed reconstructed
4D Gaussian scenes.  This module creates compact fixed 4D Gaussian scenes with
the same ingredients needed by the paper's feature-learning stage: temporally
free Gaussian primitives, linear motion, per-image inconsistent instance masks,
and optional semantic guidance vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass(frozen=True)
class SceneConfig:
    name: str
    frames: int
    views: int
    instances: int
    primitives_per_instance: int
    duration: float
    seed: int
    nonlinear: bool = False
    semantic_collision: bool = False


SCENES: Dict[str, SceneConfig] = {
    "linear_motion": SceneConfig(
        name="linear_motion",
        frames=36,
        views=4,
        instances=4,
        primitives_per_instance=36,
        duration=9.0,
        seed=42,
    ),
    "interaction_motion": SceneConfig(
        name="interaction_motion",
        frames=44,
        views=5,
        instances=5,
        primitives_per_instance=42,
        duration=8.0,
        seed=123,
        nonlinear=True,
        semantic_collision=True,
    ),
}


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / max(n, 1e-8)


def build_scene(name: str) -> Dict[str, np.ndarray]:
    if name not in SCENES:
        raise KeyError(f"Unknown scene {name!r}. Available: {sorted(SCENES)}")
    cfg = SCENES[name]
    rng = np.random.default_rng(cfg.seed)
    total = cfg.instances * cfg.primitives_per_instance
    object_id = np.repeat(np.arange(cfg.instances), cfg.primitives_per_instance)

    base_angles = np.linspace(0.0, 2.0 * np.pi, cfg.instances, endpoint=False)
    centers = np.stack(
        [1.25 * np.cos(base_angles), 1.25 * np.sin(base_angles), np.linspace(-0.4, 0.4, cfg.instances)],
        axis=1,
    )
    velocities = np.stack(
        [0.045 * np.sin(base_angles * 1.7 + 0.3), 0.05 * np.cos(base_angles * 1.3), 0.02 * np.sin(base_angles)],
        axis=1,
    )
    if cfg.nonlinear:
        velocities[:2] *= -1.8
        velocities[2:] *= 1.35

    means = np.zeros((total, 3), dtype=np.float32)
    mus_t = np.zeros(total, dtype=np.float32)
    durations = np.zeros(total, dtype=np.float32)
    primitive_velocity = np.zeros((total, 3), dtype=np.float32)
    semantic = np.zeros((total, 12), dtype=np.float32)

    semantic_centers = rng.normal(size=(cfg.instances, 12)).astype(np.float32)
    if cfg.semantic_collision and cfg.instances >= 3:
        semantic_centers[1] = semantic_centers[0] + rng.normal(scale=0.03, size=12)
    semantic_centers = semantic_centers / np.linalg.norm(semantic_centers, axis=1, keepdims=True)

    idx = 0
    for inst in range(cfg.instances):
        replacement_times = np.linspace(0, cfg.frames - 1, cfg.primitives_per_instance)
        for local in range(cfg.primitives_per_instance):
            mu_t = replacement_times[local] + rng.normal(scale=0.45)
            t_norm = mu_t / max(cfg.frames - 1, 1)
            curved = np.array(
                [
                    0.45 * np.sin(2 * np.pi * t_norm + inst),
                    0.35 * np.cos(1.7 * np.pi * t_norm + 0.4 * inst),
                    0.18 * np.sin(3.0 * np.pi * t_norm + 0.2 * inst),
                ]
            )
            if not cfg.nonlinear:
                curved *= 0.25
            center_at_time = centers[inst] + velocities[inst] * mu_t + curved
            means[idx] = center_at_time + rng.normal(scale=0.08, size=3)
            mus_t[idx] = mu_t
            durations[idx] = cfg.duration * rng.uniform(0.75, 1.25)
            primitive_velocity[idx] = velocities[inst] + rng.normal(scale=0.018 if cfg.nonlinear else 0.01, size=3)
            semantic[idx] = semantic_centers[inst] + rng.normal(scale=0.07, size=12)
            idx += 1

    semantic /= np.linalg.norm(semantic, axis=1, keepdims=True)
    view_dirs = np.stack(
        [
            [np.cos(a), np.sin(a), 0.25 * np.sin(2 * a)]
            for a in np.linspace(0, 2 * np.pi, cfg.views, endpoint=False)
        ],
        axis=0,
    ).astype(np.float32)
    view_dirs = np.stack([_unit(v) for v in view_dirs], axis=0)

    return {
        "name": np.array(cfg.name),
        "frames": np.array(cfg.frames, dtype=np.int64),
        "views": np.array(cfg.views, dtype=np.int64),
        "instances": np.array(cfg.instances, dtype=np.int64),
        "mean_xyz": means,
        "mu_t": mus_t,
        "duration": durations,
        "velocity": primitive_velocity,
        "object_id": object_id.astype(np.int64),
        "semantic": semantic.astype(np.float32),
        "view_dirs": view_dirs,
    }


def position_at(scene: Dict[str, np.ndarray], t: int) -> np.ndarray:
    return scene["mean_xyz"] + scene["velocity"] * (float(t) - scene["mu_t"])[:, None]


def temporal_opacity(scene: Dict[str, np.ndarray], t: int) -> np.ndarray:
    return np.exp(-0.5 * ((float(t) - scene["mu_t"]) / scene["duration"]) ** 2)


def active_indices(scene: Dict[str, np.ndarray], t: int, min_opacity: float = 0.22) -> np.ndarray:
    return np.flatnonzero(temporal_opacity(scene, t) >= min_opacity)


def observation_labels(scene: Dict[str, np.ndarray], view: int, t: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return active primitive indices and inconsistent per-image mask ids.

    Mask ids are deliberately permuted by view and frame.  The learning code can
    only rely on co-membership within the same image, matching the paper's
    setting where labels need not be temporally or cross-view consistent.
    """

    idx = active_indices(scene, t)
    obj = scene["object_id"][idx].copy()
    inst_count = int(scene["instances"])
    perm_rng = np.random.default_rng(10_000 + 97 * int(view) + 131 * int(t) + int(scene["frames"]))
    perm = perm_rng.permutation(inst_count)
    labels = perm[obj]

    # Controlled per-image segmentation defects: a small fraction of primitives
    # are assigned to a different mask, resembling imperfect SAM automasks.
    noise_rng = np.random.default_rng(20_000 + 31 * int(view) + 197 * int(t))
    corrupt = noise_rng.random(len(labels)) < (0.035 if int(scene["instances"]) <= 4 else 0.055)
    labels[corrupt] = noise_rng.integers(0, inst_count, size=int(corrupt.sum()))
    return idx.astype(np.int64), labels.astype(np.int64)


def iter_observations(scene: Dict[str, np.ndarray], streaming: bool = True) -> Iterable[Tuple[int, int, np.ndarray, np.ndarray]]:
    frames = range(int(scene["frames"]))
    views = range(int(scene["views"]))
    pairs: List[Tuple[int, int]] = [(t, v) for t in frames for v in views]
    if not streaming:
        rng = np.random.default_rng(777 + int(scene["frames"]))
        rng.shuffle(pairs)
    for t, v in pairs:
        idx, labels = observation_labels(scene, v, t)
        yield t, v, idx, labels


def save_all(out_dir: str = "data/generated") -> None:
    import json
    from pathlib import Path

    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    manifest = {"scenes": list(SCENES)}
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for name in SCENES:
        scene = build_scene(name)
        np.savez_compressed(path / f"{name}.npz", **scene)


if __name__ == "__main__":
    save_all()
