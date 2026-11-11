"""Hidden scene builder for the Split4D gym (never imported by method/ or data/).

It procedurally creates dynamic multi-view scenes represented, as in the paper,
by fixed Freetime Gaussian primitives (4D mean, temporal scale, linear velocity,
opacity, footprint) plus per-image observations:

* training-view pixel samples with alpha-blended contributor lists and
  independent per-image instance-mask ids from a simulated automatic
  segmenter (random ids per image, merges at contacts, part splits, boundary
  flips, random errors, dropped tiny masks), and
* hidden test-view pixel samples whose visible-instance answers are only
  written to the evaluator truth folders.

Outputs (all deterministic):
  data/scenes/<id>.npz, data/scenes/manifest.json   public inputs, no answers
  eval/train/truth/<id>.npz, manifest.json            answers for dev scenes
  eval/test/truth/<id>.npz, manifest.json             answers for test scenes

Primitive rows are randomly permuted before anything is written, so neither row
order nor any public field encodes instance identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[2]
PUBLIC_DIR = ROOT / "data" / "scenes"
TRAIN_TRUTH_DIR = ROOT / "eval" / "train" / "truth"
TEST_TRUTH_DIR = ROOT / "eval" / "test" / "truth"

SECRET = "split4d-hidden-scene-builder-v2"
FRAMES = 60
TRAIN_VIEWS = 6
TEST_VIEWS = 3
PIXELS = 200
CONTRIB = 5
GRID_DT = 0.05
T_PAD = 10.0
RENDER_MIN_OPACITY = 0.10
SEMANTIC_DIM = 12
SPLITS = {"dev": 3, "test": 5}
PROFILES = ("dynamic", "interaction")
EXPERIMENT_OF_PROFILE = {
    "dynamic": "scaled_dynamic_scene_segmentation",
    "interaction": "scaled_interaction_segmentation",
}

# Category geometry: list of parts (offset, radii, swing amplitude, share).
CATEGORIES: Dict[str, Dict[str, object]] = {
    "person": {"count": 240, "parts": [
        ((0.0, 0.0, 1.56), (0.11, 0.10, 0.12), (0.0, 0.0, 0.0), 0.14),
        ((0.0, 0.0, 1.13), (0.18, 0.13, 0.28), (0.02, 0.02, 0.0), 0.46),
        ((0.0, 0.0, 0.45), (0.16, 0.12, 0.43), (0.07, 0.05, 0.0), 0.40)]},
    "ball": {"count": 70, "parts": [((0.0, 0.0, 0.0), (0.12, 0.12, 0.12), (0.0, 0.0, 0.0), 1.0)]},
    "pan": {"count": 70, "parts": [((0.0, 0.0, 0.0), (0.17, 0.11, 0.05), (0.0, 0.0, 0.0), 1.0)]},
    "cup": {"count": 55, "parts": [((0.0, 0.0, 0.0), (0.07, 0.07, 0.09), (0.0, 0.0, 0.0), 1.0)]},
    "pet": {"count": 130, "parts": [
        ((0.0, 0.0, 0.26), (0.26, 0.11, 0.11), (0.03, 0.0, 0.02), 0.74),
        ((0.30, 0.0, 0.38), (0.08, 0.08, 0.08), (0.02, 0.0, 0.02), 0.26)]},
    "counter": {"count": 90, "parts": [((0.0, 0.0, 0.45), (0.90, 0.30, 0.45), (0.0, 0.0, 0.0), 1.0)]},
    "table": {"count": 80, "parts": [
        ((0.0, 0.0, 0.76), (0.50, 0.45, 0.04), (0.0, 0.0, 0.0), 0.62),
        ((0.0, 0.0, 0.37), (0.10, 0.10, 0.36), (0.0, 0.0, 0.0), 0.38)]},
    "shelf": {"count": 80, "parts": [((0.0, 0.0, 0.80), (0.20, 0.55, 0.80), (0.0, 0.0, 0.0), 1.0)]},
    "stool": {"count": 50, "parts": [((0.0, 0.0, 0.30), (0.17, 0.17, 0.30), (0.0, 0.0, 0.0), 1.0)]},
    "bench": {"count": 80, "parts": [((0.0, 0.0, 0.25), (0.80, 0.25, 0.25), (0.0, 0.0, 0.0), 1.0)]},
    "pillar": {"count": 60, "parts": [((0.0, 0.0, 0.70), (0.18, 0.18, 0.70), (0.0, 0.0, 0.0), 1.0)]},
}
STATIC_CATEGORIES = {"counter", "table", "shelf", "stool", "bench", "pillar"}
HANDLED = {"ball", "pan", "cup"}
SPLIT_PROB = {"person": 0.30, "pet": 0.25, "table": 0.20}


@dataclass
class Preset:
    merge_scale: float = 1.0
    split_scale: float = 1.0
    flip_prob: float = 0.25
    noise_prob: float = 0.02
    dyn_duration: Tuple[float, float] = (1.5, 3.0)


PRESETS = {
    "base": Preset(),
    "clean_masks": Preset(merge_scale=0.0, split_scale=0.0, flip_prob=0.0, noise_prob=0.02),
    "long_lived": Preset(dyn_duration=(6.0, 9.0)),
}


def merge_prob(cat_a: str, cat_b: str) -> float:
    pair = {cat_a, cat_b}
    if pair == {"person"}:
        return 0.45
    if "person" in pair and pair & HANDLED:
        return 0.50
    if pair == {"person", "pet"}:
        return 0.30
    if "person" in pair and pair & STATIC_CATEGORIES:
        return 0.12
    if pair <= STATIC_CATEGORIES:
        return 0.15
    return 0.20


@dataclass
class SceneObject:
    name: str
    category: str
    static: bool
    root: np.ndarray  # (G, 3) root position on the dense time grid


def _seed(*parts: object) -> int:
    text = ":".join(str(p) for p in (SECRET,) + parts)
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "little")


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-9)


def _disk(rng: np.random.Generator, n: int, radius: float) -> np.ndarray:
    ang = rng.uniform(0.0, 2.0 * np.pi, n)
    rad = radius * np.sqrt(rng.uniform(0.0, 1.0, n))
    return np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)


def _spline_xy(times: np.ndarray, points: np.ndarray, grid: np.ndarray) -> np.ndarray:
    xy = CubicSpline(times, points, axis=0, bc_type="natural")(grid)
    return np.column_stack([xy, np.zeros(len(grid))])


def _at(grid: np.ndarray, path: np.ndarray, t: float) -> np.ndarray:
    return np.array([np.interp(t, grid, path[:, k]) for k in range(3)])


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def interaction_objects(rng: np.random.Generator, grid: np.ndarray) -> List[SceneObject]:
    people = 4
    wp_times = np.arange(-T_PAD, FRAMES + T_PAD + 1e-6, 10.0)
    n_wp = len(wp_times)
    wps = [_disk(rng, n_wp, 1.5) for _ in range(people)]
    busy = set()
    events = 0
    for _ in range(200):
        if events >= 4:
            break
        a, b = (int(x) for x in rng.choice(people, size=2, replace=False))
        k = int(rng.integers(1, n_wp - 2))
        slots = {(a, k), (b, k), (a, k + 1), (b, k + 1)}
        if slots & busy:
            continue
        meet = _disk(rng, 1, 1.0)[0]
        d = _unit(rng.normal(size=2))
        drift = rng.normal(0.0, 0.12, size=2)
        wps[a][k] = meet + 0.18 * d
        wps[b][k] = meet - 0.18 * d
        wps[a][k + 1] = meet + 0.18 * d + drift
        wps[b][k + 1] = meet - 0.18 * d + drift
        busy |= slots
        events += 1
    roots = [_spline_xy(wp_times, wps[i], grid) for i in range(people)]
    objects = [SceneObject(f"person_{i}", "person", False, roots[i]) for i in range(people)]

    hand = np.array([0.22, 0.05, 1.0])
    ball = np.zeros((len(grid), 3))
    t = -T_PAD - 1.0
    holder = 0
    while t < FRAMES + T_PAD + 1.0:
        hold = float(rng.uniform(6.0, 10.0))
        mask = (grid >= t) & (grid < t + hold)
        ball[mask] = roots[holder][mask] + hand
        t += hold
        nxt = int(rng.choice([p for p in range(people) if p != holder]))
        fly = float(rng.uniform(3.0, 5.0))
        p0 = _at(grid, roots[holder], t) + hand
        p1 = _at(grid, roots[nxt], t + fly) + hand
        mask = (grid >= t) & (grid < t + fly)
        u = (grid[mask] - t) / fly
        ball[mask] = (1 - u)[:, None] * p0 + u[:, None] * p1
        ball[mask, 2] += 0.35 * np.sin(np.pi * u)
        t += fly
        holder = nxt
    objects.append(SceneObject("ball", "ball", False, ball))
    objects.append(SceneObject("bench", "bench", True, np.tile([0.0, -2.35, 0.0], (len(grid), 1))))
    objects.append(SceneObject("pillar", "pillar", True, np.tile([-2.25, 1.0, 0.0], (len(grid), 1))))
    return objects


def dynamic_objects(rng: np.random.Generator, grid: np.ndarray) -> List[SceneObject]:
    statics = {
        "counter": np.array([0.0, 1.30, 0.0]),
        "table": np.array([1.45, -0.35, 0.0]),
        "shelf": np.array([-1.75, 0.20, 0.0]),
        "stool": np.array([-0.45, -1.25, 0.0]),
    }
    wp_times = np.arange(-T_PAD, FRAMES + T_PAD + 1e-6, 8.0)
    n_wp = len(wp_times)
    person_wp = np.column_stack([rng.uniform(-1.0, 0.85, n_wp), rng.uniform(-0.7, 0.7, n_wp)])
    counter_spots = [np.array([-0.45, 1.05, 0.97]), np.array([0.45, 1.05, 0.99]), np.array([-0.05, 1.05, 0.98])]
    table_spot = np.array([1.30, -0.28, 0.85])
    kp1 = int(rng.integers(2, 4))
    kd1 = kp1 + 2
    kp2 = int(min(kd1 + int(rng.integers(1, 3)), n_wp - 4))
    kd2 = kp2 + 2
    plans = [
        ("pan", counter_spots[0], table_spot, kp1, kd1),
        ("cup", counter_spots[1], counter_spots[2], kp2, kd2),
    ]
    for _cat, start, end, kp, kd in plans:
        person_wp[kp] = start[:2] + np.array([0.0, -0.38])
        person_wp[kd] = end[:2] + (np.array([-0.38, 0.0]) if end[0] > 1.0 else np.array([0.0, -0.38]))
    person = _spline_xy(wp_times, person_wp, grid)
    objects = [SceneObject("person", "person", False, person)]
    hand = person + np.array([0.15, 0.15, 0.95])
    for cat, start, end, kp, kd in plans:
        tp = float(wp_times[kp])
        td = float(wp_times[kd])
        a = _smoothstep((grid - tp) / 2.0)[:, None]
        b = _smoothstep((grid - td) / 2.0)[:, None]
        carried = (1 - a) * start + a * hand
        path = (1 - b) * carried + b * end
        objects.append(SceneObject(cat, cat, False, path))
    pet_times = np.arange(-T_PAD, FRAMES + T_PAD + 1e-6, 5.0)
    pet_wp = np.column_stack([rng.uniform(-1.2, 1.0, len(pet_times)), rng.uniform(-1.0, 0.3, len(pet_times))])
    objects.append(SceneObject("pet", "pet", False, _spline_xy(pet_times, pet_wp, grid)))
    for name, pos in statics.items():
        objects.append(SceneObject(name, name, True, np.tile(pos, (len(grid), 1))))
    return objects


SUPER_CATEGORY = {"counter": "furniture", "table": "furniture", "shelf": "furniture", "stool": "furniture",
                  "bench": "furniture", "pillar": "furniture", "pan": "handheld", "cup": "handheld",
                  "ball": "handheld", "person": "animate", "pet": "animate"}


def category_embeddings() -> Tuple[Dict[str, np.ndarray], Dict[Tuple[str, int], np.ndarray]]:
    """Fixed (scene-independent) semantic prototypes: super-category + category + part."""

    supers = {s: _unit(np.random.default_rng(_seed("super", s)).normal(size=SEMANTIC_DIM))
              for s in sorted(set(SUPER_CATEGORY.values()))}
    cats: Dict[str, np.ndarray] = {}
    parts: Dict[Tuple[str, int], np.ndarray] = {}
    for name in sorted(CATEGORIES):
        rng = np.random.default_rng(_seed("category", name))
        cats[name] = 0.8 * supers[SUPER_CATEGORY[name]] + 0.5 * _unit(rng.normal(size=SEMANTIC_DIM))
        for p in range(len(CATEGORIES[name]["parts"])):
            parts[(name, p)] = _unit(rng.normal(size=SEMANTIC_DIM))
    return cats, parts


def sample_primitives(rng: np.random.Generator, objects: Sequence[SceneObject], grid: np.ndarray, preset: Preset) -> Dict[str, np.ndarray]:
    cat_emb, part_emb = category_embeddings()
    rows: Dict[str, List[np.ndarray]] = {k: [] for k in (
        "mean_xyz", "velocity", "mu_t", "duration", "opacity", "scale", "semantic", "object", "part")}

    def interp_path(path: np.ndarray, times: np.ndarray) -> np.ndarray:
        return np.stack([np.interp(times, grid, path[:, k]) for k in range(3)], axis=1)

    for obj_id, obj in enumerate(objects):
        spec = CATEGORIES[obj.category]
        parts = spec["parts"]
        n = int(round(int(spec["count"]) * rng.uniform(0.9, 1.1)))
        shares = np.array([p[3] for p in parts], dtype=float)
        part = rng.choice(len(parts), size=n, p=shares / shares.sum())
        direction = rng.normal(size=(n, 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        radius = rng.uniform(0.55, 1.0, size=n)
        offsets = np.array([p[0] for p in parts])[part]
        radii = np.array([p[1] for p in parts])[part]
        swing = np.array([p[2] for p in parts])[part]
        local = radii * direction * radius[:, None]
        omega = rng.uniform(0.45, 0.75)
        phase = rng.uniform(0, 2 * np.pi, size=len(parts))[part]
        if obj.static:
            mu = rng.uniform(-6.0, FRAMES + 5.0, size=n)
            dur = rng.uniform(25.0, 50.0, size=n)
        else:
            mu = rng.uniform(-4.0, FRAMES + 3.0, size=n)
            dur = rng.uniform(preset.dyn_duration[0], preset.dyn_duration[1], size=n)

        def world(times: np.ndarray) -> np.ndarray:
            osc = swing * np.sin(omega * times + phase)[:, None]
            return interp_path(obj.root, times) + offsets + osc + local

        pos = world(mu) + rng.normal(0.0, 0.012, size=(n, 3))
        vel = (world(mu + 0.25) - world(mu - 0.25)) / 0.5
        speed = np.linalg.norm(vel, axis=1, keepdims=True)
        if obj.static:
            vel = vel + rng.normal(0.0, 0.0008, size=(n, 3))
        else:
            vel = vel + rng.normal(size=(n, 3)) * (0.2 * speed + 0.006)
        volume = 4.0 / 3.0 * np.pi * np.prod(np.array([p[1] for p in parts]), axis=1)
        part_counts = np.bincount(part, minlength=len(parts)).astype(float)
        live_fraction = 1.0 if obj.static else 0.12
        per_prim = volume / np.maximum(part_counts * live_fraction, 1.0)
        scale = np.clip(0.75 * np.cbrt(per_prim), 0.035, 0.16)[part] * rng.uniform(0.85, 1.15, size=n)
        inst = _unit(rng.normal(size=SEMANTIC_DIM))
        sem = (cat_emb[obj.category][None, :]
               + 0.40 * np.stack([part_emb[(obj.category, int(p))] for p in part])
               + 0.25 * inst[None, :]
               + rng.normal(0.0, 0.25, size=(n, SEMANTIC_DIM)))
        sem /= np.linalg.norm(sem, axis=1, keepdims=True)
        rows["mean_xyz"].append(pos)
        rows["velocity"].append(vel)
        rows["mu_t"].append(mu)
        rows["duration"].append(dur)
        rows["opacity"].append(rng.uniform(0.55, 0.95, size=n))
        rows["scale"].append(scale)
        rows["semantic"].append(sem)
        rows["object"].append(np.full(n, obj_id))
        rows["part"].append(part)
    prims = {k: np.concatenate(v) for k, v in rows.items()}
    perm = rng.permutation(len(prims["mu_t"]))
    return {k: v[perm] for k, v in prims.items()}


def camera_axes(azimuth: float, elevation: float) -> np.ndarray:
    fwd = -np.array([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
    x_axis = _unit(np.cross(fwd, np.array([0.0, 0.0, 1.0])))
    y_axis = np.cross(x_axis, fwd)
    return np.stack([x_axis, y_axis, fwd], axis=0)


def render(prims: Dict[str, np.ndarray], t: float, axes: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Sample pixels of one image and return contributor ids and blend weights."""

    op_t = np.exp(-0.5 * ((t - prims["mu_t"]) / prims["duration"]) ** 2)
    act = np.flatnonzero(op_t >= RENDER_MIN_OPACITY)
    pos = prims["mean_xyz"][act] + prims["velocity"][act] * (t - prims["mu_t"][act])[:, None]
    uv = pos @ axes[:2].T
    depth = pos @ axes[2]
    alpha_base = prims["opacity"][act] * op_t[act]
    scale = prims["scale"][act]
    area = alpha_base * scale ** 2
    obj = prims["object"][act]
    objs = np.unique(obj)
    mass = np.array([area[obj == o].sum() for o in objs])
    share = np.sqrt(mass) / np.sqrt(mass).sum()
    spare = max(PIXELS - 6 * len(objs), 0)
    counts = 6 + np.floor(share * spare).astype(int)
    sources = []
    for o, c in zip(objs, counts):
        members = np.flatnonzero(obj == o)
        p = area[members] / area[members].sum()
        sources.append(rng.choice(members, size=int(c), p=p))
    src = np.concatenate(sources)
    pix = uv[src] + rng.normal(size=(len(src), 2)) * (0.6 * scale[src])[:, None]
    d2 = ((pix[:, None, :] - uv[None, :, :]) ** 2).sum(axis=2)
    k = min(CONTRIB, len(act))
    nn = np.argpartition(d2, k - 1, axis=1)[:, :k]
    order = np.argsort(depth[nn], axis=1)
    nn = np.take_along_axis(nn, order, axis=1)
    dd = np.take_along_axis(d2, nn, axis=1)
    sig = scale[nn]
    alpha = np.clip(alpha_base[nn] * np.exp(-dd / (2.0 * sig ** 2)), 0.0, 0.99)
    trans = np.concatenate([np.ones((len(src), 1)), np.cumprod(1.0 - alpha[:, :-1], axis=1)], axis=1)
    weight = alpha * trans
    keep = weight.sum(axis=1) >= 0.05
    contrib = act[nn][keep]
    weight = weight[keep]
    if k < CONTRIB:
        pad = CONTRIB - k
        contrib = np.concatenate([contrib, np.repeat(contrib[:, :1], pad, axis=1)], axis=1)
        weight = np.concatenate([weight, np.zeros((len(weight), pad))], axis=1)
    return contrib.astype(np.int64), weight


def object_weights(prims: Dict[str, np.ndarray], contrib: np.ndarray, weight: np.ndarray, n_objects: int) -> np.ndarray:
    objw = np.zeros((len(contrib), n_objects))
    rows = np.repeat(np.arange(len(contrib)), contrib.shape[1])
    np.add.at(objw, (rows, prims["object"][contrib].ravel()), weight.ravel())
    return objw


def simulate_masks(prims: Dict[str, np.ndarray], objects: Sequence[SceneObject], contrib: np.ndarray, weight: np.ndarray,
                   rng: np.random.Generator, preset: Preset) -> np.ndarray:
    """Per-image automatic instance masks with realistic, structured errors."""

    n_obj = len(objects)
    objw = object_weights(prims, contrib, weight, n_obj)
    order = np.argsort(-objw, axis=1)
    vis = order[:, 0]
    second = order[:, 1] if n_obj > 1 else order[:, 0]
    w1 = objw[np.arange(len(vis)), vis]
    w2 = objw[np.arange(len(vis)), second]

    parent = list(range(n_obj))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    ambiguous = w2 >= 0.3 * w1
    touch = np.zeros((n_obj, n_obj), dtype=int)
    np.add.at(touch, (vis[ambiguous], second[ambiguous]), 1)
    touch = touch + touch.T
    for a in range(n_obj):
        for b in range(a + 1, n_obj):
            if touch[a, b] >= 2 and rng.random() < preset.merge_scale * merge_prob(objects[a].category, objects[b].category):
                parent[find(a)] = find(b)
    roots = np.array([find(o) for o in range(n_obj)])
    group_size = np.bincount(roots, minlength=n_obj)
    split = np.array([
        group_size[roots[o]] == 1 and rng.random() < preset.split_scale * SPLIT_PROB.get(objects[o].category, 0.0)
        for o in range(n_obj)
    ])

    obj_c = prims["object"][contrib]
    part_c = prims["part"][contrib]

    def keys_for(target: np.ndarray) -> np.ndarray:
        masked = np.where(obj_c == target[:, None], weight, -1.0)
        best = masked.argmax(axis=1)
        part = part_c[np.arange(len(target)), best]
        return roots[target] * 16 + np.where(split[target], part, 0)

    keys = keys_for(vis)
    flip = (w2 >= 0.5 * w1) & (second != vis) & (rng.random(len(vis)) < preset.flip_prob)
    if np.any(flip):
        keys[flip] = keys_for(second)[flip]
    uniq = np.unique(keys)
    noisy = rng.random(len(keys)) < preset.noise_prob
    if np.any(noisy) and len(uniq) > 1:
        keys[noisy] = rng.choice(uniq, size=int(noisy.sum()))
    uniq, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    ids = rng.choice(30000, size=len(uniq), replace=False)
    labels = ids[inverse].astype(np.int64)
    labels[counts[inverse] < 3] = -1
    return labels


def build_scene(scene_id: str, profile: str, preset: Preset) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    rng = np.random.default_rng(_seed("scene", scene_id))
    grid = np.arange(-T_PAD - 2.0, FRAMES + T_PAD + 2.0 + 1e-9, GRID_DT)
    objects = interaction_objects(rng, grid) if profile == "interaction" else dynamic_objects(rng, grid)
    prims = sample_primitives(rng, objects, grid, preset)
    n_obj = len(objects)

    window = (grid >= 0) & (grid <= FRAMES - 1)
    dynamic = np.array([
        float(np.max(np.linalg.norm(o.root[window] - o.root[window][0], axis=1))) > 0.25 for o in objects
    ])

    az0 = float(rng.uniform(0, 2 * np.pi))
    train_axes = [camera_axes(az0 + 2 * np.pi * v / TRAIN_VIEWS, np.deg2rad(rng.uniform(12.0, 28.0))) for v in range(TRAIN_VIEWS)]
    test_axes = [camera_axes(az0 + np.pi / TRAIN_VIEWS + 2 * np.pi * v / TEST_VIEWS, np.deg2rad(20.0)) for v in range(TEST_VIEWS)]

    obs_t, obs_view, offsets, all_c, all_w, all_l = [], [], [0], [], [], []
    for t in range(FRAMES):
        for v in range(TRAIN_VIEWS):
            c, w = render(prims, float(t), train_axes[v], rng)
            lab = simulate_masks(prims, objects, c, w, rng, preset)
            obs_t.append(t)
            obs_view.append(v)
            all_c.append(c)
            all_w.append(w)
            all_l.append(lab)
            offsets.append(offsets[-1] + len(c))
    test_t, test_view, test_off, tc, tw, tv = [], [], [0], [], [], []
    for t in range(FRAMES):
        for v in range(TEST_VIEWS):
            c, w = render(prims, float(t), test_axes[v], rng)
            objw = object_weights(prims, c, w, n_obj)
            test_t.append(t)
            test_view.append(v)
            tc.append(c)
            tw.append(w)
            tv.append(objw.argmax(axis=1))
            test_off.append(test_off[-1] + len(c))

    def q8(x: np.ndarray) -> np.ndarray:
        return np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)

    public = {
        "scene_id": np.array(scene_id),
        "profile": np.array(profile),
        "frames": np.array(FRAMES, dtype=np.int64),
        "views": np.array(TRAIN_VIEWS, dtype=np.int64),
        "mean_xyz": prims["mean_xyz"].astype(np.float32),
        "velocity": prims["velocity"].astype(np.float32),
        "mu_t": prims["mu_t"].astype(np.float32),
        "duration": prims["duration"].astype(np.float32),
        "opacity": prims["opacity"].astype(np.float32),
        "scale": prims["scale"].astype(np.float32),
        "semantic": prims["semantic"].astype(np.float32),
        "view_axes": np.stack(train_axes).astype(np.float32),
        "obs_t": np.array(obs_t, dtype=np.int16),
        "obs_view": np.array(obs_view, dtype=np.int16),
        "obs_offsets": np.array(offsets, dtype=np.int64),
        "pix_contrib": np.concatenate(all_c).astype(np.int16),
        "pix_weight_u8": q8(np.concatenate(all_w)),
        "pix_label": np.concatenate(all_l).astype(np.int16),
    }
    truth = {
        "scene_id": np.array(scene_id),
        "profile": np.array(profile),
        "object_id": prims["object"].astype(np.int16),
        "object_dynamic": dynamic,
        "object_category": np.array([o.category for o in objects]),
        "test_t": np.array(test_t, dtype=np.int16),
        "test_view": np.array(test_view, dtype=np.int16),
        "test_offsets": np.array(test_off, dtype=np.int64),
        "test_contrib": np.concatenate(tc).astype(np.int16),
        "test_weight_u8": q8(np.concatenate(tw)),
        "test_visible": np.concatenate(tv).astype(np.int16),
    }
    return public, truth


def scene_ids() -> List[Tuple[str, str, str]]:
    out = []
    for profile in PROFILES:
        for split, count in SPLITS.items():
            for i in range(count):
                out.append((f"{profile}_{split}_{i}", profile, split))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--public-dir", default=str(PUBLIC_DIR))
    parser.add_argument("--train-truth-dir", default=str(TRAIN_TRUTH_DIR))
    parser.add_argument("--test-truth-dir", default=str(TEST_TRUTH_DIR))
    parser.add_argument("--preset", default="base", choices=sorted(PRESETS))
    parser.add_argument("--only-split", default=None, choices=sorted(SPLITS))
    args = parser.parse_args()
    preset = PRESETS[args.preset]
    public_dir = Path(args.public_dir)
    truth_dirs = {"dev": Path(args.train_truth_dir), "test": Path(args.test_truth_dir)}
    for d in [public_dir, *truth_dirs.values()]:
        d.mkdir(parents=True, exist_ok=True)

    manifest = {"frames": FRAMES, "train_views": TRAIN_VIEWS, "contributors_per_pixel": CONTRIB, "scenes": []}
    truth_manifests: Dict[str, Dict[str, object]] = {s: {"experiments": {}, "scenes": {}} for s in SPLITS}
    for scene_id, profile, split in scene_ids():
        if args.only_split and split != args.only_split:
            continue
        public, truth = build_scene(scene_id, profile, preset)
        np.savez_compressed(public_dir / f"{scene_id}.npz", **public)
        np.savez_compressed(truth_dirs[split] / f"{scene_id}.npz", **truth)
        n = int(len(public["mu_t"]))
        manifest["scenes"].append({"id": scene_id, "profile": profile, "split": split,
                                   "file": f"{scene_id}.npz", "primitives": n})
        exp = EXPERIMENT_OF_PROFILE[profile]
        tm = truth_manifests[split]
        tm["experiments"].setdefault(exp, []).append(scene_id)
        tm["scenes"][scene_id] = {"file": f"{scene_id}.npz", "primitives": n,
                                  "objects": int(len(truth["object_dynamic"])),
                                  "dynamic_objects": int(truth["object_dynamic"].sum())}
        print(f"built {scene_id}: primitives={n} objects={len(truth['object_dynamic'])} "
              f"dynamic={int(truth['object_dynamic'].sum())} pixels={len(public['pix_label'])}", flush=True)
    (public_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for split, tm in truth_manifests.items():
        if tm["scenes"]:
            (truth_dirs[split] / "manifest.json").write_text(json.dumps(tm, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
