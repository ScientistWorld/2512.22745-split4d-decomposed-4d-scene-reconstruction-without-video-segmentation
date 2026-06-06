"""Small-scale implementation of Split4D feature learning.

This code implements the paper's feature-stage algorithm on fixed 4D Gaussian
geometry: linear-motion Gaussian primitives, per-Gaussian learnable features,
streaming feature learning from framewise inconsistent masks, 3D tracking
regularization, semantic feature alignment, and clustering/filtering inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

from data.split4d_synthetic import SCENES, active_indices, build_scene, iter_observations, observation_labels, position_at, temporal_opacity

try:
    import hdbscan
except ImportError:  # pragma: no cover - fallback is exercised when optional dep is absent.
    hdbscan = None


def _nonzero1d(mask: torch.Tensor) -> torch.Tensor:
    return torch.nonzero(mask, as_tuple=False).flatten()


def _torch_scene(scene: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        k: torch.as_tensor(v, device=device)
        for k, v in scene.items()
        if isinstance(v, np.ndarray) and v.dtype.kind in {"f", "i", "u", "b"}
    }


def contrastive_cluster_loss(features: torch.Tensor, labels: torch.Tensor, samples_per_mask: int, generator: torch.Generator) -> torch.Tensor:
    unique = torch.unique(labels)
    chunks: List[torch.Tensor] = []
    centers: List[torch.Tensor] = []
    for lab in unique:
        local = _nonzero1d(labels == lab)
        if local.numel() < 2:
            continue
        take = local[torch.randint(local.numel(), (min(samples_per_mask, local.numel()),), device=features.device, generator=generator)]
        vals = F.normalize(features[take], dim=-1)
        chunks.append(vals)
        centers.append(F.normalize(vals.mean(dim=0), dim=0))
    if len(chunks) < 2:
        return features.sum() * 0.0
    centers_t = torch.stack(centers, dim=0)
    losses = []
    for cls, vals in enumerate(chunks):
        phi = ((vals - centers_t[cls]).pow(2).sum(dim=-1).mean() / (vals.shape[0] * np.log(vals.shape[0] + 10.0))).detach()
        phi = torch.clamp(phi, min=0.035, max=0.25)
        logits = vals @ centers_t.T / phi
        target = torch.full((vals.shape[0],), cls, dtype=torch.long, device=features.device)
        losses.append(F.cross_entropy(logits, target))
    return torch.stack(losses).mean()


def dino_alignment_loss(features: torch.Tensor, projected_semantic: torch.Tensor, labels: torch.Tensor, samples_per_mask: int, generator: torch.Generator) -> torch.Tensor:
    unique = torch.unique(labels)
    chunks: List[torch.Tensor] = []
    centers: List[torch.Tensor] = []
    temperatures: List[torch.Tensor] = []
    for lab in unique:
        local = _nonzero1d(labels == lab)
        if local.numel() < 2:
            continue
        take = local[torch.randint(local.numel(), (min(samples_per_mask, local.numel()),), device=features.device, generator=generator)]
        vals = F.normalize(features[take], dim=-1)
        chunks.append(vals)
        centers.append(F.normalize(projected_semantic[take].mean(dim=0), dim=0))
        learned_center = F.normalize(vals.mean(dim=0), dim=0)
        phi = ((vals - learned_center).pow(2).sum(dim=-1).mean() / (vals.shape[0] * np.log(vals.shape[0] + 10.0))).detach()
        temperatures.append(torch.clamp(phi, min=0.035, max=0.25))
    if len(chunks) < 2:
        return features.sum() * 0.0
    centers_t = torch.stack(centers, dim=0)
    losses = []
    for cls, vals in enumerate(chunks):
        logits = vals @ centers_t.T / temperatures[cls]
        target = torch.full((vals.shape[0],), cls, dtype=torch.long, device=features.device)
        losses.append(F.cross_entropy(logits, target))
    return torch.stack(losses).mean()


def render_pixel_features(raw_features: torch.Tensor, contrib_idx: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Render per-primitive features to synthetic 2D pixel samples."""

    rendered = (raw_features[contrib_idx] * weights.unsqueeze(-1)).sum(dim=1)
    return F.normalize(rendered, dim=-1)


def tracking_pairs(scene: Dict[str, np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    frames = int(scene["frames"])
    for t in range(frames - 1):
        a = active_indices(scene, t)
        b = active_indices(scene, t + 1)
        if len(a) == 0 or len(b) == 0:
            pairs.append((np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)))
            continue
        op_now = temporal_opacity(scene, t)[a]
        op_next = temporal_opacity(scene, t + 1)[a]
        fade = op_now - op_next
        disappearing = a[(fade > np.quantile(fade, 0.55)) & (op_now > 0.28)]
        if len(disappearing) == 0:
            disappearing = a[: min(8, len(a))]
        pos_a = position_at(scene, t)[disappearing]
        pos_b = position_at(scene, t + 1)[b]
        vel_a = scene["velocity"][disappearing]
        vel_b = scene["velocity"][b]
        pos_scale = np.median(np.linalg.norm(pos_b - np.median(pos_b, axis=0), axis=1)) + 1e-6
        vel_scale = np.median(np.linalg.norm(vel_b - np.median(vel_b, axis=0), axis=1)) + 1e-6
        pos_cost = ((pos_a[:, None, :] - pos_b[None, :, :]) ** 2).sum(axis=2) / (pos_scale**2)
        vel_cost = ((vel_a[:, None, :] - vel_b[None, :, :]) ** 2).sum(axis=2) / (vel_scale**2)
        cost = pos_cost + 0.35 * vel_cost
        cols = cost.argmin(axis=1)
        best = cost[np.arange(len(disappearing)), cols]
        robust_limit = np.median(best) + 2.5 * (np.median(np.abs(best - np.median(best))) + 1e-6)
        keep = best <= max(float(np.quantile(best, 0.88)), float(robust_limit))
        pairs.append((disappearing[keep].astype(np.int64), b[cols[keep]].astype(np.int64)))
    return pairs


def train_scene(args: argparse.Namespace, scene_name: str, method_name: str) -> Dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    scene_np = build_scene(scene_name)
    scene = _torch_scene(scene_np, device)
    n = int(scene_np["mean_xyz"].shape[0])
    generator = torch.Generator(device=device).manual_seed(args.seed + 17)

    raw_features = torch.nn.Parameter(torch.randn(n, args.feature_dim, device=device) * 0.05)
    # Reduced-scale substitute for DINOv2 image descriptors: the procedural
    # benchmark stores fixed semantic observation vectors because it has no RGB
    # frames. They are used only in the paper's early alignment-loss role.
    projection = torch.randn(scene["semantic"].shape[1], args.feature_dim, device=device)
    projection = F.normalize(projection, dim=0)
    projected_semantic = F.normalize(scene["semantic"].float() @ projection, dim=-1)
    optim = torch.optim.Adam([raw_features], lr=args.lr)
    pairs = tracking_pairs(scene_np)
    obs = list(iter_observations(scene_np, streaming=args.streaming))
    random_order_rng = np.random.default_rng(args.seed + 991)
    if not args.streaming:
        # The ablation samples observations in random temporal order rather
        # than following the paper's streaming sequence.
        obs = list(iter_observations(scene_np, streaming=False))

    history = []
    for step in range(args.steps):
        if args.lr_final > 0 and args.steps > 1:
            progress = step / float(args.steps - 1)
            lr = args.lr + (args.lr_final - args.lr) * progress
            for group in optim.param_groups:
                group["lr"] = lr
        if not args.streaming and step > 0 and step % len(obs) == 0:
            random_order_rng.shuffle(obs)
        t, _view, contrib_np, weights_np, labels_np = obs[step % len(obs)]
        contrib = torch.as_tensor(contrib_np, dtype=torch.long, device=device)
        weights = torch.as_tensor(weights_np, dtype=torch.float32, device=device)
        labels = torch.as_tensor(labels_np, dtype=torch.long, device=device)
        feats_all = F.normalize(raw_features, dim=-1)
        feats = render_pixel_features(raw_features, contrib, weights)
        loss = contrastive_cluster_loss(feats, labels, args.samples_per_mask, generator)

        if args.lambda_align > 0 and t < len(pairs):
            src_np, dst_np = pairs[t]
            if len(src_np):
                src = torch.as_tensor(src_np, dtype=torch.long, device=device)
                dst = torch.as_tensor(dst_np, dtype=torch.long, device=device)
                loss = loss + args.lambda_align * (feats_all[src] - feats_all[dst]).abs().sum(dim=-1).mean()
        if args.lambda_dino > 0 and step < int(args.steps * args.dino_fraction):
            semantic_pixels = F.normalize((projected_semantic[contrib] * weights.unsqueeze(-1)).sum(dim=1), dim=-1)
            loss = loss + args.lambda_dino * dino_alignment_loss(feats, semantic_pixels, labels, args.samples_per_mask, generator)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        if (step + 1) % max(1, args.steps // 8) == 0:
            history.append(float(loss.detach().cpu()))

    features = F.normalize(raw_features.detach(), dim=-1).cpu().numpy().astype(np.float32)
    pred = infer_labels(scene_np, features)
    out = {
        "scene": scene_name,
        "method": method_name,
        "features": features.tolist(),
        "pred_labels": pred.tolist(),
        "history": history,
        "config": vars(args),
    }
    return out


def infer_labels(scene: Dict[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    inst = infer_instance_count(scene)
    first = active_indices(scene, 0)
    if len(first) < inst:
        first = np.arange(len(features))
    elif len(first) > inst:
        # The paper clusters a 2-10% sample of first-frame Gaussians.  The
        # reduced benchmark has only hundreds of primitives, so keep a minimum
        # absolute sample floor while preserving the same HDBSCAN-first
        # inference path.
        sample_count = min(len(first), max(inst * 10, int(np.ceil(0.10 * len(first)))))
        sample_rng = np.random.default_rng(919 + len(first) * 17 + inst)
        first = np.sort(sample_rng.choice(first, size=sample_count, replace=False))

    centers = hdbscan_centers(features, first, inst)
    if centers is None:
        # Deterministic fallback for very small runs where density clustering
        # finds too few clusters after downsampling.
        kmeans = KMeans(n_clusters=inst, n_init=20, random_state=5).fit(features[first])
        centers = kmeans.cluster_centers_
    sims = features @ (centers / np.linalg.norm(centers, axis=1, keepdims=True)).T
    labels = sims.argmax(axis=1)

    pos0 = position_at(scene, 0)
    velocity = scene["velocity"]
    for c in range(inst):
        members = np.flatnonzero(labels == c)
        if len(members) < 3:
            continue
        mean_v = velocity[members].mean(axis=0)
        mean_p = pos0[members].mean(axis=0)
        dv = np.linalg.norm(velocity[members] - mean_v, axis=1)
        dp = np.linalg.norm(pos0[members] - mean_p, axis=1)
        sim = sims[members, c]
        bad = (dv > 3 * (dv.std() + 1e-6)) & (dp > 3 * (dp.std() + 1e-6)) & (sim < 0.5)
        if np.any(bad):
            alternatives = sims[members[bad]].copy()
            alternatives[:, c] = -np.inf
            labels[members[bad]] = alternatives.argmax(axis=1)
    return labels.astype(np.int64)


def infer_instance_count(scene: Dict[str, np.ndarray]) -> int:
    """Estimate object count from public per-image mask supervision."""

    labels = scene.get("obs_labels")
    if labels is not None and len(labels):
        return int(max(int(np.max(x)) for x in labels if len(x)) + 1)
    max_label = 0
    for t in range(int(scene["frames"])):
        for view in range(int(scene["views"])):
            _, obs_labels = observation_labels(scene, view, t)
            max_label = max(max_label, int(np.max(obs_labels)))
    return max_label + 1


def hdbscan_centers(features: np.ndarray, first: np.ndarray, inst: int) -> np.ndarray | None:
    """Cluster the first-frame primitive sample with the paper's HDBSCAN step."""

    if hdbscan is None or len(first) < max(inst * 3, 8):
        return None
    min_cluster_size = max(3, len(first) // max(inst * 4, 1))
    labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=2).fit_predict(features[first])
    valid = [lab for lab in np.unique(labels) if lab >= 0]
    if len(valid) < inst:
        return None

    counts = np.asarray([(labels == lab).sum() for lab in valid], dtype=np.int64)
    assigned = int(counts.sum())
    if assigned < int(0.72 * len(first)):
        return None
    if counts.min() < max(2, int(0.04 * len(first))) or counts.max() > int(0.65 * len(first)):
        return None

    valid = sorted(valid, key=lambda lab: int((labels == lab).sum()), reverse=True)[:inst]
    centers = []
    for lab in valid:
        members = first[labels == lab]
        centers.append(features[members].mean(axis=0))
    return np.stack(centers, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/split4d/results.json")
    parser.add_argument("--scenes", nargs="+", default=list(SCENES))
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--samples-per-mask", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2.5e-3)
    parser.add_argument("--lr-final", type=float, default=2e-6)
    parser.add_argument("--lambda-align", type=float, default=0.35)
    parser.add_argument("--lambda-dino", type=float, default=0.08)
    parser.add_argument("--dino-fraction", type=float, default=0.35)
    parser.add_argument("--temporal-transfer", type=float, default=0.0, help="Deprecated compatibility flag; Split4D uses ordered sampling and losses, not direct feature copying.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--method-name", default=None)
    args = parser.parse_args()

    method_name = args.method_name or ("split4d_streaming" if args.streaming else "split4d_random_order")
    results = {"method": method_name, "scenes": {}}
    for scene_name in args.scenes:
        results["scenes"][scene_name] = train_scene(args, scene_name, method_name)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results), encoding="utf-8")


if __name__ == "__main__":
    main()
