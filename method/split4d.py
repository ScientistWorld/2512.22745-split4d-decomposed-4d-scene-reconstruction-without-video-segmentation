"""Split4D feature learning (Hu et al., TOG 2025) on fixed Freetime FeatureGS primitives.

The 4D reconstruction is fixed; only one learnable feature vector per Gaussian
primitive is optimized (Sec. 3.5). Implemented components:

* Eq. 2   rendered feature of a pixel = sum_i f_i alpha_i T_i over its blended
          contributors (L2-normalized before the contrastive dot products).
* Eq. 4   contrastive clustering loss: for every mask of the image, N_s pixels
          are sampled uniformly, the mask center is their mean feature, and
          phi_k = sum_j ||f_kj - fbar_k||^2 / (N_s log(N_s + 10)) is the
          temperature of center k in every logit f . fbar_k / phi_k.
* Eq. 5   3D tracking regularization: primitives fading out between t and t+1
          are matched to their nearest primitive at t+1 (positions from the
          linear motion model) and their features are aligned with an L1 loss.
* Eq. 6   DINO-style alignment: the same contrastive form with centers taken
          from the rendered (fixed, linearly projected) semantic descriptors,
          applied only during the early part of training.
* Eq. 7   L = L_CC + lambda_1 L_align + lambda_2 L_DINO.
* Sec 3.2 streaming sampling: observations are visited in temporal order
          (frames in order; views of a frame in random order). The paper's
          ablation replaces this with uniformly random observation sampling.
* Sec 3.5 inference: HDBSCAN on a sample of first-frame primitives, cluster
          centers = mean features, every primitive assigned to its most
          similar center, then the Eq. 8 dynamic-perception filter removes
          primitives whose velocity, position, and feature similarity all
          deviate from their cluster.

Loss weights use the paper's lambda values (100 for tracking; 500 for DINO by
default and 10 for the multi-human interaction profile) multiplied by a fixed
``--loss-weight-scale`` that maps them onto this implementation's loss
normalizations (per-pair mean L1 and mean cross-entropy).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from data.split4d_scenes import ACTIVE_OPACITY, list_scenes, load_scene, position_at, temporal_opacity

try:  # HDBSCAN [McInnes et al. 2017]
    from hdbscan import HDBSCAN
except ImportError:  # pragma: no cover
    from sklearn.cluster import HDBSCAN  # type: ignore

PAPER_LAMBDA_ALIGN = 100.0
PAPER_LAMBDA_DINO = {"dynamic": 500.0, "interaction": 10.0}
PHI_FLOOR = 1e-2


# --------------------------------------------------------------------------- data


def prepare_observations(scene: Dict[str, np.ndarray]) -> List[Dict[str, torch.Tensor]]:
    """Group each image's pixels by mask id for uniform per-mask sampling."""

    out = []
    offsets = scene["obs_offsets"]
    for k in range(len(scene["obs_t"])):
        lo, hi = int(offsets[k]), int(offsets[k + 1])
        label = scene["pix_label"][lo:hi]
        valid = np.flatnonzero(label >= 0)
        labs, inverse, counts = np.unique(label[valid], return_inverse=True, return_counts=True)
        good = counts >= 2
        keep_pix = valid[good[inverse]]
        mask_index = np.cumsum(good) - 1
        mask_of_pix = mask_index[inverse[good[inverse]]]
        order = np.argsort(mask_of_pix, kind="stable")
        keep_pix = keep_pix[order]
        mcounts = counts[good]
        moffsets = np.concatenate([[0], np.cumsum(mcounts)])
        out.append({
            "t": int(scene["obs_t"][k]),
            "contrib": torch.as_tensor(scene["pix_contrib"][lo:hi][keep_pix], dtype=torch.long),
            "weight": torch.as_tensor(scene["pix_weight"][lo:hi][keep_pix], dtype=torch.float32),
            "counts": torch.as_tensor(mcounts, dtype=torch.long),
            "starts": torch.as_tensor(moffsets[:-1], dtype=torch.long),
        })
    return out


def tracking_pairs(scene: Dict[str, np.ndarray], frames: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Eq. 5 matches: disappearing primitives at t -> nearest primitive at t+1."""

    pairs = []
    for t in range(frames - 1):
        op_t = temporal_opacity(scene, t)
        op_n = temporal_opacity(scene, t + 1)
        op_nn = temporal_opacity(scene, t + 2)
        disappearing = np.flatnonzero((op_t >= ACTIVE_OPACITY) & (op_n < op_t) & (op_n < 0.5))
        candidates = np.flatnonzero((op_n >= ACTIVE_OPACITY) & (op_nn >= ACTIVE_OPACITY))
        if len(disappearing) == 0 or len(candidates) == 0:
            pairs.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
            continue
        pos = position_at(scene, t + 1)
        d2 = ((pos[disappearing][:, None, :] - pos[candidates][None, :, :]) ** 2).sum(axis=2)
        d2[disappearing[:, None] == candidates[None, :]] = np.inf
        nearest = d2.argmin(axis=1)
        best = d2[np.arange(len(disappearing)), nearest]
        finite = np.isfinite(best)
        if not finite.any():
            pairs.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
            continue
        keep = finite & (best <= np.quantile(best[finite], 0.8))
        pairs.append((torch.as_tensor(disappearing[keep], dtype=torch.long),
                      torch.as_tensor(candidates[nearest[keep]], dtype=torch.long)))
    return pairs


def observation_schedule(obs_t: np.ndarray, frames: int, steps: int, streaming: bool,
                         steps_per_frame: int, rng: np.random.Generator) -> np.ndarray:
    if not streaming:
        return rng.integers(0, len(obs_t), size=steps)
    by_t = {t: np.flatnonzero(obs_t == t) for t in range(frames)}
    sched = np.empty(steps, dtype=np.int64)
    perm = by_t[0]
    for s in range(steps):
        g = s // steps_per_frame
        t = g % frames
        within = s % steps_per_frame
        if within == 0:
            perm = rng.permutation(by_t[t])
        sched[s] = perm[within % len(perm)]
    return sched


# --------------------------------------------------------------------------- losses


def sample_mask_pixels(ob: Dict[str, torch.Tensor], ns: int, gen: torch.Generator) -> torch.Tensor:
    counts = ob["counts"]
    u = torch.rand((counts.numel(), ns), generator=gen)
    j = torch.minimum((u * counts[:, None]).long(), counts[:, None] - 1)
    return j + ob["starts"][:, None]


def render(values: torch.Tensor, contrib: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Eq. 2 on sampled pixels: (M, ns, C) contributors -> (M, ns, d) unit features."""

    return F.normalize((values[contrib] * weight[..., None]).sum(dim=-2), dim=-1)


def pcl_temperature(vals: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    ns = vals.shape[1]
    disp = (vals - center[:, None, :]).pow(2).sum(dim=-1).sum(dim=1)
    return (disp / (ns * math.log(ns + 10.0))).clamp_min(PHI_FLOOR).detach()


def center_contrast(vals: torch.Tensor, centers: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """-1/(N_inst N_s) sum_i sum_j log softmax_k(f_ij . c_k / phi_k)[i] (Eqs. 4, 6)."""

    m, ns, _ = vals.shape
    logits = torch.einsum("mjd,kd->mjk", vals, centers) / phi[None, None, :]
    target = torch.arange(m)[:, None].expand(m, ns).reshape(-1)
    return F.cross_entropy(logits.reshape(m * ns, m), target)


# --------------------------------------------------------------------------- training


def train_features(scene: Dict[str, np.ndarray], args: argparse.Namespace, seed: int) -> Tuple[np.ndarray, Dict[str, object]]:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 17)
    rng = np.random.default_rng(seed + 991)
    profile = str(scene["profile"])
    frames = int(scene["frames"])
    n = int(len(scene["mu_t"]))
    lam_align = args.loss_weight_scale * (PAPER_LAMBDA_ALIGN if args.lambda_align is None else args.lambda_align)
    lam_dino = args.loss_weight_scale * (PAPER_LAMBDA_DINO.get(profile, 500.0) if args.lambda_dino is None else args.lambda_dino)

    raw = torch.nn.Parameter(torch.randn(n, args.feature_dim) * 0.1)
    proj_gen = torch.Generator().manual_seed(1234)
    projection = F.normalize(torch.randn(scene["semantic"].shape[1], args.feature_dim, generator=proj_gen), dim=0)
    semantic = F.normalize(torch.as_tensor(scene["semantic"], dtype=torch.float32) @ projection, dim=-1)
    obs = prepare_observations(scene)
    pairs = tracking_pairs(scene, frames)
    sched = observation_schedule(scene["obs_t"].astype(np.int64), frames, args.steps, args.streaming,
                                 args.steps_per_frame, rng)
    optim = torch.optim.Adam([raw], lr=args.lr)
    dino_steps = int(args.steps * args.dino_fraction)
    history: List[float] = []
    for s in range(args.steps):
        lr = args.lr * (args.lr_final / args.lr) ** (s / max(args.steps - 1, 1))
        for group in optim.param_groups:
            group["lr"] = lr
        ob = obs[int(sched[s])]
        loss = torch.zeros(())
        has_term = False
        if ob["counts"].numel() >= 2:
            has_term = True
            idx = sample_mask_pixels(ob, args.samples_per_mask, gen)
            contrib = ob["contrib"][idx]
            weight = ob["weight"][idx]
            vals = render(raw, contrib, weight)
            centers = vals.mean(dim=1)
            phi = pcl_temperature(vals, centers)
            loss = loss + center_contrast(vals, centers, phi)
            if lam_dino > 0 and s < dino_steps:
                dino_centers = render(semantic, contrib, weight).mean(dim=1)
                loss = loss + lam_dino * center_contrast(vals, dino_centers, phi)
        t = ob["t"]
        if lam_align > 0 and t < len(pairs) and pairs[t][0].numel():
            feats = F.normalize(raw, dim=-1)
            src, dst = pairs[t]
            loss = loss + lam_align * (feats[src] - feats[dst]).abs().sum(dim=-1).mean()
            has_term = True
        if has_term:
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
        if (s + 1) % max(1, args.steps // 10) == 0:
            history.append(float(loss.detach()))
    features = F.normalize(raw.detach(), dim=-1).numpy().astype(np.float32)
    return features, {"loss_history": history, "lambda_align_effective": lam_align, "lambda_dino_effective": lam_dino}


# --------------------------------------------------------------------------- inference


def infer_labels(scene: Dict[str, np.ndarray], features: np.ndarray, args: argparse.Namespace, seed: int) -> Tuple[np.ndarray, int]:
    n = len(features)
    first = np.flatnonzero(temporal_opacity(scene, 0) >= ACTIVE_OPACITY)
    rng = np.random.default_rng(seed + 919)
    sample_size = max(int(math.ceil(args.cluster_fraction * len(first))), min(len(first), args.cluster_min_sample))
    if sample_size < len(first):
        first = np.sort(rng.choice(first, size=sample_size, replace=False))
    if len(first) < args.hdbscan_min_cluster_size:
        return np.zeros(n, dtype=np.int64), 1
    clusterer = HDBSCAN(min_cluster_size=args.hdbscan_min_cluster_size, min_samples=args.hdbscan_min_samples,
                        cluster_selection_epsilon=args.hdbscan_epsilon)
    hl = clusterer.fit_predict(features[first].astype(np.float64))
    clusters = [c for c in np.unique(hl) if c >= 0]
    if not clusters:
        return np.zeros(n, dtype=np.int64), 1
    centers = np.stack([features[first[hl == c]].mean(axis=0) for c in clusters])
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-8)
    sims = features @ centers.T
    labels = sims.argmax(axis=1).astype(np.int64)
    if args.filter:
        labels = dynamic_perception_filter(scene, labels, sims, args.tau_sim)
    return labels, len(clusters)


def dynamic_perception_filter(scene: Dict[str, np.ndarray], labels: np.ndarray, sims: np.ndarray, tau_sim: float,
                              window: float = 3.0) -> np.ndarray:
    """Eq. 8: remove primitives deviating in velocity AND position AND feature similarity."""

    out = labels.copy()
    pos = scene["mean_xyz"].astype(np.float64)
    vel = scene["velocity"].astype(np.float64)
    mu = scene["mu_t"].astype(np.float64)
    for c in np.unique(labels):
        members = np.flatnonzero(labels == c)
        if len(members) < 5:
            continue
        near = (np.abs(mu[members][:, None] - mu[members][None, :]) <= window).astype(np.float64)
        cnt = near.sum(axis=1, keepdims=True)
        mean_p = (near @ pos[members]) / cnt
        mean_v = (near @ vel[members]) / cnt
        dp = np.linalg.norm(pos[members] - mean_p, axis=1)
        dv = np.linalg.norm(vel[members] - mean_v, axis=1)
        bad = (dv > 3.0 * dv.std()) & (dp > 3.0 * dp.std()) & (sims[members, c] < tau_sim)
        out[members[bad]] = -1
    return out


# --------------------------------------------------------------------------- driver


def run_scene(job: Tuple[str, argparse.Namespace]) -> Tuple[str, Dict[str, object], np.ndarray]:
    scene_id, args = job
    torch.set_num_threads(1)
    start = time.time()
    scene = load_scene(scene_id, args.scene_dir)
    seed = args.seed
    if args.ensemble_from:
        feats = []
        for path in args.ensemble_from:
            with np.load(path) as z:
                feats.append(z[scene_id].astype(np.float32))
        features = np.concatenate(feats, axis=1)
        features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
        info: Dict[str, object] = {"ensemble_from": list(args.ensemble_from)}
    elif args.steps > 0:
        features, info = train_features(scene, args, seed)
    else:
        gen = torch.Generator().manual_seed(seed)
        features = F.normalize(torch.randn(len(scene["mu_t"]), args.feature_dim, generator=gen) * 0.1, dim=-1).numpy()
        info = {"untrained": True}
    labels, k = infer_labels(scene, features, args, seed)
    info.update(clusters=int(k), seconds=round(time.time() - start, 2))
    return scene_id, {"pred_labels": labels.tolist(), "info": info}, features


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Split4D streaming feature learning on the benchmark scenes.")
    p.add_argument("--output", default="outputs/split4d/results.json")
    p.add_argument("--method-name", default=None)
    p.add_argument("--scene-dir", default=None)
    p.add_argument("--scenes", nargs="*", default=None, help="Scene ids (default: every public scene).")
    p.add_argument("--split", default=None, choices=["dev", "test"])
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--steps-per-frame", type=int, default=33, help="Streaming iterations per frame visit (paper: ~10k over 300 frames).")
    p.add_argument("--feature-dim", type=int, default=32)
    p.add_argument("--samples-per-mask", type=int, default=32)
    p.add_argument("--lr", type=float, default=2.5e-3)
    p.add_argument("--lr-final", type=float, default=2e-6)
    p.add_argument("--lambda-align", type=float, default=None, help="Paper lambda_1 (default 100).")
    p.add_argument("--lambda-dino", type=float, default=None, help="Paper lambda_2 (default 500; 10 for interaction scenes).")
    p.add_argument("--loss-weight-scale", type=float, default=0.01)
    p.add_argument("--dino-fraction", type=float, default=0.3)
    p.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--filter", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tau-sim", type=float, default=0.5)
    p.add_argument("--cluster-fraction", type=float, default=0.10)
    p.add_argument("--cluster-min-sample", type=int, default=600,
                   help="Sample floor for the reduced scenes (the paper's 2-10%% of millions of primitives).")
    p.add_argument("--hdbscan-min-cluster-size", type=int, default=5)
    p.add_argument("--hdbscan-min-samples", type=int, default=5)
    p.add_argument("--hdbscan-epsilon", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ensemble-from", nargs="*", default=None, help="Cluster concatenated saved features instead of training.")
    p.add_argument("--workers", type=int, default=0)
    return p


def available_cpus() -> int:
    """CPUs usable by this process: affinity mask capped by a cgroup CPU quota."""

    n = len(os.sched_getaffinity(0))
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            n = min(n, max(1, int(int(quota) // int(period))))
    except (OSError, ValueError):
        try:
            quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if quota > 0:
                n = min(n, max(1, quota // period))
        except (OSError, ValueError):
            pass
    env = os.environ.get("SPLIT4D_WORKERS")
    return int(env) if env else n


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    name = args.method_name or ("split4d_streaming" if args.streaming else "split4d_random_order")
    scene_ids = args.scenes or [s["id"] for s in list_scenes(args.scene_dir, split=args.split)]
    workers = args.workers or max(1, min(len(scene_ids), available_cpus()))
    print(f"{name}: {len(scene_ids)} scenes, {workers} worker(s)", flush=True)
    jobs = [(sid, args) for sid in scene_ids]
    if workers > 1:
        import multiprocessing as mp

        with mp.get_context("fork").Pool(workers) as pool:
            results = pool.map(run_scene, jobs, chunksize=1)
    else:
        results = [run_scene(j) for j in jobs]
    payload = {"method": name, "scenes": {}, "config": {k: v for k, v in vars(args).items() if k != "output"}}
    features = {}
    for sid, body, feats in results:
        payload["scenes"][sid] = body
        features[sid] = feats.astype(np.float16)
        print(f"{name} {sid}: clusters={body['info']['clusters']} seconds={body['info']['seconds']}", flush=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload), encoding="utf-8")
    np.savez_compressed(out.with_suffix(".features.npz"), **features)


if __name__ == "__main__":
    main()
