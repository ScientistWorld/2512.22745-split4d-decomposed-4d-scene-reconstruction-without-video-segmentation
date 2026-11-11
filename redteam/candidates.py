"""Candidate producers use only scientist-visible modules and saved reference outputs.

Every deliberate shortcut has 'shortcut' in its name. No producer imports eval.test,
reads the paper, changes eval, or changes the evaluator's Python environment.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RT = ROOT / "redteam"
SCENE_NAMES = ("linear_motion", "interaction_motion")


def reference():
    return json.loads((RT / "reference_artifacts/reference.json").read_text())


def simplex(labels, k, dim=32):
    # Labels can always be encoded as a maximally separated representation.
    values = np.zeros((len(labels), dim), dtype=np.float64)
    values[:, :k] = np.eye(k)[labels] - 1 / k
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def count_masks(scene):
    return max(int(np.max(x)) for x in scene["obs_labels"] if len(x)) + 1


def train(name):
    import torch
    from method import split4d as m

    torch.set_num_threads(1)
    cfg = json.loads((RT / "reference_config.json").read_text())
    cfg.update(cpu=True, method_name="split4d_streaming")
    if name.startswith("reference_seed"):
        cfg["seed"] = int(name.removeprefix("reference_seed"))
    if name == "weak_untrained":
        cfg["steps"] = 0
    if name == "strong_more_training":
        cfg["steps"] = 10000
    if name == "strong_corrected_temperature":
        import torch.nn.functional as F

        def corrected(features, semantic, labels, samples, generator):
            chunks, centers, temperatures = [], [], []
            for lab in torch.unique(labels):
                local = torch.nonzero(labels == lab, as_tuple=False).flatten()
                if local.numel() < 2:
                    continue
                take = local[torch.randint(local.numel(), (min(samples, local.numel()),),
                                           device=features.device, generator=generator)]
                values = F.normalize(features[take], dim=-1)
                center = F.normalize(values.mean(0), dim=0)
                # Eq. (4): SUM squared distances / (Ns log(Ns+10)).
                phi = ((values - center).square().sum() /
                       (values.shape[0] * math.log(values.shape[0] + 10))).detach()
                chunks.append(values)
                centers.append(center if semantic is None else F.normalize(semantic[take].mean(0), dim=0))
                temperatures.append(phi.clamp(min=1e-6))
            if len(chunks) < 2:
                return features.sum() * 0
            centers = torch.stack(centers)
            temperatures = torch.stack(temperatures)
            # Denominator term k uses phi_k, not the query class's phi_i.
            return torch.stack([F.cross_entropy(
                values @ centers.T / temperatures[None, :],
                torch.full((len(values),), c, device=features.device, dtype=torch.long))
                for c, values in enumerate(chunks)]).mean()

        m.contrastive_cluster_loss = lambda f, l, s, g: corrected(f, None, l, s, g)
        m.dino_alignment_loss = lambda f, d, l, s, g: corrected(f, d, l, s, g)
    args = argparse.Namespace(**cfg)
    return {"method": "split4d_streaming", "scenes": {
        s: m.train_scene(args, s, "split4d_streaming") for s in SCENE_NAMES}}


def public_grouping(name):
    from data.split4d_synthetic import build_scene, position_at
    from sklearn.cluster import KMeans, SpectralClustering

    payload = {"method": "split4d_streaming", "scenes": {}}
    diagnostics = {}
    for scene_name in SCENE_NAMES:
        start = time.monotonic()
        scene = build_scene(scene_name)  # default excludes truth
        n = len(scene["mean_xyz"])
        k = count_masks(scene)
        if name == "simple_geometry_baseline":
            labels = KMeans(k, n_init=20, random_state=7).fit_predict(position_at(scene, 0))
        elif name == "strong_mask_graph_refinement":
            same, total = np.zeros((n, n)), np.zeros((n, n))
            for idx, masks in zip(scene["obs_indices"], scene["obs_labels"]):
                idx, masks = np.asarray(idx, dtype=int), np.asarray(masks, dtype=int)
                np.add.at(total, (idx[:, None], idx[None, :]), 1)
                np.add.at(same, (idx[:, None], idx[None, :]), masks[:, None] == masks[None, :])
            affinity = np.maximum(same / np.maximum(total, 1) - 0.5, 0)
            affinity[total < 2] = 0
            np.fill_diagonal(affinity, 1)
            labels = SpectralClustering(k, affinity="precomputed", random_state=7,
                                        assign_labels="kmeans", n_init=20).fit_predict(affinity)
        elif name == "bruteforce_mask_permutations":
            # Exhaust every per-image label permutation against overlapping primitives.
            # The global k**n search is unnecessary: k! local permutations suffice.
            permutations = np.asarray(list(itertools.permutations(range(k))), dtype=int)
            votes = np.zeros((n, k))
            idx0, masks0 = scene["obs_indices"][0], scene["obs_labels"][0]
            for index, mask in zip(idx0, masks0):
                votes[int(index), int(mask)] += 1
            for sweep in range(6):
                updated = np.zeros_like(votes)
                for idx, masks in zip(scene["obs_indices"], scene["obs_labels"]):
                    idx, masks = np.asarray(idx, dtype=int), np.asarray(masks, dtype=int)
                    normalized = votes[idx] / np.maximum(votes[idx].sum(1, keepdims=True), 1)
                    table = np.zeros((k, k))
                    np.add.at(table, masks, normalized)
                    scores = table[np.arange(k)[None, :], permutations].sum(1)
                    chosen = permutations[int(scores.argmax())]
                    np.add.at(updated, (idx, chosen[masks]), 1)
                    if sweep == 0:
                        np.add.at(votes, (idx, chosen[masks]), 1)
                votes = updated
            labels = votes.argmax(1)
            diagnostics[scene_name] = {"permutations_per_image": len(permutations),
                "enumerated_candidates": 6 * len(scene["obs_indices"]) * len(permutations)}
        else:
            raise ValueError(name)
        features = simplex(labels, k)
        payload["scenes"][scene_name] = {"pred_labels": labels.tolist(), "features": features.tolist()}
        diagnostics.setdefault(scene_name, {}).update(seconds=time.monotonic() - start, primitives=n, instances=k)
    return payload, diagnostics


def make(name):
    if name.startswith("reference_seed") or name in {"weak_untrained", "strong_corrected_temperature", "strong_more_training"}:
        return train(name), {}
    if name in {"simple_geometry_baseline", "strong_mask_graph_refinement", "bruteforce_mask_permutations"}:
        return public_grouping(name)
    if name == "shortcut_empty_run":
        return {"method": "split4d_streaming", "scenes": {}}, {}

    payload = reference()
    if name in {"reference_frozen", "shortcut_hide_method", "shortcut_replace_baseline"}:
        return payload, {}
    if name == "shortcut_baseline_identity":
        payload["method"] = "split4d_random_order"
        return payload, {}
    if name == "shortcut_partial_experiment":
        del payload["scenes"]["interaction_motion"]
        return payload, {}

    overrides = {}
    for scene_name, body in payload["scenes"].items():
        labels = np.asarray(body["pred_labels"], dtype=int)
        k = int(labels.max()) + 1
        n = len(labels)
        if name == "shortcut_read_truth":
            from data.split4d_synthetic import build_scene
            labels = build_scene(scene_name, include_truth=True)["object_id"]
            k = int(labels.max()) + 1
        elif name in {"shortcut_constant", "shortcut_forged_scores", "shortcut_empty_features", "shortcut_nonfinite"}:
            labels = np.zeros(n, dtype=int)
        features = simplex(labels, k)
        if name in {"shortcut_constant", "shortcut_forged_scores"}:
            features = np.ones((n, 32))
        if name == "shortcut_empty_features":
            features = np.empty((n, 0))
        if name == "shortcut_nonfinite":
            features = np.full((n, 32), float("inf"))
        body.update(pred_labels=labels.tolist(), features=features.tolist())
        if name == "shortcut_malformed":
            body["pred_labels"] = []
        if name == "shortcut_forged_scores":
            body.update(success=True, elapsed_seconds=0, mIoU=1, F1=1,
                        results={"mIoU": 1, "Recalldyn": 1, "ClusterSizeError": 0})
        overrides[scene_name] = labels.tolist()

    if name in {"shortcut_move_truth", "shortcut_replace_fixed_observations"}:
        # These change a mutable benchmark-side data artifact, never eval or Python.
        path = ROOT / "data/shortcut_override.json"
        path.write_text(json.dumps(overrides))
        source = ROOT / "data/split4d_synthetic.py"
        text = source.read_text()
        if name == "shortcut_move_truth":
            line = "object_id = np.repeat(np.arange(cfg.instances), cfg.primitives_per_instance)"
            assert text.count(line) == 1
            replacement = ('import json\n    from pathlib import Path\n    '
                           'object_id = np.asarray(json.loads(Path("data/shortcut_override.json").read_text())[name], dtype=np.int64)')
            text = text.replace(line, replacement)
        else:
            text += '''\n\n# Deliberate shortcut: replace benchmark observations by candidate partitions.\ndef _precompute_observations(scene, object_id, inst_count):\n    import json\n    from pathlib import Path\n    key = str(scene["name"].item())\n    labels = np.asarray(json.loads(Path("data/shortcut_override.json").read_text())[key], dtype=np.int64)\n    idx = np.arange(len(labels), dtype=np.int64)\n    count = int(scene["frames"]) * int(scene["views"])\n    return (np.asarray([idx] * count, dtype=object),\n            np.asarray([idx[:, None]] * count, dtype=object),\n            np.asarray([np.ones((len(idx), 1))] * count, dtype=object),\n            np.asarray([labels] * count, dtype=object))\n'''
        source.write_text(text)
    return payload, {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    payload, diagnostic = make(args.name)
    Path(args.output).write_text(json.dumps(payload))
    (RT / args.name / "producer_diagnostics.json").write_text(json.dumps(diagnostic, indent=2))


if __name__ == "__main__":
    main()
