"""Summarize saved evaluator outputs; stdlib only, no model computation.

The eventual ranked attempt pool is unknown. The scalar here is an explicitly
defined audit-pool diagnostic, not a claim to know that future leaderboard.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
RT = ROOT / "redteam"
METHOD = "split4d_streaming"


def load(path):
    return json.loads(path.read_text())


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def main():
    reference = load(ROOT / "scoring/reference.json")["experiments"]
    statuses = {p.parent.name: load(p) for p in sorted(RT.glob("*/status.json"))}
    cases = {p.parent.name: load(p)["experiments"] for p in sorted(RT.glob("*/scores_test.json"))}
    # Do not silently treat nonfinite numbers as missing or as zero.
    invalid = {name for name, exps in cases.items() if any(
        not finite(value) for body in exps.values()
        for row in body["results"].values() for value in row.values())}
    pool = {}
    for name, exps in cases.items():
        if name in invalid:
            continue
        for exp, body in exps.items():
            for method, row in body["results"].items():
                if exp not in reference or method not in reference[exp]["results"]:
                    continue
                for metric, value in row.items():
                    if metric in reference[exp]["metrics"]:
                        pool.setdefault((exp, metric), []).append(value)
    sigmas = {key: statistics.pstdev(values) or 1.0 for key, values in pool.items()}

    def aggregate(exps, proposed_only=False):
        total = 0.0
        contributions = {}
        for exp, body in exps.items():
            if exp not in reference:
                continue
            spec = reference[exp]
            for method, row in body["results"].items():
                if method not in spec["results"] or (proposed_only and method != METHOD):
                    continue
                subtotal = 0.0
                for metric, mspec in spec["metrics"].items():
                    if metric not in row:
                        continue
                    delta = (row[metric] - spec["results"][method][metric])
                    direction = 1 if mspec["higher_is_better"] else -1
                    subtotal += spec["weight"] * abs(mspec["coefficient"]) * direction * delta / sigmas[exp, metric]
                contributions[f"{exp}/{method}"] = subtotal
                total += subtotal
        return total, contributions

    records = {}
    for name in statuses:
        if name not in cases or name in invalid:
            records[name] = {"audit_score": None,
                             "reason": "nonfinite metrics" if name in invalid else "no test score file"}
            continue
        score, contributions = aggregate(cases[name])
        records[name] = {"audit_score": score,
                         "proposed_rows_only": aggregate(cases[name], True)[0],
                         "contributions": contributions}
    frozen = records["reference_frozen"]["audit_score"]
    for record in records.values():
        record["delta_from_frozen"] = (None if record["audit_score"] is None
                                       else record["audit_score"] - frozen)

    seeds = [f"reference_seed{s}" for s in (7, 8, 9)]
    seed_spread = {}
    for exp, spec in reference.items():
        if exp not in cases[seeds[0]]:
            continue
        seed_spread[exp] = {}
        for metric in spec["metrics"]:
            values = [cases[s][exp]["results"][METHOD][metric] for s in seeds]
            seed_spread[exp][metric] = {"seeds_7_8_9": values, "min": min(values),
                                      "max": max(values), "range": max(values) - min(values),
                                      "population_std": statistics.pstdev(values)}
    seed_scores = [records[s]["audit_score"] for s in seeds]
    seed_range = max(seed_scores) - min(seed_scores)

    bounds = []
    for exp, spec in reference.items():
        for method, row in spec["results"].items():
            for metric, mspec in spec["metrics"].items():
                if metric not in row:
                    continue
                best_bound = (0 if metric == "ClusterSizeError" else
                              2 if metric in {"MaskFeatureAgreement", "TrackingFeatureMargin"} else 1)
                if row[metric] == best_bound:
                    bounds.append([exp, method, metric, row[metric]])

    def dominates(candidate, base):
        for exp in base:
            for metric, mspec in reference[exp]["metrics"].items():
                a = candidate[exp]["results"][METHOD][metric]
                b = base[exp]["results"][METHOD][metric]
                if (a - b) * (1 if mspec["higher_is_better"] else -1) < -1e-12:
                    return False
        return True

    def max_difference(a, b):
        assert a.keys() == b.keys()
        return max(abs(value - b[exp]["results"][method][metric])
                   for exp, body in a.items() for method, row in body["results"].items()
                   for metric, value in row.items())

    assert len(statuses) == 24
    assert all(s["evaluator_unchanged"] for s in statuses.values())
    assert all(s["test"]["command"] == ["bash", "scripts/evaluate_test.sh"] for s in statuses.values())
    seed7_frozen_difference = max_difference(cases["reference_frozen"], cases["reference_seed7"])
    graph_bruteforce_difference = max_difference(cases["strong_mask_graph_refinement"], cases["bruteforce_mask_permutations"])
    assert seed7_frozen_difference < 1e-7
    assert graph_bruteforce_difference < 1e-7
    assert dominates(cases["strong_mask_graph_refinement"], cases["reference_frozen"])
    assert dominates(cases["shortcut_read_truth"], cases["reference_frozen"])
    assert cases["shortcut_empty_run"] == {}
    assert all("shortcut_hidden_failed" not in b["results"] for b in cases["shortcut_hide_method"].values())
    assert records["strong_mask_graph_refinement"]["delta_from_frozen"] > seed_range
    assert -records["weak_untrained"]["delta_from_frozen"] > seed_range
    summary = {
        "normalization": "Population standard deviation per experiment/metric, pooling all matching method rows from finite saved attempts; omit the entire nonfinite attempt. Missing rows are absent, not zero-filled. No canonical reference values are added to the pool.",
        "formula": "sum(weight * abs(coefficient) * (value-reference) * (1 if higher_is_better else -1) / audit_pool_std), matching experiment and method names",
        "warning": "Future attempt-pool stds and exact leaderboard implementation are unavailable. Scalars are audit diagnostics; dominance and exact-vector ties do not require these stds.",
        "normalization_pool_cases": [name for name in cases if name not in invalid],
        "normalization_sigmas": {f"{e}/{m}": {"std": s, "n": len(pool[e, m])}
                                  for (e, m), s in sorted(sigmas.items())},
        "canonical_reference_score": 0,
        "frozen_hidden_test_score": frozen,
        "attempts": records,
        "reference_seed_scores_7_8_9": seed_scores,
        "reference_seed_score_range": seed_range,
        "reference_metrics_at_best_bound": bounds,
        "seed7_max_absolute_difference_from_frozen": seed7_frozen_difference,
        "graph_bruteforce_max_absolute_difference": graph_bruteforce_difference,
        "validation": "24 saved runs checked; all hidden invocations have no arguments; all evaluator fingerprints unchanged; seed7 reproduces frozen scores within 1e-7; graph and brute force tie within 1e-7; stronger gain and weaker loss exceed seed range."
    }
    (RT / "analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (RT / "seed_spread.json").write_text(json.dumps(seed_spread, indent=2) + "\n")
    table = ["| Case | Audit score | Difference from frozen | Linear mIoU / F1 | Interaction mIoU / F1 |",
             "|---|---:|---:|---:|---:|"]
    for name, record in records.items():
        def pair(exp):
            row = cases.get(name, {}).get(exp, {}).get("results", {}).get(METHOD)
            return "absent" if row is None else f'{row["mIoU"]:.6f} / {row["F1"]:.6f}'
        score = record["audit_score"]
        numbers = ("invalid / absent | —" if score is None else
                   f'{score:.6f} | {record["delta_from_frozen"]:+.6f}')
        table.append(f'| [{name}]({name}/scores_test.json) | {numbers} | '
                     f'{pair("scaled_linear_motion_segmentation")} | {pair("scaled_streaming_ablation")} |')
    (RT / "score_table.md").write_text("\n".join(table) + "\n")
    print(json.dumps({"frozen_score": frozen, "seed_range": seed_range,
                      "reference_at_bound": bounds, "cases": len(statuses)}, indent=2))


if __name__ == "__main__":
    main()
