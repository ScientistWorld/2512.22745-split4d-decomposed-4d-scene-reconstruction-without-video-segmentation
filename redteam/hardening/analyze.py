"""Summarize the hardening cases (stdlib only; no model computation).

S is a diagnostic of the gym's rank rule: sum over experiments and metrics of
weight * |coefficient| * direction * (value - reference) / sigma, where the
reference is the seed-0 reference run's hidden-test row and sigma is the
population std of the candidate rows of all cases (failed runs included at
their worst values). The future attempt pool is unknown, so raw vectors and
per-scene paired comparisons are reported as well.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
H = ROOT / "redteam" / "hardening"
CASES = H / "cases"
REFERENCE_CASE = "reference_seed0"


def load(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    spec = load(ROOT / "scoring/reference.json")["experiments"]
    cases = {}
    for d in sorted(CASES.iterdir()):
        if (d / "status.json").exists():
            st = load(d / "status.json")
            test = load(d / "scores_test.json") if (d / "scores_test.json").exists() else None
            train = load(d / "scores_train.json") if (d / "scores_train.json").exists() else None
            cases[d.name] = {"status": st, "test": test, "train": train}

    def row(case, split="test", name=None):
        scores = cases[case][split]
        name = name or cases[case]["status"].get("candidate")
        if not scores:
            return None
        out = {}
        for exp in spec:
            r = scores["experiments"].get(exp, {}).get("results", {}).get(name)
            if r is None:
                return None
            out[exp] = r
        return out

    ref = row(REFERENCE_CASE)
    pool = {}
    for case in cases:
        r = row(case)
        if r is None:
            continue
        for exp, metrics in r.items():
            for m, v in metrics.items():
                if m in spec[exp]["metrics"] and isinstance(v, (int, float)) and math.isfinite(v):
                    pool.setdefault((exp, m), []).append(v)
    sigma = {k: (statistics.pstdev(v) or 1.0) for k, v in pool.items()}

    def score(r):
        total = 0.0
        for exp, metrics in r.items():
            for m, ms in spec[exp]["metrics"].items():
                d = (metrics[m] - ref[exp][m]) * (1 if ms["higher_is_better"] else -1)
                total += spec[exp]["weight"] * abs(ms["coefficient"]) * d / sigma[(exp, m)]
        return total

    records = {}
    for case in cases:
        st = cases[case]["status"]
        r = row(case)
        valid = None
        if cases[case]["test"]:
            valid = cases[case]["test"].get("_status", {}).get(st.get("candidate"), {}).get("valid")
            reason = cases[case]["test"].get("_status", {}).get(st.get("candidate"), {}).get("reason")
        else:
            reason = "no test score file"
        records[case] = {
            "kind": st.get("kind"), "candidate": st.get("candidate"),
            "test_returncode": st["test"]["returncode"], "train_returncode": st["train"]["returncode"],
            "valid": valid, "reason": reason,
            "S_test": None if r is None else score(r),
            "test": r, "train": row(case, "train"),
            "rows_on_test": sorted({m for b in (cases[case]["test"] or {}).get("experiments", {}).values() for m in b["results"]}),
        }

    seeds = [c for c in ("reference_seed0", "reference_seed1", "reference_seed2") if c in records and records[c]["test"]]
    seed_spread = {}
    for exp in spec:
        seed_spread[exp] = {}
        for m in spec[exp]["metrics"]:
            vals = [records[c]["test"][exp][m] for c in seeds]
            seed_spread[exp][m] = {"values": vals, "range": max(vals) - min(vals) if vals else None}
    seed_scores = [records[c]["S_test"] for c in seeds]

    # paired per-scene comparisons against the seed-0 reference
    def per_scene(case):
        t = cases[case]["test"] or {}
        return t.get("_per_scene_diagnostics", {}).get(cases[case]["status"].get("candidate"), {})

    ref_ps = per_scene(REFERENCE_CASE)
    paired = {}
    for case in cases:
        ps = per_scene(case)
        if not ps or case == REFERENCE_CASE:
            continue
        diffs = {m: [ps[s][m] - ref_ps[s][m] for s in ref_ps if s in ps] for m in ("mIoU", "F1", "Recalldyn")}
        paired[case] = {m: {"mean": statistics.mean(v), "wins": sum(x > 1e-12 for x in v), "losses": sum(x < -1e-12 for x in v),
                            "n": len(v)} for m, v in diffs.items() if v}

    bounds = []
    for exp in spec:
        for m, ms in spec[exp]["metrics"].items():
            v = ref[exp][m]
            if (ms["higher_is_better"] and v >= 1.0 - 1e-12) or (not ms["higher_is_better"] and v <= 1e-12):
                bounds.append(f"{exp}/{m}")

    summary = {
        "formula": __doc__.strip(),
        "sigma": {f"{e}/{m}": s for (e, m), s in sorted(sigma.items())},
        "reference_case": REFERENCE_CASE,
        "reference_test_row": ref,
        "reference_metrics_at_bound": bounds,
        "reference_seed_S": seed_scores,
        "reference_seed_S_range": (max(seed_scores) - min(seed_scores)) if seed_scores else None,
        "reference_seed_metric_spread": seed_spread,
        "paired_vs_reference_per_scene": paired,
        "cases": records,
    }
    (H / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["| Case | Kind | Valid | S (test) | Dyn mIoU / F1 / Rdyn (test) | Inter mIoU / F1 / Rdyn (test) | Inter mIoU (train) |",
             "|---|---|---|---:|---:|---:|---:|"]

    def triple(r, exp):
        if not r:
            return "absent"
        x = r[exp]
        return f"{x['mIoU']:.3f} / {x['F1']:.3f} / {x['Recalldyn']:.3f}"

    exps = list(spec)
    for case, rec in sorted(records.items(), key=lambda kv: -(kv[1]["S_test"] if kv[1]["S_test"] is not None else -1e9)):
        s = "—" if rec["S_test"] is None else f"{rec['S_test']:+.3f}"
        tr = rec["train"][exps[1]]["mIoU"] if rec["train"] else float("nan")
        lines.append(f"| {case} | {rec['kind']} | {rec['valid']} | {s} | {triple(rec['test'], exps[0])} | "
                     f"{triple(rec['test'], exps[1])} | {tr:.3f} |")
    (H / "score_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps({"seed_S": seed_scores, "bounds": bounds}, indent=2))


if __name__ == "__main__":
    main()
