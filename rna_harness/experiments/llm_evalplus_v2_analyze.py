"""R3-3.5 analyzer for the reproducible HumanEval+ rerun (llm_evalplus_v2.py artifacts).

Uses a FIXED pre-registered split (seed 0), not 100 averaged random splits, and reports:
  - over-acceptance at the weakest cheap rung (1 base test): joint + conditional, problem-clustered CI;
  - a composition-stable PROPORTION dial curve with CI (all tasks at every rung);
  - Goodhart-under-selection (best-of-N gold, cheap- vs strong-selected);
  - a RISK--COVERAGE--COST curve (as the cheap rung tightens: coverage down, risk down, #tests up);
  - sandbox diagnostics (extraction failures, timeouts, exceptions) so the run is auditable;
  - a small split-seed sensitivity band.
All from the saved pass vectors + diagnostics; raw completions are also on disk for inspection/re-scoring."""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

RUNS = os.path.join(os.path.dirname(__file__), "..", "runs", "llm_evalplus_v2")
PROPS = [round(0.1 * i, 1) for i in range(1, 11)]


def _clustered_ci(pp, rng, B=2000):
    nums = np.array([x[0] for x in pp], float); dens = np.array([x[1] for x in pp], float)
    td = dens.sum(); pt = 100 * nums.sum() / td if td else 0.0
    if len(nums) == 0:
        return pt, 0.0, 0.0
    idx = np.arange(len(nums)); boots = []
    for _ in range(B):
        s = rng.choice(idx, size=len(idx), replace=True); d = dens[s].sum()
        boots.append(100 * nums[s].sum() / d if d else 0.0)
    return pt, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _split(U, seed):
    rng = np.random.default_rng(seed); perm = rng.permutation(U); half = U // 2
    tune, gold = perm[:half], perm[half:]
    if len(gold) == 0:
        gold, tune = perm, perm[:0]
    return tune, gold


def analyze_model(rec, split_seed=0):
    per = rec["per"]
    rng = np.random.default_rng(split_seed)
    joint_pp, cond_pp = [], []
    cheap_gold, strong_gold = [], []
    dial = {q: [] for q in PROPS}
    rc = {q: {"acc": 0.0, "acc_fail": 0.0, "n_cand": 0, "tests": 0.0} for q in PROPS}
    diag = {"n_cand": 0, "extraction_fail": 0, "base_timeout": 0, "plus_timeout": 0, "gen_err": 0}
    for p in per:
        nb, npl = p["n_base"], p["n_plus"]
        vecs = [np.array(c["vec"], bool) for c in p["candidates"] if len(c["vec"]) == nb + npl]
        if not vecs or nb == 0 or npl == 0:
            continue
        for c in p["candidates"]:
            diag["n_cand"] += 1
            d = c.get("diag", {})
            if not d.get("extraction_ok", True):
                diag["extraction_fail"] += 1
            if d.get("base_status") == "timeout":
                diag["base_timeout"] += 1
            if d.get("plus_status") == "timeout":
                diag["plus_timeout"] += 1
            if isinstance(c.get("completion"), str) and c["completion"].startswith("__GEN_ERR__"):
                diag["gen_err"] += 1
        base = np.stack([v[:nb] for v in vecs]); plus = np.stack([v[nb:nb + npl] for v in vecs])
        U = plus.shape[1]
        tune, gold = _split(U, split_seed)
        gold_ok = plus[:, gold].all(1)
        bperm = rng.permutation(nb)                        # fixed per-task ordering (this split seed)
        passk1 = base[:, bperm[0]]                          # weakest rung = 1 base test
        joint_pp.append((float((passk1 & ~gold_ok).sum()), float(len(vecs))))
        cond_pp.append((float((passk1 & ~gold_ok).sum()), float(passk1.sum())))
        cheap_pick = int(np.lexsort((np.arange(len(vecs)), -base.sum(1)))[0])
        strong_pick = int(np.lexsort((np.arange(len(vecs)), -(base.sum(1) + plus[:, tune].sum(1))))[0])
        cheap_gold.append((float(gold_ok[cheap_pick]), 1.0)); strong_gold.append((float(gold_ok[strong_pick]), 1.0))
        for q in PROPS:
            kk = max(1, int(np.ceil(q * nb)))
            passq = base[:, bperm[:kk]].all(1)
            dial[q].append((float((passq & ~gold_ok).sum()), float(passq.sum())))
            rc[q]["acc"] += float(passq.sum()); rc[q]["acc_fail"] += float((passq & ~gold_ok).sum())
            rc[q]["n_cand"] += len(vecs); rc[q]["tests"] += kk
    j = _clustered_ci(joint_pp, rng); c = _clustered_ci(cond_pp, rng)
    cg = _clustered_ci(cheap_gold, rng); sg = _clustered_ci(strong_gold, rng)
    dcurve = {q: _clustered_ci(dial[q], rng) for q in PROPS}
    risk_cov = {q: {"coverage": round(rc[q]["acc"] / rc[q]["n_cand"], 4) if rc[q]["n_cand"] else 0.0,
                    "risk_cond_fp": round(rc[q]["acc_fail"] / rc[q]["acc"], 4) if rc[q]["acc"] else 0.0,
                    "mean_tests": round(rc[q]["tests"] / len(joint_pp), 2) if joint_pp else 0.0}
                for q in PROPS}
    return {"n_problems": len(joint_pp), "model": rec.get("model"), "model_key": rec.get("model_key"),
            "temperature": rec.get("temperature"), "n_dropped": len(rec.get("dropped", [])),
            "joint_mass": j, "conditional_rate": c, "cheap_gold": cg, "strong_gold": sg,
            "raw": {"joint_num": round(sum(x[0] for x in joint_pp)), "joint_den": round(sum(x[1] for x in joint_pp)),
                    "cond_num": round(sum(x[0] for x in cond_pp)), "cond_den": round(sum(x[1] for x in cond_pp))},
            "dial_curve_prop": dcurve, "risk_coverage": risk_cov, "diagnostics": diag,
            "split_seed": split_seed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--sensitivity-seeds", type=int, nargs="+", default=[1, 2, 3])
    a = ap.parse_args()
    out = {}
    files = [f for f in glob.glob(os.path.join(RUNS, "*.json")) if os.path.basename(f) != "analysis_v2.json"]
    print("=== HumanEval+ v2 (fixed pre-registered split; all artifacts on disk) ===")
    print(f"{'model':16} | conditional k=1 (num/den) | joint | gold cheap->strong | extraction_fail")
    for f in sorted(files):
        rec = json.load(open(f))
        if "per" not in rec:
            continue
        r = analyze_model(rec, a.split_seed)
        # sensitivity over alternative split seeds (conditional rate range)
        cond_alt = [analyze_model(rec, s)["conditional_rate"][0] for s in a.sensitivity_seeds]
        r["conditional_rate_sensitivity"] = [round(x, 1) for x in cond_alt]
        out[r["model_key"]] = r
        j, c, cg, sg, d = r["joint_mass"], r["conditional_rate"], r["cheap_gold"], r["strong_gold"], r["diagnostics"]
        print(f"{r['model_key']:16} | {c[0]:5.1f}% [{c[1]:.1f},{c[2]:.1f}] ({r['raw']['cond_num']}/{r['raw']['cond_den']}) "
              f"| {j[0]:5.1f}% | {cg[0]:.1f}->{sg[0]:.1f} ({sg[0]-cg[0]:+.1f}) | {d['extraction_fail']}/{d['n_cand']}")
    json.dump(out, open(os.path.join(RUNS, "analysis_v2.json"), "w"), indent=1)
    print("\n  risk-coverage-cost (per model), q -> coverage / cond-FP / mean#tests:")
    for k, r in out.items():
        print(f"    {k}:")
        for q in PROPS:
            rcq = r["risk_coverage"][q]
            print(f"      q={q}: cov={rcq['coverage']:.3f} risk={rcq['risk_cond_fp']:.3f} tests={rcq['mean_tests']}")
    print(f"\n  wrote {os.path.join(RUNS, 'analysis_v2.json')}")


if __name__ == "__main__":
    main()
