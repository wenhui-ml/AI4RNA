"""R3.5 -- statistical backing for the controlled-Goodhart claim.

The paper reports that in the CHEAP arm robustness (P(target)>=theta) rises then FALLS with budget while the
proxy (exact match) keeps rising, and that the two arms' robustness trajectories diverge. A reviewer asks for
a proper test that (a) the cheap-arm decline is not sampling noise and (b) the arm x budget INTERACTION is
significant, both respecting the hierarchical design (18 targets x 5 seeds; outcomes clustered by target).

We provide two nonparametric, cluster-aware tests on the saved per-unit rows
(budget, arm, pid=target, seed, bp, p_target):

  * CLUSTER BOOTSTRAP (resample TARGETS with replacement): CIs for
      - cheap-arm robustness decline: R_cheap(peak budget) - R_cheap(max budget)
      - difference-in-differences (INTERACTION): [R_strong(max)-R_strong(min)] - [R_cheap(max)-R_cheap(min)]
  * PERMUTATION test of the interaction: within each (target,seed) permute the ARM label between the two
      arms' trajectories and recompute the DiD statistic; p = fraction of permutations with |DiD*| >= |DiD|.

Also emits PER-TARGET cheap-arm robustness trajectories (to show heterogeneity: how many targets exhibit the
peak-then-decline vs monotone vs flat)."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

RUN = os.path.join(os.path.dirname(__file__), "..", "runs", "goodhart_multiseed", "goodhart_multiseed.json")


def _load(path, theta):
    d = json.load(open(path))
    rows = d["rows"]
    budgets = sorted(set(r["budget"] for r in rows))
    targets = sorted(set(r["pid"] for r in rows))
    seeds = sorted(set(r["seed"] for r in rows))
    # robust indicator per (arm,budget,target,seed)
    R = {}
    for r in rows:
        R[(r["arm"], r["budget"], r["pid"], r["seed"])] = int(r["p_target"] >= theta)
    return R, budgets, targets, seeds


def _rate(R, arm, budget, targets, seeds):
    v = [R.get((arm, budget, t, s)) for t in targets for s in seeds]
    v = [x for x in v if x is not None]
    return 100 * np.mean(v) if v else float("nan")


def _rate_on(R, arm, budget, tsel, seeds):
    v = [R.get((arm, budget, t, s)) for t in tsel for s in seeds if R.get((arm, budget, t, s)) is not None]
    return 100 * np.mean(v) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=RUN)
    ap.add_argument("--theta", type=float, default=0.9)
    ap.add_argument("--B", type=int, default=5000, help="bootstrap resamples")
    ap.add_argument("--P", type=int, default=10000, help="permutations")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    R, budgets, targets, seeds = _load(a.inp, a.theta)
    bmin, bmax = budgets[0], budgets[-1]
    print(f"Goodhart stats: {len(targets)} targets x {len(seeds)} seeds; budgets {budgets}; theta={a.theta}")

    # point estimates
    cheap = {b: _rate(R, "cheap", b, targets, seeds) for b in budgets}
    strong = {b: _rate(R, "strong", b, targets, seeds) for b in budgets}
    peak_b = max(budgets, key=lambda b: cheap[b])
    print("  cheap-arm robustness:", {b: round(cheap[b], 1) for b in budgets}, f"(peak @ {peak_b})")
    print("  strong-arm robustness:", {b: round(strong[b], 1) for b in budgets})

    decline = cheap[peak_b] - cheap[bmax]                      # cheap peak -> max decline
    did = (strong[bmax] - strong[bmin]) - (cheap[bmax] - cheap[bmin])  # interaction (DiD)

    # ---- cluster bootstrap over targets
    decl_bs, did_bs = [], []
    tarr = np.array(targets)
    for _ in range(a.B):
        ts = list(rng.choice(tarr, size=len(tarr), replace=True))
        c_peak = _rate_on(R, "cheap", peak_b, ts, seeds); c_max = _rate_on(R, "cheap", bmax, ts, seeds)
        c_min = _rate_on(R, "cheap", bmin, ts, seeds)
        s_max = _rate_on(R, "strong", bmax, ts, seeds); s_min = _rate_on(R, "strong", bmin, ts, seeds)
        decl_bs.append(c_peak - c_max)
        did_bs.append((s_max - s_min) - (c_max - c_min))
    def ci(x): return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
    decl_ci, did_ci = ci(decl_bs), ci(did_bs)

    # ---- permutation test of the INTERACTION: within each (target,seed), swap arm labels
    did0 = abs(did)
    ge = 0
    for _ in range(a.P):
        Rp = {}
        for t in targets:
            for s in seeds:
                swap = rng.random() < 0.5
                for b in budgets:
                    ca = ("strong" if swap else "cheap"); sa = ("cheap" if swap else "strong")
                    Rp[("cheap", b, t, s)] = R.get((ca, b, t, s))
                    Rp[("strong", b, t, s)] = R.get((sa, b, t, s))
        cmax = _rate(Rp, "cheap", bmax, targets, seeds); cmin = _rate(Rp, "cheap", bmin, targets, seeds)
        smax = _rate(Rp, "strong", bmax, targets, seeds); smin = _rate(Rp, "strong", bmin, targets, seeds)
        if abs((smax - smin) - (cmax - cmin)) >= did0:
            ge += 1
    p_perm = (ge + 1) / (a.P + 1)

    # ---- per-target cheap-arm trajectories (heterogeneity)
    traj = {}
    shape = {"peak_then_decline": 0, "monotone_up": 0, "monotone_down": 0, "flat": 0, "other": 0}
    for t in targets:
        y = [np.mean([R.get(("cheap", b, t, s)) for s in seeds]) * 100 for b in budgets]
        traj[str(t)] = [round(v, 1) for v in y]
        pk = int(np.argmax(y))
        if max(y) == min(y):
            shape["flat"] += 1
        elif 0 < pk < len(y) - 1 and y[-1] < y[pk]:
            shape["peak_then_decline"] += 1
        elif all(y[i] <= y[i + 1] for i in range(len(y) - 1)):
            shape["monotone_up"] += 1
        elif all(y[i] >= y[i + 1] for i in range(len(y) - 1)):
            shape["monotone_down"] += 1
        else:
            shape["other"] += 1

    out = {"theta": a.theta, "budgets": budgets, "cheap": cheap, "strong": strong, "peak_budget": peak_b,
           "cheap_decline": {"point": decline, "ci": decl_ci},
           "interaction_DiD": {"point": did, "ci": did_ci, "p_perm": p_perm},
           "per_target_shape": shape, "per_target_traj": traj}
    od = os.path.join(os.path.dirname(a.inp), "goodhart_stats.json")
    json.dump(out, open(od, "w"), indent=1)

    print(f"\n=== R3.5 RESULTS ===")
    print(f"  cheap-arm decline peak(@{peak_b})->max(@{bmax}): {decline:+.1f} pts  95% CI {decl_ci[0]:+.1f},{decl_ci[1]:+.1f}")
    print(f"  INTERACTION DiD [strong slope - cheap slope]: {did:+.1f} pts  95% CI {did_ci[0]:+.1f},{did_ci[1]:+.1f}")
    print(f"  permutation p (|DiD*| >= |DiD|): {p_perm:.4f}  ({a.P} perms)")
    print(f"  per-target cheap-arm shape (of {len(targets)}): {shape}")
    print(f"  wrote {od}")


if __name__ == "__main__":
    main()
