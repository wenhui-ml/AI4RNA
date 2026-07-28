#!/usr/bin/env python3
"""How strong is each claim, actually? Intervals and multiplicity for every headline on this branch.

The point estimates are in the run logs. What is missing is (a) an interval on every ratio, bootstrapped
over the unit that was randomised, and (b) an honest count of how many tests were run, because a p = 0.02
among twenty comparisons is not the same object as a p = 7.6e-06.

Bootstrap unit matters and differs per experiment:
  gate         over the 8 SEEDS (the replication unit) and separately over the 60 TARGETS
  trajectory   over the 15 REPLICATES (disjoint seed blocks, so they are independent draws)
  robustness   over the 64 TARGETS, paired
"""
import json, os
from collections import defaultdict
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)
B = 10000


def ci(v, a=2.5):
    return float(np.percentile(v, a)), float(np.percentile(v, 100 - a))


print("=== 1. ShapeKnots gate: capability ratio, bootstrapped two ways ===")
recs = [json.load(open(os.path.join(H, "gate", f))) for f in os.listdir(os.path.join(H, "gate"))
        if f.endswith(".json")]
ok = [r for r in recs if "error" not in r]
seeds = sorted({r["seed"] for r in ok})
tids = sorted({r["id"] for r in ok})
win = defaultdict(int)
for r in ok:
    if (r.get("panel") or {}).get("unanimous"):
        win[(r["arm"], r["id"], r["seed"])] = 1
for unit, pool in (("seeds", seeds), ("targets", tids)):
    rs = []
    for _ in range(B):
        pick = rng.choice(len(pool), len(pool))
        a = b = 0
        for i in pick:
            u = pool[i]
            for t in (tids if unit == "seeds" else seeds):
                k = (u, t) if unit == "targets" else (t, u)
                a += win.get(("off_150", k[0], k[1]), 0)
                b += win.get(("on_400", k[0], k[1]), 0)
        rs.append(b / max(a, 1e-9))
    lo, hi = ci(rs)
    print(f"  bootstrap over {unit:8}: ratio {np.median(rs):.2f}  95% CI [{lo:.2f}, {hi:.2f}]")
a = sum(v for (arm, _, _), v in win.items() if arm == "off_150")
b = sum(v for (arm, _, _), v in win.items() if arm == "on_400")
print(f"  point: {b}/{a} = {b/a:.2f}x   (McNemar 18 discordant cells, all one way, p = 7.6e-06)")
print(f"  the falsifier: off_400 = {sum(v for (arm,_,_),v in win.items() if arm=='off_400')} "
      f"identical to off_150, 0 discordant cells, p = 1")

print("\n=== 2. trajectory allocation: which comparisons are actually significant? ===")
S = json.load(open(os.path.join(H, "seeds_result.json")))
arms = list(S)
for arm in arms:
    s = np.array(S[arm]["solves"], float); c = np.array(S[arm]["calls"], float)
    print(f"  {arm:9} solves {s.mean():5.2f} +/- {s.std()/np.sqrt(len(s)):.2f}  "
          f"calls {c.mean():7.0f}  per 1000 {1000*s.mean()/c.mean():5.2f}")
d0 = np.array(S["depth1"]["solves"], float)
c0 = np.array(S["depth1"]["calls"], float)
for arm in ("delta_S2", "delta_S4", "delta_S8", "rand_S4"):
    s = np.array(S[arm]["solves"], float); c = np.array(S[arm]["calls"], float)
    idx = rng.integers(0, len(s), (B, len(s)))
    eff = [(s[i].mean() / c[i].mean()) / (d0[i].mean() / c0[i].mean()) for i in idx]
    sr = [s[i].mean() / max(d0[i].mean(), 1e-9) for i in idx]
    lo, hi = ci(eff); slo, shi = ci(sr)
    print(f"  {arm:9} vs depth1: solves x{s.mean()/d0.mean():.2f} [{slo:.2f},{shi:.2f}]   "
          f"efficiency x{(s.mean()/c.mean())/(d0.mean()/c0.mean()):.2f} [{lo:.2f},{hi:.2f}]")
print("  NOTE: only delta_S8 vs depth1 reaches p<0.05 on SOLVES (11/2, p=0.0225).")
print("  delta_S4 vs depth1 is 8/4, p=0.39 -- not significant. The 2.26x equal-solves figure is an")
print("  INTERPOLATION between delta_S2 and delta_S4 and carries no test of its own.")
print("  What IS strong is the dose-response: S2->S4 11/0 p=0.0010, S4->S8 13/0 p=0.0002.")

print("\n=== 3. Eterna ensemble objective: paired ratio over targets ===")
R = json.load(open(os.path.join(H, "robustness.json")))
G = defaultdict(dict)
for r in R:
    if r.get("p") is not None:
        G[r["grp"]][r["id"]] = r["p"]
keys = sorted(set(G["ours"]) & set(G["inv_pf"]))
o = np.array([G["ours"][k] for k in keys]); q = np.array([G["inv_pf"][k] for k in keys])
idx = rng.integers(0, len(keys), (B, len(keys)))
rs = [np.median(q[i]) / max(np.median(o[i]), 1e-12) for i in idx]
lo, hi = ci(rs)
print(f"  median P: ours {np.median(o):.4f}  inv_pf {np.median(q):.4f}")
print(f"  ratio {np.median(q)/np.median(o):.2f}x  95% CI [{lo:.2f}, {hi:.2f}]  "
      f"({int((q>o).sum())}/{len(keys)} targets, Wilcoxon p = 6.5e-12)")
print("  LIMIT: this is measured on the 64 targets we ALREADY solve. It shows our existing solutions")
print("  could be far more robust. It does NOT show inv_pf solves any of the 36 we fail, so it is not")
print("  yet a capability claim on 55/100.")

print("\n=== 4. multiplicity: how many tests were run on this branch? ===")
tests = [
    ("step-2 frontier, delta vs uniform", 4), ("step-2, delta vs random", 4),
    ("step-2, delta vs length", 4), ("band, Fisher", 1), ("p0, MWU + Spearman", 4),
    ("trajectory, arms vs depth1", 4), ("trajectory, dose-response", 2),
    ("gate, McNemar", 3), ("robustness, Wilcoxon", 3), ("who_rejects, MWU", 2),
]
n = sum(k for _, k in tests)
print(f"  approximately {n} hypothesis tests across the branch:")
for lab, k in tests:
    print(f"    {k:2d}  {lab}")
print(f"  Bonferroni threshold at alpha=0.05 with {n} tests: p < {0.05/n:.2e}")
print("  survives:  gate McNemar 7.6e-06  |  dose-response 2e-04 and 1e-03  |  robustness 6.5e-12")
print("             |  who_rejects (descriptive, not a test)")
print("  does NOT survive:  step-2's one positive (1.75x, p=0.031)  |  delta_S8 vs depth1 (p=0.0225)")
print("                     |  delta vs random on solves (p=0.092)")
