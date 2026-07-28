#!/usr/bin/env python3
"""Score the ShapeKnots gate. Written before the run finished.

The headline the project cares about: panel-unanimous solves per execution, out of 60. The committed
number is 3.2/60 with 21/60 ever solved across 15 executions.

Three things have to hold for the gate to be the cause:

  1. on_400 beats off_150 on independently adjudicated panel-unanimous solves.
  2. off_400 does NOT -- more pKiss budget alone must not lift the panel rate. This is the falsifier, and
     it is a compute-matched control, so a win here would mean the gate is irrelevant.
  3. The gate must not simply be trading false positives for nothing: report CLAIMED solves against
     ADJUDICATED unanimous solves per arm, so an arm that lies is visible.

Paired within (target, seed) throughout, with an exact McNemar test on the discordant pairs -- the right
test here because each cell is a matched binary outcome on the same target and seed.
"""
import json, os
from collections import defaultdict
import numpy as np
from scipy.stats import binom

H = os.path.dirname(os.path.abspath(__file__))
G = os.path.join(H, "gate")
ARMS = ["off_150", "off_400", "on_400"]


def mcnemar(a, b):
    """Exact McNemar on paired binary vectors: b wins where b=1,a=0 and vice versa."""
    n01 = int(sum(1 for x, y in zip(a, b) if not x and y))
    n10 = int(sum(1 for x, y in zip(a, b) if x and not y))
    n = n01 + n10
    p = 1.0 if n == 0 else min(1.0, 2 * binom.sf(max(n01, n10) - 1, n, 0.5))
    return n01, n10, p


def cp(k, n, a=0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


recs = [json.load(open(os.path.join(G, f))) for f in os.listdir(G) if f.endswith(".json")]
errs = [r for r in recs if "error" in r]
print(f"{len(recs)} attempts on disk, {len(errs)} errors")
if errs:
    from collections import Counter
    for e, c in Counter(r["error"][:70] for r in errs).most_common(3):
        print(f"   {c:4d}x  {e}")
ok = [r for r in recs if "error" not in r]
seeds = sorted({r["seed"] for r in ok})
tids = sorted({r["id"] for r in ok}, key=lambda t: int(t.split(":")[-1]))
cell = {(r["arm"], r["id"], r["seed"]): r for r in ok}
print(f"{len(tids)} targets x {len(seeds)} seeds\n")

def vec(arm, key):
    out = []
    for t in tids:
        for s in seeds:
            r = cell.get((arm, t, s))
            if r is None:
                out.append(0); continue
            if key == "unan":
                out.append(int(bool((r.get("panel") or {}).get("unanimous"))))
            elif key == "claimed":
                out.append(int(bool(r.get("claimed"))))
            elif key == "pkpb":
                p = r.get("panel") or {}
                out.append(int(bool(p.get("pkiss")) and bool(p.get("probknot"))))
            else:
                out.append(r.get(key, 0))
    return np.array(out, float)

print("=== 1. the headline: panel-unanimous solves per execution, out of 60 ===")
print(f"  {'arm':8} {'claimed':>8} {'adjudicated':>12} {'false positives':>16} "
      f"{'per execution':>14} {'95% CI':>16} {'ever/60':>8}")
res = {}
for arm in ARMS:
    cl, un = vec(arm, "claimed"), vec(arm, "unan")
    fp = int(((cl > 0) & (un == 0)).sum())
    ever = len({t for t in tids if any((cell.get((arm, t, s)) or {}).get("panel", {}) and
                                       (cell[(arm, t, s)]["panel"] or {}).get("unanimous")
                                       for s in seeds)})
    per = un.sum() / len(seeds)
    lo, hi = cp(int(un.sum()), len(un))
    res[arm] = {"unan": un, "claimed": cl, "per": per}
    print(f"  {arm:8} {int(cl.sum()):8d} {int(un.sum()):12d} {fp:16d} {per:14.2f} "
          f"[{lo*60:5.2f},{hi*60:5.2f}] {ever:8d}")
print(f"  reference: the committed 15-cell sweep gives 3.20/60 per execution, 21/60 ever")

print("\n=== 2. paired comparisons, exact McNemar on (target, seed) cells ===")
for a, b in (("off_150", "on_400"), ("off_150", "off_400"), ("off_400", "on_400")):
    n01, n10, p = mcnemar(res[a]["unan"], res[b]["unan"])
    ra, rb = res[a]["per"], res[b]["per"]
    print(f"  {b:8} vs {a:8}: {rb:5.2f} vs {ra:5.2f} per execution  ratio "
          f"{rb/max(ra,1e-9):5.2f}   {b} wins {n01:3d} cells, {a} wins {n10:3d}   p = {p:.3g}")
print("  falsifier: off_400 vs off_150 is the compute-matched control. If extra pKiss budget alone")
print("  lifts the panel rate, the gate is not what did the work.")

print("\n=== 3. cost: is the gain paid for, and by how much? ===")
print(f"  {'arm':8} {'pKiss':>8} {'ProbKnot':>9} {'ShapeKnots':>11} {'oracle s':>9} "
      f"{'s/attempt':>10} {'oracle s per unanimous':>23}")
for arm in ARMS:
    pk, pb, sk = vec(arm, "n_pkiss"), vec(arm, "n_probknot"), vec(arm, "n_shapeknots")
    os_, w = vec(arm, "oracle_s"), vec(arm, "s")
    un = res[arm]["unan"].sum()
    print(f"  {arm:8} {pk.mean():8.1f} {pb.mean():9.1f} {sk.mean():11.1f} {os_.mean():9.1f} "
          f"{w.mean():10.1f} {os_.sum()/max(un,1):23.1f}")
b0 = vec("off_150", "oracle_s").sum() / max(res["off_150"]["unan"].sum(), 1)
for arm in ARMS[1:]:
    b = vec(arm, "oracle_s").sum() / max(res[arm]["unan"].sum(), 1)
    print(f"  {arm} costs {b/b0:.2f}x per unanimous solve relative to off_150")

print("\n=== 4. did the gate merely stop lying, or actually find more? ===")
for arm in ARMS:
    cl, un, pp = res[arm]["claimed"], res[arm]["unan"], vec(arm, "pkpb")
    print(f"  {arm:8} claimed {int(cl.sum()):4d}  of which unanimous {int(((cl>0)&(un>0)).sum()):4d} "
          f"({100*((cl>0)&(un>0)).sum()/max(cl.sum(),1):5.1f}%)   "
          f"pKiss+ProbKnot agreements returned {int(pp.sum()):4d}")
print("  off_* arms return a sequence whenever pKiss+ProbKnot agree, so their claimed count includes")
print("  the 90.9% that ShapeKnots rejects. on_400 returns only three-oracle agreements.")

print("\n=== 5. gate internals: how many candidates were screened and rejected? ===")
sk, rej = vec("on_400", "n_shapeknots"), vec("on_400", "n_sk_reject")
scr = sk[sk > 0]
print(f"  attempts that screened at least one candidate: {int((sk>0).sum())}/{len(sk)}")
if len(scr):
    print(f"  candidates screened per such attempt: median {np.median(scr):.1f} "
          f"mean {scr.mean():.1f} max {scr.max():.0f}")
    print(f"  rejection rate among screened candidates: {rej.sum()/max(sk.sum(),1):.3f} "
          f"(the stored sweep implied 0.909)")
    print(f"  attempts that hit the max_shapeknots=60 cap: {int((sk>=60).sum())}")

print("\n=== 6. per-target: which targets the gate unlocked ===")
print(f"  {'id':>4} " + " ".join(f"{a:>8}" for a in ARMS))
new = []
for t in tids:
    row = []
    for arm in ARMS:
        row.append(sum(int(bool((cell.get((arm, t, s)) or {}).get("panel") or {}) and
                           ((cell[(arm, t, s)]["panel"] or {}).get("unanimous")))
                       for s in seeds if (arm, t, s) in cell))
    if any(row):
        flag = "  <== gate only" if row[2] > 0 and row[0] == 0 else ""
        print(f"  {t.split(':')[-1]:>4} " + " ".join(f"{v:8d}" for v in row) + flag)
        if row[2] > 0 and row[0] == 0:
            new.append(t)
print(f"  targets solved by on_400 but never by off_150: {len(new)} "
      f"{[t.split(':')[-1] for t in new]}")

json.dump({a: {"unan": res[a]["unan"].tolist(), "claimed": res[a]["claimed"].tolist(),
               "n_pkiss": vec(a, "n_pkiss").tolist(), "oracle_s": vec(a, "oracle_s").tolist()}
           for a in ARMS}, open(os.path.join(H, "gate_result.json"), "w"), indent=1)
