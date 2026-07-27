#!/usr/bin/env python3
"""Score the trajectory-allocation validation. Written before the run finished.

The arms are deliberately NOT budget-matched, because at this budget they cannot be: 30 targets at one
trajectory each already exhausts it, and cost per trajectory rises from 42 calls on the lowest-Delta 11
to 133 on the full 30. So the comparison is on REALISED calls, with two summaries:

  * solves per 1000 realised calls -- efficiency, the quantity the project cares about
  * the dose-response curve over S = 2, 4, 8 with NESTED seeds, which is the mechanism test: if
    independent trajectories are what convert, solves must rise with S on the same 11 targets

and the equal-cost comparison by interpolation on the delta curve against the depth1 baseline.

Paired within replicate throughout. 15 replicates with disjoint seed blocks, so replicates are
independent draws and the paired differences are exchangeable.
"""
import json, os
from collections import defaultdict
import numpy as np
from scipy.stats import binom

H = os.path.dirname(os.path.abspath(__file__))
S = os.path.join(H, "seeds")
ARMS = ["depth1", "delta_S2", "delta_S4", "delta_S8", "rand_S4"]


def sign_test(d):
    pos = int(sum(1 for x in d if x > 0)); neg = int(sum(1 for x in d if x < 0))
    n = pos + neg
    return pos, neg, (1.0 if n == 0 else min(1.0, 2 * binom.sf(max(pos, neg) - 1, n, 0.5)))


recs = [json.load(open(os.path.join(S, f))) for f in os.listdir(S) if f.endswith(".json")]
errs = [r for r in recs if "error" in r]
plan = json.load(open(os.path.join(H, "seeds_plan.json")))
NREP = plan["nrep"]
print(f"{len(recs)} trajectories on disk, {len(errs)} errors, k={plan['k']}, S={plan['S']}, "
      f"budget {plan['budget']} calls/replicate")
if errs:
    from collections import Counter
    for e, c in Counter(r["error"][:70] for r in errs).most_common(3):
        print(f"   {c:4d}x  {e}")

# per (arm, rep): solves = number of DISTINCT targets with at least one panel-unanimous trajectory
per = defaultdict(lambda: {"calls": 0, "won": set(), "pk": set(), "traj": 0})
for r in recs:
    if "error" in r:
        continue
    a = per[(r["arm"], r["rep"])]
    a["calls"] += r.get("n_pkiss", 0)
    a["traj"] += 1
    if (r.get("panel") or {}).get("unanimous"):
        a["won"].add(r["id"])
    if r.get("solved"):
        a["pk"].add(r["id"])

def S_(arm, key):
    if key == "solves":
        return np.array([len(per[(arm, i)]["won"]) for i in range(NREP)], float)
    if key == "pkiss":
        return np.array([len(per[(arm, i)]["pk"]) for i in range(NREP)], float)
    return np.array([per[(arm, i)][key] for i in range(NREP)], float)

print("\n=== results, per replicate (solves = distinct targets passing the full panel) ===")
print(f"  {'arm':9} {'traj':>5} {'calls':>7} {'x budget':>9} {'solves':>7} {'sd':>5} "
      f"{'per 1000 calls':>15} {'in-loop pKiss':>14}")
base = None
for arm in ARMS:
    c, s, t, p = S_(arm, "calls"), S_(arm, "solves"), S_(arm, "traj"), S_(arm, "pkiss")
    eff = 1000 * s.mean() / max(c.mean(), 1e-9)
    if arm == "depth1":
        base = (c.mean(), s.mean(), eff)
    print(f"  {arm:9} {t.mean():5.0f} {c.mean():7.0f} {c.mean()/plan['budget']:9.2f} "
          f"{s.mean():7.2f} {s.std():5.2f} {eff:15.2f} {p.mean():14.2f}")

print(f"\n=== efficiency vs the depth1 baseline ({base[1]:.2f} solves at {base[0]:.0f} calls) ===")
print(f"  {'arm':9} {'solves ratio':>13} {'cost ratio':>11} {'efficiency ratio':>17} "
      f"{'paired +/-/p':>15}")
for arm in ARMS[1:]:
    c, s = S_(arm, "calls"), S_(arm, "solves")
    pos, neg, pv = sign_test(s - S_("depth1", "solves"))
    print(f"  {arm:9} {s.mean()/max(base[1],1e-9):13.2f} {c.mean()/base[0]:11.2f} "
          f"{(1000*s.mean()/c.mean())/base[2]:17.2f} {pos:4d}/{neg:3d}/{pv:6.4f}")

print("\n=== the mechanism test: dose-response on the SAME 11 targets, nested seeds ===")
print(f"  {'S':>3} {'trajectories':>13} {'calls':>7} {'solves':>7} {'marginal solves':>16} "
      f"{'marginal calls':>15} {'solves per 1000 marginal calls':>31}")
prev = None
for arm, s_ in (("delta_S2", 2), ("delta_S4", 4), ("delta_S8", 8)):
    c, s, t = S_(arm, "calls").mean(), S_(arm, "solves").mean(), S_(arm, "traj").mean()
    if prev is None:
        print(f"  {s_:3d} {t:13.0f} {c:7.0f} {s:7.2f} {'-':>16} {'-':>15} {'-':>31}")
    else:
        dm, dc = s - prev[1], c - prev[0]
        print(f"  {s_:3d} {t:13.0f} {c:7.0f} {s:7.2f} {dm:+16.2f} {dc:+15.0f} "
              f"{1000*dm/max(dc,1e-9):31.2f}")
    prev = (c, s)
d2, d4, d8 = (S_(f"delta_S{i}", "solves") for i in (2, 4, 8))
for a, b, la, lb in ((d2, d4, "S2", "S4"), (d4, d8, "S4", "S8")):
    pos, neg, pv = sign_test(b - a)
    print(f"  paired {lb} - {la}: mean {(b-a).mean():+.2f}   {pos}/{neg} replicates   p = {pv:.4f}")

print("\n=== is it Delta, or just fewer-targets-more-seeds? (delta_S4 vs rand_S4) ===")
a, b = S_("delta_S4", "solves"), S_("rand_S4", "solves")
ca, cb = S_("delta_S4", "calls"), S_("rand_S4", "calls")
pos, neg, pv = sign_test(a - b)
print(f"  delta_S4 {a.mean():.2f} solves at {ca.mean():.0f} calls   "
      f"rand_S4 {b.mean():.2f} at {cb.mean():.0f}")
print(f"  paired difference {(a-b).mean():+.2f}  {pos}/{neg} replicates  p = {pv:.4f}")
print(f"  efficiency: delta {1000*a.mean()/ca.mean():.2f} vs random "
      f"{1000*b.mean()/cb.mean():.2f} solves per 1000 calls")

print("\n=== equal-solves cost, interpolated on the delta curve ===")
xs = [S_(f"delta_S{i}", "calls").mean() for i in (2, 4, 8)]
ys = [S_(f"delta_S{i}", "solves").mean() for i in (2, 4, 8)]
tgt = base[1]
if tgt <= max(ys):
    j = int(np.searchsorted(ys, tgt))
    if j == 0:
        need = xs[0] * tgt / max(ys[0], 1e-9)
    else:
        w = (tgt - ys[j - 1]) / max(ys[j] - ys[j - 1], 1e-9)
        need = xs[j - 1] + w * (xs[j] - xs[j - 1])
    dcost = 8 * plan["k"]
    print(f"  depth1 gets {tgt:.2f} solves at {base[0]:.0f} calls")
    print(f"  delta curve reaches it at {need:.0f} calls  =  {base[0]/need:.2f}x cheaper")
    print(f"  charging Delta's own probe ({dcost} calls, N=8 x {plan['k']} targets): "
          f"{base[0]/(need+dcost):.2f}x single-run, {base[0]/(need+dcost/NREP):.2f}x amortised")
else:
    print(f"  depth1's {tgt:.2f} solves is above the delta curve's maximum {max(ys):.2f} -- "
          f"the policy never reaches the baseline, and the efficiency claim fails")

print("\n=== which targets carried it (union over replicates) ===")
for arm in ARMS:
    u = defaultdict(int)
    for i in range(NREP):
        for t in per[(arm, i)]["won"]:
            u[t] += 1
    tot = sum(u.values())
    print(f"  {arm:9} {len(u):2d} distinct targets ever won, {tot:3d} replicate-wins: "
          + ", ".join(f"{t.split(':')[-1]}:{c}" for t, c in sorted(u.items(), key=lambda x: -x[1])))

json.dump({arm: {k: (S_(arm, k)).tolist() for k in ("calls", "solves", "traj", "pkiss")}
           for arm in ARMS}, open(os.path.join(H, "seeds_result.json"), "w"), indent=1)
