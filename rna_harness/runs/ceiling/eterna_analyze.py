#!/usr/bin/env python3
"""Eterna100 on the same S(B) surface: S_inf, B_half, U, in CPU-seconds, adjudicated under umfe.

Mirrors metric.py so the two benchmarks are directly comparable. Reports both predicates, because the
community standard for Eterna100 is 'match' and the honest one is 'umfe', and the gap between them is
itself a result (the archive gave 64 under match and 55 under umfe).
"""
import json, os
from collections import defaultdict
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(H, "eterna_sb")
recs = [json.load(open(os.path.join(D, f))) for f in os.listdir(D) if f.endswith(".json")]
errs = [r for r in recs if "error" in r]
ok = [r for r in recs if "error" not in r]
print(f"{len(recs)} attempts, {len(errs)} errors")
if errs:
    from collections import Counter
    for e, c in Counter(r["error"][:70] for r in errs).most_common(3):
        print(f"   {c:4d}x  {e}")

tids = sorted({r["id"] for r in ok}, key=lambda t: int(t.split(":")[1]))
arms = sorted({r["arm"] for r in ok})
nseed = len({r["seed"] for r in ok})
print(f"{len(tids)} targets x {nseed} seeds x {len(arms)} arms\n")


def water_fill(p, c, B, bmax=2000):
    p = np.asarray(p, float); c = np.asarray(c, float)
    live = np.where(p > 0)[0]
    if len(live) == 0 or B <= 0:
        return 0.0
    b = np.zeros(len(p)); gain = p[live] / c[live]; spent = 0.0
    for _ in range(bmax * max(1, len(live))):
        k = int(np.argmax(gain)); i = live[k]
        if gain[k] <= 0:
            break
        if spent + c[i] > B:
            aff = [j for j, idx in enumerate(live) if spent + c[idx] <= B and gain[j] > 0]
            if not aff:
                break
            k = max(aff, key=lambda j: gain[j]); i = live[k]
        b[i] += 1; spent += c[i]
        gain[k] = p[i] * (1 - p[i]) ** b[i] / c[i]
        if b[i] >= bmax:
            gain[k] = -1.0
    return float(np.sum(1.0 - (1.0 - p) ** b))


GRID = np.logspace(0, 5.5, 60)
print("=== per-arm summary, both predicates ===")
print(f"  {'arm':11} {'pred':>6} {'one-shot /100':>14} {'S_inf (ever)':>13} {'B_half (cpu-s)':>15} "
      f"{'U':>7} {'cpu-s/attempt':>14}")
SUM = {}
for arm in arms:
    R = [r for r in ok if r["arm"] == arm]
    for pred in ("match", "umfe"):
        w = defaultdict(list); c = defaultdict(list)
        for r in R:
            w[r["id"]].append(int(bool((r.get("adj") or {}).get(pred))))
            c[r["id"]].append(float(r.get("cpu_s", 0.0)))
        p = np.array([np.mean(w[t]) for t in tids])
        cc = np.array([max(np.mean(c[t]), 1e-6) for t in tids])
        s = np.array([water_fill(p, cc, B) for B in GRID])
        s_inf = int(np.sum(p > 0))
        one = float(np.mean([np.mean(w[t]) for t in tids]) * len(tids))
        half = s_inf / 2.0
        i = int(np.searchsorted(s, half))
        if i == 0:
            bh = GRID[0]
        elif i >= len(GRID):
            bh = float("inf")
        else:
            ww = (half - s[i - 1]) / max(s[i] - s[i - 1], 1e-12)
            bh = float(np.exp(np.log(GRID[i - 1]) + ww * (np.log(GRID[i]) - np.log(GRID[i - 1]))))
        U = float(np.trapezoid(s, np.log(GRID)) / (np.log(GRID[-1]) - np.log(GRID[0])))
        SUM[(arm, pred)] = {"p": p, "c": cc, "S": s, "s_inf": s_inf, "bh": bh, "U": U}
        print(f"  {arm:11} {pred:>6} {one:14.2f} {s_inf:13d} {bh:15.0f} {U:7.2f} {cc.mean():14.2f}")
print("  reference: the archive's one-shot run gave 64/100 under match and 55/100 under umfe")

print("\n=== the honest number: how much does 'ever' beat 'one-shot'? ===")
for arm in arms:
    for pred in ("match", "umfe"):
        S = SUM[(arm, pred)]
        one = float(np.sum(S["p"]))
        print(f"  {arm:11} {pred:>6}: one-shot {one:5.2f}/100   ever-solved (S_inf) {S['s_inf']:3d}/100"
              f"   ratio {S['s_inf']/max(one,1e-9):5.2f}x")

print("\n=== does the umfe gate buy anything here? (expected: little -- umfe rejects ~14%) ===")
a, b = "remc_plain", "remc_gate"
for pred in ("match", "umfe"):
    A, Bb = SUM[(a, pred)], SUM[(b, pred)]
    print(f"  {pred:>6}: S_inf {A['s_inf']} -> {Bb['s_inf']}   U {A['U']:.2f} -> {Bb['U']:.2f}   "
          f"cost/attempt {A['c'].mean():.2f} -> {Bb['c'].mean():.2f} cpu-s")
R = [r for r in ok if r["arm"] == "remc_gate"]
tr = np.array([r.get("tries", 1) for r in R], float)
rj = np.array([r.get("rejects", 0) for r in R], float)
print(f"  gate internals: mean tries {tr.mean():.2f}, mean rejects {rj.mean():.2f}, "
      f"attempts that ever rejected {int((rj>0).sum())}/{len(R)}")

print("\n=== cost to reach a fixed number of solved targets (umfe) ===")
def B_at(arm, k):
    S = SUM[(arm, "umfe")]["S"]
    if S[-1] < k:
        return float("inf")
    i = int(np.searchsorted(S, k))
    if i == 0:
        return GRID[0]
    w = (k - S[i - 1]) / max(S[i] - S[i - 1], 1e-12)
    return float(np.exp(np.log(GRID[i - 1]) + w * (np.log(GRID[i]) - np.log(GRID[i - 1]))))
print(f"  {'targets':>8} " + " ".join(f"{a:>13}" for a in arms))
for k in (20, 40, 55, 60, 65, 70, 75):
    cells = []
    for a in arms:
        v = B_at(a, k)
        cells.append(f"{v:13.0f}" if v < float("inf") else f"{'unreachable':>13}")
    print(f"  {k:8d} " + " ".join(cells))

print("\n=== by length: where is the ceiling actually lost? ===")
n_of = {r["id"]: r["n"] for r in ok}
p = SUM[("remc_plain", "umfe")]["p"]
for lo, hi in ((0, 60), (60, 100), (100, 150), (150, 250), (250, 401)):
    idx = [i for i, t in enumerate(tids) if lo <= n_of[t] < hi]
    if not idx:
        continue
    reach = sum(1 for i in idx if p[i] > 0)
    print(f"  {lo:3d}-{hi:3d}nt  n={len(idx):3d}  reachable {reach:3d}  "
          f"mean p {np.mean([p[i] for i in idx]):.3f}  "
          f"mean cpu-s {np.mean([SUM[('remc_plain','umfe')]['c'][i] for i in idx]):7.1f}")

json.dump({f"{a}|{p_}": {"s_inf": SUM[(a, p_)]["s_inf"], "bh": SUM[(a, p_)]["bh"],
                         "U": SUM[(a, p_)]["U"], "p": SUM[(a, p_)]["p"].tolist(),
                         "c": SUM[(a, p_)]["c"].tolist(), "S": SUM[(a, p_)]["S"].tolist(),
                         "grid": GRID.tolist()}
           for a in arms for p_ in ("match", "umfe")},
          open(os.path.join(H, "eterna_sb.json"), "w"), indent=1)
print("\nwrote eterna_sb.json")
