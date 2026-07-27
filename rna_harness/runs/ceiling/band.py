#!/usr/bin/env python3
"""Is the right rule a two-sided BAND in Delta rather than a one-sided threshold?

The allocation experiment failed at 10% of budget (0.20x against a predicted 3.04x) and won at 20%
(1.75x). Per-funded-target arithmetic says why: at m = 6 the Delta-ranked arm hit the in-loop oracle on
0.878 of its funded targets against uniform's 0.184 -- a 4.8x gain -- while its panel pass rate CONDITIONAL
on an in-loop solve fell to 0.013 against uniform's 0.060, a 4.5x loss. The two cancel. At m = 13 they do
not cancel and the arm wins on both stages.

Mechanism, and it is the same one step 0b measured on all 60 targets: Delta_bp -> 0 means pKiss is very
easily satisfied on this target, which is exactly the condition under which pKiss can be satisfied
SPURIOUSLY. So the extreme low-Delta tail should be pKiss-solvable and panel-impossible, making Delta an
ANTI-predictor there and a good predictor in the middle -- non-monotone, not weak.

If that is right, the rule is a band [lo, hi] on the Delta rank, not a threshold. This script tests it
with no new oracle calls:

  1. Is the relation non-monotone? Panel win rate by Delta rank bin, on ids 0-29 and 30-59 separately.
  2. Fit the band edges on ids 0-29 ONLY, apply to 30-59, and compare against the one-sided rule the
     experiment actually ran.
  3. Score the resulting funded set against every arm already measured, on realised calls.

Point 2 is the honest test. Choosing the band on all 60 and then reporting 30-59 would be fitting the
answer to its own test set, which is the mistake this whole branch is trying not to make.
"""
import json, os, sys
from collections import defaultdict
import numpy as np
from scipy.stats import fisher_exact

H = os.path.dirname(os.path.abspath(__file__))
pk = {r["id"]: r for r in json.load(open(os.path.join(H, "pk_ev.json")))["rows"]}
att = json.load(open(os.path.join(H, "pk_ev.json")))["attempts"]
cure = json.load(open(os.path.join(H, "cure.json")))
plan = json.load(open(os.path.join(H, "alloc_plan.json")))
ND = cure["ndraw"]
NCELL = plan["ncells"]
idx = lambda t: int(t.split(":")[-1])
ALL = sorted(pk, key=idx)
FIT = [t for t in ALL if idx(t) < 30]
HELD = [t for t in ALL if idx(t) >= 30]
delta = {t: min(pk[t]["draws_norm"][:ND]) for t in ALL}


def rank(pool):
    """Delta rank within a pool, ties broken by length then id -- the ordering the experiment used."""
    return sorted(pool, key=lambda t: (delta[t], pk[t]["n"], t))


print("=== 1. is the relation non-monotone in Delta rank? (stored 15-cell sweep) ===")
print("    pKiss = target ever solved in-loop; panel = panel-unanimous wins out of 15 executions")
for lab, pool in (("ids 0-29 (fit)", FIT), ("ids 30-59 (held)", HELD)):
    r = rank(pool)
    print(f"  {lab}")
    print(f"    {'rank bin':>10} {'n':>3} {'median Delta':>13} {'pKiss-solvable':>15} "
          f"{'panel wins':>11} {'wins/target/cell':>17}")
    for a, b in ((0, 6), (6, 12), (12, 18), (18, 24), (24, 30)):
        g = r[a:b]
        if not g:
            continue
        ps = sum(1 for t in g if pk[t]["pk_solved"])
        pw = sum(pk[t]["panel_win"] for t in g)
        print(f"    {f'{a+1}-{b}':>10} {len(g):3d} {np.median([delta[t] for t in g]):13.3f} "
              f"{ps:>10d}/{len(g):<4d} {pw:11d} {pw/(len(g)*NCELL):17.4f}")

print("\n=== 2. the mechanism, tested directly on the fit split ===")
r = rank(FIT)
bot, mid = r[:6], r[6:18]
for lab, g in (("bottom 6 by Delta", bot), ("ranks 7-18", mid)):
    ps = sum(1 for t in g if pk[t]["pk_solved"])
    ever = sum(1 for t in g if pk[t]["panel_win"] > 0)
    pw = sum(pk[t]["panel_win"] for t in g)
    print(f"  {lab:20} n={len(g):2d}  pKiss-solvable {ps}/{len(g)}  panel ever {ever}/{len(g)}  "
          f"panel wins {pw:3d}/{len(g)*NCELL}")
tab = [[sum(1 for t in bot if pk[t]["panel_win"] > 0), sum(1 for t in bot if pk[t]["panel_win"] == 0)],
       [sum(1 for t in mid if pk[t]["panel_win"] > 0), sum(1 for t in mid if pk[t]["panel_win"] == 0)]]
try:
    print(f"  Fisher exact, ever-panel   bottom-6 vs ranks 7-18: p = {fisher_exact(tab)[1]:.4f}  {tab}")
except Exception as e:
    print(f"  Fisher failed: {e}")

print("\n=== 3. fit the band on ids 0-29, by expected panel wins per call ===")
# cost proxy: realised calls per attempt in the stored sweep, per target
cost = defaultdict(list)
for a in att:
    if a["arm"] == "panel_auto" and a["n_pkiss"] > 0:
        cost[a["id"]].append(a["n_pkiss"])
mcost = {t: float(np.mean(cost[t])) if cost[t] else float('nan') for t in ALL}
rf = rank(FIT)
best = None
print(f"    {'band':>10} {'n':>3} {'panel wins':>11} {'calls':>8} {'wins/1000 calls':>16}")
for lo in range(0, 13):
    for hi in range(lo + 4, 31):
        g = rf[lo:hi]
        w = sum(pk[t]["panel_win"] for t in g)
        c = sum(mcost[t] for t in g) * NCELL
        if c <= 0:
            continue
        score = 1000 * w / c
        if best is None or score > best[0]:
            best = (score, lo, hi, w, c)
for lo, hi in ((0, 6), (0, 13), (0, 18), (0, 25), (0, 30), (6, 18), (best[1], best[2])):
    g = rf[lo:hi]
    w = sum(pk[t]["panel_win"] for t in g)
    c = sum(mcost[t] for t in g) * NCELL
    tag = "   <== fitted optimum" if (lo, hi) == (best[1], best[2]) else ""
    print(f"    {f'{lo+1}-{hi}':>10} {len(g):3d} {w:11d} {c:8.0f} {1000*w/c:16.2f}{tag}")
LO, HI = best[1], best[2]
print(f"  fitted band on ids 0-29: ranks {LO+1}-{HI}  ({HI-LO} of 30 targets)")

print(f"\n=== 4. APPLY ranks {LO+1}-{HI} to ids 30-59 and compare with what was measured ===")
rh = rank(HELD)
band = rh[LO:HI]
print(f"  band targets: {[t.split(':')[-1] for t in band]}")
res = defaultdict(lambda: defaultdict(lambda: {"u": 0, "p": 0, "c": 0}))
for f in os.listdir(os.path.join(H, "alloc")):
    r0 = json.load(open(os.path.join(H, "alloc", f)))
    a = res[(r0["arm"], round(r0["frac"], 2))][r0["id"]]
    a["u"] += int(bool((r0.get("panel") or {}).get("unanimous")))
    a["p"] += int(bool(r0.get("solved")))
    a["c"] += r0.get("n_pkiss", 0)

print("\n  what the already-run arms did, restricted to the band's targets"
      "\n  (the delta arm at m>=HI funded all of them, so this is a re-scoring, not a prediction)")
print(f"    {'arm':8} {'B/B0':>5} {'m':>3} | {'all funded: calls':>17} {'solves':>7} | "
      f"{'band subset: calls':>18} {'solves':>7}")
for frac in (0.10, 0.20, 0.50, 1.00):
    for arm in ("uniform", "delta"):
        A = res[(arm, frac)]
        ca = sum(v["c"] for v in A.values()) / NCELL
        ua = sum(v["u"] for v in A.values()) / NCELL
        cb = sum(v["c"] for t, v in A.items() if t in band) / NCELL
        ub = sum(v["u"] for t, v in A.items() if t in band) / NCELL
        m = plan["planned"][f"0|{arm}|{frac}"]["m"]
        print(f"    {arm:8} {frac:5.2f} {m:3d} | {ca:17.0f} {ua:7.2f} | {cb:18.0f} {ub:7.2f}")

print("\n=== 5. the claim to be tested by a new run ===")
u = {frac: (sum(v["c"] for v in res[("uniform", frac)].values()) / NCELL,
            sum(v["u"] for v in res[("uniform", frac)].values()) / NCELL)
     for frac in (0.10, 0.20, 0.50, 1.00)}
d = {frac: (sum(v["c"] for v in res[("delta", frac)].values()) / NCELL,
            sum(v["u"] for v in res[("delta", frac)].values()) / NCELL)
     for frac in (0.10, 0.20, 0.50, 1.00)}
print("  measured so far, on realised calls:")
for frac in (0.10, 0.20, 0.50, 1.00):
    print(f"    B/B0={frac:4.2f}  uniform {u[frac][1]:5.2f} @ {u[frac][0]:6.0f}   "
          f"delta {d[frac][1]:5.2f} @ {d[frac][0]:6.0f}")
# the band arm's cost, extrapolated from the delta arm's per-target realised cost at each frac
for frac in (0.20, 0.50):
    A = res[("delta", frac)]
    funded = [t for t in A if A[t]["c"] > 0]
    per = np.mean([A[t]["c"] for t in funded]) / NCELL if funded else float('nan')
    print(f"    band arm at the m={HI-LO} scale, frac {frac}: "
          f"~{per*(HI-LO):.0f} calls/cell if per-target cost holds "
          f"(delta spent {d[frac][0]:.0f} on {len(funded)})")
print(f"\n  PREDICTION to test: funding ranks {LO+1}-{HI} of ids 30-59 reaches at least the delta arm's "
      f"solve count at strictly fewer realised calls, because the {LO} dropped targets are "
      f"pKiss-solvable and panel-dead and their calls are pure waste.")
