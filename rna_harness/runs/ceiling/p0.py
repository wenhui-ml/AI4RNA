#!/usr/bin/env python3
"""The decisive free test: does p0 discriminate INSIDE the Delta = 0 group, where all the winners are?

per_target.py established three things about the deterministic run:

  * The cap is not binding (used/cap 0.43-0.50, only 16-33% of attempts reach it). The operator
    self-terminates, so depth is not purchasable.
  * Wins per seed are UNDER-dispersed relative to Poisson (var 0.38 vs mean 0.53, and so on), and a
    target's win count is nearly constant across a 12x budget range. Outcome is a function of
    (target, seed), not of budget.
  * On ranks 1-6 the stored non-deterministic sweep got 5 wins at comparable call counts where the
    deterministic run gets 1.

So the resource that converts into panel-unanimous solves is INDEPENDENT TRAJECTORIES, not oracle calls,
and the allocation experiment held trajectories fixed (the same 15 seeds for every arm) while varying
calls. That is the design error.

The blocking problem for a trajectory-allocation policy: 13 of the 30 held-out targets have Delta_bp = 0
at N = 8, and that group contains every winner (54, 58, 35, 44, 36) AND most of the zeros (56, 41, 31,
32, 40, 46, 42, 59). Delta cannot rank inside it, because a minimum saturates at its bound.

Candidate coordinate: p0(T) = fraction of the 200 stored draws reaching Delta_bp = 0. It estimates the
MEASURE of the satisfying set rather than whether the set is non-empty, and under a restart-and-climb
search the per-trajectory success probability scales with that measure -- which also gives the cost model
the Weibull-in-calls fit never had (expected trajectories to succeed ~ 1/p0).

Falsifiable, and this script can kill it: if p0's AUC against per-target win rate is no better than the
minimum's, or if it fails to separate winners from zeros inside the Delta = 0 group, the idea is dead and
the trajectory reframe needs a different coordinate.
"""
import json, os
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu

H = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(H, "pk_ev.json")))
pk = {r["id"]: r for r in D["rows"]}
plan = json.load(open(os.path.join(H, "alloc_plan.json")))
NCELL = plan["ncells"]
idx = lambda t: int(t.split(":")[-1])
ALL = sorted(pk, key=idx)
HELD = [t for t in ALL if idx(t) >= 30]
FIT = [t for t in ALL if idx(t) < 30]

dmin = {t: min(pk[t]["draws_norm"][:8]) for t in ALL}
p0 = {t: float(np.mean([1.0 if d == 0 else 0.0 for d in pk[t]["draws"]])) for t in ALL}
p0_8 = {t: float(np.mean([1.0 if d == 0 else 0.0 for d in pk[t]["draws"][:8]])) for t in ALL}
nd = {t: len(pk[t]["draws"]) for t in ALL}

# outcome in the NEW deterministic run: wins/15 under the uniform arm at full budget (all 30 funded)
wins = defaultdict(int)
for f in os.listdir(os.path.join(H, "alloc")):
    r = json.load(open(os.path.join(H, "alloc", f)))
    if r["arm"] == "uniform" and round(r["frac"], 2) == 1.00:
        wins[r["id"]] += int(bool((r.get("panel") or {}).get("unanimous")))
# and in the stored wall-clock sweep, which had genuine trajectory diversity
sw = {t: pk[t]["panel_win"] for t in ALL}


def auc(pos, neg):
    if not pos or not neg:
        return float('nan')
    return sum(1 if a > b else .5 if a == b else 0 for a in pos for b in neg) / (len(pos) * len(neg))


print("=== 0. how much of the population does the minimum fail to rank? ===")
for lab, pool in (("ids 0-29", FIT), ("ids 30-59", HELD)):
    z = [t for t in pool if dmin[t] == 0]
    w = [t for t in z if sw[t] > 0]
    print(f"  {lab}: {len(z)}/{len(pool)} targets have min-Delta = 0 at N = 8; "
          f"{len(w)} of them ever win in the stored sweep "
          f"({sum(sw[t] for t in z)} of {sum(sw[t] for t in pool)} total wins)")

print("\n=== 1. p0 vs the minimum, whole held-out set, both outcomes ===")
print(f"  {'coordinate':>12} {'AUC vs new-run win':>19} {'AUC vs stored win':>18} "
      f"{'Spearman(new)':>14} {'Spearman(stored)':>17}")
for lab, v in (("min-Delta(8)", {t: -dmin[t] for t in HELD}),      # sign so larger = easier
               ("p0 (N=8)", p0_8), ("p0 (N=200)", p0)):
    pos = [v[t] for t in HELD if wins[t] > 0]
    neg = [v[t] for t in HELD if wins[t] == 0]
    pos2 = [v[t] for t in HELD if sw[t] > 0]
    neg2 = [v[t] for t in HELD if sw[t] == 0]
    s1 = spearmanr([v[t] for t in HELD], [wins[t] for t in HELD]).statistic
    s2 = spearmanr([v[t] for t in HELD], [sw[t] for t in HELD]).statistic
    print(f"  {lab:>12} {auc(pos, neg):19.3f} {auc(pos2, neg2):18.3f} {s1:14.3f} {s2:17.3f}")

print("\n=== 2. THE TEST: inside the min-Delta = 0 group, does p0 separate winners from zeros? ===")
for lab, pool, out in (("ids 30-59, new run", HELD, wins), ("ids 30-59, stored", HELD, sw),
                       ("ids 0-29, stored", FIT, sw)):
    z = [t for t in pool if dmin[t] == 0]
    W = [t for t in z if out[t] > 0]
    Z = [t for t in z if out[t] == 0]
    if not W or not Z:
        print(f"  {lab}: degenerate ({len(W)} winners, {len(Z)} zeros)")
        continue
    a = auc([p0[t] for t in W], [p0[t] for t in Z])
    try:
        u = mannwhitneyu([p0[t] for t in W], [p0[t] for t in Z], alternative="greater").pvalue
    except Exception:
        u = float('nan')
    print(f"  {lab}: n={len(z)} at Delta=0, {len(W)} winners / {len(Z)} zeros   "
          f"AUC(p0) = {a:.3f}   one-sided MWU p = {u:.4f}")
    print(f"    winners p0: {sorted(round(p0[t],3) for t in W)}")
    print(f"    zeros   p0: {sorted(round(p0[t],3) for t in Z)}")

print("\n=== 3. the cost model p0 implies: expected trajectories to a win ~ 1/p0? ===")
print(f"  {'id':>4} {'rank':>4} {'p0(200)':>8} {'1/p0':>7} {'stored wins/15':>15} "
      f"{'new wins/15':>12} {'per-traj rate':>14}")
for t in sorted(HELD, key=lambda x: -p0[x])[:16]:
    r = 1 + sorted(HELD, key=lambda x: (dmin[x], pk[x]["n"], x)).index(t)
    print(f"  {t.split(':')[-1]:>4} {r:4d} {p0[t]:8.3f} "
          f"{(1/p0[t] if p0[t] > 0 else float('inf')):7.1f} {sw[t]:15d} {wins[t]:12d} "
          f"{wins[t]/NCELL:14.3f}")
w = [t for t in HELD if p0[t] > 0]
if len(w) > 3:
    s = spearmanr([p0[t] for t in w], [wins[t] for t in w])
    print(f"  Spearman(p0, new wins) over the {len(w)} targets with p0 > 0: "
          f"rho = {s.statistic:+.3f}, p = {s.pvalue:.4f}")

print("\n=== 4. what a trajectory budget buys, from the measured per-trajectory costs ===")
cost = defaultdict(list)
for f in os.listdir(os.path.join(H, "alloc")):
    r = json.load(open(os.path.join(H, "alloc", f)))
    if r["arm"] == "uniform" and round(r["frac"], 2) == 1.00:
        cost[r["id"]].append(r.get("n_pkiss", 0))
mc = {t: float(np.mean(cost[t])) for t in cost}
tot = sum(mc.values())
print(f"  uniform at full budget: {tot:.0f} calls/cell for 30 trajectories "
      f"= {tot/30:.1f} calls per trajectory")
print(f"  the same {tot:.0f} calls spent as trajectories on the best target by p0:")
for t in sorted(HELD, key=lambda x: -p0[x])[:5]:
    k = tot / max(mc.get(t, 1), 1)
    pr = wins[t] / NCELL
    print(f"    {t.split(':')[-1]:>4} p0={p0[t]:.3f} cost/traj {mc.get(t,0):5.1f} -> {k:5.1f} trajectories, "
          f"per-traj win rate {pr:.3f} -> E[wins] {k*pr:5.2f}   (uniform gets "
          f"{sum(wins.values())/NCELL:.2f})")
