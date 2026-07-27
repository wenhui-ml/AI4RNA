#!/usr/bin/env python3
"""The anomaly the aggregates cannot resolve: m = 6 undershot its OWN expectation eightfold.

band.py refuted the two-sided-band hypothesis. What survives both splits is a cliff after Delta rank ~12
(ranks 1-12 win at 0.11-0.12 per target per cell in the stored sweep, ranks 13-30 at 0.00-0.056). Taking
that at face value, funding 6 targets should yield about 0.6 panel-unanimous solves per cell and funding
13 about 1.2. Measured: m = 13 gave 0.93, close to expectation; m = 6 gave 0.07, an eightfold undershoot.
Target selection cannot explain it -- the two arms ran at nearly identical realised depth (26.2 vs 28.2
calls per funded target).

Something about the m = 6 configuration destroys wins that the same targets produce elsewhere. Three
candidates, separable per target with data already on disk:

  (a) DEPTH TRUNCATION. The panel operator is lexicographic: drive pKiss to 0 first, then close ProbKnot
      without breaking pKiss. `max_pkiss` gates the whole loop, so capping it truncates the SECOND phase --
      exactly the one that produces panel agreement. Low-Delta targets reach pKiss = 0 fastest and so
      spend the largest share of their cap in the phase being cut.
  (b) WRONG TARGETS. The 6 are simply panel-dead in deterministic mode, whatever the stored wall-clock
      sweep says.
  (c) DETERMINISM. The stored sweep's wins came from wall-clock runs; with a fixed seed and a counted
      budget the operator lands in one basin per seed and the 15 executions are less independent than the
      stored sweep's were.

(a) predicts wins appear once the cap rises on the SAME targets. (b) predicts they never appear at any
cap. (c) predicts wins concentrate in few seeds and the per-seed variance collapses. All three are
distinguishable here. No new oracle calls.
"""
import json, os
from collections import defaultdict
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(H, "pk_ev.json")))
pk = {r["id"]: r for r in D["rows"]}
cure = json.load(open(os.path.join(H, "cure.json")))
plan = json.load(open(os.path.join(H, "alloc_plan.json")))
ND, NCELL = cure["ndraw"], plan["ncells"]
idx = lambda t: int(t.split(":")[-1])
HELD = sorted([t for t in pk if idx(t) >= 30], key=idx)
delta = {t: min(pk[t]["draws_norm"][:ND]) for t in pk}
rank = {t: i for i, t in enumerate(sorted(HELD, key=lambda x: (delta[x], pk[x]["n"], x)))}

cells = defaultdict(dict)          # (arm,frac) -> tid -> list of per-cell records
for f in os.listdir(os.path.join(H, "alloc")):
    r = json.load(open(os.path.join(H, "alloc", f)))
    cells[(r["arm"], round(r["frac"], 2))].setdefault(r["id"], []).append(r)

print("=== 1. who actually wins in the NEW deterministic run? (panel-unanimous, /15 cells) ===")
print(f"  {'rank':>4} {'id':>4} {'Delta':>7} {'stored':>7} | " +
      " | ".join(f"{a[:4]}{int(f*100):03d}" for a in ("uniform", "delta") for f in (0.10, 0.20, 0.50, 1.00)))
for t in sorted(HELD, key=lambda x: rank[x]):
    row = []
    for arm in ("uniform", "delta"):
        for frac in (0.10, 0.20, 0.50, 1.00):
            R = cells[(arm, frac)].get(t)
            if not R:
                row.append("   -")
            else:
                w = sum(int(bool((x.get("panel") or {}).get("unanimous"))) for x in R)
                row.append(f"{w:4d}")
    print(f"  {rank[t]+1:4d} {t.split(':')[-1]:>4} {delta[t]:7.3f} {pk[t]['panel_win']:7d} | " +
          " | ".join(row))

print("\n=== 2. (a) depth: does the SAME target win once its cap rises? (delta arm) ===")
print(f"  {'rank':>4} {'id':>4} | " + " | ".join(f"B/B0={f:<4.2f} cap/used/win" for f in (0.10, 0.20, 0.50, 1.00)))
for t in sorted(HELD, key=lambda x: rank[x])[:14]:
    row = []
    for frac in (0.10, 0.20, 0.50, 1.00):
        R = cells[("delta", frac)].get(t)
        if not R:
            row.append("      -       ")
            continue
        cap = R[0]["cap"]
        used = np.mean([x.get("n_pkiss", 0) for x in R])
        w = sum(int(bool((x.get("panel") or {}).get("unanimous"))) for x in R)
        row.append(f"{cap:4d}/{used:5.1f}/{w:2d}  ")
    print(f"  {rank[t]+1:4d} {t.split(':')[-1]:>4} | " + " | ".join(row))

print("\n=== 3. is the cap binding, or does the operator stop on its own? ===")
print(f"  {'arm':8} {'B/B0':>5} {'cap':>5} {'used (mean)':>12} {'used/cap':>9} "
      f"{'frac hitting cap':>17} {'in-loop pKiss':>14}")
for arm in ("uniform", "delta"):
    for frac in (0.10, 0.20, 0.50, 1.00):
        R = [x for v in cells[(arm, frac)].values() for x in v]
        cap = R[0]["cap"]
        used = np.mean([x.get("n_pkiss", 0) for x in R])
        athit = np.mean([1.0 if x.get("n_pkiss", 0) >= cap else 0.0 for x in R])
        ps = np.mean([1.0 if x.get("solved") else 0.0 for x in R])
        print(f"  {arm:8} {frac:5.2f} {cap:5d} {used:12.1f} {used/cap:9.2f} {athit:17.2f} {ps:14.2f}")

print("\n=== 4. (c) determinism: how concentrated are wins across the 15 seeds? ===")
for arm in ("uniform", "delta"):
    for frac in (0.20, 1.00):
        per = defaultdict(int)
        for t, R in cells[(arm, frac)].items():
            for x in R:
                if (x.get("panel") or {}).get("unanimous"):
                    per[x["cell"]] += 1
        v = [per.get(c, 0) for c in range(NCELL)]
        print(f"  {arm:8} B/B0={frac:4.2f}  wins per seed {v}  "
              f"mean {np.mean(v):.2f} var {np.var(v):.2f} (Poisson would give var=mean)")

print("\n=== 5. stored sweep vs new run on the SAME targets, ranks 1-6 ===")
top6 = sorted(HELD, key=lambda x: rank[x])[:6]
sw = json.load(open(os.path.join(H, "pk_ev.json")))["attempts"]
for t in top6:
    A = [a for a in sw if a["arm"] == "panel_auto" and a["id"] == t]
    used = np.mean([a["n_pkiss"] for a in A]) if A else float('nan')
    w = sum(a["unanimous"] for a in A)
    new = cells[("delta", 1.00)].get(t) or cells[("delta", 0.50)].get(t) or []
    nu = np.mean([x.get("n_pkiss", 0) for x in new]) if new else float('nan')
    nw = sum(int(bool((x.get("panel") or {}).get("unanimous"))) for x in new)
    print(f"  {t.split(':')[-1]:>4} rank {rank[t]+1:2d}  stored: {used:6.1f} calls, {w:2d}/15 wins   "
          f"new(best cap): {nu:6.1f} calls, {nw:2d}/15 wins")
