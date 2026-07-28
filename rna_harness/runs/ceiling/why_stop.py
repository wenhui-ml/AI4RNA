#!/usr/bin/env python3
"""The highest-value free measurement left: WHAT makes the operator stop at 40-50% of its allowance?

Across 5190 design cells the panel operator used 43-50% of its pKiss allowance and only 16-33% of
attempts ever reached the cap. That is the largest, best-powered waste signal in anything measured on this
branch, and unlike the allocation policy it is an OPERATOR change, not a scheduling change.

The operator has four stopping conditions (pk_decompose_panel.py):

    best_dist <= 0                 the target is hit -- success, correctly terminal for pKiss
    n_pkiss   >= max_pkiss         the counted budget
    it        >= max_repair_iters  800 repair iterations
    n_probknot >= max_probknot     120 ProbKnot calls  <- the PANEL-CLOSING phase's own budget

Panel-unanimous success needs the second phase (close ProbKnot without breaking pKiss). If what actually
binds is max_probknot or max_repair_iters rather than the pKiss budget, then handing the operator more
pKiss calls can never help -- which is exactly what step 2 measured -- and the lever is the OTHER knob.

Reads stored stats only. No oracle calls.
"""
import json, os
from collections import Counter, defaultdict
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
CC = os.path.abspath(os.path.join(H, "..", "cc_ascent"))

rows = []
for e in range(3):
    for s in range(5):
        f = os.path.join(CC, f"gvp_e{e}_s{s}", "sweep.json")
        if not os.path.exists(f):
            continue
        for r in json.load(open(f))["rows"]:
            v = r["arms"].get("panel_auto") or {}
            st = v.get("stats") or {}
            if not st:
                continue
            rows.append({"id": r["id"], "cell": f"e{e}s{s}", "n": r["n"],
                         "pk": st.get("n_pkiss", 0), "pb": st.get("n_probknot", 0),
                         "it": st.get("n_iters", 0), "cand": st.get("n_candidates", 0),
                         "solved": bool(v.get("solved")),
                         "unan": bool((v.get("panel") or {}).get("unanimous")),
                         "rep_s": st.get("repair_wall_s", 0), "skel_s": st.get("skel_wall_s", 0)})
print(f"{len(rows)} stored panel_auto attempts (wall-clock regime, the committed sweep)")

LIM = {"pb": 120, "it": 800}
print("\n=== 1. which limit is actually reached? ===")
print(f"  {'group':>22} {'n':>5} {'pKiss':>8} {'ProbKnot':>9} {'iters':>7} {'cands':>7} "
      f"{'at PB cap':>10} {'at iter cap':>12}")
for lab, sel in (("all", lambda r: True),
                 ("in-loop solved", lambda r: r["solved"]),
                 ("in-loop failed", lambda r: not r["solved"]),
                 ("panel-unanimous", lambda r: r["unan"]),
                 ("solved but panel-NO", lambda r: r["solved"] and not r["unan"])):
    g = [r for r in rows if sel(r)]
    if not g:
        continue
    print(f"  {lab:>22} {len(g):5d} {np.mean([r['pk'] for r in g]):8.1f} "
          f"{np.mean([r['pb'] for r in g]):9.1f} {np.mean([r['it'] for r in g]):7.1f} "
          f"{np.mean([r['cand'] for r in g]):7.1f} "
          f"{np.mean([1.0 if r['pb'] >= LIM['pb'] else 0.0 for r in g]):10.2f} "
          f"{np.mean([1.0 if r['it'] >= LIM['it'] else 0.0 for r in g]):12.2f}")

print("\n=== 2. the wall-clock guards: were they the real stopping rule? ===")
print("    (in this committed sweep deterministic=False, so repair_wall_s=90 / skel_wall_s=12 apply)")
for lab, sel in (("all", lambda r: True), ("in-loop failed", lambda r: not r["solved"])):
    g = [r for r in rows if sel(r)]
    at_rep = np.mean([1.0 if r["rep_s"] >= 89.0 else 0.0 for r in g])
    print(f"  {lab:>16} n={len(g):5d}  fraction with repair_wall_s >= 89 (i.e. timed out): {at_rep:.2f}"
          f"   mean repair_wall_s {np.mean([r['rep_s'] for r in g]):6.1f}")

print("\n=== 3. how many CANDIDATES does an attempt produce, and does volume buy panel agreement? ===")
print(f"  {'candidate count':>18} {'n':>5} {'in-loop rate':>13} {'panel rate':>11} "
      f"{'panel | in-loop':>16}")
qs = [0, 20, 50, 100, 200, 400, 10 ** 9]
for a, b in zip(qs, qs[1:]):
    g = [r for r in rows if a <= r["cand"] < b]
    if not g:
        continue
    s = np.mean([1.0 if r["solved"] else 0.0 for r in g])
    u = np.mean([1.0 if r["unan"] else 0.0 for r in g])
    ns = sum(1 for r in g if r["solved"])
    print(f"  {f'{a}-{b if b < 10**9 else 'inf'}':>18} {len(g):5d} {s:13.3f} {u:11.3f} "
          f"{(sum(1 for r in g if r['unan'])/max(ns,1)):16.3f}")

print("\n=== 4. is ProbKnot use the thing that separates panel-YES from panel-NO? ===")
Y = [r for r in rows if r["unan"]]
N = [r for r in rows if r["solved"] and not r["unan"]]
for lab, g in (("panel-unanimous", Y), ("solved, panel-NO", N)):
    print(f"  {lab:>18} n={len(g):4d}  ProbKnot calls: median "
          f"{np.median([r['pb'] for r in g]) if g else float('nan'):6.1f}  "
          f"mean {np.mean([r['pb'] for r in g]) if g else float('nan'):6.1f}  "
          f"pKiss: median {np.median([r['pk'] for r in g]) if g else float('nan'):6.1f}")
if Y and N:
    from scipy.stats import mannwhitneyu
    p = mannwhitneyu([r["pb"] for r in Y], [r["pb"] for r in N]).pvalue
    print(f"  Mann-Whitney on ProbKnot calls, YES vs NO: p = {p:.2e}")
    p2 = mannwhitneyu([r["pk"] for r in Y], [r["pk"] for r in N]).pvalue
    print(f"  Mann-Whitney on pKiss    calls, YES vs NO: p = {p2:.2e}")

print("\n=== 5. the event-rate problem, stated as a power calculation ===")
u = np.mean([1.0 if r["unan"] else 0.0 for r in rows])
per_cell = u * 60
print(f"  panel-unanimous rate per attempt {u:.4f}  ->  {per_cell:.2f} of 60 targets per execution")
for ratio in (1.2, 1.36, 1.5, 2.0):
    # paired per-replicate counts ~ Poisson(m); detect a ratio with two-sided alpha=0.05, power 0.8
    m = per_cell
    need = ((1.96 + 0.84) ** 2) * (1 + ratio) / (m * (ratio - 1) ** 2)
    print(f"  to detect {ratio:.2f}x at 80% power you need about {need:.0f} replicates "
          f"({need*2101*0.22/3600:.1f} core-h per arm at the step-2 budget)")
