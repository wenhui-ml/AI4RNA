#!/usr/bin/env python3
"""Oracle cost against LENGTH. sk_cost.py measured one length band and I then used it as universal.

sk_cost.py timed 12 operator-produced sequences of 21-42 nt and reported medians pKiss 210.1 ms,
ProbKnot 120.1, ShapeKnots 64.8 -- ratio 0.31 -- and I wrote "ShapeKnots is the cheapest of the three"
without a length qualifier, then used those constants for every dollar figure on this branch and to order
a short-circuit conjunction cheapest-first.

A DesiRNA smoke test broke that: on pseudobase:24 the ShapeKnots-first screen spent 92.6 s over 145 calls,
which is 0.64 s per call -- ten times the figure I was quoting -- and made the "optimised" ordering 3.3x
WORSE than the one it replaced on that target, while being 2.7-4.1x better on two others.

So the ordering is length-dependent and a fixed order is wrong. This measures all three across the actual
Pseudobase range so the cost model, the dollar axis, and the short-circuit order rest on a curve rather
than on one band.
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
from rna_harness.benchmarks.rnainvbench import load_pseudobase                      # noqa: E402
from rna_harness.oracles_pk import (pkiss_pairs, probknot_pairs, shapeknots_pairs,   # noqa: E402
                                    rnastructure_available)

REPS = int(os.environ.get("REPS", "3"))
WC = [('A', 'U'), ('U', 'A'), ('G', 'C'), ('C', 'G'), ('G', 'U'), ('U', 'G')]


def compatible(db, rng):
    import random
    OP = {'(': ')', '[': ']', '{': '}', '<': '>'}
    CL = {v: k for k, v in OP.items()}
    st = {k: [] for k in OP}; p = {}
    for i, c in enumerate(db):
        if c in OP:
            st[c].append(i)
        elif c in CL:
            j = st[CL[c]].pop(); p[j] = i; p[i] = j
    s = ['A'] * len(db)
    for i in range(len(db)):
        if i not in p:
            s[i] = rng.choice('ACGU')
    for i, j in p.items():
        if i < j:
            a, b = rng.choice(WC); s[i], s[j] = a, b
    return "".join(s)


def main():
    import random
    print(f"RNAstructure available: {rnastructure_available()}")
    tasks = load_pseudobase(limit=60)
    tasks = sorted(tasks, key=lambda t: len(t.target))
    step = max(1, len(tasks) // 14)
    sample = tasks[::step][:14]
    rng = random.Random(0)
    print(f"timing {len(sample)} targets, lengths {len(sample[0].target)}-{len(sample[-1].target)} nt, "
          f"{REPS} reps each, on random COMPATIBLE sequences\n")
    print(f"  {'n':>4} {'pKiss ms':>10} {'ProbKnot ms':>12} {'ShapeKnots ms':>14} "
          f"{'cheapest':>11} {'sk/pk':>7}")
    rows = []
    for t in sample:
        seq = compatible(t.target, rng)
        r = {}
        for lab, fn in (("pk", pkiss_pairs), ("pb", probknot_pairs), ("sk", shapeknots_pairs)):
            ts = []
            for _ in range(REPS):
                t0 = time.time()
                try:
                    fn(seq)
                except Exception:
                    pass
                ts.append(time.time() - t0)
            r[lab] = float(np.median(ts)) * 1000
        r["n"] = len(t.target); r["id"] = t.task_id
        rows.append(r)
        cheap = min(("pKiss", r["pk"]), ("ProbKnot", r["pb"]), ("ShapeKnots", r["sk"]),
                    key=lambda x: x[1])[0]
        print(f"  {r['n']:4d} {r['pk']:10.1f} {r['pb']:12.1f} {r['sk']:14.1f} {cheap:>11} "
              f"{r['sk']/max(r['pk'],1e-9):7.2f}", flush=True)
    json.dump(rows, open(os.path.join(HERE, "oracle_cost.json"), "w"), indent=1)

    n = np.array([r["n"] for r in rows], float)
    print("\n=== scaling: fit ms = a * n^b ===")
    for lab, k in (("pKiss", "pk"), ("ProbKnot", "pb"), ("ShapeKnots", "sk")):
        v = np.array([r[k] for r in rows], float)
        m = v > 0
        b, a = np.polyfit(np.log(n[m]), np.log(v[m]), 1)
        print(f"  {lab:>11}  exponent {b:5.2f}   at n=30 {np.exp(a)*30**b:8.1f} ms   "
              f"at n=60 {np.exp(a)*60**b:8.1f} ms   at n=100 {np.exp(a)*100**b:9.1f} ms")

    print("\n=== which oracle is cheapest, by length band ===")
    for lo, hi in ((0, 35), (35, 50), (50, 80), (80, 1000)):
        g = [r for r in rows if lo <= r["n"] < hi]
        if not g:
            continue
        med = {k: float(np.median([r[k] for r in g])) for k in ("pk", "pb", "sk")}
        order = sorted(med, key=med.get)
        nm = {"pk": "pKiss", "pb": "ProbKnot", "sk": "ShapeKnots"}
        print(f"  {lo:3d}-{hi if hi < 1000 else '+':>4}nt  n={len(g):2d}  "
              + "  <  ".join(f"{nm[k]} {med[k]:.0f}ms" for k in order))

    print("\n=== consequence for the short-circuit order ===")
    print("  A conjunction should test cheapest-first, but 'cheapest' flips with length, so the order")
    print("  must be chosen per target from the measured curve rather than fixed. And the ratio that")
    print("  justified adding ShapeKnots at all -- cheapest AND strictest -- holds only where it is")
    print("  actually cheapest; above that band it buys the same strictness at a higher price.")
    print("\n=== consequence for every dollar figure on this branch ===")
    sk30 = float(np.median([r["sk"] for r in rows if r["n"] < 35] or [np.nan]))
    skhi = float(np.median([r["sk"] for r in rows if r["n"] >= 50] or [np.nan]))
    if sk30 == sk30 and skhi == skhi:
        print(f"  ShapeKnots median {sk30:.0f} ms below 35 nt against {skhi:.0f} ms above 50 nt "
              f"({skhi/max(sk30,1e-9):.1f}x). The USD numbers in metric.py used oracle-SECONDS measured")
        print("  per attempt, so they are unaffected -- those were realised times, not modelled from a")
        print("  constant. What WAS wrong is every statement of the form 'ShapeKnots costs 0.31x pKiss'.")


if __name__ == "__main__":
    main()
