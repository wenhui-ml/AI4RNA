#!/usr/bin/env python3
"""The one unknown that decides whether a ShapeKnots gate is affordable.

bottleneck.py: of 530 in-loop solves, pKiss and ProbKnot each agree 530/530 (both are in-loop) and
ShapeKnots agrees 48/530. Every one of the 482 failures is ShapeKnots alone -- pKiss and ProbKnot kill
zero otherwise-passing solutions. So the fix is a late-stage ShapeKnots gate: run the existing loop to
pKiss+ProbKnot agreement (~37 pKiss calls), screen with ShapeKnots, and on rejection keep searching for a
DIFFERENT pKiss+ProbKnot solution.

At the measured 9.1% pass rate that needs ~11 screened candidates per success. Whether that is a 3x or a
30x cost increase depends entirely on what one ShapeKnots call costs relative to one pKiss call, and the
stored stats never recorded it (only n_pkiss and n_probknot).

Measures all three oracles on the same sequences, at the same lengths, so the ratio is apples to apples.
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
from rna_harness.benchmarks.rnainvbench import load_pseudobase                   # noqa: E402
from rna_harness.oracles_pk import (pkiss_pairs, probknot_pairs, shapeknots_pairs,  # noqa: E402
                                    rnastructure_available)

REPS = int(os.environ.get("REPS", "3"))


def main():
    print(f"RNAstructure available: {rnastructure_available()}")
    tasks = load_pseudobase(limit=60)
    # real sequences the operator actually produced, so lengths and composition are representative
    seqs = []
    CC = os.path.abspath(os.path.join(HERE, "..", "cc_ascent"))
    for e in range(3):
        for s in range(5):
            f = os.path.join(CC, f"gvp_e{e}_s{s}", "sweep.json")
            if not os.path.exists(f):
                continue
            for r in json.load(open(f))["rows"]:
                v = r["arms"].get("panel_auto") or {}
                if v.get("seq"):
                    seqs.append((r["id"], len(v["seq"]), v["seq"]))
    # de-duplicate and stratify by length
    seen, pool = set(), []
    for tid, n, s in sorted(seqs, key=lambda x: x[1]):
        if s in seen:
            continue
        seen.add(s); pool.append((tid, n, s))
    step = max(1, len(pool) // 12)
    sample = pool[::step][:12]
    print(f"{len(pool)} distinct operator-produced sequences; timing {len(sample)} of them "
          f"(lengths {sample[0][1]}-{sample[-1][1]}), {REPS} reps each\n")

    print(f"  {'id':>14} {'n':>4} {'pKiss ms':>10} {'ProbKnot ms':>12} {'ShapeKnots ms':>14} "
          f"{'SK/pKiss':>9}")
    acc = {"pk": [], "pb": [], "sk": [], "n": []}
    for tid, n, s in sample:
        row = {}
        for lab, fn in (("pk", pkiss_pairs), ("pb", probknot_pairs), ("sk", shapeknots_pairs)):
            ts = []
            for _ in range(REPS):
                t0 = time.time()
                try:
                    fn(s)
                except Exception:
                    pass
                ts.append(time.time() - t0)
            row[lab] = float(np.median(ts)) * 1000
        acc["n"].append(n)
        for k in ("pk", "pb", "sk"):
            acc[k].append(row[k])
        print(f"  {tid:>14} {n:4d} {row['pk']:10.1f} {row['pb']:12.1f} {row['sk']:14.1f} "
              f"{row['sk']/max(row['pk'],1e-9):9.2f}", flush=True)

    pk, pb, sk = (np.array(acc[k]) for k in ("pk", "pb", "sk"))
    print(f"\n  median   pKiss {np.median(pk):8.1f} ms   ProbKnot {np.median(pb):8.1f} ms   "
          f"ShapeKnots {np.median(sk):8.1f} ms")
    print(f"  ShapeKnots / pKiss cost ratio: median {np.median(sk/pk):.2f}, "
          f"range {np.min(sk/pk):.2f}-{np.max(sk/pk):.2f}")

    print("\n=== is a late-stage ShapeKnots gate affordable? ===")
    # current: ~37 pKiss + ~33 ProbKnot per in-loop solve, 9.1% of which pass ShapeKnots
    PK_PER_SOLVE, PB_PER_SOLVE, PASS = 37.3, 33.3, 0.091
    cand = 1.0 / PASS
    cur_ms = PK_PER_SOLVE * np.median(pk) + PB_PER_SOLVE * np.median(pb)
    gate_ms = cand * (PK_PER_SOLVE * np.median(pk) + PB_PER_SOLVE * np.median(pb) + np.median(sk))
    print(f"  today, per in-loop solve:      {PK_PER_SOLVE:.0f} pKiss + {PB_PER_SOLVE:.0f} ProbKnot "
          f"= {cur_ms/1000:8.2f} s   (9.1% of these pass the panel)")
    print(f"  gate, per PANEL-PASSING solve: {cand:.1f} candidates x (same + 1 ShapeKnots) "
          f"= {gate_ms/1000:8.2f} s")
    print(f"  cost per panel-passing solve today: {cur_ms/PASS/1000:8.2f} s "
          f"(you must already pay for {cand:.1f} attempts, you just cannot SEE which one worked)")
    print(f"  -> the gate's overhead is only the ShapeKnots calls themselves: "
          f"{cand*np.median(sk)/1000:.2f} s per success = "
          f"{100*cand*np.median(sk)/(cur_ms/PASS):.1f}% on top")
    print("\n  NOTE the asymmetry: today the 11 candidates are spread over ~11 separate EXECUTIONS and")
    print("  only 21/60 targets ever land one. A gate spends them inside ONE execution, which is why the")
    print("  per-execution rate (3.2/60) can move toward the ever-solved rate (21/60) at nearly no extra")
    print("  oracle cost -- the search work is already being done and thrown away.")


if __name__ == "__main__":
    main()
