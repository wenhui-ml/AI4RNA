#!/usr/bin/env python3
"""If every available judge is wrong most of the time, what should "confirmed" mean? Calibrate on ground truth.

THE PROBLEM, measured. The shipped criterion is "all three pseudoknot predictors reproduce the target
exactly". The archived native-sequence control (runs/axis1_native_control, 250 Pseudobase targets) shows what
that criterion does to KNOWN-CORRECT answers -- fold each wild-type sequence and ask whether it recovers its
own published structure:

    exact recovery      pKiss 32.4%   ProbKnot 5.6%   ShapeKnots 7.6%
    crossing recall     pKiss 0.85    ProbKnot 0.62   ShapeKnots 0.63

So unanimous-exact rejects most of nature. A design that fails it has not been shown to be wrong; it has been
shown to be unrecognisable to instruments that mostly cannot recognise biology either. Calling that
population "false" was an overclaim of mine, and tightening or loosening the criterion by taste would just be
a different overclaim.

THE FIX IS NOT A NEW ORACLE, IT IS A CALIBRATION SET. Natural structures are ground truth: the wild-type
sequence really does form the published structure. So a confirmation rule can be CALIBRATED -- choose it so
that it recognises a declared fraction of known-correct structures, then report that fraction alongside every
certificate. "This design passes a criterion that recognises 80% of known natural structures" is a statement
a non-expert can act on. "Three heuristics agreed" is not.

CRITERION FAMILY, two interpretable knobs:
    accept_o(seq, T; d)  :=  bp_dist(fold_o(seq), T) <= d          per oracle, tolerance d
    CONFIRMED(k, d)      :=  at least k of the 3 oracles accept
(k=3, d=0) is exactly what ships today, so the current rule is a point in the family rather than a baseline
being argued against.

SENSITIVITY WITHOUT SPECIFICITY IS MEANINGLESS, and loosening d until natives pass is exactly how one would
manufacture a flattering number. So the same grid is evaluated on a NULL: a random ACGU sequence of the same
length as each native target. A tolerance at which random sequences also pass is not a confirmation rule, it
is a rubber stamp. The reported quantity is therefore the pair (sensitivity on natives, false-accept rate on
random), and a rule is only usable where the two separate.

THIS SCRIPT DOES THE CALIBRATION HALF ONLY -- it is free, it runs on the archive, and it decides whether the
approach is viable before any compute is spent re-scanning designs. If no (k, d) reaches useful sensitivity,
the idea dies here at zero cost.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
NATIVE = os.path.join(HERE, "agent", "native_control.json")
ORACLES = ("pkiss", "probknot", "shapeknots")
TARGET_SENSITIVITY = 0.80          # declared before looking: the rule must recognise 80% of natives


def _null_rows(rows, seed=0):
    """Fold ONE random sequence per target under all three oracles -> the false-accept null for each (k, d).
    Cached, because it costs real oracle time and nothing about it changes between runs."""
    cache = os.path.join(HERE, "calibrate_null.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    import random
    from concurrent.futures import ProcessPoolExecutor
    from rna_harness.benchmarks.rnainvbench import load_pseudobase
    tgt = {t.task_id: t.target for t in load_pseudobase()}
    rng = random.Random(seed)
    items = [(r["id"], tgt.get(r["id"]), "".join(rng.choice("ACGU") for _ in range(r["n"])))
             for r in rows]
    items = [it for it in items if it[1] and len(it[1]) == len(it[2])]
    with ProcessPoolExecutor(max_workers=16) as ex:
        out = list(ex.map(_null_one, items))
    out = [o for o in out if o]
    json.dump(out, open(cache, "w"))
    return out


def _null_one(item):
    tid, target, seq = item
    from rna_harness.benchmarks.rnainvbench import pk_bp_distance
    from rna_harness.oracles_pk import pkiss_pairs, probknot_pairs, shapeknots_pairs
    import rna_harness.benchmarks.rnainvbench as rib
    fns = {"pkiss": pkiss_pairs, "probknot": probknot_pairs, "shapeknots": shapeknots_pairs}
    rec = {"id": tid, "n": len(target)}
    for o, fn in fns.items():
        ps = fn(seq)
        if ps is None:
            rec[o] = {"ok": False, "bp_dist": None}
            continue
        tp = rib.pk_pairs(target)
        rec[o] = {"ok": True, "bp_dist": len(ps ^ tp)}
    return rec


def main():
    rows = json.load(open(NATIVE))["rows"]
    rows = [r for r in rows if all(r.get(o, {}).get("ok") for o in ORACLES)]
    print(f"calibration set: {len(rows)} natural structures with all three oracles run")
    print(f"  ground truth = the wild-type sequence really does form the published structure\n")

    print(f"  per-oracle native recovery at tolerance d (bp_dist <= d):")
    print(f"  {'d':>4} " + "".join(f"{o:>12}" for o in ORACLES))
    DS = [0, 1, 2, 3, 5, 8, 12, 20]
    for d in DS:
        cells = []
        for o in ORACLES:
            acc = sum(1 for r in rows if r[o].get("bp_dist") is not None and r[o]["bp_dist"] <= d)
            cells.append(f"{acc/len(rows):>11.1%}")
        print(f"  {d:>4} " + "".join(cells))

    print(f"\n=== sensitivity of CONFIRMED(k, d) on ground truth ===")
    print(f"  {'d':>4} " + "".join(f"{'k=' + str(k):>10}" for k in (1, 2, 3)))
    grid = {}
    for d in DS:
        cells = []
        for k in (1, 2, 3):
            n = 0
            for r in rows:
                acc = sum(1 for o in ORACLES
                          if r[o].get("bp_dist") is not None and r[o]["bp_dist"] <= d)
                n += (acc >= k)
            sens = n / len(rows)
            grid[(k, d)] = sens
            cells.append(f"{sens:>9.1%}")
        print(f"  {d:>4} " + "".join(cells))

    print(f"\n=== the rule, chosen by a declared criterion rather than by taste ===")
    print(f"  requirement: sensitivity >= {TARGET_SENSITIVITY:.0%} on known natural structures")
    print(f"  among those, take the STRICTEST (largest k, then smallest d)")
    ok = [(k, d, s) for (k, d), s in grid.items() if s >= TARGET_SENSITIVITY]
    shipped = grid[(3, 0)]
    print(f"  currently shipped rule CONFIRMED(k=3, d=0): sensitivity {shipped:.1%}")
    if not ok:
        best = max(grid.items(), key=lambda kv: kv[1])
        print(f"  NO (k, d) reaches {TARGET_SENSITIVITY:.0%}. Best available is "
              f"k={best[0][0]}, d={best[0][1]} at {best[1]:.1%}.")
        print(f"  -> Confirmation by agreement among THESE THREE predictors cannot be calibrated to a")
        print(f"     useful sensitivity at any threshold. That is a statement about the instruments, and it")
        print(f"     means the honest product move is to report the disagreement profile, not a verdict.")
    else:
        ok.sort(key=lambda t: (-t[0], t[1]))
        k, d, s = ok[0]
        print(f"  -> CONFIRMED(k={k}, d={d}) at sensitivity {s:.1%}")
        print(f"     a certificate can then say: 'passes a criterion that recognises {s:.0%} of known"
              f" natural structures'")

    print(f"\n=== the missing half: false-accept rate on RANDOM sequences (same lengths) ===")
    null = _null_rows(rows)
    print(f"  null set: {len(null)} random ACGU sequences")
    print(f"  {'d':>4} " + "".join(f"{'k=' + str(k):>22}" for k in (1, 2, 3)))
    print(f"  {'':>4} " + "".join(f"{'sens':>10}{'  random':>12}" for _ in (1, 2, 3)))
    usable = []
    for d in DS:
        cells = []
        for k in (1, 2, 3):
            fa = 0
            for r in null:
                acc = sum(1 for o in ORACLES if r[o].get("bp_dist") is not None and r[o]["bp_dist"] <= d)
                fa += (acc >= k)
            fa /= max(len(null), 1)
            cells.append(f"{grid[(k, d)]:>9.1%}{fa:>12.1%}")
            if grid[(k, d)] >= TARGET_SENSITIVITY and fa <= 0.05:
                usable.append((k, d, grid[(k, d)], fa))
        print(f"  {d:>4} " + "".join(cells))
    print(f"\n  usable rules (sensitivity >= {TARGET_SENSITIVITY:.0%} AND random false-accept <= 5%): "
          f"{usable if usable else 'NONE'}")
    if not usable:
        print(f"  -> No tolerance separates known-correct structures from RANDOM SEQUENCES while keeping")
        print(f"     useful sensitivity. Confirmation by agreement among these three predictors cannot be")
        print(f"     calibrated at all. The honest product move is to report the disagreement profile and")
        print(f"     the instruments' measured recovery, not to issue a verdict dressed as ground truth.")

    print(f"\n=== what the shipped rule costs in sensitivity ===")
    for k in (3, 2, 1):
        print(f"  k={k}: " + "  ".join(f"d={d} {grid[(k, d)]:.0%}" for d in DS))

    json.dump({"n_calibration": len(rows), "target_sensitivity": TARGET_SENSITIVITY,
               "grid": {f"k{k}_d{d}": s for (k, d), s in grid.items()},
               "shipped_k3_d0": shipped,
               "viable_by_sensitivity_only": bool(ok),
               "usable_with_specificity": usable},
              open(os.path.join(HERE, "calibrate.json"), "w"), indent=1)
    print(f"\n  wrote rna_harness/runs/ceiling/calibrate.json")


if __name__ == "__main__":
    main()
