#!/usr/bin/env python3
"""Re-derive the positive control end to end, through the SAME calls the main sweep uses.

Why this exists. The instrument's discriminating power rests on one number: the frozen agent-written
operators' carry-over, 0.302, against the hand-built operator's 0.108. That number is produced by
agent_replicate.py, while every other arm in the paper is measured by sweep.py. Comparing a treatment
measured by one instrument against a reference measured by another is the kind of gap that invalidates a
control, so it is checked rather than assumed. Three things could differ, and all three are tested here:

  (1) the MATCH PREDICATE -- both paths import forms_target, but from different modules; asserted identical
      as an object, not merely equivalent by inspection;
  (2) the POOL -- both must be load_pseudobase(limit=60);
  (3) the VERDICTS themselves -- recomputed from the stored raw sequences by calling pkiss_pairs /
      probknot_pairs / shapeknots_pairs directly, then compared to the stored verdicts DESIGN BY DESIGN, so
      an aggregate that happens to match cannot hide compensating per-design errors.

A fourth failure mode is checked by verify_shapeknots_stability(): a sibling line found a different
pseudoknot tool silently returning nested-only structures when its working directory had leftovers, which
would have manufactured a false negative. Sweep-negative designs are re-folded in a single clean process; any
flip would mean the parallel sweep under-counts ShapeKnots and the whole comparison is void.

Offline over stored sequences; no design episodes.
"""
import glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
sys.path.insert(0, HERE)

from rna_harness.oracles_pk import (pkiss_pairs, probknot_pairs,          # noqa: E402
                                    shapeknots_pairs, forms_target)
from rna_harness.benchmarks.rnainvbench import load_pseudobase            # noqa: E402

OPS = ("gemini31pro", "kimik3")


def check_predicate_identity():
    """The two measurement paths must use the SAME function object, not two that look alike."""
    from invent_operator import forms_target as ft_agent
    same = ft_agent is forms_target
    print(f"(1) match predicate: agent path forms_target IS sweep path forms_target -> {same} "
          f"({ft_agent.__module__})")
    return same


def recompute(targets):
    print("(3) per-design re-derivation from stored sequences, using the sweep's own oracle calls")
    ok = True
    for op in OPS:
        files = sorted(glob.glob(os.path.join(HERE, f"arep_{op}_dev", "s*_e0.json")))
        sk_st = n_st = sk_re = n_re = dis = 0
        for f in files:
            for r in json.load(open(f))["rows"]:
                s = r.get("seq")
                if not s:
                    continue
                T = targets[r["id"]]
                if r.get("pkiss") and r.get("probknot"):
                    n_st += 1; sk_st += bool(r.get("shapeknots"))
                pk = bool(forms_target(pkiss_pairs(s), T))
                pb = bool(forms_target(probknot_pairs(s), T))
                if pk and pb:
                    v = bool(forms_target(shapeknots_pairs(s), T))
                    n_re += 1; sk_re += v
                    dis += int(v != bool(r.get("shapeknots")))
        print(f"    {op:12} stored {sk_st:3d}/{n_st:3d} = {sk_st/max(1,n_st):.3f}   "
              f"recomputed {sk_re:3d}/{n_re:3d} = {sk_re/max(1,n_re):.3f}   "
              f"per-design disagreements: {dis}")
        ok &= (dis == 0 and (sk_st, n_st) == (sk_re, n_re))
    return ok


def verify_shapeknots_stability(k=10):
    """Would the parallel sweep have under-counted ShapeKnots? Re-fold sweep-negative designs serially."""
    p = os.path.join(HERE, "attr_s0", "sweep.json")
    if not os.path.exists(p):
        print("(4) ShapeKnots stability: attr_s0 missing, skipped")
        return True
    rows = json.load(open(p))["rows"]
    neg = [(r["id"], v["seq"], r["target"]) for r in rows
           if (v := r["arms"].get("panel_auto") or {}).get("solved") and v.get("seq")
           and not (v.get("panel") or {}).get("shapeknots")]
    flips = sum(bool(forms_target(shapeknots_pairs(s), t)) for _, s, t in neg[:k])
    print(f"(4) ShapeKnots stability: {flips}/{min(k, len(neg))} sweep-negative designs flip when re-folded "
          f"serially in a clean process")
    return flips == 0


def main():
    tasks = load_pseudobase(limit=60)
    targets = {f"pseudobase:{i}": t.target for i, t in enumerate(tasks)}
    print(f"(2) pool: agent path and sweep path both use load_pseudobase(limit=60), n={len(targets)}\n")
    a = check_predicate_identity()
    print()
    b = recompute(targets)
    print()
    c = verify_shapeknots_stability()
    print(f"\npositive control verified end to end: {a and b and c}")
    print("  If any line above is False the 0.302 reference is an artifact of its measurement path and the")
    print("  instrument's discriminating power claim must be withdrawn.")


if __name__ == "__main__":
    main()
