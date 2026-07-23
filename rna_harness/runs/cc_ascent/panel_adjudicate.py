#!/usr/bin/env python3
"""Cross-oracle PANEL adjudication of the AUTONOMOUS operator's solves (chomsky_climb positive side).

Reads a track_b_pk run.json, reloads the matching Pseudobase targets by id, and re-folds each
autonomous-solved sequence under pKiss / ProbKnot / ShapeKnots. Reports the pKiss-reference count, the
panel-unanimous count, and (implicitly) the envelope -- the honest, model-relative positive rate the
paper's certificate demands. Predictors that are unavailable are reported as such, NEVER silently counted.
"""
import json
import os
import sys

# make the repo root importable regardless of how this script is invoked (path vs -m)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from rna_harness.benchmarks.rnainvbench import load_pseudobase
from rna_harness.oracles_pk import (forms_target, pkiss_pairs, probknot_pairs,
                                     rnastructure_available, shapeknots_pairs)


def main(run_json, limit=60, arm="autonomous"):
    run = json.load(open(run_json))
    id2t = {t.task_id: t.target for t in load_pseudobase(limit=limit)}
    rows = run["rows"]
    solved = [(r["id"], r["by_policy"][arm]["seq"]) for r in rows
              if arm in r["by_policy"] and r["by_policy"][arm]["solved"]
              and r["by_policy"][arm].get("seq")]
    rnastr = rnastructure_available()
    print(f"panel[{arm}]: {len(solved)} solved seqs; ProbKnot/ShapeKnots available={rnastr}", flush=True)
    res, n_pk, n_unan = [], 0, 0
    for tid, seq in solved:
        target = id2t.get(tid)
        if not target:
            continue
        pk = forms_target(pkiss_pairs(seq), target)
        pb = forms_target(probknot_pairs(seq), target) if rnastr else None
        sk = forms_target(shapeknots_pairs(seq), target) if rnastr else None
        unan = (bool(pk) and bool(pb) and bool(sk)) if rnastr else None
        n_pk += int(bool(pk))
        n_unan += int(bool(unan))
        res.append({"id": tid, "pkiss": pk, "probknot": pb, "shapeknots": sk, "unanimous": unan})
        print(f"  {tid:16} pkiss={pk} probknot={pb} shapeknots={sk} unanimous={unan}", flush=True)
    out = {"arm": arm, "n_solved": len(solved), "rnastructure_available": rnastr,
           "pkiss_reconfirmed": n_pk, "panel_unanimous": (n_unan if rnastr else None), "rows": res}
    outp = os.path.join(os.path.dirname(run_json), f"panel_{arm}.json")
    json.dump(out, open(outp, "w"), indent=1)
    envelope = "n/a (ProbKnot/ShapeKnots unavailable)" if not rnastr else f"unanimous={n_unan}/{len(solved)}"
    print(f"\npanel[{arm}] summary: pKiss-reconfirmed={n_pk}/{len(solved)}  {envelope}  -> {outp}", flush=True)


if __name__ == "__main__":
    rj = sys.argv[1] if len(sys.argv) > 1 else "rna_harness/runs/cc_ascent/generous/run.json"
    arm = sys.argv[2] if len(sys.argv) > 2 else "autonomous"
    main(rj, arm=arm)
