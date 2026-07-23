#!/usr/bin/env python3
"""Flexible held-out sweep for chomsky_climb (reused for the 2nd pool and extra seeds).

Loads Pseudobase[offset:offset+limit], runs the requested arms at a given seed, adjudicates each solved
sequence under the full panel (pKiss/ProbKnot/ShapeKnots), and saves cc_ascent/<tag>/sweep.json. Mirrors the
proven family_ood.py adjudication; no LLM (search-only operators), so no $ cost.

Usage: sweep.py <tag> <offset> <limit> <seed> <arms-comma>
  e.g. sweep.py pool2 60 50 0 always_cheap,panel_auto      # 2nd disjoint held-out pool (C)
       sweep.py seed3 0 60 3 panel_auto                    # extra seed (T3)
"""
import json, os, sys, time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from rna_harness.experiments.track_b_pk import build_policies
from rna_harness.benchmarks.rnainvbench import load_pseudobase
from rna_harness.oracles_pk import (forms_target, pkiss_pairs, probknot_pairs,
                                     shapeknots_pairs, rnastructure_available, pk_pairs)

HERE = os.path.dirname(os.path.abspath(__file__))


def panel(seq, target, rnastr):
    pk = forms_target(pkiss_pairs(seq), target)
    pb = forms_target(probknot_pairs(seq), target) if rnastr else None
    sk = forms_target(shapeknots_pairs(seq), target) if rnastr else None
    unan = (bool(pk) and bool(pb) and bool(sk)) if rnastr else None
    return {"pkiss": bool(pk), "probknot": (None if pb is None else bool(pb)),
            "shapeknots": (None if sk is None else bool(sk)), "unanimous": unan}


def main(tag, offset, limit, seed, arms):
    os.environ.setdefault("DESIRNA_CMD",
                          "/home/ubuntu/miniconda3/envs/rna/bin/python /data/wenhui/DesiRNA/DesiRNA.py")
    out = os.path.join(HERE, tag); os.makedirs(out, exist_ok=True)
    rnastr = rnastructure_available()
    tasks = load_pseudobase(limit=offset + limit)[offset:offset + limit]
    pols = build_policies(arms)
    print(f"sweep[{tag}]: Pseudobase[{offset}:{offset+limit}] ({len(tasks)} targets) seed={seed} "
          f"arms={list(pols)} rnastr={rnastr}", flush=True)
    rows = []
    for t in tasks:
        rec = {"id": t.task_id, "n": t.n, "n_pk_pairs": t.meta.get("n_pk_pairs"),
               "target": t.target, "arms": {}}
        for name, pol in pols.items():
            t0 = time.time()
            try:
                r = pol.solve(t.target, seed=seed)
                seq = r.get("seq")
                adj = panel(seq, t.target, rnastr) if (r.get("solved") and seq) else None
                rec["arms"][name] = {"solved": bool(r.get("solved")), "seq": seq, "panel": adj,
                                     "s": round(time.time() - t0, 1)}
            except Exception as e:
                rec["arms"][name] = {"solved": None, "error": repr(e)[:200], "s": round(time.time() - t0, 1)}
            a = rec["arms"][name]; u = (a.get("panel") or {}).get("unanimous")
            print(f"  {t.task_id:14} g={rec['n_pk_pairs']} {name:12} solved={a.get('solved')} unan={u} ({a.get('s')}s)", flush=True)
        rows.append(rec)
        json.dump({"tag": tag, "offset": offset, "limit": limit, "seed": seed, "rows": rows},
                  open(os.path.join(out, "sweep.json"), "w"), indent=1)
    # summary
    for name in pols:
        ncols = [r["arms"][name] for r in rows]
        solved = sum(1 for a in ncols if a.get("solved"))
        unan = sum(1 for a in ncols if (a.get("panel") or {}).get("unanimous"))
        print(f"  [{tag}] {name}: solved={solved}/{len(rows)}  panel-unanimous={unan}/{len(rows)}", flush=True)
    print(f"done -> {tag}/sweep.json", flush=True)


if __name__ == "__main__":
    tag = sys.argv[1]; offset = int(sys.argv[2]); limit = int(sys.argv[3]); seed = int(sys.argv[4])
    arms = sys.argv[5].split(",")
    main(tag, offset, limit, seed, arms)
