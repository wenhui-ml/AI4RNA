#!/usr/bin/env python3
"""Flexible held-out sweep for chomsky_climb, PARALLELIZED across targets (256-core box).

Loads Pseudobase[offset:offset+limit], runs the requested arms at a given seed, adjudicates each solved
sequence under the full panel (pKiss/ProbKnot/ShapeKnots), saves cc_ascent/<tag>/sweep.json. The design
operators are single-process and time-boxed, so target-level parallelism gives near-linear speedup: a
multiprocessing Pool of NWORKERS runs many targets at once. No LLM (search-only), so no $ cost.

Usage: sweep.py <tag> <offset> <limit> <seed> <arms-comma>   [env: NWORKERS (default 48), PKP_*]
"""
import json, os, sys, time
import multiprocessing as mp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from rna_harness.experiments.track_b_pk import build_policies
from rna_harness.benchmarks.rnainvbench import load_pseudobase
from rna_harness.oracles_pk import (forms_target, pkiss_pairs, probknot_pairs,
                                     shapeknots_pairs, rnastructure_available)

HERE = os.path.dirname(os.path.abspath(__file__))
_POLS = None
_RNASTR = None
_SEED = 0
_ARMS = None


def _panel(seq, target):
    pk = forms_target(pkiss_pairs(seq), target)
    pb = forms_target(probknot_pairs(seq), target) if _RNASTR else None
    sk = forms_target(shapeknots_pairs(seq), target) if _RNASTR else None
    unan = (bool(pk) and bool(pb) and bool(sk)) if _RNASTR else None
    return {"pkiss": bool(pk), "probknot": (None if pb is None else bool(pb)),
            "shapeknots": (None if sk is None else bool(sk)), "unanimous": unan}


def _init(arms, seed):
    """Per-worker: build policies once (reads PKP_* from the inherited env) and cache rnastr."""
    global _POLS, _RNASTR, _SEED, _ARMS
    os.environ.setdefault("DESIRNA_CMD",
                          "/home/ubuntu/miniconda3/envs/rna/bin/python /data/wenhui/DesiRNA/DesiRNA.py")
    _ARMS, _SEED = arms, seed
    _RNASTR = rnastructure_available()
    _POLS = build_policies(arms)


def _work(item):
    tid, npk, n, target = item
    rec = {"id": tid, "n": n, "n_pk_pairs": npk, "target": target, "arms": {}}
    for name in _ARMS:
        t0 = time.time()
        try:
            r = _POLS[name].solve(target, seed=_SEED)
            seq = r.get("seq")
            adj = _panel(seq, target) if (r.get("solved") and seq) else None
            rec["arms"][name] = {"solved": bool(r.get("solved")), "seq": seq, "panel": adj,
                                 "s": round(time.time() - t0, 1)}
        except Exception as e:
            rec["arms"][name] = {"solved": None, "error": repr(e)[:200], "s": round(time.time() - t0, 1)}
    return rec


def main(tag, offset, limit, seed, arms):
    out = os.path.join(HERE, tag); os.makedirs(out, exist_ok=True)
    nworkers = int(os.environ.get("NWORKERS", "48"))
    tasks = load_pseudobase(limit=offset + limit)[offset:offset + limit]
    items = [(t.task_id, t.meta.get("n_pk_pairs"), t.n, t.target) for t in tasks]
    order = {t.task_id: i for i, t in enumerate(tasks)}
    print(f"sweep[{tag}]: Pseudobase[{offset}:{offset+limit}] ({len(tasks)} targets) seed={seed} "
          f"arms={arms} workers={nworkers}", flush=True)
    results = {}
    with mp.Pool(nworkers, initializer=_init, initargs=(arms, seed)) as pool:
        for rec in pool.imap_unordered(_work, items):
            results[rec["id"]] = rec
            u = {k: (v.get("panel") or {}).get("unanimous") for k, v in rec["arms"].items()}
            print(f"  [{len(results):>3}/{len(items)}] {rec['id']:14} g={rec['n_pk_pairs']} unan={u}", flush=True)
            rows = [results[t.task_id] for t in tasks if t.task_id in results]
            json.dump({"tag": tag, "offset": offset, "limit": limit, "seed": seed, "rows": rows},
                      open(os.path.join(out, "sweep.json"), "w"), indent=1)
    rows = [results[tid] for tid in sorted(results, key=lambda x: order[x])]
    for name in arms:
        solved = sum(1 for r in rows if r["arms"][name].get("solved"))
        unan = sum(1 for r in rows if (r["arms"][name].get("panel") or {}).get("unanimous"))
        print(f"  [{tag}] {name}: solved={solved}/{len(rows)}  panel-unanimous={unan}/{len(rows)}", flush=True)
    print(f"done -> {tag}/sweep.json", flush=True)


if __name__ == "__main__":
    tag = sys.argv[1]; offset = int(sys.argv[2]); limit = int(sys.argv[3]); seed = int(sys.argv[4])
    arms = sys.argv[5].split(",")
    main(tag, offset, limit, seed, arms)
