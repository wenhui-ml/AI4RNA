#!/usr/bin/env python3
"""Genuine (top-level) CF-floor measurement for the cross-source (bpRNA) and second (pool2) pools.

The sweep.py `always_cheap` arm ERRORS inside its mp.Pool workers -- replica_exchange's own process pool is
refused by the fork-bomb guard (nested multiprocessing) -- so the previously-logged cross/pool2 "floor 0" was
error-backed, not measured. Here we run `always_cheap` (RemcTool, a CF-confined optimizer) at TOP LEVEL, where
replica_exchange can spawn its replica pool, and adjudicate each design under the full panel. Expected: 0
panel-unanimous crossings (Prop. 1: a pk-free/CF-confined optimizer cannot realize a crossing target). Serial
over targets; each REMC call uses its own replica pool. Usage: xsrc_floor.py <cross|pool2> [limit].
"""
import json, os, sys, time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from rna_harness.experiments.track_b_pk import build_policies
from rna_harness.benchmarks.rnainvbench import load_pseudobase
from rna_harness.oracles_pk import (forms_target, pkiss_pairs, probknot_pairs,
                                     shapeknots_pairs, rnastructure_available)

HERE = os.path.dirname(os.path.abspath(__file__))


def load_pool(name):
    if name == "cross":
        rows = json.load(open(os.path.join(HERE, "bprna_pk/targets.json")))["rows"]
        return [(r["id"], r["target"]) for r in rows]
    if name == "pool2":
        tasks = load_pseudobase(limit=110)[60:110]
        return [(t.task_id, t.target) for t in tasks]
    if name == "bigpool":                                  # lever-b enlarged pool: full 251 crossing set
        tasks = load_pseudobase(limit=251)[0:251]
        return [(t.task_id, t.target) for t in tasks]
    raise ValueError(f"unknown pool {name!r}")


def run(pool_name, seed=0, limit=None):
    rnastr = rnastructure_available()
    remc = build_policies(["always_cheap"])["always_cheap"]
    items = load_pool(pool_name)
    if limit:
        items = items[:limit]
    print(f"[floor:{pool_name}] {len(items)} targets; rnastr={rnastr}", flush=True)
    rows, npk, nunan = [], 0, 0
    for tid, target in items:
        t0 = time.time()
        try:
            r = remc.solve(target, seed=seed)                       # CF-confined optimizer on the FULL target
            seq = r.get("seq")
            pk = bool(forms_target(pkiss_pairs(seq), target)) if seq else False
            pb = bool(forms_target(probknot_pairs(seq), target)) if (seq and rnastr) else None
            sk = bool(forms_target(shapeknots_pairs(seq), target)) if (seq and rnastr) else None
            unan = (pk and bool(pb) and bool(sk)) if rnastr else None
            npk += int(pk); nunan += int(bool(unan))
            rows.append({"id": tid, "pkiss": pk, "probknot": pb, "shapeknots": sk,
                         "unanimous": unan, "s": round(time.time() - t0, 1)})
            print(f"  {tid:16} pkiss={pk} unan={unan}", flush=True)
        except Exception as e:
            rows.append({"id": tid, "error": repr(e)[:200]})
            print(f"  {tid:16} ERROR {repr(e)[:90]}", flush=True)
        json.dump({"pool": pool_name, "n": len(items), "floor_pkiss": npk,
                   "floor_panel_unanimous": nunan, "rows": rows},
                  open(os.path.join(HERE, f"xsrc_floor_{pool_name}.json"), "w"), indent=1)
    print(f"[floor:{pool_name}] DONE floor_pkiss={npk}/{len(items)} "
          f"panel-unanimous={nunan}/{len(items)}", flush=True)


if __name__ == "__main__":
    pool = sys.argv[1] if len(sys.argv) > 1 else "cross"
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run(pool, limit=lim)
