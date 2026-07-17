"""RSSD smoke test: does the distilled DecomposePKTool crack pseudoknot residual the {REMC,DesiRNA}
toolbox couldn't? Runs the skill on a chosen set of Pseudobase++ task ids (default: the bp<=2 near-miss
residual from the pk251 baseline) and reports solved/bp. Cheap, parallel, informative. See docs/RSSD_design.md.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..benchmarks.rnainvbench import load_pseudobase, pk_pairs, pkiss_available
from ..cognitive.skills.pk_decompose import DecomposePKTool

OUT_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "runs", "rssd_smoke")


def _residual_ids(near_only: bool = True):
    """Ids DesiRNA failed on in the cached pk251 run (optionally only the bp<=2 near-misses)."""
    p = os.path.join(os.path.dirname(__file__), "..", "runs", "pk_desirna", "pk_desirna.json")
    rows = json.load(open(p))["rows"]
    res = [r for r in rows if not r["pk_verified_solved"]]
    if near_only:
        res = [r for r in res if r["pk_best_bp"] is not None and r["pk_best_bp"] <= 2]
    res.sort(key=lambda r: (r["pk_best_bp"], r["n"]))
    return [int(r["task_id"].split(":")[1]) for r in res], {int(r["task_id"].split(":")[1]): r for r in rows}


def _one(idx, target, skel_wall_s, repair_wall_s, max_iters, replicas, seed):
    tool = DecomposePKTool(skel_wall_s=skel_wall_s, repair_wall_s=repair_wall_s,
                           max_repair_iters=max_iters, n_replicas=replicas)
    r = tool.attempt(target, seed=seed)
    return idx, {"solved": r.solved, "bp": r.bp, "cost_usd": r.cost_usd, "wall_s": r.wall_s,
                 "oracle_s": r.oracle_s, "seq": r.seq}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, nargs="+", help="explicit task ids; default = pk251 near-miss residual")
    ap.add_argument("--all-residual", action="store_true", help="use all 71 residual, not just bp<=2")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--replicas", type=int, default=8)
    ap.add_argument("--skel-wall-s", type=float, default=15.0)
    ap.add_argument("--repair-wall-s", type=float, default=60.0)
    ap.add_argument("--max-iters", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=OUT_DEFAULT)
    a = ap.parse_args()

    if not pkiss_available():
        raise SystemExit("pKiss unavailable -- set PKISS_CMD or install conda env 'pk'")
    os.makedirs(a.outdir, exist_ok=True)
    tasks = load_pseudobase()                                   # all 251, index-aligned to task_id
    by_idx = {int(t.task_id.split(":")[1]): t for t in tasks}

    if a.ids:
        ids = a.ids
        base = {}
    else:
        ids, base = _residual_ids(near_only=not a.all_residual)
    print(f"RSSD smoke: {len(ids)} targets; skill=pkdecomp skel={a.skel_wall_s}s repair={a.repair_wall_s}s "
          f"workers={a.workers} replicas={a.replicas}", flush=True)

    results = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_one, i, by_idx[i].target, a.skel_wall_s, a.repair_wall_s, a.max_iters,
                          a.replicas, a.seed): i for i in ids if i in by_idx}
        for fut in as_completed(futs):
            idx, res = fut.result()
            results[idx] = res
            b = base.get(idx, {})
            was = b.get("pk_best_bp")
            tag = "  <== CRACKED (skill solved residual)" if res["solved"] else ""
            print(f"  pseudobase:{idx:<4} n={by_idx[idx].n:<3} pk_pairs={len(pk_pairs(by_idx[idx].target)):<2} "
                  f"desirna_bp={was} -> pkdecomp solved={int(res['solved'])} bp={res['bp']} "
                  f"${res['cost_usd']:.2e} ({res['oracle_s']:.0f}s pKiss){tag}", flush=True)
            json.dump({"config": vars(a), "results": results},
                      open(os.path.join(a.outdir, "smoke.json"), "w"), indent=1)

    cracked = sum(1 for r in results.values() if r["solved"])
    print(f"\n=== RSSD smoke: pkdecomp CRACKED {cracked}/{len(results)} of the (DesiRNA-residual) targets ===",
          flush=True)
    if base:
        improved = sum(1 for i, r in results.items() if not r["solved"]
                       and r["bp"] is not None and base[i].get("pk_best_bp") is not None
                       and r["bp"] < base[i]["pk_best_bp"])
        print(f"    of the still-unsolved, {improved} improved bp below DesiRNA's best (partial progress)",
              flush=True)


if __name__ == "__main__":
    main()
