"""F5 -- verifier-cost routing: does a CHEAP oracle pre-screen cut EXPENSIVE-oracle cost at equal solve?

The pseudoknot oracle pKiss costs ~0.5-2s/call; the ViennaRNA MFE oracle costs microseconds. During
pkdecomp's repair, every candidate is judged by pKiss. F5 inserts an MFE pre-screen: fold the candidate
with MFE (cheap) and only spend a pKiss call if its NESTED part is within `mfe_screen` bp of the skeleton
(a candidate whose nested part is broken cannot be a real pseudoknot solve, so the pKiss call is wasted).
Sweeps the screen strictness and reports total pKiss wall (the dominant cost) + solves -- a cheaper
verification at equal quality is the win; a strict screen that loses solves is the honest failure mode.
This is the cost-of-the-VERIFIER axis (Paper D's 'which oracle' brought to the efficiency side)."""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..benchmarks.rnainvbench import load_pseudobase, pkiss_available
from ..cognitive.skills.pk_decompose import DecomposePKTool

OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "rssd_verifier_cost")


def _sample_ids():
    """A mix: solvable (pkdecomp should solve, so the screen must not cost solves) + near-miss residual."""
    p = os.path.join(os.path.dirname(__file__), "..", "runs", "pk_desirna", "pk_desirna.json")
    rows = json.load(open(p))["rows"]
    solved = [int(r["task_id"].split(":")[1]) for r in rows if r["pk_verified_solved"]]
    resid = [int(r["task_id"].split(":")[1]) for r in rows
             if not r["pk_verified_solved"] and r["pk_best_bp"] is not None and r["pk_best_bp"] <= 2]
    return sorted(solved)[:20] + sorted(resid)[:15]


def _one(idx, target, screen, skel_wall, repair_wall, max_iters, seed):
    tool = DecomposePKTool(skel_wall_s=skel_wall, repair_wall_s=repair_wall, max_repair_iters=max_iters,
                           skel_restarts=6)
    r = tool.attempt(target, seed=seed, mfe_screen=(None if screen < 0 else screen))
    return (idx, screen), {"solved": bool(r.solved), "bp": r.bp, "oracle_s": r.oracle_s, "cost": r.cost_usd}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screens", type=int, nargs="+", default=[-1, 12, 8, 4], help="-1 = no screen (baseline)")
    ap.add_argument("--skel-wall-s", type=float, default=8.0)
    ap.add_argument("--repair-wall-s", type=float, default=40.0)
    ap.add_argument("--max-iters", type=int, default=250)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=OUT)
    a = ap.parse_args()
    if not pkiss_available():
        raise SystemExit("pKiss unavailable")
    os.makedirs(a.outdir, exist_ok=True)
    tasks = {int(t.task_id.split(":")[1]): t.target for t in load_pseudobase()}
    ids = _sample_ids()
    print(f"F5 verifier-cost routing: {len(ids)} targets x screens {a.screens} "
          f"(-1=no screen); repair={a.repair_wall_s}s", flush=True)
    jobs = [(i, tasks[i], sc) for i in ids for sc in a.screens]
    res = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_one, i, tgt, sc, a.skel_wall_s, a.repair_wall_s, a.max_iters, a.seed): (i, sc)
                for i, tgt, sc in jobs}
        for fut in as_completed(futs):
            key, r = fut.result()
            res[f"{key[0]}:{key[1]}"] = r
            json.dump(res, open(os.path.join(a.outdir, "verifier_cost.json"), "w"), indent=0)

    print(f"\n=== F5 VERIFIER-COST ROUTING (pKiss wall + solves per screen strictness) ===", flush=True)
    base_solved = sum(1 for i in ids if res[f"{i}:-1"]["solved"])
    for sc in a.screens:
        tot_oracle = sum(res[f"{i}:{sc}"]["oracle_s"] for i in ids)
        solved = sum(1 for i in ids if res[f"{i}:{sc}"]["solved"])
        lbl = "no-screen (baseline)" if sc < 0 else f"MFE-screen<={sc}bp"
        print(f"  {lbl:22} solved={solved}/{len(ids)}  total pKiss wall={tot_oracle:.0f}s", flush=True)
    base_oracle = sum(res[f"{i}:-1"]["oracle_s"] for i in ids)
    print(f"\n  interpretation: a screen that keeps solved={base_solved} while cutting pKiss wall below "
          f"{base_oracle:.0f}s is a free verification saving; one that drops solves is too strict.", flush=True)
    json.dump({"config": vars(a), "ids": ids, "results": res},
              open(os.path.join(a.outdir, "verifier_cost.json"), "w"), indent=1)
    print(f"  wrote {os.path.join(a.outdir, 'verifier_cost.json')}", flush=True)


if __name__ == "__main__":
    main()
