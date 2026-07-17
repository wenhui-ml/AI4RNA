"""Measure DesiRNA's REAL per-target wall+cost on a sample of solvable PK targets, so the RSSD cost
comparison (pkdecomp vs DesiRNA at equal solve) uses measured DesiRNA cost, not a nominal upper bound.
Runs each target as its own DesiRNA invocation (replicas processes) concurrently. Honest accounting."""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..benchmarks.rnainvbench import load_pseudobase
from ..cognitive.tools import DesirnaTool

OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "rssd_desirna_cost.json")


def _one(idx, target, tl, replicas, seed):
    r = DesirnaTool(timelimit_s=tl, replicas=replicas).attempt(target, seed=seed)
    return idx, {"solved": r.solved, "bp": r.bp, "wall_s": r.wall_s, "cores": r.cores,
                 "cost_usd": r.cost_usd, "oracle_s": r.oracle_s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, nargs="+", required=True)
    ap.add_argument("--timelimit-s", type=int, default=300)
    ap.add_argument("--replicas", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tasks = load_pseudobase(); by = {int(t.task_id.split(":")[1]): t for t in tasks}
    print(f"DesiRNA real-cost on {len(a.ids)} targets; -t {a.timelimit_s} replicas {a.replicas}", flush=True)
    out = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_one, i, by[i].target, a.timelimit_s, a.replicas, a.seed): i
                for i in a.ids if i in by}
        for fut in as_completed(futs):
            idx, res = fut.result()
            out[idx] = res
            print(f"  pseudobase:{idx:<4} n={by[idx].n:<3} solved={int(res['solved'])} bp={res['bp']} "
                  f"REAL wall={res['wall_s']:.1f}s x{res['cores']}cores cost=${res['cost_usd']:.2e}", flush=True)
            json.dump(out, open(OUT, "w"), indent=1)
    walls = [r["wall_s"] for r in out.values()]
    costs = [r["cost_usd"] for r in out.values()]
    import statistics as st
    print(f"\n  DesiRNA measured: median wall={st.median(walls):.1f}s  median cost=${st.median(costs):.2e}  "
          f"(n={len(out)})", flush=True)


if __name__ == "__main__":
    main()
