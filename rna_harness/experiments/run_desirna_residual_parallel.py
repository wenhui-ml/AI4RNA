"""Plan-A measurement (PARALLEL, -p 2004 corrected): run DesiRNA across the 42 model-unsolved
Eterna100-V2 targets to get the true ORCHESTRATED solve total.

Supersedes the serial `run_desirna_residual.py`. Two things changed since v1 (which gave 70/100):
  1. the adapter now passes `-p 2004` (task.oracle.param_set) so DesiRNA optimises under the SAME
     energy model our oracle verifies with -- v1 ran DesiRNA's 1999 default and lost 5 easy targets
     (16,37,59,66,82) to a cross-oracle near-miss (all recovered under 2004);
  2. we fan targets across the box with a process pool. Each DesiRNA run uses R replicas (~R cores);
     with --workers concurrent targets the peak core use is ~workers*R. Default 16*10=160 (authorized).

Every returned sequence is still re-verified by OUR primary oracle inside the adapter -- the pool only
changes scheduling, not the accept test. Requires DESIRNA_CMD set (see reference-desirna-integration).
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..benchmarks.eterna100 import load_eterna100
from ..generators.desirna import DesiRNAGenerator, desirna_available
from ..task import DesignTask

OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "desirna_residual")
BASE_SUMMARY = os.path.join(os.path.dirname(__file__), "..", "runs", "glm5.2_eterna100", "summary.json")
MODEL_SOLVED = 58  # base GLM-5.2 solve count (glm5.2_eterna100)


def _solve_one(pid: int, name: str, target: str, replicas: int, results_number: int) -> dict:
    """Worker: one DesiRNA call (oracle-verified, -p 2004 via task.oracle). Runs in a pool process;
    DesiRNA itself is a subprocess here, so no nested multiprocessing fork from this worker."""
    n = len(target)
    tl = min(180, int(30 + 0.5 * n))
    t0 = time.time()
    r = DesiRNAGenerator(replicas=replicas, timelimit_s=tl, results_number=results_number).design(
        DesignTask(task_id=str(pid), target=target))
    return {"pid": pid, "name": name, "len": n, "timelimit_s": tl,
            "bp": r.best_bp_distance, "solved": bool(r.solved), "candidates": r.n_evals,
            "available": r.meta.get("available"), "error": r.meta.get("error"),
            "wall_s": round(time.time() - t0, 1)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16, help="concurrent targets (~workers*replicas cores)")
    ap.add_argument("--replicas", type=int, default=10)
    ap.add_argument("--results-number", type=int, default=20)
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    if not desirna_available():
        raise SystemExit("DESIRNA_CMD not set / DesiRNA not found -- cannot run.")
    puz = {p["id"]: p for p in load_eterna100(version="V2")}
    summ = json.load(open(BASE_SUMMARY))
    # largest-first: longest jobs start earliest -> better pool load balance (shortest tail)
    unsolved = sorted([int(r["task_id"].split(":")[1]) for r in summ["records"] if not r["solved"]],
                      key=lambda i: len(puz[i]["target"]), reverse=True)
    print(f"DesiRNA (PARALLEL, -p 2004) on {len(unsolved)} model-unsolved targets; "
          f"workers={a.workers} replicas={a.replicas} (~{a.workers * a.replicas} cores peak); "
          f"per-size timelimit=min(180,30+0.5n)s; every candidate re-verified by our primary oracle.",
          flush=True)
    rows, solved = [], 0
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_solve_one, pid, puz[pid]["name"], puz[pid]["target"],
                          a.replicas, a.results_number): pid for pid in unsolved}
        for fut in as_completed(futs):
            pid = futs[fut]
            try:
                r = fut.result()
            except Exception as e:  # never let one target kill the sweep
                r = {"pid": pid, "name": puz[pid]["name"], "len": len(puz[pid]["target"]),
                     "bp": None, "solved": False, "candidates": 0, "available": None,
                     "error": f"worker-crash: {e}", "wall_s": None}
            solved += int(r["solved"])
            rows.append(r)
            done = len(rows)
            print(f"  [{done:>2}/{len(unsolved)}] pid {r['pid']:>3} n={r['len']:>4} "
                  f"tl={r.get('timelimit_s','?')}s | DesiRNA bp={r['bp']} solved={r['solved']} "
                  f"cand={r['candidates']} ({r.get('wall_s')}s)", flush=True)
            rows_sorted = sorted(rows, key=lambda x: x["pid"])
            json.dump({"model_solved": MODEL_SOLVED, "n_desirna_solved_of_unsolved": solved,
                       "orchestrated_total": MODEL_SOLVED + solved,
                       "config": {"workers": a.workers, "replicas": a.replicas, "param": "2004"},
                       "rows": rows_sorted},
                      open(os.path.join(OUT, "desirna_residual.json"), "w"), indent=1)
    resid = sorted([r["pid"] for r in rows if not r["solved"]])
    print(f"\nWall time: {time.time() - t_start:.0f}s.", flush=True)
    print(f"DesiRNA solved {solved}/{len(unsolved)} of the model-unsolved set.", flush=True)
    print(f"ORCHESTRATED TOTAL = {MODEL_SOLVED} (model) + {solved} (DesiRNA) = "
          f"{MODEL_SOLVED + solved}/100", flush=True)
    print(f"still unsolved within budget ({len(resid)}): {resid}", flush=True)


if __name__ == "__main__":
    main()
