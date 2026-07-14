"""A-class pass 2: close the bp=1 hardcore stragglers that pass 1 (DesiRNA 10min + close_pair) plateaued.

Different attack, not more-of-the-same:
  (1) MULTI-SEED DesiRNA -- many independent REMC trajectories; a bp=1 near-miss is one lucky basin
      away, and different seeds explore different basins;
  (2) coordinated_endgame on each seed's best near-miss -- the JOINT multi-base repair (break the
      non-local intruder + build the missing target pair, lexicographic bp->ensemble_defect) that
      single-position close_pair provably cannot reach. This is the operator matched to long-range
      frustration (min_pair_prob~0), which is exactly the pass-1 failure signature.
Every candidate re-verified by OUR primary oracle. Parallel over (pid x seed) with a concurrency cap.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import RNA

from ..benchmarks.eterna100 import load_eterna100
from ..generators.desirna import DesiRNAGenerator, desirna_available
from ..task import DesignTask
from .anneal import coordinated_endgame

OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "a_class_solve")
STRAGGLERS_DEFAULT = [71, 79, 81, 87]


def _attempt(pid: int, target: str, seed: int, replicas: int, timelimit: int) -> dict:
    """One (pid, seed): DesiRNA run, then coordinated_endgame on its best near-miss. Oracle-verified."""
    n = len(target)
    r = DesiRNAGenerator(replicas=replicas, timelimit_s=timelimit, results_number=40).design(
        DesignTask(task_id=str(pid), target=target), seed=seed)
    bp, seq, arm = r.best_bp_distance, r.best_sequence, "desirna"
    if bp > 0 and seq:  # joint multi-base repair on the near-miss (matched to long-range frustration)
        eg = coordinated_endgame(seq, target, trigger_bp=6, sys_top=4,
                                 stoch_iters=6000, restarts=15, seed=seed)
        if eg.get("bp", 99) < bp:
            bp, seq, arm = eg["bp"], eg["seq"], "desirna+coord_endgame"
    solved = bp == 0 and seq and RNA.fold(seq)[0] == target
    return {"pid": pid, "seed": seed, "n": n, "bp": bp, "solved": bool(solved),
            "arm": arm if solved else None, "seq": seq if solved else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids", default="", help="comma pids (default: the 4 bp=1 stragglers)")
    ap.add_argument("--seeds", type=int, default=10, help="independent DesiRNA seeds per target")
    ap.add_argument("--replicas", type=int, default=10)
    ap.add_argument("--timelimit", type=int, default=400)
    ap.add_argument("--max-workers", type=int, default=20, help="~max_workers*replicas cores peak")
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    if not desirna_available():
        raise SystemExit("DESIRNA_CMD not set / DesiRNA not found -- cannot run.")
    pids = [int(x) for x in a.pids.split(",") if x.strip()] or list(STRAGGLERS_DEFAULT)
    puz = {p["id"]: p for p in load_eterna100(version="V2")}
    jobs = [(pid, s) for pid in pids for s in range(1, a.seeds + 1)]
    print(f"A-class pass 2 on {pids}: {a.seeds} seeds each ({len(jobs)} runs), DesiRNA R={a.replicas} "
          f"-t={a.timelimit}s + coordinated_endgame; workers={a.max_workers} "
          f"(~{a.max_workers*a.replicas} cores peak); oracle-verified.", flush=True)
    best = {pid: {"pid": pid, "n": len(puz[pid]["target"]), "bp": 99, "solved": False,
                  "arm": None, "seq": None, "by_seed": {}} for pid in pids}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.max_workers) as ex:
        futs = {ex.submit(_attempt, pid, puz[pid]["target"], s, a.replicas, a.timelimit): (pid, s)
                for (pid, s) in jobs}
        done = 0
        for fut in as_completed(futs):
            pid, s = futs[fut]
            done += 1
            try:
                r = fut.result()
            except Exception as e:
                print(f"  ({done}/{len(jobs)}) pid {pid} seed {s} CRASH: {e}", flush=True)
                continue
            b = best[pid]
            b["by_seed"][s] = r["bp"]
            if r["bp"] < b["bp"] or (r["solved"] and not b["solved"]):
                b.update({"bp": r["bp"], "solved": r["solved"] or b["solved"],
                          "arm": r["arm"] or b["arm"], "seq": r["seq"] or b["seq"]})
            flag = "SOLVED" if r["solved"] else f"bp={r['bp']}"
            print(f"  ({done}/{len(jobs)}) pid {pid:>3} seed {s:>2} -> {flag} "
                  f"[best bp={b['bp']} solved={b['solved']}]", flush=True)
            json.dump({"pids": pids, "config": vars(a),
                       "n_solved": sum(1 for v in best.values() if v["solved"]),
                       "results": {str(k): v for k, v in best.items()}},
                      open(os.path.join(OUT, "a_class_pass2.json"), "w"), indent=1)
    nsolved = sum(1 for v in best.values() if v["solved"])
    resid = sorted(pid for pid, v in best.items() if not v["solved"])
    print(f"\nWall {time.time()-t0:.0f}s. pass-2 SOLVED {nsolved}/{len(pids)}.", flush=True)
    for pid in pids:
        v = best[pid]
        print(f"  pid {pid:>3} n={v['n']:>4}: bp={v['bp']} solved={v['solved']} arm={v['arm']}", flush=True)
    print(f"still unsolved: {resid}", flush=True)


if __name__ == "__main__":
    main()
