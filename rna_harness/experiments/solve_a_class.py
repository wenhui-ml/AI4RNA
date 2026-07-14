"""Close the A-class near-miss residual to bp=0.

These are the small (n<=214), bp<=2 targets DesiRNA reached but did not close at the minute-level
budget of the residual sweep -- "last-mile" long-range structural frustration (min_pair_prob~0, the
competing pairing is non-local). Here we push them with a MUCH larger DesiRNA budget in parallel
(longer -t, more replicas), then (arm 2, for any straggler) a targeted local endgame seeded from
DesiRNA's best near-miss sequence: op_close_pair hits exactly the mispaired positions, which global
REMC dilutes. Every candidate is re-verified by OUR primary oracle -- solved := oracle MFE == target.

Usage (escalating passes -- caller bumps budget/seed):
  python -m rna_harness.experiments.solve_a_class --timelimit 600 --replicas 20 --seed 1
  python -m rna_harness.experiments.solve_a_class --pids 71,92 --timelimit 1800 --replicas 30 --seed 7 --endgame
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

OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "a_class_solve")
A_CLASS_DEFAULT = [60, 64, 71, 79, 81, 87, 89, 92]


def _endgame(target: str, seed_seq: str, wall_s: float = 120.0) -> tuple:
    """Arm 2: targeted local last-mile from DesiRNA's best near-miss. close_pair repairs the specific
    mispaired positions; bounded_solve rounds add restart diversity. Never calls inverse_fold."""
    from ..mcp.finisher import _fold
    from .self_revise import op_close_pair
    best_seq, best_bp = seed_seq, RNA.bp_distance(_fold(seed_seq), target)
    t0 = time.time()
    while best_bp > 0 and time.time() - t0 < wall_s:
        cp = op_close_pair(best_seq, target, budget={"iters": 12000})
        if cp.get("bp_distance", 99) < best_bp:
            best_bp, best_seq = cp["bp_distance"], cp["seq"]
        else:
            break  # close_pair converged; no further local gain
    return best_seq, best_bp


def _solve_one(pid: int, name: str, target: str, replicas: int, timelimit: int,
               seed: int, endgame: bool) -> dict:
    n = len(target)
    t0 = time.time()
    r = DesiRNAGenerator(replicas=replicas, timelimit_s=timelimit, results_number=40).design(
        DesignTask(task_id=str(pid), target=target), seed=seed)
    bp, seq, arm = r.best_bp_distance, r.best_sequence, "desirna"
    if bp > 0 and endgame and seq:  # arm 2: targeted local endgame from DesiRNA's best near-miss
        eseq, ebp = _endgame(target, seq)
        if ebp < bp:
            bp, seq, arm = ebp, eseq, "desirna+endgame"
    solved = bp == 0 and seq and RNA.fold(seq)[0] == target
    return {"pid": pid, "name": name, "len": n, "bp": bp, "solved": bool(solved),
            "arm": arm if solved else None, "candidates": r.n_evals,
            "seq": seq if solved else None, "wall_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids", default="", help="comma pids (default: the 8 A-class)")
    ap.add_argument("--replicas", type=int, default=20)
    ap.add_argument("--timelimit", type=int, default=600, help="DesiRNA -t seconds per target")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--endgame", action="store_true", help="arm 2: local close_pair endgame on stragglers")
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    if not desirna_available():
        raise SystemExit("DESIRNA_CMD not set / DesiRNA not found -- cannot run.")
    pids = [int(x) for x in a.pids.split(",") if x.strip()] or list(A_CLASS_DEFAULT)
    puz = {p["id"]: p for p in load_eterna100(version="V2")}
    print(f"A-class push on {pids}; DesiRNA -t={a.timelimit}s R={a.replicas} seed={a.seed} "
          f"endgame={a.endgame} (~{len(pids)*a.replicas} cores peak); oracle-verified.", flush=True)
    rows, solved = [], 0
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=len(pids)) as ex:
        futs = {ex.submit(_solve_one, pid, puz[pid]["name"], puz[pid]["target"],
                          a.replicas, a.timelimit, a.seed, a.endgame): pid for pid in pids}
        for fut in as_completed(futs):
            pid = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"pid": pid, "name": puz[pid]["name"], "len": len(puz[pid]["target"]),
                     "bp": None, "solved": False, "arm": None, "candidates": 0, "seq": None,
                     "wall_s": None, "error": f"worker-crash: {e}"}
            solved += int(r["solved"])
            rows.append(r)
            print(f"  [{len(rows)}/{len(pids)}] pid {r['pid']:>3} n={r['len']:>4} | bp={r['bp']} "
                  f"solved={r['solved']} arm={r.get('arm')} ({r.get('wall_s')}s)", flush=True)
            json.dump({"pids": pids, "config": vars(a), "n_solved": solved,
                       "rows": sorted(rows, key=lambda x: x["pid"])},
                      open(os.path.join(OUT, "a_class_solve.json"), "w"), indent=1)
    resid = sorted(r["pid"] for r in rows if not r["solved"])
    print(f"\nWall {time.time()-t_start:.0f}s. SOLVED {solved}/{len(pids)} of the A-class near-misses.",
          flush=True)
    print(f"still unsolved: {resid}", flush=True)


if __name__ == "__main__":
    main()
