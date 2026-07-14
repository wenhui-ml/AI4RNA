"""Bounded, hang-proof classical solver + TRUE-RESIDUAL measurement.

The frozen-budget 'priors' pipeline -- canonical GC-stiff decompose assembly (no search) + bounded
`parallel_finish` (multi-restart mispaired-targeted hillclimb) -- already SOLVED hard targets 36/49/6
and never calls the unbounded `RNA.inverse_fold` (the 45-min-hang source). Here we make it a reliable,
WALL-BOUNDED solver: iterate `parallel_finish` (+ a `close_pair` endgame near a solve) in rounds, with
a small per-round perturbation for restart diversity, checking the wall clock BETWEEN rounds so it can
never hang. Then run it across the model-unsolved set to find the TRUE RESIDUAL: the targets that even
a strong bounded classical search cannot crack. That residual -- not the whole 'model-unsolved' list,
most of which is orchestration/hang failure, not search failure -- is the real frontier that would
justify a learned policy (Phase-B). Cheap and decisive BEFORE any GPU/RL.
"""
from __future__ import annotations

import random
import time
from typing import Dict

import RNA

from ..mcp.finisher import _fold, parallel_finish
from .retrieval import RetrievalLibrary
from .self_revise import op_close_pair

_BASES = "ACGU"
_COMP = {"A": "U", "U": "A", "G": "C", "C": "G"}


def _perturb(seq: str, target_pt, k: int, rng: random.Random) -> str:
    """Mutate k random positions (keeping intended pairs complementary) for restart diversity."""
    s = list(seq)
    n = len(s)
    for _ in range(k):
        i = rng.randrange(n)
        s[i] = rng.choice(_BASES)
        j = target_pt[i]
        if j >= 0 and j != i:
            s[j] = _COMP[s[i]]
    return "".join(s)


def bounded_solve(target: str, wall_s: float = 60.0, restarts: int = 8, iters: int = 8000,
                  seed: int = 0, stall_rounds: int = 15) -> Dict:
    """Canonical-assembly seed + wall-bounded rounds of parallel_finish (+ close_pair endgame). Never
    calls inverse_fold, so it cannot hang. Returns best bp, solved, rounds, oracle-time."""
    from ..structure_utils import pair_table
    t0 = time.time()
    n = len(target)
    pt = pair_table(target)
    seed_seq, _, _ = RetrievalLibrary().warm_start(target)     # GC-stiff canonical assembly, no search
    best_seq = seed_seq
    best_bp = RNA.bp_distance(_fold(seed_seq), target)
    from .guards import BudgetGuard, IterationGuard, guard_main_only
    guard_main_only("bounded_solve process pool")
    budget = BudgetGuard(wall_s=wall_s)                        # unified wall bound (E4)
    stall = IterationGuard(max_iters=200, stall_patience=stall_rounds)  # stop perturbation wheel-spin
    rng = random.Random(seed)
    rnd = 0
    stop = "solved"
    # Scale per-round iters DOWN with target size so a single parallel_finish call stays time-bounded
    # (a 400nt fold is ~50x a 100nt fold; fixed iters made one round take ~7min on 400nt). Keeps each
    # round ~10-20s at any size, so the wall check between rounds actually caps total time.
    round_iters = max(400, min(iters, int(150000 / max(1, n))))
    while best_bp > 0 and budget.ok():
        start = seed_seq if rnd == 0 else _perturb(best_seq, pt, rng.randint(2, 6), rng)
        r = parallel_finish(start, target, n_restarts=restarts, iters=round_iters)
        if r.get("bp_distance", 99) < best_bp:
            best_bp, best_seq = r["bp_distance"], r["seq"]
        if best_bp == 0:
            break
        if best_bp <= 6 and budget.ok():
            cp = op_close_pair(best_seq, target, budget={"iters": 6000})
            if cp.get("bp_distance", 99) < best_bp:
                best_bp, best_seq = cp["bp_distance"], cp["seq"]
        rnd += 1
        cont, why = stall.advance(best_bp)   # stop if best_bp unchanged for stall_rounds rounds
        if not cont:
            stop = why
            break
    if best_bp > 0 and stop == "solved":
        stop = budget.reason()
    solved = best_bp == 0 and _fold(best_seq) == target
    return {"bp": best_bp, "solved": bool(solved), "rounds": rnd + 1, "stop": stop,
            "t_s": round(time.time() - t0, 1), "seq": best_seq if solved else None}


if __name__ == "__main__":
    import argparse
    import json
    import os
    from ..benchmarks.eterna100 import load_eterna100

    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["unsolved", "all"], default="unsolved")
    ap.add_argument("--wall-s", type=float, default=60.0)
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "..", "runs", "residual"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    puz = {p["id"]: p for p in load_eterna100(version="V2")}
    summ = json.load(open(os.path.join(os.path.dirname(__file__), "..", "runs",
                                       "glm5.2_eterna100", "summary.json")))
    unsolved = [int(r["task_id"].split(":")[1]) for r in summ["records"] if not r["solved"]]
    ids = unsolved if a.set == "unsolved" else sorted(puz)
    ids = sorted(ids, key=lambda i: len(puz[i]["target"]))
    print(f"bounded_solve on {len(ids)} targets ({a.set}); wall={a.wall_s}s restarts={a.restarts} "
          f"iters={a.iters}", flush=True)
    rows, solved = [], 0
    for pid in ids:
        t = puz[pid]["target"]
        r = bounded_solve(t, wall_s=a.wall_s, restarts=a.restarts, iters=a.iters)
        solved += int(r["solved"])
        rows.append({"pid": pid, "name": puz[pid]["name"], "len": len(t), **{k: r[k] for k in
                     ("bp", "solved", "rounds", "t_s")}})
        print(f"  pid {pid:>3} n={len(t):>4} bp={r['bp']:>3} solved={r['solved']} "
              f"rounds={r['rounds']:>2} {r['t_s']:>5.1f}s", flush=True)
        json.dump({"config": vars(a), "n_solved": solved, "rows": rows},
                  open(os.path.join(a.outdir, "residual.json"), "w"), indent=1)
    resid = [r["pid"] for r in rows if not r["solved"]]
    print(f"\nSOLVED {solved}/{len(ids)} by bounded classical search.", flush=True)
    print(f"TRUE RESIDUAL ({len(resid)} targets, even bounded search can't crack): {resid}", flush=True)
    print(f"  -> these, not the full model-unsolved list, are the real frontier for a learned policy.",
          flush=True)
