#!/usr/bin/env python3
"""Stage A of executing decision A: score every dumped candidate with RibonanzaNet, once, to a file.

RUNS IN THE RibonanzaNet venv (/data/wenhui/rnet_env/bin/python). The harness venv has no torch and this one
has no ViennaRNA, so the judges cannot run here -- that is stage B's job (`selector_heldout.py`, `rna` env).

WHY A SEPARATE FILE RATHER THAN REUSING selector_replay.py's INLINE SCORING. That script scores the pool and
then evaluates every arm with the SAME RibonanzaNet score it selected on, which is why its own docstring
calls the `oracle` arm an upper bound "not a reachable target" and why the project's synthetic null showed
max-of-outcome inflating by about +3.67. Decision A puts RibonanzaNet INTO the selection loop, so from here
on RibonanzaNet can no longer be the thing that says whether the selection was good. The scores are needed
to CHOOSE; something never in the loop has to JUDGE. Hence: dump the scores, judge elsewhere.

WHAT IS WRITTEN. {puzzle: {sequence: crossed_pair_F1}} for every unique candidate in every dumped pool,
using `pool_rerank_score._score_chunk` unchanged so the numbers are the same ones every prior replay used.

COST: $0. Local CPU, one torch thread per worker.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=os.path.join(HERE, "pool_dump"))
    ap.add_argument("--run", default=os.path.join(HERE, "leaderboard_replay.json"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(HERE, "selector_heldout_rnet.json"))
    a = ap.parse_args()

    from selector_replay import load_dump
    from pool_rerank_score import _score_chunk

    R = json.load(open(a.run))
    T = {p: v["target"] for p, v in R["targets"].items()}
    P = load_dump(a.dump, T)
    n = sum(len(v) for v in P.values())
    print(f"落盘池 {len(P)}/{len(T)} 个靶标,唯一候选合计 {n},{a.workers} 进程", flush=True)

    jobs = [((p, i), r["seq"], T[p]) for p in sorted(P) for i, r in enumerate(P[p])]
    from concurrent.futures import ProcessPoolExecutor
    CH = max(1, len(jobs) // (a.workers * 8))
    sc, t0, done = {}, time.time(), 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for res in ex.map(_score_chunk, [jobs[i:i + CH] for i in range(0, len(jobs), CH)]):
            sc.update(dict(res))
            done += 1
            if done % 10 == 0:
                print(f"    {len(sc)}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"  打分完成 {len(sc)}/{len(jobs)},用时 {(time.time()-t0)/60:.1f} 分钟")

    out = {p: {P[p][i]["seq"]: sc[(p, i)] for i in range(len(P[p])) if (p, i) in sc} for p in sorted(P)}
    json.dump({"n_scored": len(sc), "n_candidates": len(jobs), "workers": a.workers,
               "minutes": (time.time() - t0) / 60, "f1c": out}, open(a.out, "w"))
    print(f"写出 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
