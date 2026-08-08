#!/usr/bin/env python3
"""Score every pool candidate and KEEP the per-candidate row, so the generation gap can be dissected.

RUNS IN THE RibonanzaNet venv.

The two previous scoring passes each threw away exactly what is now needed. `pool_rerank_score.py` kept the
pool maximum and median; `rerank_audit_ourpools.py` kept the saturation and the tie count. Neither kept the
per-candidate score joined to the ladder tier that produced it, and that join is what separates the two
live explanations of why our designs sit at 0.816 while the field's learned designers sit at 0.93-0.98:

    search has not converged   -> the 60 s tier should produce visibly better candidates than the 5 s tier
    the criterion is misaimed  -> the tiers should look alike, because more search finds more of the same
                                  thing, and the thing itself is not what RibonanzaNet rewards

This script only produces the table. `generation_gap.py` does the testing, so that no analysis choice can be
made while looking at these numbers.

COST: $0. Local CPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pool_rerank_score import _score_chunk                                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default=os.path.join(HERE, "pool_rerank_pools.json"))
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--out", default=os.path.join(HERE, "gap_score_dump.json"))
    a = ap.parse_args()

    P = json.load(open(a.pools))
    T = P["targets"]
    pools = {r["puzzle"]: r["pool"] for r in P["rows"] if "pool" in r}
    jobs = [((p, i), c["seq"], T[p]["target"])
            for p in sorted(pools) for i, c in enumerate(pools[p])]
    print(f"{len(pools)} 个池,{len(jobs)} 条候选,{a.workers} 进程", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    CH = max(1, len(jobs) // (a.workers * 8))
    chunks = [jobs[i:i + CH] for i in range(0, len(jobs), CH)]
    sc, t0, done = {}, time.time(), 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for res in ex.map(_score_chunk, chunks):
            for k, v in res:
                sc[k] = v
            done += 1
            if done % 40 == 0 or done == len(chunks):
                print(f"    {done}/{len(chunks)}  {(time.time()-t0)/60:.1f} min", flush=True)

    rows = []
    for p in sorted(pools):
        for i, c in enumerate(pools[p]):
            if (p, i) in sc:
                rows.append({"puzzle": p, "i": i, "tier": c["tier"], "f1c": sc[(p, i)],
                             "seq": c["seq"]})
    json.dump({"n": len(rows), "rows": rows}, open(a.out, "w"), ensure_ascii=False)
    print(f"\n  wrote {a.out}  ({len(rows)} 行)")


if __name__ == "__main__":
    main()
