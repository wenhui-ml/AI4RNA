#!/usr/bin/env python3
"""PRE-REGISTERED: does the M1 win generalise beyond composition, or was GC a coincidence?

The boundary map says a wrapper beats the solver at fixed budget exactly when the request carries a
requirement the solver does not optimise. Composition is one instance (3/72 against 68/72). If the map is
right, the SAME pattern must appear for any other unoptimised requirement; if it does not, the map is wrong
and the GC result is a one-off.

The cleanest second instance is STRICTNESS, because it is not an extra preference bolted onto the request --
it is a stricter reading of the same request, and a perfectly ordinary scientific one:

    match   the MFE structure equals the target                    what the solver optimises
    umfe    ...and no other structure shares that minimum energy    "fold uniquely into this shape"
    pfe     ...and P(target) >= 0.90 in the ensemble                "fold into this shape almost always"

Archived survival on Eterna100 is 1.000 / 0.902 / 0.066, so `match` and `pfe` are very different questions
and nothing in the solver distinguishes them.

DECLARED BEFORE THE RUN: same solver, same seconds, same pool, both arms judged by the same exact oracle.
PREDICTION: the tool's own answer fails pfe far more often than screening its pool does. If the two arms come
out equal, M1 does not generalise and the boundary map above must be rewritten.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)

SHAPES = {
    "hairpin-6": "((((((....))))))",
    "hairpin-10": "((((((((((......))))))))))",
    "two-hairpins": "((((....))))....((((....))))",
    "junction-2": "(((((..((((....))))..((((....))))..)))))",
    "junction-3": "((((.(((....))).(((....))).(((....))).))))",
    "internal-loop": "((((..((((....))))..))))",
}


def run_one(shape, target, rung, timelimit, seed):
    from rna_harness.cognitive.certify import _nested_rungs
    from rna_harness.generators.desirna import DesiRNAGenerator
    from rna_harness.task import DesignTask, OracleConfig
    t0 = time.time()
    res = DesiRNAGenerator(replicas=10, timelimit_s=timelimit, results_number=50).design(
        DesignTask(task_id="rung", target=target,
                   oracle=OracleConfig(param_set="turner2004", temperature=37.0, dangles=2)), seed=seed)
    pool, seen = [], set()
    for s in ([res.best_sequence] if res.best_sequence else []) + list(res.candidates or []):
        if s and len(s) == len(target) and s not in seen:
            seen.add(s)
            pool.append(s)
    tool = res.best_sequence
    tool_rungs = _nested_rungs(tool, target) if tool else []
    ours = next((s for s in pool if rung in _nested_rungs(s, target)), None)
    return {"shape": shape, "n": len(target), "rung": rung, "seed": seed,
            "wall_s": round(time.time() - t0, 1), "pool": len(pool),
            "tool_rungs": tool_rungs, "tool_ok": rung in tool_rungs,
            "ours_ok": bool(ours), "ours_sequence": ours,
            "pool_qualifying": sum(1 for s in pool if rung in _nested_rungs(s, target))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timelimit", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    jobs = [(sh, t, rung, a.timelimit, s)
            for sh, t in SHAPES.items() for rung in ("umfe", "pfe") for s in range(a.seeds)]
    print(f"strictness benchmark: {len(SHAPES)} shapes x 2 rungs x {a.seeds} seeds = {len(jobs)} runs\n",
          flush=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    R, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, f in enumerate(as_completed([ex.submit(run_one, *j) for j in jobs]), 1):
            R.append(f.result())
            if n % 8 == 0:
                print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)

    print(f"\n=== {len(R)} runs, same solver, same seconds, same pool ===")
    out = {}
    for rung in ("umfe", "pfe"):
        s = [r for r in R if r["rung"] == rung]
        t_ok = sum(1 for r in s if r["tool_ok"])
        o_ok = sum(1 for r in s if r["ours_ok"])
        b = sum(1 for r in s if r["ours_ok"] and not r["tool_ok"])
        c = sum(1 for r in s if r["tool_ok"] and not r["ours_ok"])
        n = b + c
        p = (sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n * 2) if n else 1.0
        print(f"  {rung:<5}  tool's own answer {t_ok:>2}/{len(s)}   screening its pool {o_ok:>2}/{len(s)}   "
              f"McNemar {b}-{c}, p={p:.2e}")
        out[rung] = {"tool_ok": t_ok, "ours_ok": o_ok, "n": len(s), "b": b, "c": c, "p": p}
    print(f"\n  {'shape':<14} {'rung':<5} {'tool rungs':<22} {'tool ok':>8} {'ours ok':>8} {'pool ok':>8}")
    for r in sorted(R, key=lambda r: (r["rung"], r["shape"], r["seed"])):
        print(f"  {r['shape']:<14} {r['rung']:<5} {','.join(r['tool_rungs']) or '-':<22} "
              f"{str(r['tool_ok']):>8} {str(r['ours_ok']):>8} {r['pool_qualifying']:>8}")
    json.dump({"n": len(R), "by_rung": out, "rows": R},
              open(os.path.join(HERE, "rung_bench.json"), "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote rna_harness/runs/ceiling/rung_bench.json")


if __name__ == "__main__":
    main()
