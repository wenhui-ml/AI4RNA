#!/usr/bin/env python3
"""Melting temperature: a requirement DesiRNA cannot express, tested in BOTH directions.

WHY BOTH DIRECTIONS. The rule earned from three previous benchmarks is that a wrapper pays exactly when
satisfying the requirement FORCES A LOSS in the solver's own objective. That rule makes a DIRECTIONAL
prediction here, and a directional prediction is falsifiable in a way a one-sided one is not:

    a LOW Tm request  costs stability -> the objective must be given up -> predict a LARGE gap
    a HIGH Tm request buys stability  -> it is with the grain          -> predict NO gap

If both directions show a gap, the rule is wrong and the wins are just "the tool has no flag for it".
If neither does, the rule is wrong the other way. Only the asymmetry confirms it.

WHAT THE TOOL HAS: nothing. `-tmin`/`-tmax` are replica-exchange temperatures (a search parameter), `-tm` is
targeted mutations (a mutation heuristic), and there is no folding-temperature option at all -- DesiRNA
designs at 37 C, full stop. Verified in its argument parser and source.

Tm IS DEFINED EXACTLY, not by a heuristic formula: the temperature at which the target structure's
probability in the Boltzmann ensemble falls to 0.5, found by bisection on ViennaRNA's partition function.
That is a statement about the structure the user asked for, not about a generic duplex.

FAIR BY CONSTRUCTION: same solver, same seconds, same pool; both arms judged by fold == target AND
|Tm - request| <= tolerance.
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
    "hairpin-10": "((((((((((......))))))))))",
    "two-hairpins": "((((....))))....((((....))))",
    "junction-2": "(((((..((((....))))..((((....))))..)))))",
    "internal-loop": "((((..((((....))))..))))",
    "hairpin-6": "((((((....))))))",
}
REQUESTS = [(55, 5), (65, 5), (75, 5), (85, 5)]      # target Tm in C, tolerance


def p_at(seq, target, T):
    import RNA
    md = RNA.md()
    md.temperature = float(T)
    fc = RNA.fold_compound(seq, md)
    fc.pf()
    return fc.pr_structure(target)


def melting_point(seq, target, lo=5.0, hi=120.0, iters=12):
    """The temperature at which P(target) crosses 0.5. Exact given the model, no closed-form approximation."""
    if p_at(seq, target, lo) < 0.5:
        return lo
    if p_at(seq, target, hi) >= 0.5:
        return hi
    for _ in range(iters):
        mid = (lo + hi) / 2
        if p_at(seq, target, mid) >= 0.5:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 1)


def run_one(shape, target, want, tol, timelimit, seed):
    from rna_harness.generators.desirna import DesiRNAGenerator
    from rna_harness.task import DesignTask, OracleConfig
    import RNA
    t0 = time.time()
    res = DesiRNAGenerator(replicas=10, timelimit_s=timelimit, results_number=50).design(
        DesignTask(task_id="tm", target=target,
                   oracle=OracleConfig(param_set="turner2004", temperature=37.0, dangles=2)), seed=seed)
    pool, seen = [], set()
    for s in ([res.best_sequence] if res.best_sequence else []) + list(res.candidates or []):
        if s and len(s) == len(target) and s not in seen:
            seen.add(s)
            pool.append(s)
    folds = [s for s in pool if RNA.fold(s)[0] == target]     # Tm is only meaningful if it folds at all
    tms = {s: melting_point(s, target) for s in folds}

    tool = res.best_sequence
    tool_tm = tms.get(tool) if (tool and tool in tms) else (
        melting_point(tool, target) if tool and RNA.fold(tool)[0] == target else None)
    tool_ok = tool_tm is not None and abs(tool_tm - want) <= tol
    ours = next((s for s in folds if abs(tms[s] - want) <= tol), None)
    got = [tms[s] for s in folds]
    return {"shape": shape, "n": len(target), "want": want, "tol": tol, "seed": seed,
            "wall_s": round(time.time() - t0, 1), "pool": len(pool), "pool_folding": len(folds),
            "tool_tm": tool_tm, "tool_ok": tool_ok,
            "ours_tm": tms.get(ours) if ours else None, "ours_ok": bool(ours),
            "pool_qualifying": sum(1 for s in folds if abs(tms[s] - want) <= tol),
            "pool_tm_min": min(got) if got else None, "pool_tm_max": max(got) if got else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timelimit", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    jobs = [(sh, t, w, tol, a.timelimit, s)
            for sh, t in SHAPES.items() for w, tol in REQUESTS for s in range(a.seeds)]
    print(f"melting-temperature benchmark: {len(SHAPES)} shapes x {len(REQUESTS)} requests x {a.seeds} seeds "
          f"= {len(jobs)} runs\n", flush=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    R, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, f in enumerate(as_completed([ex.submit(run_one, *j) for j in jobs]), 1):
            R.append(f.result())
            if n % 10 == 0:
                print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)

    nat = [r["tool_tm"] for r in R if r["tool_tm"] is not None]
    nat.sort()
    print(f"\n=== {len(R)} runs, same solver, same seconds, same pool ===")
    print(f"  求解器不受约束时的天然 Tm:中位 {nat[len(nat)//2]:.1f}C  区间 {nat[0]:.1f}-{nat[-1]:.1f}C")
    print(f"\n  {'请求 Tm':>8} {'方向':>6} {'工具达标':>10} {'筛选其池':>10} {'池中合格(中位)':>16}")
    out = {}
    for w, tol in REQUESTS:
        sub = [r for r in R if r["want"] == w]
        t_ok = sum(1 for r in sub if r["tool_ok"])
        o_ok = sum(1 for r in sub if r["ours_ok"])
        q = sorted(r["pool_qualifying"] for r in sub)
        direction = "逆向" if w < nat[len(nat) // 2] else "顺向"
        print(f"  {w:>6}±{tol} {direction:>6} {str(t_ok) + '/' + str(len(sub)):>10} "
              f"{str(o_ok) + '/' + str(len(sub)):>10} {q[len(q)//2]:>16}")
        out[str(w)] = {"tool": t_ok, "ours": o_ok, "n": len(sub), "direction": direction}
    b = sum(1 for r in R if r["ours_ok"] and not r["tool_ok"])
    c = sum(1 for r in R if r["tool_ok"] and not r["ours_ok"])
    n = b + c
    p = (sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n * 2) if n else 1.0
    print(f"\n  合计: 工具 {sum(1 for r in R if r['tool_ok'])}/{len(R)}   "
          f"筛选 {sum(1 for r in R if r['ours_ok'])}/{len(R)}   McNemar {b}-{c}, p={p:.2e}")
    json.dump({"n": len(R), "natural_tm_median": nat[len(nat) // 2], "by_request": out,
               "mcnemar": {"b": b, "c": c, "p": p}, "rows": R},
              open(os.path.join(HERE, "tm_bench.json"), "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote rna_harness/runs/ceiling/tm_bench.json")


if __name__ == "__main__":
    main()
