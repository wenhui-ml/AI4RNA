#!/usr/bin/env python3
"""A problem class the standard tool cannot deliver: fold correctly AND meet a composition target.

THE POINT OF THIS FILE. Everything else measured here compares how HARD a target is. This compares what a
request can ASK FOR. DesiRNA takes one target and optimises folding; composition is a soft steer through
-acgu_content, and a steer is not a guarantee. So "fold to this shape AND land within +/-5 of 50% GC" is a
well-posed design problem -- not an ill-posed or adversarial one -- that its interface does not deliver.

FAIR BY CONSTRUCTION, and this is what makes it evidence rather than a strawman:
  * the composition flag is invoked the way its own parser documents (-acgu on -acgu_content A,C,G,U).
    An earlier measurement of ours predates finding that -acgu_content alone makes argparse EXIT, so that
    number was our bug; this one is not.
  * BOTH arms get the SAME solver, the SAME seconds and the SAME candidate pool. Arm B does not search
    harder or longer. The only difference is what is done with the output.
  * both arms are judged by the same exact oracle: ViennaRNA MFE == target, AND |GC - request| <= tolerance.
    A sequence that meets the composition and folds wrong is not an answer, and is not counted as one.

So a win here is not "we have a better search". It is "the tool found the answer and its own selection
discarded it, because it was selecting on a different question than the one asked".
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

SHAPES = {
    "hairpin-6": "((((((....))))))",
    "hairpin-10": "((((((((((......))))))))))",
    "two-hairpins": "((((....))))....((((....))))",
    "junction-2": "(((((..((((....))))..((((....))))..)))))",
    "junction-3": "((((.(((....))).(((....))).(((....))).))))",
    "internal-loop": "((((..((((....))))..))))",
}
REQUESTS = [(30, 5), (40, 5), (50, 5), (60, 5)]        # GC percent, tolerance


def _gc(s):
    return 100.0 * sum(1 for c in s if c in "GC") / len(s)


def run_one(shape, target, want, tol, timelimit, seed):
    from rna_harness.generators.desirna import DesiRNAGenerator
    from rna_harness.task import DesignTask, OracleConfig
    import RNA
    a = [want, want, 100 - 2 * want, 0] if False else None            # placeholder, see below
    # A,C,G,U as integers summing to 100: split the requested GC evenly over G and C, the rest over A and U.
    g = c = want // 2
    g += want - (g + c)
    a_ = u = (100 - want) // 2
    a_ += (100 - want) - (a_ + u)
    os.environ["DESIRNA_ARGS"] = f"-acgu on -acgu_content {a_},{c},{g},{u}"
    t0 = time.time()
    res = DesiRNAGenerator(replicas=10, timelimit_s=timelimit, results_number=50).design(
        DesignTask(task_id="gc", target=target,
                   oracle=OracleConfig(param_set="turner2004", temperature=37.0, dangles=2)), seed=seed)
    os.environ.pop("DESIRNA_ARGS", None)
    wall = time.time() - t0
    pool, seen = [], set()
    for s in ([res.best_sequence] if res.best_sequence else []) + list(res.candidates or []):
        if s and len(s) == len(target) and s not in seen:
            seen.add(s)
            pool.append(s)

    def qualifies(s):
        return RNA.fold(s)[0] == target and abs(_gc(s) - want) <= tol

    tool = res.best_sequence
    tool_ok = bool(tool) and qualifies(tool)
    ours = next((s for s in pool if qualifies(s)), None)
    return {"shape": shape, "n": len(target), "want": want, "tol": tol, "seed": seed,
            "wall_s": round(wall, 1), "pool": len(pool),
            "tool_sequence": tool, "tool_gc": round(_gc(tool), 1) if tool else None,
            "tool_folds": bool(tool) and RNA.fold(tool)[0] == target, "tool_ok": tool_ok,
            "ours_sequence": ours, "ours_gc": round(_gc(ours), 1) if ours else None, "ours_ok": bool(ours),
            "pool_qualifying": sum(1 for s in pool if qualifies(s))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timelimit", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    jobs = [(sh, t, w, tol, a.timelimit, s)
            for sh, t in SHAPES.items() for w, tol in REQUESTS for s in range(a.seeds)]
    print(f"composition benchmark: {len(SHAPES)} shapes x {len(REQUESTS)} requests x {a.seeds} seeds "
          f"= {len(jobs)} runs, {a.timelimit}s each\n", flush=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    R, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, f in enumerate(as_completed([ex.submit(run_one, *j) for j in jobs]), 1):
            R.append(f.result())
            if n % 10 == 0:
                print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)

    tool = sum(1 for r in R if r["tool_ok"])
    ours = sum(1 for r in R if r["ours_ok"])
    both_fold = sum(1 for r in R if r["tool_folds"])
    print(f"\n=== {len(R)} runs, same solver, same seconds, same pool ===")
    print(f"  DesiRNA's own answer  folds AND meets the composition : {tool}/{len(R)}")
    print(f"  screening its pool    folds AND meets the composition : {ours}/{len(R)}")
    print(f"  (DesiRNA's own answer folds correctly, ignoring GC)   : {both_fold}/{len(R)}")
    disc = [r for r in R if r["ours_ok"] and not r["tool_ok"]]
    print(f"\n  runs where the answer WAS IN ITS POOL and it returned something else: {len(disc)}/{len(R)}")
    print(f"\n  {'shape':<14} {'want':>5} {'tool GC':>8} {'ok':>4} {'ours GC':>8} {'ok':>4} {'pool ok':>8}")
    for r in sorted(R, key=lambda r: (r["shape"], r["want"], r["seed"]))[:24]:
        print(f"  {r['shape']:<14} {r['want']:>5} {str(r['tool_gc']):>8} {str(r['tool_ok']):>4} "
              f"{str(r['ours_gc']):>8} {str(r['ours_ok']):>4} {r['pool_qualifying']:>8}")
    json.dump({"n": len(R), "tool_ok": tool, "ours_ok": ours, "tool_folds": both_fold,
               "discarded_own_answer": len(disc), "rows": R},
              open(os.path.join(HERE, "gc_bench.json"), "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote rna_harness/runs/ceiling/gc_bench.json")


if __name__ == "__main__":
    main()
