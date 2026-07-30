#!/usr/bin/env python3
"""A second problem class the standard tool cannot deliver: FORBID a subsequence, as a hard rule.

WHY THIS ONE. The composition result (3/72 against 68/72) needed a second instance before it could be called
a class rather than a quirk, and the corrected M1 rule says where to look: a requirement that pulls AGAINST
the solver's objective. Forbidding G-runs does exactly that -- the objective drives GC up (unconstrained
median 67.9%, up to 93.8%), and every G-run removed is stability given away.

IT IS ALSO A REAL REQUIREMENT, not a constructed one. G-quadruplex-forming runs, homopolymer runs that break
solid-phase synthesis, and restriction sites in the DNA template are all things a working RNA engineer must
exclude, and excluding them is not negotiable -- "mostly absent" is not a usable answer.

WHAT THE TOOL HAS, and it is a soft penalty, verified in its source: `-motifs KEY,VALUE` adds VALUE to the
scoring function when the motif is present (sequence_utils.py::score_motifs -> scoring_function += bonus).
Present-or-absent, once, added to an energy-scale objective. Nothing forbids it.

FAIR BY CONSTRUCTION:
  * the tool is given ITS OWN mechanism, correctly invoked, at a penalty large enough to dominate the energy
    terms -- and a second arm at a 10x larger penalty, so "the penalty was too small" cannot explain a loss.
  * both arms get the SAME solver, the SAME seconds and the SAME candidate pool.
  * both are judged by the same exact rule: ViennaRNA MFE == target AND the forbidden motif does not occur.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
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
# (name, the regex a sequence must NOT match, what the tool would be told to penalise)
FORBIDDEN = [
    ("no-GGGG", r"GGGG", "GGGG"),                     # G-quadruplex risk
    ("no-CCCC", r"CCCC", "CCCC"),                     # the partner run: same pressure, other strand
    ("no-homopolymer-5", r"(A{5}|C{5}|G{5}|U{5})", "AAAAA,GGGGG,CCCCC,UUUUU"),   # synthesis failure
]
PENALTIES = [10.0, 100.0]


def run_one(shape, target, rule, pattern, motif_arg, penalty, timelimit, seed):
    from rna_harness.generators.desirna import DesiRNAGenerator
    from rna_harness.task import DesignTask, OracleConfig
    import RNA
    rx = re.compile(pattern)
    keys = motif_arg.split(",")
    os.environ["DESIRNA_ARGS"] = "-motifs " + ",".join(f"{k},{penalty}" for k in keys)
    t0 = time.time()
    res = DesiRNAGenerator(replicas=10, timelimit_s=timelimit, results_number=50).design(
        DesignTask(task_id="forbid", target=target,
                   oracle=OracleConfig(param_set="turner2004", temperature=37.0, dangles=2)), seed=seed)
    os.environ.pop("DESIRNA_ARGS", None)
    pool, seen = [], set()
    for s in ([res.best_sequence] if res.best_sequence else []) + list(res.candidates or []):
        if s and len(s) == len(target) and s not in seen:
            seen.add(s)
            pool.append(s)

    def ok(s):
        return RNA.fold(s)[0] == target and not rx.search(s)

    tool = res.best_sequence
    ours = next((s for s in pool if ok(s)), None)
    return {"shape": shape, "n": len(target), "rule": rule, "penalty": penalty, "seed": seed,
            "wall_s": round(time.time() - t0, 1), "pool": len(pool),
            "tool_sequence": tool, "tool_folds": bool(tool) and RNA.fold(tool)[0] == target,
            "tool_clean": bool(tool) and not rx.search(tool), "tool_ok": bool(tool) and ok(tool),
            "ours_sequence": ours, "ours_ok": bool(ours),
            "pool_qualifying": sum(1 for s in pool if ok(s))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timelimit", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    jobs = [(sh, t, rule, pat, marg, pen, a.timelimit, s)
            for sh, t in SHAPES.items() for rule, pat, marg in FORBIDDEN
            for pen in PENALTIES for s in range(a.seeds)]
    print(f"forbidden-motif benchmark: {len(SHAPES)} shapes x {len(FORBIDDEN)} rules x "
          f"{len(PENALTIES)} penalties x {a.seeds} seeds = {len(jobs)} runs\n", flush=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    R, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, f in enumerate(as_completed([ex.submit(run_one, *j) for j in jobs]), 1):
            R.append(f.result())
            if n % 15 == 0:
                print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)

    def mc(sub):
        b = sum(1 for r in sub if r["ours_ok"] and not r["tool_ok"])
        c = sum(1 for r in sub if r["tool_ok"] and not r["ours_ok"])
        n = b + c
        p = (sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n * 2) if n else 1.0
        return b, c, p

    b, c, p = mc(R)
    print(f"\n=== {len(R)} runs, same solver, same seconds, same pool ===")
    print(f"  the tool's own answer   folds AND obeys the ban : {sum(1 for r in R if r['tool_ok'])}/{len(R)}")
    print(f"  screening its pool      folds AND obeys the ban : {sum(1 for r in R if r['ours_ok'])}/{len(R)}")
    print(f"  (the tool's answer folds correctly, ignoring the ban): "
          f"{sum(1 for r in R if r['tool_folds'])}/{len(R)}")
    print(f"  McNemar exact, paired: {b}-{c}, p={p:.2e}")
    print(f"\n  {'rule':<18} {'penalty':>8} {'tool':>7} {'ours':>7}")
    out = {}
    for rule, _, _ in FORBIDDEN:
        for pen in PENALTIES:
            sub = [r for r in R if r["rule"] == rule and r["penalty"] == pen]
            print(f"  {rule:<18} {pen:>8.0f} {sum(1 for r in sub if r['tool_ok']):>3}/{len(sub):<3} "
                  f"{sum(1 for r in sub if r['ours_ok']):>3}/{len(sub):<3}")
            out[f"{rule}@{pen:.0f}"] = {"tool": sum(1 for r in sub if r["tool_ok"]),
                                        "ours": sum(1 for r in sub if r["ours_ok"]), "n": len(sub)}
    dry = [r for r in R if not r["ours_ok"]]
    print(f"\n  我们也失败(池中 0 合格): {len(dry)}/{len(R)}"
          + (f"   {sorted({(r['shape'], r['rule']) for r in dry})}" if dry else ""))
    json.dump({"n": len(R), "tool_ok": sum(1 for r in R if r["tool_ok"]),
               "ours_ok": sum(1 for r in R if r["ours_ok"]),
               "tool_folds": sum(1 for r in R if r["tool_folds"]),
               "mcnemar": {"b": b, "c": c, "p": p}, "by_cell": out, "rows": R},
              open(os.path.join(HERE, "forbid_bench.json"), "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote rna_harness/runs/ceiling/forbid_bench.json")


if __name__ == "__main__":
    main()
