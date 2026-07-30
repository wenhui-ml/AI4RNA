#!/usr/bin/env python3
"""A requirement the standard tool has NO FORMULATION for: a SET, not a sequence.

WHERE THIS COMES FROM. Two pre-registered predictions have now failed (strictness, forbidden motifs) and one
succeeded hugely (composition), and together they give a rule that can be applied before spending:

    a wrapper pays exactly when satisfying the requirement FORCES A LOSS in the solver's own objective.
      implied by the objective    umfe/pfe        18/18 vs 18/18      no gap
      free inside a level set     banned 4-mer   106/108 vs 108/108   no gap
      costs objective value       GC 30-50%        3/72 vs 68/72      huge gap

An ORTHOGONAL SET is the sharpest remaining case on both axes at once:

  * DesiRNA cannot express it. Its input file is one structure and one sequence restraint; there is no way to
    say "and these N sequences must differ from each other". This is not soft-versus-hard -- the concept is
    absent from the interface.
  * It must cost objective value. Forcing N sequences to be mutually dissimilar drives most of them off the
    optimum by construction, so no per-sequence steer can produce it.
  * It is a real requirement: orthogonal riboregulator libraries, molecular barcodes, and any multiplexed
    assay need sequences that fold the same way and do not talk to each other.

FAIR BY CONSTRUCTION. Both arms consume the SAME N runs of the SAME solver at the SAME seconds. Arm A takes
what the tool returns from each run -- exactly what a user gets by running it N times with different seeds.
Arm B greedily reads the pools those same runs already produced. No extra search, no extra compute.

JUDGED EXACTLY: every member must fold to the target (ViennaRNA MFE), every pair must be at most MAX_ID
identical, and no pair may form a duplex stronger than MAX_DUPLEX kcal/mol (RNA.duplexfold).
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
    "hairpin-10": "((((((((((......))))))))))",
    "two-hairpins": "((((....))))....((((....))))",
    "junction-2": "(((((..((((....))))..((((....))))..)))))",
    "junction-3": "((((.(((....))).(((....))).(((....))).))))",
    "internal-loop": "((((..((((....))))..))))",
}
WANT = 8               # how many mutually orthogonal sequences the request asks for
MAX_ID = 0.60          # pairwise identity ceiling
MAX_DUPLEX = -12.0     # a pair may not form a duplex stronger than this (kcal/mol)


def _ident(a, b):
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def _compatible(s, chosen):
    import RNA
    for t in chosen:
        if _ident(s, t) > MAX_ID:
            return False
        if RNA.duplexfold(s, t).energy < MAX_DUPLEX:
            return False
    return True


def run_one(shape, target, timelimit, n_runs, gc_band=None):
    from rna_harness.generators.desirna import DesiRNAGenerator
    from rna_harness.task import DesignTask, OracleConfig
    import RNA
    oc = OracleConfig(param_set="turner2004", temperature=37.0, dangles=2)
    t0 = time.time()
    tool_picks, pool, seen = [], [], set()
    for seed in range(n_runs):
        res = DesiRNAGenerator(replicas=10, timelimit_s=timelimit,
                               results_number=50).design(
            DesignTask(task_id="ortho", target=target, oracle=oc), seed=seed)
        if res.best_sequence:
            tool_picks.append(res.best_sequence)          # what a user gets from run `seed`
        for s in ([res.best_sequence] if res.best_sequence else []) + list(res.candidates or []):
            if s and len(s) == len(target) and s not in seen:
                seen.add(s)
                pool.append(s)
    wall = time.time() - t0

    folds = [s for s in pool if RNA.fold(s)[0] == target]
    if gc_band is not None:
        # THE CONTROL. Three wins so far all deliver lower-GC sequences than the tool does, so "we have one
        # gap wearing three costumes" is a live reading. Restricting BOTH arms to the same narrow GC band
        # removes composition as an explanation: if the tool still cannot assemble the set from sequences of
        # the same composition as ours, orthogonality is a separate gap. If it now can, it was GC all along.
        lo, hi = gc_band
        def _gc(x):
            return 100.0 * sum(1 for ch in x if ch in "GC") / len(x)
        folds = [x for x in folds if lo <= _gc(x) <= hi]
        tool_picks = [x for x in tool_picks if lo <= _gc(x) <= hi]

    def greedy(cands):
        out = []
        for s in cands:
            if len(out) >= WANT:
                break
            if _compatible(s, out):
                out.append(s)
        return out

    tool_set = greedy([s for s in tool_picks if RNA.fold(s)[0] == target])
    tool_in_band = len(tool_picks)
    ours_set = greedy(folds)
    return {"shape": shape, "n": len(target), "n_runs": n_runs, "wall_s": round(wall, 1),
            "pool": len(pool), "pool_folding": len(folds),
            "tool_returned": tool_in_band, "tool_set": len(tool_set), "tool_ok": len(tool_set) >= WANT,
            "gc_band": gc_band,
            "ours_set": len(ours_set), "ours_ok": len(ours_set) >= WANT,
            "ours_sequences": ours_set}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timelimit", type=int, default=15)
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--gc-band", default="", help="e.g. 45,55 -- restrict BOTH arms to this GC band")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    band = tuple(float(x) for x in a.gc_band.split(",")) if a.gc_band else None
    print(f"orthogonal-set benchmark: {len(SHAPES)} shapes, asking for {WANT} sequences each that all fold, "
          f"are pairwise <= {MAX_ID:.0%} identical and form no duplex below {MAX_DUPLEX} kcal/mol")
    print(f"  both arms consume the same {a.runs} runs x {a.timelimit}s of the same solver\n", flush=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    R, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_one, sh, t, a.timelimit, a.runs, band) for sh, t in SHAPES.items()]
        for n, f in enumerate(as_completed(futs), 1):
            R.append(f.result())
            print(f"    {n}/{len(futs)}  {(time.time()-t0)/60:.1f} min", flush=True)

    print(f"\n=== {len(R)} shapes, same solver, same {a.runs} runs, same pools ===")
    print(f"  {'shape':<14} {'池':>5} {'折叠对':>7} {'工具给的':>9} {'工具集合':>9} {'我们集合':>9}")
    for r in sorted(R, key=lambda r: r["shape"]):
        print(f"  {r['shape']:<14} {r['pool']:>5} {r['pool_folding']:>7} {r['tool_returned']:>9} "
              f"{str(r['tool_set']) + '/' + str(WANT):>9} {str(r['ours_set']) + '/' + str(WANT):>9}")
    print(f"\n  工具凑齐 {WANT} 条正交序列: {sum(1 for r in R if r['tool_ok'])}/{len(R)}")
    print(f"  筛选其池凑齐            : {sum(1 for r in R if r['ours_ok'])}/{len(R)}")
    json.dump({"want": WANT, "max_id": MAX_ID, "max_duplex": MAX_DUPLEX,
               "tool_ok": sum(1 for r in R if r["tool_ok"]), "ours_ok": sum(1 for r in R if r["ours_ok"]),
               "n": len(R), "rows": R},
              open(os.path.join(HERE, f"orthoset_bench{('_' + a.tag) if a.tag else ''}.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"\n  wrote rna_harness/runs/ceiling/orthoset_bench.json")


if __name__ == "__main__":
    main()
