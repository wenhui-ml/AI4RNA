#!/usr/bin/env python3
"""Is every gap we have found just LOW GC in disguise? A requirement that costs stability WITHOUT touching GC.

THE HYPOTHESIS BEING TESTED, against ourselves. Three benchmarks now show a large gap, and in all three the
sequences we deliver are markedly lower in GC than the ones the tool returns (median 50.0% against 67.9%,
50.0% against 73.1%, and the Tm gap collapses once the request stops asking for less stability). If every
win is the same GC-stability trade wearing a different costume, then this project has ONE capability, not
four, and it should say so.

THE DISCRIMINATING REQUIREMENT: stay close to a GIVEN sequence.

    "here is a natural sequence that folds to the wrong thing; give me one that folds to MY target and is
     still at least X% identical to it"

That is minimal-mutation redesign -- keeping a natural sequence's identity while fixing its fold -- and it is
an ordinary request in RNA engineering. It costs objective value by construction, because the optimum is
almost never inside a small neighbourhood of an arbitrary sequence. And it says NOTHING about composition:
the reference sequences here are drawn across the GC range, so a low-GC answer is no help unless the
reference is low-GC too.

WHAT THE TOOL HAS: `>seed_seq` sets a STARTING POINT, not a constraint -- the search is free to walk away
from it, and does. There is no way to bound the distance. (`>seq_restr` can pin individual positions, but
pinning is not the same requirement: it fixes WHICH positions may change, while this bounds HOW MANY.)

PREDICTION, declared before the run: a large gap, because the requirement costs objective value. If instead
there is no gap, the honest conclusion is that our capability is the GC trade alone.
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
KEEP = [0.60, 0.70]          # the answer must be at least this identical to the reference


def _ident(a, b):
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def reference_for(target, seed):
    """A reference sequence drawn deterministically ACROSS the GC range, so the test is not secretly a GC
    test: with GC 30/50/70% references, a low-GC answer only helps for the low-GC reference."""
    import random
    rng = random.Random(1000 + seed)
    gc_frac = (0.30, 0.50, 0.70)[seed % 3]
    out = []
    for _ in range(len(target)):
        out.append(rng.choice("GC") if rng.random() < gc_frac else rng.choice("AU"))
    return "".join(out)


def run_one(shape, target, keep, timelimit, seed):
    from rna_harness.generators.desirna import DesiRNAGenerator
    from rna_harness.task import DesignTask, OracleConfig
    import RNA
    ref = reference_for(target, seed)
    t0 = time.time()
    os.environ["DESIRNA_SEED_SEQ"] = ref            # honoured by the adapter if it supports >seed_seq
    res = DesiRNAGenerator(replicas=10, timelimit_s=timelimit, results_number=50).design(
        DesignTask(task_id="near", target=target,
                   oracle=OracleConfig(param_set="turner2004", temperature=37.0, dangles=2),
                   meta={"seed_seq": ref}), seed=seed)
    os.environ.pop("DESIRNA_SEED_SEQ", None)
    pool, seen = [], set()
    for s in ([res.best_sequence] if res.best_sequence else []) + list(res.candidates or []):
        if s and len(s) == len(target) and s not in seen:
            seen.add(s)
            pool.append(s)

    def ok(s):
        return RNA.fold(s)[0] == target and _ident(s, ref) >= keep

    tool = res.best_sequence
    ours = max((s for s in pool if ok(s)), key=lambda s: _ident(s, ref), default=None)
    best_id = max((_ident(s, ref) for s in pool if RNA.fold(s)[0] == target), default=None)
    return {"shape": shape, "n": len(target), "keep": keep, "seed": seed,
            "ref_gc": round(100 * sum(1 for c in ref if c in "GC") / len(ref), 1),
            "wall_s": round(time.time() - t0, 1), "pool": len(pool),
            "tool_ident": round(_ident(tool, ref), 3) if tool else None,
            "tool_folds": bool(tool) and RNA.fold(tool)[0] == target,
            "tool_ok": bool(tool) and ok(tool),
            "ours_ident": round(_ident(ours, ref), 3) if ours else None, "ours_ok": bool(ours),
            "best_ident_in_pool": round(best_id, 3) if best_id is not None else None,
            "pool_qualifying": sum(1 for s in pool if ok(s))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timelimit", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    jobs = [(sh, t, k, a.timelimit, s)
            for sh, t in SHAPES.items() for k in KEEP for s in range(a.seeds)]
    print(f"stay-near-a-reference benchmark: {len(SHAPES)} shapes x {len(KEEP)} thresholds x {a.seeds} "
          f"references = {len(jobs)} runs\n", flush=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    R, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, f in enumerate(as_completed([ex.submit(run_one, *j) for j in jobs]), 1):
            R.append(f.result())
            if n % 10 == 0:
                print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)

    b = sum(1 for r in R if r["ours_ok"] and not r["tool_ok"])
    c = sum(1 for r in R if r["tool_ok"] and not r["ours_ok"])
    n = b + c
    p = (sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n * 2) if n else 1.0
    print(f"\n=== {len(R)} runs, same solver, same seconds, same pool ===")
    print(f"  工具自己的答案  折叠对 AND 相似度达标 : {sum(1 for r in R if r['tool_ok'])}/{len(R)}")
    print(f"  筛选它的池      折叠对 AND 相似度达标 : {sum(1 for r in R if r['ours_ok'])}/{len(R)}")
    print(f"  McNemar {b}-{c}, p={p:.2e}")
    ti = [r["tool_ident"] for r in R if r["tool_ident"] is not None]
    bi = [r["best_ident_in_pool"] for r in R if r["best_ident_in_pool"] is not None]
    ti.sort(); bi.sort()
    print(f"\n  工具答案与参考序列的相同度  中位 {ti[len(ti)//2]:.0%}")
    print(f"  它自己池中能达到的最高相同度 中位 {bi[len(bi)//2]:.0%}")
    print(f"\n  {'门槛':>6} {'工具':>8} {'筛选':>8}")
    out = {}
    for k in KEEP:
        sub = [r for r in R if r["keep"] == k]
        print(f"  {k:>6.0%} {str(sum(1 for r in sub if r['tool_ok'])) + '/' + str(len(sub)):>8} "
              f"{str(sum(1 for r in sub if r['ours_ok'])) + '/' + str(len(sub)):>8}")
        out[str(k)] = {"tool": sum(1 for r in sub if r["tool_ok"]),
                       "ours": sum(1 for r in sub if r["ours_ok"]), "n": len(sub)}
    print(f"\n  按参考序列的 GC 分层(检验这是不是又一次 GC 缺口):")
    for g in (30.0, 50.0, 70.0):
        sub = [r for r in R if abs(r["ref_gc"] - g) < 12]
        if sub:
            print(f"    参考 GC≈{g:.0f}%  工具 {sum(1 for r in sub if r['tool_ok'])}/{len(sub)}   "
                  f"筛选 {sum(1 for r in sub if r['ours_ok'])}/{len(sub)}")
    json.dump({"n": len(R), "tool_ok": sum(1 for r in R if r["tool_ok"]),
               "ours_ok": sum(1 for r in R if r["ours_ok"]),
               "mcnemar": {"b": b, "c": c, "p": p}, "by_threshold": out, "rows": R},
              open(os.path.join(HERE, "nearby_bench.json"), "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote rna_harness/runs/ceiling/nearby_bench.json")


if __name__ == "__main__":
    main()
