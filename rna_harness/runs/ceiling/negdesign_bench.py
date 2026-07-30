#!/usr/bin/env python3
"""Is there a SECOND gap, or is everything we win just low GC? A requirement about the LANDSCAPE, not the bases.

WHY THIS REQUIREMENT. Three confirmed wins all deliver lower-GC sequences than the tool does, and the control
(orthogonal sets restricted to GC 45-58%) showed the tool contributing 0-1 sequences in the band at all. So
"one gap in three costumes" is the current honest reading. Settling it needs a requirement that costs the
solver objective value WITHOUT being a composition requirement.

Negative design is that requirement. "Fold to the target AND keep this NAMED alternative structure at least
G kcal/mol above the minimum" is about DISCRIMINATION between two folds, not about stability: raising GC
stabilises the target and the competitor alike, so composition is close to neutral here. If a gap appears,
this project has two capabilities. If it does not, the one-gap reading survives another attempt to break it.

THE TOOL HAS A MECHANISM FOR THIS, and it gets it: `>alt_sec_struct` names the competitor and `-nd on` turns
on negative design. As with composition, having a mechanism and delivering are different questions, and only
the second one is measured here.

THE COMPETITOR IS REAL, NOT INVENTED: for each shape it is the first SUBOPTIMAL structure of the tool's own
seed-0 answer -- the fold that actually competes with the target for that solver's output. Fixed once per
shape and given identically to both arms.

JUDGED EXACTLY: MFE == target, and eval_structure(competitor) - MFE >= gap.
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
GAPS = [2.0, 4.0]          # kcal/mol the named competitor must sit above the minimum


def competitor_for(target, timelimit):
    """The fold that actually competes: the first suboptimal structure of the tool's own seed-0 answer."""
    from rna_harness.generators.desirna import DesiRNAGenerator
    from rna_harness.task import DesignTask, OracleConfig
    import RNA
    res = DesiRNAGenerator(replicas=10, timelimit_s=timelimit, results_number=10).design(
        DesignTask(task_id="comp", target=target,
                   oracle=OracleConfig(param_set="turner2004", temperature=37.0, dangles=2)), seed=0)
    seq = res.best_sequence
    if not seq:
        return None
    from rna_harness.cognitive.certify import _bp_distance
    fc = RNA.fold_compound(seq)
    fc.mfe()
    n_pairs = target.count("(")
    far = max(4, n_pairs // 3)
    # STRUCTURALLY DISTINCT, not merely next in energy. The first suboptimal structure is by definition the
    # nearest neighbour -- for every shape here it was the target with one outermost pair breathing open --
    # and demanding that a one-pair breath sit 3-6 kcal/mol above the minimum is impossible for anyone. Both
    # arms scored 0/30 on that version, which measured the requirement, not the tool.
    for s in sorted(fc.subopt(2000), key=lambda x: x.energy):
        if s.structure.count("(") > 0 and _bp_distance(s.structure, target) >= far:
            return s.structure
    return None


def run_one(shape, target, competitor, gap, timelimit, seed):
    from rna_harness.generators.desirna import DesiRNAGenerator
    from rna_harness.task import DesignTask, OracleConfig
    import RNA
    os.environ["DESIRNA_ARGS"] = "-nd on"
    t0 = time.time()
    res = DesiRNAGenerator(replicas=10, timelimit_s=timelimit, results_number=50).design(
        DesignTask(task_id="neg", target=target,
                   oracle=OracleConfig(param_set="turner2004", temperature=37.0, dangles=2),
                   meta={"alt_sec_struct": [competitor]}), seed=seed)
    os.environ.pop("DESIRNA_ARGS", None)
    pool, seen = [], set()
    for s in ([res.best_sequence] if res.best_sequence else []) + list(res.candidates or []):
        if s and len(s) == len(target) and s not in seen:
            seen.add(s)
            pool.append(s)

    def margin(s):
        fc = RNA.fold_compound(s)
        mfe = fc.mfe()[1]
        return fc.eval_structure(competitor) - mfe

    def ok(s):
        return RNA.fold(s)[0] == target and margin(s) >= gap

    tool = res.best_sequence
    ours = next((s for s in pool if ok(s)), None)
    def _gc(x):
        return 100.0 * sum(1 for ch in x if ch in "GC") / len(x)
    return {"shape": shape, "gap": gap, "seed": seed, "wall_s": round(time.time() - t0, 1),
            "pool": len(pool), "competitor": competitor,
            "tool_margin": round(margin(tool), 2) if tool else None,
            "tool_folds": bool(tool) and RNA.fold(tool)[0] == target,
            "tool_ok": bool(tool) and ok(tool), "tool_gc": round(_gc(tool), 1) if tool else None,
            "ours_ok": bool(ours), "ours_margin": round(margin(ours), 2) if ours else None,
            "ours_gc": round(_gc(ours), 1) if ours else None,
            "pool_qualifying": sum(1 for s in pool if ok(s))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timelimit", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    print("choosing a STRUCTURALLY DISTINCT competitor for each shape (bp distance >= n_pairs/3)...",
          flush=True)
    from rna_harness.cognitive.certify import _bp_distance
    comps = {sh: competitor_for(t, a.timelimit) for sh, t in SHAPES.items()}
    for sh, c in comps.items():
        print(f"  {sh:<14} {c}   bp距离={_bp_distance(c, SHAPES[sh]) if c else '-'}")
    jobs = [(sh, t, comps[sh], g, a.timelimit, s)
            for sh, t in SHAPES.items() if comps[sh]
            for g in GAPS for s in range(a.seeds)]
    print(f"\nnegative-design benchmark: {len(jobs)} runs\n", flush=True)
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
    print(f"\n=== {len(R)} runs, same solver, same seconds, same pool, tool given -nd on + alt_sec_struct ===")
    print(f"  工具自己的答案  折叠对 AND 间隔达标 : {sum(1 for r in R if r['tool_ok'])}/{len(R)}")
    print(f"  筛选它的池      折叠对 AND 间隔达标 : {sum(1 for r in R if r['ours_ok'])}/{len(R)}")
    print(f"  McNemar {b}-{c}, p={p:.2e}")
    print(f"\n  {'间隔要求':>8} {'工具':>8} {'筛选':>8}")
    for g in GAPS:
        sub = [r for r in R if r["gap"] == g]
        print(f"  {g:>6.0f}   {str(sum(1 for r in sub if r['tool_ok'])) + '/' + str(len(sub)):>8} "
              f"{str(sum(1 for r in sub if r['ours_ok'])) + '/' + str(len(sub)):>8}")
    tg = [r["tool_gc"] for r in R if r["tool_gc"]]
    og = [r["ours_gc"] for r in R if r["ours_gc"]]
    print(f"\n  这是不是又一次 GC 缺口?  工具 GC 中位 {sorted(tg)[len(tg)//2]:.1f}%   "
          f"我们 GC 中位 {sorted(og)[len(og)//2]:.1f}%" if og else "\n  (我们没有合格解,无法比较 GC)")
    json.dump({"n": len(R), "tool_ok": sum(1 for r in R if r["tool_ok"]),
               "ours_ok": sum(1 for r in R if r["ours_ok"]), "mcnemar": {"b": b, "c": c, "p": p},
               "competitors": comps, "rows": R},
              open(os.path.join(HERE, "negdesign_bench.json"), "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote rna_harness/runs/ceiling/negdesign_bench.json")


if __name__ == "__main__":
    main()
