#!/usr/bin/env python3
"""Stage B: does more solver time make our designs better, worse, or neither?

RUNS IN THE RibonanzaNet venv. Stage A ran the solver at 5 / 15 / 60 / 300 s with the tiers kept
independent, so this is the version of the tier question that `pool_rerank.py`'s cross-tier dedup made
unanswerable.

The prediction under test was fixed before stage A ran, and it is now paired with a measured fact rather
than a guess. `criterion_vs_tube.py` found that our thermodynamic panel predicts wet-lab behaviour at
within-target rho +0.263 while RibonanzaNet manages +0.421 on the same 4,560 molecules. Our solver spends
its seconds driving the weaker of those two. If that is what limits us, extra seconds buy nothing on the
metric we are graded by, and may cost.

    S1  median quality, 300 s tier minus 5 s tier, paired within target, Wilcoxon, TWO-SIDED
    S2  best candidate in the tier, same test
    S3  the monotonicity across all four tiers

COST: $0. Local CPU.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pool_rerank_score import _score_chunk                                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default=os.path.join(HERE, "search_convergence_pools.json"))
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--out", default=os.path.join(HERE, "search_convergence_score.json"))
    a = ap.parse_args()

    from m2r_pairwise import wilcoxon

    P = json.load(open(a.pools))
    T, tiers = P["targets"], P["tiers"]
    jobs = []
    for r in P["rows"]:
        for i, s in enumerate(r["pool"]):
            jobs.append(((r["puzzle"], r["tier"], i), s, T[r["puzzle"]]["target"]))
    print(f"{len(jobs)} 条候选,{len(tiers)} 档 {tiers},{a.workers} 进程\n", flush=True)

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

    per = {}
    for r in P["rows"]:
        v = [sc[(r["puzzle"], r["tier"], i)] for i in range(len(r["pool"]))
             if (r["puzzle"], r["tier"], i) in sc]
        if v:
            per.setdefault(r["puzzle"], {})[r["tier"]] = {
                "median": st.median(v), "max": max(v), "n": len(v)}

    print(f"\n=== S3 各档时限下的质量 ===")
    print(f"  {'时限':>6s} {'靶标':>4s} {'候选数中位':>10s} {'池中位的中位':>12s} {'池最好的中位':>12s}")
    agg = {}
    for tl in tiers:
        ps = [p for p in per if tl in per[p]]
        med = st.median([per[p][tl]["median"] for p in ps])
        mx = st.median([per[p][tl]["max"] for p in ps])
        nn = st.median([per[p][tl]["n"] for p in ps])
        agg[tl] = {"n_targets": len(ps), "median_of_medians": med, "median_of_max": mx, "n_cand": nn}
        print(f"  {tl:5d}s {len(ps):4d} {nn:10.0f} {med:12.3f} {mx:12.3f}")

    lo, hi = tiers[0], tiers[-1]
    out = {"tiers": tiers, "per_tier": agg}
    for label, key in (("S1 池中位", "median"), ("S2 池最好", "max")):
        ps = [p for p in per if lo in per[p] and hi in per[p]]
        d = [per[p][hi][key] - per[p][lo][key] for p in ps]
        w, pv, _ = wilcoxon(d, "two-sided")
        up = sum(1 for x in d if x > 0)
        print(f"\n  {label}:{hi}s − {lo}s 配对差中位 {st.median(d):+.3f}({len(ps)} 个靶标),"
              f"变好 {up},变差 {sum(1 for x in d if x<0)},p = {'n/a' if pv is None else f'{pv:.3g}'}")
        out[key] = {"n_targets": len(ps), "median_diff": st.median(d), "p": pv,
                    "better": up, "worse": sum(1 for x in d if x < 0)}

    md = out["median"]["median_diff"]
    pv = out["median"]["p"]
    sig = pv is not None and pv < 0.05
    out["verdict"] = ("搜索未收敛,加时间是便宜的真实收益" if sig and md > 0.02 else
                      "加时间反而变差 —— 求解器在过度优化一个我们并不据以受评的目标" if sig and md < -0.02 else
                      "加时间无差别 —— 搜索已收敛,天花板是目标函数本身")
    print(f"\n  判读:{out['verdict']}")
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
