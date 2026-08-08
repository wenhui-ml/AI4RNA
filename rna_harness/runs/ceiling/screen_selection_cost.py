#!/usr/bin/env python3
"""How much does the new screen need to ADJUDICATE before it can answer? The cost/quality curve.

RUNS IN THE RibonanzaNet venv.

THE DESIGN DECISION THIS SETTLES. The rescue that took coverage from 43/57 to 57/57 selected by adjudicating
the WHOLE pool and then ranking the survivors by (panel_k, panel_mean). The shipped `certify_design` does
something much cheaper: it walks the pool and returns the FIRST candidate that satisfies the criterion. The
difference is not cosmetic. At 240 nt one panel adjudication costs about 55 s, so ranking a 200-candidate
pool is roughly three hours for one target, against seconds for early exit. Goal 2 is "faster and cheaper";
a coverage gain that costs three hours per hard target is not obviously a gain at all.

So before changing the product, measure what the extra adjudications actually buy:

    first-pass      the first candidate in pool order clearing the screen -- what early exit would return
    best-of-N       the best by (panel_k, panel_mean) among the first N adjudicated, for N on a grid
    best-of-all     the rescue's rule, ranking every survivor -- the expensive end
    pool-best       the best candidate by RNet F1, an ORACLE bound no selector can beat, for scale

Judged by RibonanzaNet crossed-pair F1. The panel screens and ranks; RNet only judges, never enters the
loop, and cannot have been fitted to by a screen calibrated on other people's molecules.

    S1  Is best-of-all better than first-pass? Paired within target, Wilcoxon, one-sided greater.
        If it is NOT, the cheap rule is the right product and the expensive one was wasted effort.
    S2  Where does the curve flatten? The smallest N whose median is within 0.02 of best-of-all. That N is
        the adjudication budget worth paying for, and it converts directly into seconds per target.
    S3  What does that N cost in wall time, per target, at the measured per-adjudication cost by length.

COST: $0. Local CPU.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pool_rerank_score import _score_chunk                                      # noqa: E402

GRID = [1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200]


def adj_cost_s(n_nt):
    """Measured panel cost by length: 0.4 s at 48 nt, 3.5 s at 100 nt, 60 s at 240 nt."""
    return 3.5 * (n_nt / 100.0) ** 3.2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "rescue_notfound.json"))
    ap.add_argument("--thr", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--out", default=os.path.join(HERE, "screen_selection_cost.json"))
    a = ap.parse_args()

    from m2r_pairwise import wilcoxon

    S = json.load(open(a.src))
    T, rows = S["targets"], S["rows"]
    jobs = [((i,), r["seq"], T[r["puzzle"]]["target"]) for i, r in enumerate(rows)]
    print(f"{len(jobs)} 条候选打分,{a.workers} 进程\n", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    CH = max(1, len(jobs) // (a.workers * 8))
    chunks = [jobs[i:i + CH] for i in range(0, len(jobs), CH)]
    sc = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for res in ex.map(_score_chunk, chunks):
            for k, v in res:
                sc[k[0]] = v
    for i, r in enumerate(rows):
        r["f1c"] = sc.get(i)

    by = collections.defaultdict(list)
    for r in rows:                       # rows are in pool order per puzzle, preserved by ex.map
        if r.get("f1c") is not None:
            by[r["puzzle"]].append(r)

    def pick_best(cands):
        return max(cands, key=lambda r: (r["panel_k"], r["panel_mean"]))

    ps = sorted(by)
    first, best_all, oracle, curve = {}, {}, {}, {n: {} for n in GRID}
    for p in ps:
        pool = by[p]
        surv = [r for r in pool if r["panel_mean"] >= a.thr]
        oracle[p] = max((r["f1c"] for r in pool), default=0.0)
        first[p] = surv[0]["f1c"] if surv else 0.0
        best_all[p] = pick_best(surv)["f1c"] if surv else 0.0
        for n in GRID:
            head = [r for r in pool[:n] if r["panel_mean"] >= a.thr]
            curve[n][p] = pick_best(head)["f1c"] if head else 0.0

    print(f"=== S1 「取第一个过筛的」对「把整池排完序」 ===")
    d = [best_all[p] - first[p] for p in ps]
    w, p1, _ = wilcoxon(d, "greater")
    print(f"  {len(ps)} 个靶标:第一个过筛 中位 {st.median([first[p] for p in ps]):.3f},"
          f"整池排序 中位 {st.median([best_all[p] for p in ps]):.3f}")
    print(f"  配对差 中位 {st.median(d):+.3f},排序更好 {sum(1 for x in d if x>0)}/{len(ps)},"
          f"p = {'n/a' if p1 is None else f'{p1:.3g}'}")
    print(f"  (对照:池中最好 中位 {st.median([oracle[p] for p in ps]):.3f} —— 任何选择器都达不到的上界)")

    print(f"\n=== S2/S3 只裁决前 N 条,质量与代价 ===")
    print(f"  {'N':>5s} {'交付中位':>9s} {'距整池排序':>11s} {'240nt 单靶标耗时':>16s} {'100nt':>9s}")
    target_all = st.median([best_all[p] for p in ps])
    knee = None
    for n in GRID:
        m = st.median([curve[n][p] for p in ps])
        gap = m - target_all
        if knee is None and gap >= -0.02:
            knee = n
        print(f"  {n:5d} {m:9.3f} {gap:+11.3f} {n*adj_cost_s(240):15.0f}s {n*adj_cost_s(100):8.0f}s")
    print(f"\n  拐点 N = {knee}(中位数进入整池排序的 0.02 以内)")
    if knee:
        print(f"  -> 240nt 靶标 {knee*adj_cost_s(240):.0f}s,100nt 靶标 {knee*adj_cost_s(100):.0f}s"
              f";整池 200 条则要 {200*adj_cost_s(240)/3600:.1f} 小时 / {200*adj_cost_s(100)/60:.0f} 分钟")

    json.dump({"threshold": a.thr, "n_targets": len(ps),
               "S1": {"first_median": st.median([first[p] for p in ps]),
                      "best_all_median": target_all,
                      "oracle_median": st.median([oracle[p] for p in ps]),
                      "paired_diff": st.median(d), "p": p1},
               "S2": {"knee_N": knee,
                      "curve": {str(n): st.median([curve[n][p] for p in ps]) for n in GRID}},
               "per_target": {p: {"first": first[p], "best_all": best_all[p], "oracle": oracle[p]}
                              for p in ps}},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
