#!/usr/bin/env python3
"""Does the better operating point survive being chosen on one set of targets and judged on another?

WHY THIS EXISTS. `panel_operating_point.py` swept a grid of thresholds and reported the best one: mean
crossing recall >= 0.3 accepts 74.8% of real designs against the shipped 57.8%, and separates the tube
slightly BETTER (+1.97 against +1.90). That is a large coverage gain for no measured loss of meaning -- and
it was selected from a grid scored on the very data used to judge it. Picking the maximum of seven options
and then quoting that maximum's p-value is the error that killed the pool-rerank headline this morning and
the C3 placement flip yesterday. Twice is enough; this one gets held out before it is believed.

Two additional worries the sweep raised, both handled here rather than left implicit:

    THE WINNER SAT AT THE EDGE OF THE GRID. 0.3 was the lowest threshold tried, so the sweep cannot tell a
    genuine optimum from "looser is always better until the criterion accepts everything". The grid is
    extended downward. If separation holds while acceptance climbs toward 100%, the criterion is dissolving
    rather than improving, and the accept rate at which it dissolves is the thing to report.

    THE FRONTIER IS NOT MONOTONE. Separation runs +1.97 / +1.71 / +1.58 / +2.36 / -0.07 / -0.28 / -0.26
    across 0.3 to 0.9. Beyond 0.7 the panel stops identifying better molecules altogether. A criterion that
    inverts when tightened is either telling us something real about what strong three-oracle agreement
    selects for, or it is small-sample noise; the cell counts (51, 42, 33) argue against pure noise, so it
    is reported as a finding in its own right.

PRE-REGISTERED. Split the 57 targets into two halves at random, choose the threshold on half A purely by
its separation there, then measure that threshold's accept rate and separation on half B, which had no part
in choosing it. Roles swapped, and repeated over many splits so the answer does not hinge on one draw.
Cells never straddle a split: a (target, method) cell belongs to whichever half its target is in.

    H1  HELD-OUT SEPARATION. Median over splits of the chosen threshold's separation on the unseen half.
        PASS: >= 1.5 OpenKnot points and better than the shipped point's separation on the SAME unseen half
        on a majority of splits.
        KILL: it collapses out of sample -> the gain was grid-search overfitting, the shipped operating
        point stands, and coverage has to come from generation instead.

    H2  HELD-OUT COVERAGE. The accept-rate gain on the unseen half. A gain that survives H1 but shrinks to
        nothing here is not worth changing a shipped criterion for.

    H3  WHERE THE CRITERION DISSOLVES. Accept rate and separation for thresholds down to 0.0, so the point
        at which "accept almost everything" stops separating is visible rather than assumed.

COST: $0. No oracle calls; every number reads adjudications already on disk.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MIN_GROUP = 3
GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def sep(designs, accept, min_group=MIN_GROUP):
    cells = collections.defaultdict(list)
    for d in designs:
        cells[(d["puzzle"], d["method"])].append(d)
    out = []
    for _, rs in cells.items():
        yes = [r["wet"] for r in rs if accept(r)]
        no = [r["wet"] for r in rs if not accept(r)]
        if len(yes) >= min_group and len(no) >= min_group:
            out.append(st.median(yes) - st.median(no))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "criterion_vs_tube.json"))
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "panel_operating_point_holdout.json"))
    a = ap.parse_args()

    from m2r_pairwise import wilcoxon

    D = json.load(open(a.src))["designs"]
    puzzles = sorted({d["puzzle"] for d in D})
    print(f"真实分子 {len(D)} 条,{len(puzzles)} 个靶标,{a.splits} 次随机对半划分\n")

    # ---- H3, on everything, so the shape is visible before any selection happens ---------------------
    print(f"=== H3 判据在什么地方「化掉」(全量,先看形状)===")
    print(f"  {'阈值':>8s} {'通过率':>8s} {'分离度':>8s} {'格子':>5s}")
    shape = []
    for t in GRID:
        v = sep(D, lambda d, th=t: d["panel_mean"] >= th)
        rate = sum(1 for d in D if d["panel_mean"] >= t) / len(D)
        shape.append({"t": t, "rate": rate, "sep": (st.median(v) if len(v) >= 6 else None),
                      "cells": len(v)})
        s = f"{st.median(v):+8.2f}" if len(v) >= 6 else f"{'格子不足':>8s}"
        print(f"  {t:8.2f} {rate:7.1%} {s} {len(v):5d}")
    base_rate = sum(1 for d in D if d["confirmed"]) / len(D)
    base_sep = st.median(sep(D, lambda d: d["confirmed"]))
    print(f"  {'出厂':>8s} {base_rate:7.1%} {base_sep:+8.2f}")

    # ---- H1 / H2 -------------------------------------------------------------------------------------
    rng = random.Random(a.seed)
    chosen, held_sep, held_base, held_rate, held_rate_base = [], [], [], [], []
    for _ in range(a.splits):
        ps = puzzles[:]
        rng.shuffle(ps)
        half = set(ps[:len(ps) // 2])
        A = [d for d in D if d["puzzle"] in half]
        B = [d for d in D if d["puzzle"] not in half]
        best_t, best_v = None, None
        for t in GRID:
            v = sep(A, lambda d, th=t: d["panel_mean"] >= th)
            if len(v) < 6:
                continue
            m = st.median(v)
            if best_v is None or m > best_v:
                best_t, best_v = t, m
        if best_t is None:
            continue
        vb = sep(B, lambda d, th=best_t: d["panel_mean"] >= th)
        vc = sep(B, lambda d: d["confirmed"])
        if len(vb) < 6 or len(vc) < 6:
            continue
        chosen.append(best_t)
        held_sep.append(st.median(vb))
        held_base.append(st.median(vc))
        held_rate.append(sum(1 for d in B if d["panel_mean"] >= best_t) / len(B))
        held_rate_base.append(sum(1 for d in B if d["confirmed"]) / len(B))

    cnt = collections.Counter(chosen)
    print(f"\n=== H1 在 A 半边选阈值,到没参与选择的 B 半边上检验({len(held_sep)} 次有效划分)===")
    print(f"  被选中的阈值分布:{dict(sorted(cnt.items()))}")
    print(f"  留出半边 分离度:选出的点 {st.median(held_sep):+.2f} 分,出厂点 {st.median(held_base):+.2f} 分")
    d = [x - y for x, y in zip(held_sep, held_base)]
    wins = sum(1 for x in d if x > 0)
    print(f"  逐次划分配对差 中位 {st.median(d):+.2f},选出的点更好 {wins}/{len(d)} 次")
    h1 = st.median(held_sep) >= 1.5 and wins > len(d) / 2
    print(f"  {'通过 —— 增益经得起留出检验' if h1 else 'KILL —— 增益是网格搜索的过拟合,出厂工作点不动,覆盖只能靠生成'}")

    print(f"\n=== H2 留出半边的覆盖变化 ===")
    dr = [x - y for x, y in zip(held_rate, held_rate_base)]
    print(f"  通过率:出厂 {st.median(held_rate_base):.1%} -> 选出的点 {st.median(held_rate):.1%}"
          f"(配对差中位 {st.median(dr):+.1%})")

    json.dump({"H3_shape": shape, "baseline": {"rate": base_rate, "sep": base_sep},
               "H1": {"n_splits": len(held_sep), "chosen_hist": dict(cnt),
                      "held_sep_median": st.median(held_sep),
                      "held_base_sep_median": st.median(held_base),
                      "paired_diff_median": st.median(d), "wins": wins, "pass": bool(h1)},
               "H2": {"held_rate_median": st.median(held_rate),
                      "held_base_rate_median": st.median(held_rate_base),
                      "paired_rate_diff_median": st.median(dr)}},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
