#!/usr/bin/env python3
"""Can we accept more without the certificate meaning less? The operating point, calibrated against a tube.

WHY THIS IS THE FIRST MOVE ON THE CEILING. Our shipped criterion accepts a sequence when at least one of the
three oracles reproduces ALL of the target's crossing pairs (tau = 1.0). That operating point was chosen in
silico -- 0.4% false-accept against random sequences -- and has never once been calibrated against an
external outcome. On the clean baseline it leaves 14 of 57 OpenKnot targets undelivered. Those 14 are not
targets we cannot design for; they are targets this particular threshold refuses.

Loosening a criterion to claim more is normally self-deception, and that is exactly what makes this
answerable now rather than earlier: since `criterion_vs_tube.py` we hold 4,560 real molecules that were
actually synthesised, each carrying both our panel's scores and a measured OpenKnot score. So an operating
point can be chosen by what it predicts IN A TUBE instead of by how many of our own targets it lets through.

    The quantity being preserved is the one G3 established, in the form the review corrected it to:
    designs our panel accepts score higher in the tube than designs it rejects, +1.90 points measured
    WITHIN (target, method) cells. The uncontrolled figure was +4.75 and 60% of it was method composition,
    so the controlled estimator is the only one used here.

PRE-REGISTERED, before any operating point was scored. Independent unit is the (target, method) cell,
never the design; correlations and separations are computed inside cells and combined across them.

    W1  THE FRONTIER. For each candidate operating point -- k >= 1 (what ships), k >= 2, k >= 3, and a grid
        on the panel's mean crossing recall -- report the accept rate and the method-controlled wet-lab
        separation between accepted and rejected designs.

    W2  IS THERE A DOMINATING POINT? A point that accepts materially more while separating at least as well.
        PASS: some point reaches accept rate >= current + 10 percentage points with separation >= 1.5
        OpenKnot points (the current controlled estimate is 1.90; a small loss is allowed, a collapse is
        not) at p < 0.025.
        KILL: no such point -> the shipped threshold already sits on the frontier, coverage cannot be
        bought without spending certificate meaning, and the ceiling has to be raised by GENERATION
        instead. That is a real answer and it redirects the plan rather than ending it.

    W3  THE OTHER DIRECTION, reported whatever W2 says. Is there a STRICTER point that separates much
        better? If tightening to k >= 2 buys a large gain in separation, the honest product may be a
        two-level certificate rather than a looser one.

    A guard against the obvious way to cheat this: accept rate is measured on the FIELD's designs, not on
    our own targets, so an operating point cannot be tuned to let our specific failures through. Whether it
    actually rescues any of our 14 is a separate question, asked afterwards and on its own evidence.

COST: $0. Local CPU, no oracle calls -- every number reads from the adjudications already on disk.
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

ALPHA = 0.025
MIN_GROUP = 3


def separation(designs, accept, min_group=MIN_GROUP):
    """Method-controlled wet-lab separation: accepted minus rejected, inside (target, method) cells."""
    cells = collections.defaultdict(list)
    for d in designs:
        cells[(d["puzzle"], d["method"])].append(d)
    diffs = []
    for k, rs in sorted(cells.items()):
        yes = [r["wet"] for r in rs if accept(r)]
        no = [r["wet"] for r in rs if not accept(r)]
        if len(yes) >= min_group and len(no) >= min_group:
            diffs.append(st.median(yes) - st.median(no))
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "criterion_vs_tube.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "panel_operating_point.json"))
    a = ap.parse_args()

    from m2r_pairwise import wilcoxon

    D = json.load(open(a.src)).get("designs")
    assert D, "criterion_vs_tube.json has no per-design rows"
    print(f"真实分子 {len(D)} 条,{len({d['puzzle'] for d in D})} 个靶标,"
          f"{len({d['method'] for d in D})} 个方法\n")

    # `confirmed` IS the shipped k>=1 point. The k>=2 / k>=3 rungs need `panel_k`, which the first dump did
    # not keep; they are skipped with a note rather than silently omitted, because W3's graded-certificate
    # question is exactly about those rungs and a reader must see that it went unanswered.
    has_k = "panel_k" in D[0]
    points = [("出厂 k>=1 (tau=1.0)", lambda d: d["confirmed"])]
    if has_k:
        points += [("k>=2", lambda d: d["panel_k"] >= 2),
                   ("k>=3 (三家全对)", lambda d: d["panel_k"] >= 3)]
    else:
        print("  注:本次 dump 未保存 panel_k,k>=2 / k>=3 两级无法计算(W3 分级证书问题因此悬空);"
              "已在 criterion_vs_tube.py 补上字段,下次重跑即可补齐。\n")
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        points.append((f"平均召回 >= {t:.1f}", (lambda th: lambda d: d["panel_mean"] >= th)(t)))

    print(f"=== W1 工作点前沿(分离度 = 同靶标 x 同方法内,通过组 − 未通过组 的湿实验中位差)===")
    print(f"  {'工作点':22s} {'通过率':>7s} {'分离度':>8s} {'格子':>5s} {'通过更好':>9s} {'p':>10s}")
    rows = []
    for name, fn in points:
        rate = sum(1 for d in D if fn(d)) / len(D)
        diffs = separation(D, fn)
        if len(diffs) < 6:
            print(f"  {name:22s} {rate:6.1%} {'(格子不足)':>8s} {len(diffs):5d}")
            rows.append({"name": name, "accept_rate": rate, "n_cells": len(diffs),
                         "separation": None, "p": None})
            continue
        w, pv, _ = wilcoxon(diffs, "greater")
        print(f"  {name:22s} {rate:6.1%} {st.median(diffs):+8.2f} {len(diffs):5d} "
              f"{sum(1 for x in diffs if x>0):6d}/{len(diffs):<3d} "
              f"{'n/a' if pv is None else f'{pv:.3g}':>10s}")
        rows.append({"name": name, "accept_rate": rate, "n_cells": len(diffs),
                     "separation": st.median(diffs), "p": pv,
                     "wins": sum(1 for x in diffs if x > 0)})

    cur = next(r for r in rows if r["name"].startswith("出厂"))
    print(f"\n  出厂工作点:通过率 {cur['accept_rate']:.1%},分离度 {cur['separation']:+.2f} 分")

    # ---- W2 ------------------------------------------------------------------------------------------
    cand = [r for r in rows if r["separation"] is not None
            and r["accept_rate"] >= cur["accept_rate"] + 0.10
            and r["separation"] >= 1.5 and r["p"] is not None and r["p"] < ALPHA]
    print(f"\n=== W2 有没有「收得更多、判得不差」的点 ===")
    if cand:
        best = max(cand, key=lambda r: r["accept_rate"])
        print(f"  通过 —— `{best['name']}`:通过率 {cur['accept_rate']:.1%} -> {best['accept_rate']:.1%}"
              f"(+{100*(best['accept_rate']-cur['accept_rate']):.0f} 个点),"
              f"分离度 {cur['separation']:+.2f} -> {best['separation']:+.2f},p = {best['p']:.3g}")
        print(f"  -> 下一步:把这个点接到 14 个 not_found 上,看救回几个")
    else:
        best = None
        print(f"  KILL —— 没有任何工作点能在保住分离度的前提下多收 10 个点以上。")
        print(f"  出厂阈值已经在前沿上;覆盖买不来,天花板只能靠【生成】抬。")

    # ---- W3 ------------------------------------------------------------------------------------------
    tighter = [r for r in rows if r["separation"] is not None
               and r["accept_rate"] < cur["accept_rate"]
               and r["separation"] >= cur["separation"] + 0.5]
    print(f"\n=== W3 反方向:有没有「收得更少、但判得明显更准」的点 ===")
    if tighter:
        t = max(tighter, key=lambda r: r["separation"])
        print(f"  `{t['name']}`:通过率 {t['accept_rate']:.1%},分离度 {t['separation']:+.2f} "
              f"(出厂 {cur['separation']:+.2f})")
        print(f"  -> 提示产品该出【分级证书】,而不是单一阈值")
    else:
        print(f"  没有明显更准的收紧点;分级证书的依据不足")

    json.dump({"current": cur, "frontier": rows,
               "W2_best": best, "W2_pass": bool(cand),
               "W3_tighter": (max(tighter, key=lambda r: r["separation"]) if tighter else None)},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
