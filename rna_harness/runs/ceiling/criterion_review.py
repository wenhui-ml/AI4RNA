#!/usr/bin/env python3
"""Review of G3: is "+4.75 wet-lab points" our criterion working, or our criterion reading the pipeline?

WHY THIS EXISTS. G3 in `criterion_vs_tube.py` is the strongest positive result the project has: designs our
shipped panel confirms score +4.75 OpenKnot points higher in the tube than designs it does not, on 43 of 51
targets, p = 4.9e-08. It was computed WITHIN TARGET, which controls for how hard the puzzle is. It was not
computed within METHOD, and it needs to be, because the methods differ from each other in wet-lab score by
much more than the effect being claimed:

    Eterna 87.4    codesign-RFdiff 89.9    Struct2SeQ-SHAPE 89.6    Rosetta 74.5    3DRNA 56.9

If our panel confirms Eterna designs more readily than Rosetta designs -- for any reason, including reasons
that have nothing to do with whether a molecule works -- then a comparison of confirmed against unconfirmed
inside a target is partly a comparison of Eterna against Rosetta, and the +4.75 would be an artifact of
which pipeline made the design rather than evidence that our certificate means anything.

    R1  CONFIRMATION RATE BY METHOD. If the panel confirms every method at the same rate, no confound is
        possible and G3 stands as computed. Descriptive, run first.

    R2  G3 CONTROLLED FOR METHOD. The same confirmed-minus-unconfirmed difference, computed inside
        (target, method) cells so that every comparison is between designs from the SAME pipeline for the
        SAME puzzle. Cluster unit is the cell. Wilcoxon over cells, one-sided greater.
        PASS: p < 0.025 and the median difference stays positive -> G3 survives, and the certificate's
        external validation is real.
        KILL: the effect collapses -> G3 was reading method composition, the headline is withdrawn, and the
        project has no external validation of its certificate after all.

    R3  HOW MUCH OF THE RAW EFFECT WAS COMPOSITION. The uncontrolled and controlled medians side by side.

    R4  G2 CONTROLLED THE SAME WAY. The panel-versus-RibonanzaNet correlation comparison, recomputed inside
        method cells. This one is less exposed -- both criteria are scored on the same designs, so
        composition shifts both -- but it costs nothing to check.

Reads the per-design rows `criterion_vs_tube.py` now saves. No adjudication is repeated.

COST: $0. Local CPU, no oracle calls.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "criterion_vs_tube.json"))
    ap.add_argument("--min-group", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(HERE, "criterion_review.json"))
    a = ap.parse_args()

    from m2r_pairwise import wilcoxon, spearman

    S = json.load(open(a.src))
    D = S.get("designs")
    assert D, "criterion_vs_tube.json has no per-design rows; re-run it with the current version"
    print(f"逐条设计 {len(D)} 行,{len({r['puzzle'] for r in D})} 个靶标,"
          f"{len({r['method'] for r in D})} 个方法")

    # ---- R1 -----------------------------------------------------------------------------------------
    bym = collections.defaultdict(list)
    for r in D:
        bym[r["method"]].append(r)
    print(f"\n=== R1 面板对各方法的确认率(若各方法差别很大,混杂就是活的)===")
    print(f"  {'方法':22s} {'n':>5s} {'确认率':>8s} {'湿实验中位':>10s}")
    rates = {}
    for m, rs in sorted(bym.items(), key=lambda kv: -len(kv[1])):
        cr = sum(1 for r in rs if r["confirmed"]) / len(rs)
        rates[m] = cr
        print(f"  {m:22s} {len(rs):5d} {cr:8.1%} {st.median([r['wet'] for r in rs]):10.1f}")
    spread = max(rates.values()) - min(rates.values())
    print(f"  确认率极差 {spread:.1%} —— {'混杂可能存在,R2 必须做' if spread > 0.1 else '各方法确认率接近,混杂空间很小'}")

    # ---- R2 / R3 ------------------------------------------------------------------------------------
    cells = collections.defaultdict(list)
    for r in D:
        cells[(r["puzzle"], r["method"])].append(r)
    diffs, ns = [], []
    for k, rs in sorted(cells.items()):
        yes = [r["wet"] for r in rs if r["confirmed"]]
        no = [r["wet"] for r in rs if not r["confirmed"]]
        if len(yes) >= a.min_group and len(no) >= a.min_group:
            diffs.append(st.median(yes) - st.median(no))
            ns.append(len(rs))
    w, p2, n2 = wilcoxon(diffs, "greater")
    print(f"\n=== R2 把方法也控制住之后(在【同靶标 x 同方法】内比)===")
    print(f"  合格单元 {len(diffs)} 个(每组至少 {a.min_group} 条),单元中位样本量 {st.median(ns):.0f}")
    print(f"  确认组 − 未确认组 湿实验中位差 {st.median(diffs):+.2f} 分,"
          f"确认组更好 {sum(1 for x in diffs if x>0)}/{len(diffs)},"
          f"p = {'n/a' if p2 is None else f'{p2:.3g}'}")
    ok = bool(p2 is not None and p2 < ALPHA and st.median(diffs) > 0)
    print(f"  {'通过 —— G3 在控制方法后依然成立,证书的外部验证是真的' if ok else 'KILL —— G3 读的是方法构成,头条撤回'}")

    raw = S["G3"]["wetlab_diff_median"]
    print(f"\n=== R3 未控制 {raw:+.2f} 分  ->  控制方法后 {st.median(diffs):+.2f} 分 ===")
    print(f"  其中由方法构成贡献的部分约 {raw - st.median(diffs):+.2f} 分"
          f"({abs(raw - st.median(diffs))/abs(raw):.0%} 的原始效应)")

    # ---- R4 -----------------------------------------------------------------------------------------
    pr, pp = [], []
    for k, rs in sorted(cells.items()):
        if len(rs) < 15:
            continue
        wet = [r["wet"] for r in rs]
        if len(set(wet)) < 3:
            continue
        s1 = spearman([r["f1c"] for r in rs], wet)
        s2 = spearman([r["panel_mean"] for r in rs], wet)
        if s1 is not None and s2 is not None:
            pr.append(s1)
            pp.append(s2)
    if pr:
        d4 = [x - y for x, y in zip(pr, pp)]
        w, p4, _ = wilcoxon(d4, "greater")
        print(f"\n=== R4 G2 在同靶标 x 同方法内重算 ===")
        print(f"  {len(pr)} 个单元:RNet rho 中位 {st.median(pr):+.3f},面板 rho 中位 {st.median(pp):+.3f}")
        print(f"  配对差 {st.median(d4):+.3f},RNet 更好 {sum(1 for x in d4 if x>0)}/{len(d4)},"
              f"p = {'n/a' if p4 is None else f'{p4:.3g}'}")
    else:
        p4 = None
        print(f"\n=== R4 没有足够大的 (靶标 x 方法) 单元来重算 G2 ===")

    json.dump({"R1_confirm_rate_by_method": rates, "R1_spread": spread,
               "R2": {"n_cells": len(diffs), "median_diff": st.median(diffs), "p": p2, "pass": ok,
                      "wins": sum(1 for x in diffs if x > 0)},
               "R3": {"uncontrolled": raw, "controlled": st.median(diffs)},
               "R4": ({"n_cells": len(pr), "rho_rnet": st.median(pr), "rho_panel": st.median(pp),
                       "paired_diff": st.median(d4), "p": p4} if pr else None)},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
