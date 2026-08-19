#!/usr/bin/env python3
"""Three ways the confirmed result could still be an artefact. Run because it PASSED, not because it failed.

THE DECLARED TEST PASSED: on 8,787 fresh molecules the top region's per-cell fraction is 56.0% [50.5, 61.3],
clearing 0.5 from above, and the bottom region does not clear, so the one-sidedness holds. Twice today a
number I reported in the morning was retracted in the afternoon, and both times the cause was an estimator
choice rather than the data. So the checks below are run on a PASS, where the temptation is to stop:

    S1  THE LOWER BOUND CLEARS BY HALF A POINT. 50.5% against a 0.5 null, from 2,000 bootstrap draws. The
        interval endpoints carry Monte Carlo error of their own, so the pass could be a seed. Repeated at
        20,000 draws across five seeds; if any seed puts the lower bound below 0.5, the pass is not stable and
        must be reported as marginal rather than as a pass.
    S2  ONE CELL HOLDS 1,900 PAIRS AND SOME HOLD ONE. One vote per cell is the estimator that survives today's
        audit, but a cell contributing a single pair contributes a 0 or a 1, which is noise given a vote equal
        to a cell holding hundreds. Repeated requiring a minimum number of pairs per cell inside the region.
        This trades bias for variance in the OTHER direction from pair-weighting, so agreement between the two
        is the thing worth knowing.
    S3  THE FRESH SAMPLE SAYS THE TUBE IS NOISIER THAN THE MARGIN ASSUMES. Replicate spread came out at 4.1
        points against the 3.0 the declaration froze. 3.0 was NOT changed -- it was declared -- but a margin
        below the true noise admits pairs the tube cannot actually referee, which adds noise and, if the
        effect is real, works AGAINST it. Repeated at 4.1 and at 5.0.

WHAT WOULD OVERTURN THE RESULT: S1 unstable across seeds, or the effect vanishing at the honest margin. What
would NOT: the effect shrinking, which is expected of a confirmatory estimate.

A LIMIT NO CHECK HERE REMOVES, stated because the word "holdout" oversells it: the fresh sample is fresh
MOLECULES on the SAME 57 targets. There are no fresh targets to be had -- these are the only pseudoknot
targets carrying large-scale tube measurement. The frozen cut is a single scalar fitted on 7,858 pairs, so
target-level leakage through it is small, but it is not zero.

COST: $0.
"""
from __future__ import annotations

import collections, json, os, random, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", ".."))); sys.path.insert(0, HERE)

from arbitration_audit import cell_pairs                                    # noqa: E402
from arbitration_confirm import load_fresh                                  # noqa: E402


def cluster(percell, draws, rng, unweighted=True):
    """One vote per cell, cluster bootstrap. Same estimator as arbitration_confirm.cluster, made fast.

    With one vote per cell each cell's fraction is FIXED, so a draw is a sample of those fractions and their
    mean -- O(cells) rather than O(pairs). That is what makes 20,000 draws over 34,284 pairs affordable, and
    it is the identical estimator, not an approximation. Asserted against the slow path in __main__.
    """
    fs = [sum(1 for p in v if p["learned_right"]) / len(v) for v in percell.values()]
    if not fs:
        return None, None, None
    n = len(fs)
    out = sorted(sum(fs[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws))
    return sum(fs) / n, out[int(0.025 * draws)], out[min(draws - 1, int(0.975 * draws))]

RNET_OPTIMISED = "Struct2SeQ-SHAPE"


_CELLCACHE = {}


def _cells(rec, margin):
    if margin not in _CELLCACHE:
        g = collections.defaultdict(list)
        for r in rec:
            g[(r["puzzle"], r["method"])].append(r)
        _CELLCACHE[margin] = {k: v for k, v in ((k, cell_pairs(v, margin)) for k, v in g.items()) if v}
    return _CELLCACHE[margin]


def regions(rec, margin, cut_lo, cut_hi, min_pairs=0):
    cells = _cells(rec, margin)
    out = {}
    for name, lo, hi in (("底档", -1e9, cut_lo), ("顶档", cut_hi, 1e9)):
        pc = {k: [p for p in v if lo < (p["rl"] - p["rp"]) <= hi] for k, v in cells.items()}
        pc = {k: v for k, v in pc.items() if len(v) > max(0, min_pairs - 1)}
        out[name] = pc
    return out, sum(len(v) for v in cells.values()), len(cells)


def main() -> int:
    D = json.load(open(os.path.join(HERE, "arbitration_confirm_declaration.json")))
    lo_c, hi_c = D["cut_rank_bottom_quartile"], D["cut_rank_top_quartile"]
    rec, spreads, meta = load_fresh(os.path.join(HERE, "panel_f1_fresh.json"), "panel_f1")
    print(f"新样本 {len(rec)} 条分子;复测噪声中位 {st.median(spreads):.1f} 分")
    res = {"declaration_cuts": [lo_c, hi_c]}

    # the fast estimator must equal the one the confirmatory run used, or nothing below is comparable
    from arbitration_confirm import cluster as slow
    _reg, _, _ = regions(rec, D["margin_tube_noise"], lo_c, hi_c)
    a = cluster(_reg["顶档"], 2000, random.Random(0))
    b = slow(_reg["顶档"], 2000, random.Random(1), True)
    assert abs(a[0] - b[0]) < 1e-12, f"fast estimator disagrees on the point value: {a[0]} vs {b[0]}"
    print(f"  快慢两种实现的点估计一致({a[0]:.4%});区间抽样误差 {abs(a[1]-b[1]):.2%}\n")

    print("=== S1 通过是不是一个种子的事(顶档,每格一票,20000 抽) ===")
    reg, _, _ = regions(rec, D["margin_tube_noise"], lo_c, hi_c)
    s1 = []
    for seed in (0, 1, 2, 3, 4):
        pt, lo, hi = cluster(reg["顶档"], 20000, random.Random(seed), True)
        s1.append({"seed": seed, "point": pt, "ci": [lo, hi], "clears": bool(lo > 0.5)})
        print(f"  种子 {seed}: {pt:.1%} [{lo:.2%},{hi:.2%}]  {'✓' if lo > 0.5 else '✗ 下界掉到 0.5 以下'}")
    stable = all(x["clears"] for x in s1)
    print(f"  {'五个种子全部越过 —— 通过是稳定的' if stable else '存在种子不越过 —— 只能记作边缘,不能记作通过'}")
    res["S1"] = {"runs": s1, "stable": bool(stable)}

    print("\n=== S2 要求格内至少若干配对(顶档 / 底档,每格一票) ===")
    res["S2"] = []
    for mp in (1, 3, 5, 10, 20):
        reg, _, _ = regions(rec, D["margin_tube_noise"], lo_c, hi_c, min_pairs=mp)
        row = {"min_pairs": mp}
        line = f"  >={mp:2d} 对:"
        for nm in ("顶档", "底档"):
            pc = reg[nm]
            if sum(len(v) for v in pc.values()) < 20:
                line += f"  {nm} 太少"; continue
            pt, lo, hi = cluster(pc, 4000, random.Random(0), True)
            row[nm] = {"n_cells": len(pc), "point": pt, "ci": [lo, hi], "clears_above": bool(lo > 0.5)}
            line += f"  {nm} 格{len(pc):3d} {pt:.1%} [{lo:.1%},{hi:.1%}]{'✓' if lo > 0.5 else '跨0.5'}"
        print(line)
        res["S2"].append(row)

    print("\n=== S3 把边距提到新数据实测的噪声之上(顶档 / 底档,每格一票) ===")
    res["S3"] = []
    for m in (3.0, 4.1, 5.0):
        reg, npair, ncell = regions(rec, m, lo_c, hi_c)
        line = f"  边距 {m:.1f}(配对 {npair:6d},格 {ncell:3d}):"
        row = {"margin": m, "n_pairs": npair, "n_cells": ncell}
        for nm in ("顶档", "底档"):
            pc = reg[nm]
            if sum(len(v) for v in pc.values()) < 20:
                line += f"  {nm} 太少"; continue
            pt, lo, hi = cluster(pc, 4000, random.Random(0), True)
            row[nm] = {"n_cells": len(pc), "point": pt, "ci": [lo, hi], "clears_above": bool(lo > 0.5)}
            line += f"  {nm} {pt:.1%} [{lo:.1%},{hi:.1%}]{'✓' if lo > 0.5 else '跨0.5'}"
        print(line)
        res["S3"].append(row)

    surv = (res["S1"]["stable"]
            and all(r.get("顶档", {}).get("clears_above", True) for r in res["S2"])
            and all(r.get("顶档", {}).get("clears_above", True) for r in res["S3"]))
    print(f"\n=== 三项之后 ===")
    print(f"  顶档效应 {'在全部三项下都越过 0.5' if surv else '在某一项下不再越过 0.5 —— 见上表,结论按最弱的那一格写'}")
    res["survives_all_three"] = bool(surv)
    json.dump(res, open(os.path.join(HERE, "arbitration_confirm_robust.json"), "w"), indent=1)
    print("\n写出 arbitration_confirm_robust.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
