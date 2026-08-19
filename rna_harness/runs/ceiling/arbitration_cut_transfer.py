#!/usr/bin/env python3
"""Does the frozen operating point transfer across cell size? Run BEFORE the confirmatory data is scored.

WHY IT HAD TO BE NOW. `arbitration_confirm_declaration.json` freezes an ABSOLUTE cut (rank-difference 0.23134)
taken from the exploratory sample, and the confirmatory sample deliberately has larger cells -- 160 designs per
target instead of 80, median 7 designs per cell instead of 4. In-cell rank differences are normalised by
(n - 1), which should make them scale-free, but "should" is not a measurement.

If the cut's meaning drifted with cell size, then a confirmatory FAILURE would be ambiguous: genuinely not
replicating, or the threshold no longer sitting where it was fitted. Measuring it after seeing the result is
how a null gets explained away, so it is measured here, on the OLD data only, while the fresh panel run is
still folding and nobody can know the answer yet. Whatever this says, it says it in advance.

The test: bucket the exploratory cells by how many DESIGNS they hold, and in each bucket report what fraction
of discordant pairs sit above the frozen cut. If the cut is scale-free that fraction is flat near 25%.

COST: $0.
"""
from __future__ import annotations

import collections, json, os, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", ".."))); sys.path.insert(0, HERE)

from arbitration_audit import cell_pairs, TUBE_NOISE, RNET_OPTIMISED   # noqa: E402
from arbitration import load                                          # noqa: E402

BUCKETS = [(3, 4), (5, 6), (7, 9), (10, 14), (15, 10 ** 6)]


def main() -> int:
    rec, _ = load("panel_f1")
    rec = [r for r in rec if r["method"] != RNET_OPTIMISED]
    g = collections.defaultdict(list)
    for r in rec:
        g[(r["puzzle"], r["method"])].append(r)
    cells = {k: v for k, v in ((k, cell_pairs(v, TUBE_NOISE)) for k, v in g.items()) if v}
    ndes = {k: len(g[k]) for k in cells}
    cut = json.load(open(os.path.join(HERE, "arbitration_confirm_declaration.json")))["cut_rank_top_quartile"]
    print(f"冻结的秩差顶四分位切点 = {cut:.5f}(取自旧样本全体)\n")
    print(f"{'格内设计条数':>12s} {'格数':>5s} {'配对数':>7s} {'秩差中位':>9s} {'75分位':>8s} {'切点以上占比':>14s}")
    rows = []
    for lo, hi in BUCKETS:
        ks = [k for k in cells if lo <= ndes[k] <= hi]
        ps = [p for k in ks for p in cells[k]]
        label = f"{lo}-{'+' if hi > 10 ** 5 else hi}"
        if len(ps) < 20:
            print(f"{label:>12s} {len(ks):5d} {len(ps):7d}  样本太少,不给数")
            rows.append({"bucket": label, "n_cells": len(ks), "n_pairs": len(ps)})
            continue
        d = sorted(p["rl"] - p["rp"] for p in ps)
        above = sum(1 for x in d if x > cut) / len(d)
        print(f"{label:>12s} {len(ks):5d} {len(ps):7d} {st.median(d):9.3f} {d[int(0.75*len(d))]:8.3f} {above:13.1%}")
        rows.append({"bucket": label, "n_cells": len(ks), "n_pairs": len(ps), "median": st.median(d),
                     "q75": d[int(0.75 * len(d))], "frac_above_cut": above})
    fr = [r["frac_above_cut"] for r in rows if "frac_above_cut" in r]
    spread = max(fr) - min(fr)
    print(f"\n  极差 {spread:.1%} —— {'切点与格大小无关,尺度免疫成立' if spread < 0.10 else '切点随格大小漂移,确认失败将无法归因'}")
    print(f"  旧样本每格设计条数中位 {st.median(list(ndes.values())):.0f};确认样本预计更大(160/靶标 vs 80/靶标)")
    print(f"  记录在案:这条检查在新面板还在折叠时跑的,结果不可能被结果影响。")
    json.dump({"cut": cut, "buckets": rows, "spread": spread, "scale_free": bool(spread < 0.10),
               "run_before_confirmatory_data_existed": True},
              open(os.path.join(HERE, "arbitration_cut_transfer.json"), "w"), indent=1)
    print(f"\n写出 arbitration_cut_transfer.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
