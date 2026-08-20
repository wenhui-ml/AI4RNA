#!/usr/bin/env python
"""P2 -- the blind re-check of the row P1 closed, and the first experiment this week that can LOWER the score.

Pre-registration: p2_kladder_blind_declaration.json.

P1 closed the k-ladder's upper half, and its verdict survived audit -- but both of the power claims I built
around it did not, and I retracted them the same night.  So this is the check that should have accompanied
the closure: does B stay at zero on cells nobody has looked at?

Rule 35 is applied to this design rather than discovered afterwards.  The fifth sample taught it: I wrote
that pooling would take the cell count from 110 to 170 by adding 110 and 60, never checked whether the
grouping keys overlapped, and all sixty of the fifth's cells turned out to be inside the fourth's.  Measured
first this time: at per_cell=10 the sixth sample opens 130 cells, seventy of them shared with the fourth or
fifth, leaving SIXTY that no prior sample has entered.  That is the blind arm, and the union across samples
really is 170 -- derived now, not added.

The primary arm is the blind sixty, and NOT because it has more power.  Ten molecules per cell makes
degeneracy much likelier, and whether sixty noisy cells beat sixty-eight quieter ones cannot be settled
before folding.  It is primary because it is blind, and because 110 of the pooled arm's 170 cells have
already been through A13, A14, A15 and P1.  I have already retracted one primary chosen on a power claim.

Two of the four declared outcomes send objective 4 back from 82.8% to 80%.  P1 had a quirk I flagged at the
time -- pending costs more than a settled negative, so it raised the score whichever way it landed.  P2 does
not have that quirk, and that is stated here rather than after the numbers arrive.
"""
from __future__ import annotations

import argparse, collections, json, os, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)

from arbitration_confirm import load_fresh                                       # noqa: E402
from a14_analyse import line, FLOOR                                              # noqa: E402
from p1_kladder_power import units_for                                           # noqa: E402

QUAL = {"rnet": "f1c", "tube": "wet"}


def spread(vals):
    """Rule 35's second half: a median bootstrap interval is pinned by ties, so exact-tie counts and
    percentiles must be reported next to it or a narrow interval reads as power it does not have."""
    s = sorted(vals); n = len(s)
    z = sum(1 for x in s if abs(x) < 1e-12)
    q = {f"p{int(p*100)}": round(s[min(n - 1, int(p * n))], 4) for p in (0.10, 0.25, 0.75, 0.90)}
    return {"n": n, "exact_zeros": z, "exact_zero_frac": round(z / n, 4) if n else None,
            "centre5": [round(x, 4) for x in s[max(0, n // 2 - 2):n // 2 + 3]], **q}


def arm(tag, rec, draws, seed, out):
    res = {}
    for jn, qk in QUAL.items():
        units, deg = units_for(rec, qk)
        live = [u for u in units if u["both_nondegenerate"]]
        print(f"\n  --- {tag} / 判官 = {jn}({qk}) ---")
        print(f"  格 {len(units)};【三层能分开】的格 {len(live)};声明门槛 {FLOOR} → "
              f"{'过' if len(live) >= FLOOR else '不过,记无数据'}")
        if deg:
            print(f"  退化原因 {dict(deg)}")
        if len(live) < FLOOR:
            res[jn] = {"no_data": True, "n_live": len(live), "degenerate": dict(deg)}
            continue
        r = {"n_live": len(live), "degenerate": dict(deg)}
        r["A"] = line("A = conf均值 − 格均值", [u["A"] for u in live], draws, seed)
        r["B"] = line("B = 块均值 − conf均值 【主】", [u["B"] for u in live], draws, seed + 1)
        sp = spread([u["B"] for u in live])
        r["B_spread"] = sp
        print(f"    B 的并列与分位(第 35 条后半):精确为 0 {sp['exact_zeros']}/{sp['n']} "
              f"({sp['exact_zero_frac']:.1%});中央五个 {sp['centre5']};"
              f"10/25/75/90 = {sp['p10']}/{sp['p25']}/{sp['p75']}/{sp['p90']}")
        byp = collections.defaultdict(list)
        for u in live:
            byp[u["puzzle"]].append(u["B"])
        r["B_by_puzzle"] = line("B 按靶标聚合", [st.mean(v) for v in byp.values()], draws, seed + 2)
        res[jn] = r
    out[tag] = res
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sixth", default=os.path.join(HERE, "panel_f1_sixth.json"))
    ap.add_argument("--fifth", default=os.path.join(HERE, "panel_f1_fifth.json"))
    ap.add_argument("--fourth", default=os.path.join(HERE, "panel_f1_fourth.json"))
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "p2_kladder_blind.json"))
    a = ap.parse_args()

    r6, _s, m6 = load_fresh(a.sixth, "panel_recall_product")
    r5, _s, _m = load_fresh(a.fifth, "panel_recall_product")
    r4, _s, m4 = load_fresh(a.fourth, "panel_recall_product")
    assert m6.get("sixth_per_cell"), "这不是第六批(sixth_per_cell 缺失),拒绝分析"
    k6 = {(r["puzzle"], r["seq"], r["method"]) for r in r6}
    prior = {(r["puzzle"], r["seq"], r["method"]) for r in r4 + r5}
    assert not (k6 & prior), f"第六批与前批重叠 {len(k6 & prior)} 条,拒绝分析"

    prior_cells = {(r["puzzle"], r["method"]) for r in r4 + r5}
    blind = [r for r in r6 if (r["puzzle"], r["method"]) not in prior_cells]
    bc = {(r["puzzle"], r["method"]) for r in blind}
    print(f"第六批 {len(r6)} 条 / {len({(r['puzzle'],r['method']) for r in r6})} 格;"
          f"与前批设计交集 0(断言)")
    print(f"【盲臂】没有任何前批碰过的格 {len(bc)} 个,{len(blind)} 条 —— 这是主检验的人群")
    print(f"合池臂:格 {len(prior_cells | {(r['puzzle'],r['method']) for r in r6})} 个"
          f"(其中 {len(prior_cells)} 个已被 A13/A14/A15/P1 看过)")

    out = {}
    bl = arm("盲臂(主检验 —— 没人看过的 60 个格)", blind, a.draws, a.seed, out)
    po = arm("合池臂(次要,最大功效但含已看过的数据)", r4 + r5 + r6, a.draws, a.seed + 100, out)

    print("\n=== 判读(按声明:主检验 = 盲臂 x RibonanzaNet x B)===")
    bb = (bl.get("rnet") or {}).get("B"); pbb = (po.get("rnet") or {}).get("B")
    v, why = "no_data", ""
    if (bl.get("rnet") or {}).get("no_data") or bb is None or bb.get("no_data"):
        v = "no_data"
        why = ("盲臂三层能分开的格 < 20 -> 无数据。按声明,那一行【退回 ⏳】,④ 从 24/29 = 82.8% "
               "降回 24/30 = 80% —— 因为 P1 的关闭理由需要盲臂支持")
    elif bb["excludes_zero"] and bb["median"] > 0:
        v = "REOPENS_the_row"
        why = (f"盲臂 B = {bb['median']:+.4f} CI {bb['ci']} 【排除 0 且为正】-> P1 的关闭是【过早的】,"
               "那一行退回 ⏳,④ 从 24/29 = 82.8% 降回 24/30 = 80%,并在周报里就地撤回 P1 的判定")
    elif bb["excludes_zero"] and bb["median"] < 0:
        v = "reverses"
        why = (f"盲臂 B = {bb['median']:+.4f} CI {bb['ci']} 【排除 0 且为负】-> k=max 偏好是【有害】的。"
               "这比 null 更强,而且直接指向产品:那条规则应当被【删】,不是被简化")
    else:
        v = "confirms_P1"
        why = (f"盲臂 B = {bb['median']:+.4f} CI {bb['ci']} 跨 0 -> P1 的关闭成立,而且这次是"
               "【在盲数据上】成立。④ 保持 24/29 = 82.8%")
    if pbb and not pbb.get("no_data") and bb and not bb.get("no_data"):
        if (pbb["median"] > 0) != (bb["median"] > 0):
            why += (f";【两臂方向相反】合池 {pbb['median']:+.4f} vs 盲臂 {bb['median']:+.4f} —— "
                    "按声明以盲臂为准")
    print(f"  结论 {v}\n  {why}")
    print("  跑之前就写在声明里的一句:P1 有个怪癖 —— 成立与否都涨分,因为现行口径下 ⏳ 比")
    print("  『已解决的负向』更伤分。【P2 没有这个怪癖】,它四个结局里有两个让分数降回 80%。")
    print("  循环性提醒:本次【没有】用 IPknot+RNAPKplex 做判官 —— 它们与面板共用能量模型族,")
    print("  A15 的 +0.0181 就是那层循环,而这一行卡在 ⏳ 正是因为它。")

    json.dump({"verdict": v, "why": why, "n_blind_cells": len(bc), "n_blind_designs": len(blind),
               "arms": out}, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n写出 {a.out}")


if __name__ == "__main__":
    main()
