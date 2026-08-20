#!/usr/bin/env python
"""P1 -- the k-ladder's upper half, with the power plan corrected before it was run.

Pre-registration: p1_kladder_power_declaration.json.

Two things had to be fixed before this could be a real experiment.

The report's next-step plan said to make cells bigger, quoting 82 targets able to supply 40 designs each.
That figure predates the fourth sample's 2,640 draws.  After samples 1-4 the library holds 12,610 unused
designs in 466 cells, and every workable cut gives FEWER cells than the fourth sample's 110: sixty at >=24,
fifty at >=30, thirty-three at >=40.  Since power in the B term scales with sqrt(n_cells), a bigger-cell
fifth sample on its own is worse, not better.  So the cell size stays at 24 and the cells get POOLED.

And the blocker was never really n.  A15's B does exclude zero, at +0.0181 [+0.0002, +0.0524], but its
judges share an energy-model family with the panel being measured, so agreement there cannot be read as the
panel being right -- which is exactly why the row is still open.  The primary judge here is RibonanzaNet:
learned, no energy model, no shared error path with the panel, and ceiling_probe measured it as the highest
signal-to-noise independent column available (0.164 alone against 0.103 for all twenty-one thermodynamic
columns together).  The tube travels as a secondary with its plus-or-minus 1.5 resolution stated.

This is also the first run under rule 34, written earlier today: the primary population is chosen for POWER,
and the cost is stated rather than hidden -- the fourth-sample half is not blind, having already been through
A13, A14 and A15.  The fifth sample alone is reported as the fresh arm, and if the two disagree in direction
the fresh arm wins, because it is the only half nobody has looked at.
"""
from __future__ import annotations

import argparse, collections, json, os, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)

from arbitration_confirm import load_fresh                                       # noqa: E402
from tiebreak_direction import grade                                             # noqa: E402
from a14_analyse import line, FLOOR                                              # noqa: E402

QUAL = {"rnet": "f1c", "tube": "wet"}


def units_for(rec, qkey):
    cells = collections.defaultdict(list)
    for r in rec:
        cells[(r["puzzle"], r["method"])].append(r)
    units, deg = [], collections.Counter()
    for cell, rw in cells.items():
        conf = [r for r in rw if (grade(r) or 0) >= 1]
        if len(rw) < 2:
            deg["格 < 2 条"] += 1; continue
        if not conf:
            deg["无 confirmed(整格无法进检验)"] += 1; continue
        if any(r.get(qkey) is None for r in rw):
            deg[f"缺 {qkey} 列"] += 1; continue
        kmax = max(grade(r) for r in conf)
        blk = [r for r in conf if grade(r) == kmax]
        a0, b0 = len(conf) == len(rw), len(blk) == len(conf)
        if a0: deg["conf == 整格 → A 结构性为 0"] += 1
        if b0: deg["块 == conf → B 结构性为 0"] += 1
        pq = [r[qkey] for r in rw]; cq = [r[qkey] for r in conf]; bq = [r[qkey] for r in blk]
        units.append({"cell": cell, "puzzle": cell[0],
                      "A": st.mean(cq) - st.mean(pq), "B": st.mean(bq) - st.mean(cq),
                      "identity": st.mean(bq) - st.mean(pq),
                      "both_nondegenerate": (not a0) and (not b0)})
    bad = [u for u in units if abs(u["A"] + u["B"] - u["identity"]) > 1e-9]
    assert not bad, "恒等式 A+B ≡ 块均值−格均值 不成立,结果不可读"
    return units, deg


def report(tag, rec, draws, seed, out):
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
        r["A_plus_B"] = line("A+B = 块均值 − 格均值(恒等)", [u["A"] + u["B"] for u in live], draws, seed + 2)
        byp = collections.defaultdict(list)
        for u in live: byp[u["puzzle"]].append(u["B"])
        r["B_by_puzzle"] = line("B 按靶标聚合", [st.mean(v) for v in byp.values()], draws, seed + 3)
        res[jn] = r
    out[tag] = res
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fifth", default=os.path.join(HERE, "panel_f1_fifth.json"))
    ap.add_argument("--fourth", default=os.path.join(HERE, "panel_f1_fourth.json"))
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "p1_kladder_power.json"))
    a = ap.parse_args()

    r5, _s, m5 = load_fresh(a.fifth, "panel_recall_product")
    r4, _s, m4 = load_fresh(a.fourth, "panel_recall_product")
    assert m5.get("fifth_per_cell"), "这不是第五批(fifth_per_cell 缺失),拒绝分析"
    assert m4.get("fourth_per_cell"), "这不是第四批,拒绝分析"
    k5 = {(r["puzzle"], r["seq"], r["method"]) for r in r5}
    k4 = {(r["puzzle"], r["seq"], r["method"]) for r in r4}
    print(f"第五批 {len(r5)} 条 / {len({(r['puzzle'],r['method']) for r in r5})} 格;"
          f"第四批 {len(r4)} 条 / {len({(r['puzzle'],r['method']) for r in r4})} 格;"
          f"交集 {len(k5 & k4)}(断言 0)")
    assert not (k5 & k4), "第五批与第四批重叠,拒绝分析"

    out = {}
    fresh = report("第五批(新鲜臂,唯一没人看过的一半)", r5, a.draws, a.seed, out)
    pooled = report("合池(主人群,按功效选 —— §八 第 34 条)", r4 + r5, a.draws, a.seed + 100, out)

    print("\n=== 判读(按声明:主判官 = RibonanzaNet,主量 = B)===")
    pb = (pooled.get("rnet") or {}).get("B"); fb = (fresh.get("rnet") or {}).get("B")
    v, why = "no_data", ""
    if pb is None or pb.get("no_data") or (pooled.get("rnet") or {}).get("no_data"):
        v, why = "no_data", "合池在 RNet 判官下三层能分开的格不足门槛 —— 【库供不起这个功效】,那一行仍是 ⏳"
    elif fb and (pb["median"] > 0) != (fb["median"] > 0):
        v = "fresh_arm_wins"
        why = (f"合池 B {pb['median']:+.4f} CI {pb['ci']} 与新鲜臂 B {fb['median']:+.4f} CI {fb['ci']} "
               "【方向相反】。按声明以新鲜臂为准 —— 它是唯一没被看过的那一半")
        if not (fb["excludes_zero"] and fb["median"] > 0):
            v = "closes_as_refuted"
            why += ";新鲜臂不排除 0 ⇒ 该行判为已测的负向结论"
    elif pb["excludes_zero"] and pb["median"] > 0:
        v = "closes_as_capability"
        why = (f"RNet 判官下 B = {pb['median']:+.4f} CI {pb['ci']} 排除 0 且为正 -> ④ 那一行 ⏳ 变 ✅,"
               "24/30 -> 25/30 = 83.3%")
    else:
        v = "closes_as_refuted"
        why = (f"RNet 判官下 B = {pb['median']:+.4f} CI {pb['ci']} 跨 0 -> 该行判为【已测的负向结论】,"
               "进背景,分母降到 29 -> 24/29 = 82.8%")
    print(f"  结论 {v}\n  {why}")
    tb = (pooled.get("tube") or {}).get("B")
    if tb and not tb.get("no_data"):
        print(f"  次口径(试管,分辨率 ±1.5 原样带出):B = {tb['median']:+.4f} CI {tb['ci']}")
    print("  【两个结局都涨分】,因为现行口径下 ⏳ 比『已解决的负向』更伤分。这一句必须随结论一起写,")
    print("  否则读起来像我在挑一个必赢的实验。")
    print("  循环性提醒:IPknot+RNAPKplex 与面板共用能量模型族,本次【没有】用它们做主判官 —— ")
    print("  A15 的 B 排除 0 正是因为那层循环,所以它不能关掉这一行。")

    json.dump({"verdict": v, "why": why, "arms": out}, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n写出 {a.out}")


if __name__ == "__main__":
    main()
