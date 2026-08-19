#!/usr/bin/env python3
"""A14: the confirmatory read of A13's three-layer decomposition, on cells whose SIZE is set by declaration.

Declared in `a14_declaration.json`, committed and pushed before the fourth sample was judged.

WHAT WENT WRONG IN A13, IN ONE SENTENCE. Its power floor was written as "fewer than 20 usable units -> no
data", when it should have been "fewer than 20 units whose three layers can DIFFER". The tube cells had a
median of eight molecules, so 89 of 287 had the confirmed set equal to the whole cell -- term A was a
structural zero, not a measurement -- and the median then reported how big the cells were rather than what the
ruler knows. Same class of defect as A9's declaration, which set a floor on the total and not on what actually
entered the test.

WHAT THIS RUN CHANGES AND WHAT IT DOES NOT. The cell is still (target, method), because A and B are within-set
contrasts and pooling methods would let a method that is both more confirmable and better in the tube pose as
a boundary carrying information. What changes is that the fourth sample draws 24 per CELL, so cell size is a
declared constant instead of whatever share that method happened to hold in the library. Cells that could not
supply 24 are dropped whole, which costs 358 of 468 cells and limits the claim to well-represented methods.

NOT A BLIND TEST, AND LABELLED AS SUCH. A13's population-2 means (A +0.368 / +0.355, B -0.333 / -0.146) were
on screen before this file existed. They are not used as a prior and the direction of B is still undeclared,
but the honest label is "a new population for a question whose first answer I have already seen".

C IS NOT MEASURED HERE. The tie-breaker term needs entropy, which is not in these files, and A12 measured it
on its own equal-grade population (null). Recorded as not measured, never as zero.

COST: $0, offline.
"""
from __future__ import annotations

import argparse, collections, json, os, random, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)

from arbitration_confirm import load_fresh                                       # noqa: E402
from tiebreak_direction import grade                                             # noqa: E402

FLOOR = 20


def boot(vals, draws, seed, stat=st.median):
    rng = random.Random(seed)
    v = list(vals); n = len(v)
    o = sorted(stat([v[rng.randrange(n)] for _ in range(n)]) for _ in range(draws))
    return stat(v), o[int(0.025 * draws)], o[min(draws - 1, int(0.975 * draws))]


def line(label, vals, draws, seed):
    if len(vals) < 5:
        print(f"  {label:34s} 单位 {len(vals):4d} —— 太少,不给数")
        return {"n": len(vals), "no_data": True}
    pt, lo, hi = boot(vals, draws, seed)
    pos = sum(1 for x in vals if x > 0); neg = sum(1 for x in vals if x < 0)
    ex = lo > 0 or hi < 0
    print(f"  {label:34s} 单位 {len(vals):4d}  中位 {pt:+.4f}  均值 {st.mean(vals):+.4f}  "
          f"正/负 {pos:4d}/{neg:4d}  CI [{lo:+.4f}, {hi:+.4f}]{'  排除 0' if ex else '  跨过 0'}")
    return {"n": len(vals), "median": pt, "mean": st.mean(vals), "n_pos": pos, "n_neg": neg,
            "ci": [lo, hi], "excludes_zero": bool(ex),
            "excludes_zero_positive": bool(ex and lo > 0),
            "excludes_zero_negative": bool(ex and hi < 0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "panel_f1_fourth.json"))
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "a14.json"))
    a = ap.parse_args()
    D = json.load(open(os.path.join(HERE, "a14_declaration.json")))
    rec, _sp, meta = load_fresh(a.src, "panel_recall_product")
    assert meta.get("fourth_per_cell"), "这不是第四批(fourth_per_cell 缺失),拒绝分析"

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
        kmax = max(grade(r) for r in conf)
        blk = [r for r in conf if grade(r) == kmax]
        a0, b0 = len(conf) == len(rw), len(blk) == len(conf)
        if a0:
            deg["conf == 整格 → A 结构性为 0"] += 1
        if b0:
            deg["块 == conf → B 结构性为 0"] += 1
        pq = [r["wet"] for r in rw]; cq = [r["wet"] for r in conf]; bq = [r["wet"] for r in blk]
        units.append({"cell": cell, "puzzle": cell[0],
                      "A": st.mean(cq) - st.mean(pq), "B": st.mean(bq) - st.mean(cq),
                      "identity": st.mean(bq) - st.mean(pq),
                      "n_pool": len(rw), "n_conf": len(conf), "n_block": len(blk),
                      "both_nondegenerate": (not a0) and (not b0)})
    bad = [u for u in units if abs(u["A"] + u["B"] - u["identity"]) > 1e-9]
    assert not bad, "恒等式 A+B ≡ 块均值−格均值 不成立,结果不可读"

    live = [u for u in units if u["both_nondegenerate"]]
    print(f"=== 第四批:{len(rec)} 条去重分子,{len(cells)} 个格(每格声明为 24 条)===")
    print(f"  层大小中位:整格 {st.median([u['n_pool'] for u in units]):.0f}  "
          f"confirmed {st.median([u['n_conf'] for u in units]):.0f}  "
          f"顶端块 {st.median([u['n_block'] for u in units]):.0f}")
    print(f"  退化情况(按声明必须打印):{dict(deg)}")
    print(f"  **三层能分开的格 {len(live)} / {len(units)}**;声明门槛 {FLOOR}"
          f" → {'过' if len(live) >= FLOOR else '不过,记无数据'}")
    res = {"declaration": D, "n_designs": len(rec), "n_cells": len(units),
           "n_both_nondegenerate": len(live), "degenerate": dict(deg), "floor": FLOOR}

    if len(live) < FLOOR:
        res["verdict"] = "no_data"
        print(f"\n  【结论】no_data —— {D['outcomes_fixed_in_advance']['no_data']}")
        json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
        return 0

    print(f"\n  --- 主检验,人群 = 【三层能分开的格】(按声明){len(live)} 个 ---")
    res["primary"] = {k: line({"A": "A  confirmed 边界(k>=1)", "B": "B  等级取最高(k=max)"}[k],
                              [u[k] for u in live], a.draws, a.seed + ord(k)) for k in ("A", "B")}
    res["primary"]["A_plus_B"] = line("A+B = 块均值 − 格均值", [u["A"] + u["B"] for u in live],
                                      a.draws, a.seed)
    print(f"  --- 参照:全部 {len(units)} 个格(含结构性零,A13 就是这么读的,会被零稀释)---")
    res["all_cells"] = {k: line({"A": "A 全部格", "B": "B 全部格"}[k], [u[k] for u in units],
                                a.draws, a.seed + ord(k)) for k in ("A", "B")}
    print(f"  --- 更保守:按【靶标】聚类(格之间共用靶标)---")
    byp = collections.defaultdict(list)
    for u in live:
        byp[u["puzzle"]].append(u)
    res["by_puzzle"] = {k: line({"A": "A 按靶标", "B": "B 按靶标"}[k],
                                [st.median([y[k] for y in v]) for v in byp.values()],
                                a.draws, a.seed + 20) for k in ("A", "B")}
    print(f"  (靶标数 {len(byp)};每靶标平均 {len(live)/len(byp):.2f} 格)")

    A, B = res["primary"]["A"], res["primary"]["B"]
    if not A["excludes_zero_positive"]:
        v = "null"
    elif B["excludes_zero_negative"]:
        v = "grade_ladder_is_harmful"
    elif B["excludes_zero_positive"]:
        v = "gradient_at_the_top"
    else:
        v = "gradient_at_the_low_end"
    print(f"\n=== 判定(按声明)===")
    print(f"  A 中位 {A['median']:+.4f} [{A['ci'][0]:+.4f},{A['ci'][1]:+.4f}]"
          f"{' 排除 0 为正' if A['excludes_zero_positive'] else ' 跨过 0'}")
    print(f"  B 中位 {B['median']:+.4f} [{B['ci'][0]:+.4f},{B['ci'][1]:+.4f}]"
          f"{' 排除 0' if B['excludes_zero'] else ' 跨过 0'}")
    print(f"  【结论】{v} —— {D['outcomes_fixed_in_advance'][v]}")
    print(f"  对照 A13 人群1(真实池,RNet 判):A +0.1036 [+0.0584,+0.1293] 排除 0;"
          f"B +0.0150 [+0.0000,+0.0719] 跨过 0")
    res["verdict"] = v
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n写出 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
