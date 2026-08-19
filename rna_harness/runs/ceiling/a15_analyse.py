#!/usr/bin/env python3
"""A15 stage 2: the same three-layer decomposition, quality = a FIXED pair of never-in-loop judges.

Declared in `a15_declaration.json`. The estimator is word-for-word A14's; the only thing that changes is what
"quality" means, which is the whole point -- three judge families on one question, each one's judge being the
others' weakness:

    A13 pop1   our own candidate pools          judged by RibonanzaNet (learned)
    A14        the fourth sample                judged by MEASURED OpenKnot scores
    A15 (here) the same fourth sample           judged by IPknot + RNAPKplex (thermodynamic, never in loop)

A15's weakness is stated up front and must travel with any conclusion: these judges SHARE an energy-model
family with the panel being measured, so agreement here can never be read as "the panel is right". What it can
do is say whether the LAYER structure survives a change of judge.

WHY THE JUDGE SET IS FIXED AT TWO. `mean3`-style "average whatever succeeded" is not reproducible: Knotty is
wall-clock bounded and its availability moved between two runs of the same code on the same 91 sequences
(37 -> 36 targets). IPknot and RNAPKplex failed 0/91. Knotty is read only as a coverage-reported secondary.
"""
from __future__ import annotations

import argparse, collections, json, os, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)

from tiebreak_direction import grade                                             # noqa: E402
from a14_analyse import line, FLOOR                                              # noqa: E402


def decompose(cells, q, draws, seed, tag):
    units, deg = [], collections.Counter()
    for cell, rw in cells.items():
        rw = [r for r in rw if q.get((r["puzzle"], r["seq"])) is not None]
        conf = [r for r in rw if (grade(r) or 0) >= 1]
        if len(rw) < 2:
            deg["格 < 2 条(判官失败后)"] += 1; continue
        if not conf:
            deg["无 confirmed"] += 1; continue
        kmax = max(grade(r) for r in conf)
        blk = [r for r in conf if grade(r) == kmax]
        a0, b0 = len(conf) == len(rw), len(blk) == len(conf)
        if a0:
            deg["conf == 整格 → A 结构性为 0"] += 1
        if b0:
            deg["块 == conf → B 结构性为 0"] += 1
        f = lambda g: st.mean([q[(r["puzzle"], r["seq"])] for r in g])            # noqa: E731
        units.append({"cell": cell, "puzzle": cell[0], "A": f(conf) - f(rw), "B": f(blk) - f(conf),
                      "identity": f(blk) - f(rw), "both_nondegenerate": (not a0) and (not b0)})
    bad = [u for u in units if abs(u["A"] + u["B"] - u["identity"]) > 1e-9]
    assert not bad, "恒等式不成立,结果不可读"
    live = [u for u in units if u["both_nondegenerate"]]
    print(f"\n=== {tag} ===")
    print(f"  退化情况(按声明必须打印):{dict(deg)}")
    print(f"  **三层能分开的格 {len(live)} / {len(units)}**;门槛 {FLOOR}"
          f" → {'过' if len(live) >= FLOOR else '不过,记无数据'}")
    row = {"n_cells": len(units), "n_live": len(live), "degenerate": dict(deg)}
    if len(live) < FLOOR:
        row["verdict"] = "no_data"; return row
    print(f"  --- 主检验,人群 = 三层能分开的格 ---")
    row["primary"] = {k: line({"A": "A  confirmed 边界(k>=1)", "B": "B  等级取最高(k=max)"}[k],
                              [u[k] for u in live], draws, seed + ord(k)) for k in ("A", "B")}
    byp = collections.defaultdict(list)
    for u in live:
        byp[u["puzzle"]].append(u)
    print(f"  --- 更保守:按【靶标】聚类({len(byp)} 个靶标)---")
    row["by_puzzle"] = {k: line({"A": "A 按靶标", "B": "B 按靶标"}[k],
                                [st.median([y[k] for y in v]) for v in byp.values()],
                                draws, seed + 20) for k in ("A", "B")}
    A, B = row["primary"]["A"], row["primary"]["B"]
    row["verdict"] = ("null" if not A["excludes_zero_positive"]
                      else "B_negative" if B["excludes_zero_negative"]
                      else "B_positive" if B["excludes_zero_positive"] else "B_crosses_zero")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "panel_f1_fourth.json"))
    ap.add_argument("--judged", default=os.path.join(HERE, "a15_judged_ipknot_pkplex.json"))
    ap.add_argument("--knotty", default=os.path.join(HERE, "a15_judged_knotty.json"))
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "a15.json"))
    a = ap.parse_args()
    D = json.load(open(os.path.join(HERE, "a15_declaration.json")))
    from arbitration_confirm import load_fresh
    rec, _sp, meta = load_fresh(a.src, "panel_recall_product")
    assert meta.get("fourth_per_cell"), "不是第四批,拒绝"
    cells = collections.defaultdict(list)
    for r in rec:
        cells[(r["puzzle"], r["method"])].append(r)

    J = {tuple(k.split("\t")): v for k, v in json.load(open(a.judged)).items()}
    both = {k: st.mean([v["ipknot"], v["pkplex"]]) for k, v in J.items()
            if v.get("ipknot") is not None and v.get("pkplex") is not None}
    print(f"第四批 {len(rec)} 条去重分子,{len(cells)} 个格;"
          f"IPknot+RNAPKplex 都成功的 {len(both)}/{len(J)} = {len(both)/max(1,len(J)):.1%}")
    res = {"declaration": D, "n_judged": len(J), "n_both_ok": len(both)}
    res["main"] = decompose(cells, both, a.draws, a.seed, "主口径 q = (IPknot + RNAPKplex)/2")

    if os.path.exists(a.knotty):
        K = {tuple(k.split("\t")): v for k, v in json.load(open(a.knotty)).items()}
        kq = {k: v["knotty"] for k, v in K.items() if v.get("knotty") is not None}
        print(f"\n  Knotty 覆盖率:{len(kq)}/{len(K)} = {len(kq)/max(1,len(K)):.1%}"
              f"  —— 按声明只作次口径,【不进】主口径的平均")
        res["knotty_secondary"] = decompose(cells, kq, a.draws, a.seed, "次口径 q = Knotty 单家(覆盖率如实上报)")
        res["knotty_coverage"] = len(kq) / max(1, len(K))
    else:
        print(f"\n  Knotty 文件还没有({a.knotty}) —— 次口径留空,不是零")
        res["knotty_secondary"] = None

    print(f"\n=== 判定(按声明)===")
    m = res["main"]
    print(f"  主口径:{m['verdict']}")
    if m.get("primary"):
        for k in ("A", "B"):
            x = m["primary"][k]
            print(f"    {k} 中位 {x['median']:+.4f} [{x['ci'][0]:+.4f},{x['ci'][1]:+.4f}]"
                  f"{' 排除 0' if x['excludes_zero'] else ' 跨过 0'}")
    print(f"  对照 A13 人群1(真实池,RNet):A +0.1036 排除 0;B +0.0150 跨过 0")
    print(f"  与 A14(试管)的对比要人工写进周报 —— 两路方向一致 = 结构性结论;相反 = 判官依赖,以试管为准。")
    print(f"  【本次的弱点,必须随结论一起走】:这两家判官与面板共用能量模型族,所以一致【不能】读成「面板是对的」。")
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n写出 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
