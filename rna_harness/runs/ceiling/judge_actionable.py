#!/usr/bin/env python3
"""Step 2 of the bigger circle: does judge_share CHANGE A DECISION, or is it only bookkeeping?

Step 1 established that the share of a delivery verdict attributable to judge-choice exists and is stable --
97.1% over the deployment population, FIT 97.5% against TEST 96.7%, and the panel is not secretly one oracle.
That says the quantity is real. It does not say anyone should care.

This is the make-or-break. If judge_share is worth putting on a certificate, then certificates the panel
AGREES on must survive outside review better than certificates it splits on. RNAPKplex is the right outside
reviewer: different algorithm, different codebase, installed in a separate environment, and it has never been
part of any rule or selection in this project.

    agreement rises with the number of confirming oracles  ->  judge_share is an actionable confidence label
    agreement is flat                                      ->  it is bookkeeping. The frame stays descriptive
                                                               and the report must say so plainly

PRE-REGISTERED before any number here was computed:

    Primary: external agreement for unanimously-confirmed certificates minus split ones, > 10 points with a
    95% interval clear of zero.
    Secondary and stronger: agreement is monotone in the number of confirming oracles (1 -> 2 -> 3).

    KILL: gap under 10 points, or an interval straddling zero, means judge_share carries no information about
    whether an outside judge will agree.

THE CONFOUND, named before running because instrument.py cannot catch this class -- it catches wrong
implementations, not rigged designs. Unanimous certificates may simply sit on EASIER targets. Topology alone
moves external agreement enormously: 29.9% on H-type against 60.8% on kissing. So the comparison is
stratified by topology, and the number of crossing pairs is reported for both groups. An aggregate-only
result here would be worth nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)

from rna_harness.instrument import Clustered, constancy                        # noqa: E402

MIN_GAP = 0.10
MIN_PER_CELL = 8


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def boot_gap(a, b, n_boot=4000, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    A, B = np.array(a, float), np.array(b, float)
    if len(A) < 3 or len(B) < 3:
        return None
    d = [rng.choice(A, len(A)).mean() - rng.choice(B, len(B)).mean() for _ in range(n_boot)]
    d.sort()
    return {"mean": float(np.mean(d)), "lo95": float(d[int(0.025 * n_boot)]),
            "hi95": float(d[int(0.975 * n_boot)])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "judge_actionable.json"))
    a = ap.parse_args()

    ext = json.load(open(os.path.join(HERE, "external_judge.json")))
    ours = [r for r in ext["records"] if r.get("pop") == "ours" and r.get("ran")]
    print(f"总体:{len(ours)} 条我们交付的证书,外部裁判 {ext.get('judge')}(从未进入任何规则)\n")

    from rna_harness.benchmarks.rnainvbench import load_pseudobase
    from rna_harness.structure_utils import classify_topology
    # load_pseudobase yields DesignTask objects, not dicts
    tgt = {}
    for row in load_pseudobase():
        tid = str(getattr(row, "task_id", None) or getattr(row, "id", ""))
        t = str(getattr(row, "target", "") or "")
        if tid and t:
            tgt[tid] = t
    topo = {k: classify_topology(v)["topology"] for k, v in tgt.items()}

    rows = []
    for r in ours:
        cb = r.get("ours_confirmed_by") or []
        # external_judge stores ids as "pseudobase:11#ours"; the loader's task_id is "pseudobase:11".
        # The first version joined the raw strings, matched 0 of 243, and the script then printed
        # "not testable" -- a broken join reported as a finding, which is the exact class of defect this
        # project keeps catching. The join rate is asserted below rather than left to be read.
        base = str(r["id"]).split("#")[0]
        rows.append({"id": r["id"], "task_id": base, "k": len(cb), "confirmed_by": sorted(cb),
                     "external": bool(r.get("confirmed")),
                     "topology": topo.get(base, "unknown"),
                     "n_cross": r.get("n_cross")})
    miss = sum(1 for r in rows if r["topology"] == "unknown")
    print(f"  拓扑可判定 {len(rows)-miss}/{len(rows)}" + (f"(缺 {miss} 条,单列不并入)" if miss else ""))
    if miss > 0.2 * len(rows):
        raise SystemExit(f"!! 拓扑连接只成功 {len(rows)-miss}/{len(rows)} —— 这是 join 坏了,"
                         f"不是数据缺失。先修 join,不许把它报成【不可测】。")

    def rate(g):
        k = sum(1 for r in g if r["external"])
        lo, hi = wilson(k, len(g))
        return k, len(g), (k / len(g) if g else 0.0), lo, hi

    print(f"\n=== 主检验:一致 vs 分裂(先看合计,但合计【不作数】)===")
    un = [r for r in rows if r["k"] == 3]
    sp = [r for r in rows if 1 <= r["k"] <= 2]
    for nm, g in (("三家一致", un), ("分裂", sp)):
        k, n, p, lo, hi = rate(g)
        print(f"  {nm:8s} {k:>3}/{n:<3} = {p:6.1%}  [{lo:.1%}, {hi:.1%}]")
    agg_gap = (rate(un)[2] - rate(sp)[2]) if un and sp else None
    print(f"  合计差 {agg_gap:+.1%}" if agg_gap is not None else "  合计:样本不足")

    print(f"\n=== 单调性:外部同意率随确认家数怎么走 ===")
    mono = []
    for kk in (1, 2, 3):
        g = [r for r in rows if r["k"] == kk]
        if not g:
            continue
        k, n, p, lo, hi = rate(g)
        mono.append({"k": kk, "n": n, "rate": p, "lo": lo, "hi": hi})
        print(f"  {kk} 家确认  {k:>3}/{n:<3} = {p:6.1%}  [{lo:.1%}, {hi:.1%}]")
    is_mono = all(mono[i]["rate"] <= mono[i + 1]["rate"] for i in range(len(mono) - 1)) if len(mono) > 1 else False
    print(f"  单调上升:{is_mono}")

    print(f"\n=== 分层(决定性的一格:一致的证书会不会只是落在更容易的拓扑上)===")
    by = defaultdict(list)
    for r in rows:
        by[r["topology"]].append(r)
    strat, usable, need = [], 0, []
    for t, g in sorted(by.items(), key=lambda x: -len(x[1])):
        u = [r for r in g if r["k"] == 3]
        s = [r for r in g if 1 <= r["k"] <= 2]
        if len(u) < MIN_PER_CELL or len(s) < MIN_PER_CELL:
            print(f"  {t:16s} n={len(g):>3}  一致 {len(u):>2} / 分裂 {len(s):>3}  "
                  f"-> 单元格 <{MIN_PER_CELL},不做检验(不报为 null)")
            continue
        usable += 1
        ku, nu, pu, *_ = rate(u)
        ks, ns, ps, *_ = rate(s)
        b = boot_gap([r["external"] for r in u], [r["external"] for r in s])
        strat.append({"topology": t, "n_unan": nu, "n_split": ns, "rate_unan": pu,
                      "rate_split": ps, "gap": pu - ps, "boot": b})
        print(f"  {t:16s} 一致 {ku}/{nu} = {pu:.1%}   分裂 {ks}/{ns} = {ps:.1%}   "
              f"差 {pu-ps:+.1%}  95% [{b['lo95']:+.1%}, {b['hi95']:+.1%}]" if b else "")

    # composition balance, so an unstratified reader can see whether the groups were comparable at all
    nc_u = [r["n_cross"] for r in rows if r["k"] == 3 and r["n_cross"] is not None]
    nc_s = [r["n_cross"] for r in rows if 1 <= r["k"] <= 2 and r["n_cross"] is not None]
    if nc_u and nc_s:
        print(f"\n  交叉对数 一致 中位 {sorted(nc_u)[len(nc_u)//2]}   分裂 中位 {sorted(nc_s)[len(nc_s)//2]}")
    cl = Clustered(values=[1.0 if r["external"] else 0.0 for r in rows],
                   clusters=[r["task_id"] for r in rows])
    print(f"  重复层级:{cl.n_obs} 条 / {cl.n_clusters} 个靶标 —— {cl.warn_if_inflated() or '无膨胀'}")
    print(f"  外部判决本身有没有动态范围:{constancy([1.0 if r['external'] else 0.0 for r in rows])['where']}")

    ok = [s for s in strat if s["boot"] and s["boot"]["lo95"] > 0 and s["gap"] > MIN_GAP]
    print(f"\n=== 预注册判定 ===")
    print(f"  可做检验的拓扑 {usable} 个;其中差 >{MIN_GAP:.0%} 且区间不跨零的 {len(ok)} 个")
    verdict = usable > 0 and len(ok) == usable
    print(f"  -> {'judge_share 是可据以行动的置信标签' if verdict else 'judge_share 不携带外部正确性信息 —— 框架停在描述层,照实写'}")
    if usable == 0:
        print("     (注意:这是【不可测】而不是【已否定】—— 每个拓扑里一致的证书都太少)")
        # A null without a power statement is not a result. Say what it would actually take.
        need = []
        for t, g in sorted(by.items(), key=lambda x: -len(x[1])):
            u = sum(1 for r in g if r["k"] == 3)
            if len(g) >= 20:
                rate_u = u / len(g)
                req = int(MIN_PER_CELL / rate_u) if rate_u > 0 else None
                need.append({"topology": t, "n_now": len(g), "unan_now": u,
                             "unanimity_rate": rate_u, "n_designs_needed": req})
                print(f"     {t:16s} 现有 {len(g):>3} 条给出 {u} 条一致(率 {rate_u:.1%});"
                      f" 要凑够 {MIN_PER_CELL} 条需要约 {req if req else '∞'} 条设计")
        print("     -> 这不是想法问题,是【要多跑多少条设计】的问题,而这个数字现在有了。")

    # The non-monotonicity is a finding in its own right and must not be buried under the aggregate.
    if len(mono) >= 3 and mono[1]["rate"] < mono[0]["rate"]:
        print(f"\n  !! 单调性反了一段:1 家确认 {mono[0]['rate']:.1%} -> 2 家 {mono[1]['rate']:.1%}"
              f"(更差),只有 3 家那格是 {mono[2]['rate']:.1%} 而 n={mono[2]['n']}。")
        print("     所以【确认家数越多越可信】这个直觉,在能测的那两格上是【被推翻的】;")
        print("     合计那个 +64.4% 完全由 n=7 的一格撑着,这正是分层要防的东西。")

    json.dump({"n": len(rows), "power_requirement": (need if usable == 0 else None), "aggregate": {"unan": rate(un)[:3], "split": rate(sp)[:3], "gap": agg_gap},
               "monotone": mono, "is_monotone": is_mono, "stratified": strat,
               "n_testable_strata": usable, "n_passing": len(ok), "passes": bool(verdict),
               "rows": rows}, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
