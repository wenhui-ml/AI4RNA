#!/usr/bin/env python3
"""Now that the screen decides WHETHER to answer, what should decide WHICH answer?

THE GAP THIS ADDRESSES. With the screen shipped, we deliver all 57 OpenKnot targets at a median crossed-pair
F1 of 0.826 while the best candidate sitting in the same pool is around 0.95. That 0.13 is now the single
largest remaining gap, and it is a SELECTION gap: the sequences are already generated and already
adjudicated, we simply hand back the wrong one. Three separate measurements this week said the panel decides
whether to answer far better than which answer to give -- the grade selects no better than random at the
product's pool size, ranking screened survivors by (k, mean recall) was not better than taking the first one
through, and the panel's wet-lab correlation is roughly half RibonanzaNet's.

WHY IT CAN BE ANSWERED WITHOUT RUNNING ANYTHING. `criterion_vs_tube.json` holds 4,560 designs that were
actually synthesised, each carrying our panel's scores, RibonanzaNet's crossed-pair F1, AND a measured
OpenKnot score. So the product's pipeline -- screen, then rank -- can be replayed on real molecules with the
outcome known, for every candidate ranking rule, at no cost.

    ARMS, all applied only to candidates that clear the shipped screen (mean recall >= 0.1):
      first        pool order -- what `knot_screened` does today. Field pool order is arbitrary, so this
                   doubles as the random-selection control.
      grade        (panel_k, then mean recall) -- what `knot_confirmed` does today
      recall       mean crossing recall alone
      rnet         RibonanzaNet crossed-pair F1
      oracle       the best wet-lab score available -- an upper bound no selector can reach, for scale

PRE-REGISTERED. Independent unit is the target. Within each target, draw N candidates at random, apply the
screen, select by each rule, record the SELECTED design's measured wet-lab score; repeat, average within
target, then compare across targets. Never pool draws -- that is the pseudo-replication that produced this
project's retracted p = 2.05e-28.

    V1  Does ranking by RibonanzaNet beat the grade we ship? Paired within target, Wilcoxon, one-sided.
        PASS: >= 1.0 OpenKnot point and p < 0.025.
    V2  Does ANY panel-internal rule beat first/random? If not, the panel cannot rank at all and the only
        way to close the gap is an outside model.
    V3  How much is on the table -- oracle minus first.

    THE DECISION THIS FORCES, recorded before the numbers so it cannot be rationalised afterwards. If V1
    passes, switching the selector to RibonanzaNet costs us something real: RNet is currently our only
    external-ish yardstick, and a model cannot both choose our answer and grade it. Adopting it as the
    selector means every future leaderboard number we quote is circular, and we would need a different judge
    -- the wet-lab-anchored panel separation, or an outside predictor we do not use in the loop. That price
    is not obviously worth paying for a gain of one or two OpenKnot points, and this run is what tells us the
    size of the thing being bought.

COST: $0. No solver, no oracle calls, no model inference -- every quantity is already on disk.
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

ALPHA = 0.025
SCREEN = 0.1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "criterion_vs_tube.json"))
    ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "selector_after_screen.json"))
    a = ap.parse_args()

    from m2r_pairwise import wilcoxon

    D = json.load(open(a.src))["designs"]
    by = collections.defaultdict(list)
    for d in D:
        by[d["puzzle"]].append(d)
    print(f"真实分子 {len(D)} 条,{len(by)} 个靶标;筛 = 平均交叉召回 >= {SCREEN}\n")

    rules = {
        "first":  lambda c: 0.0,                                   # pool order == arbitrary == random
        "grade":  lambda c: (c["panel_k"], c["panel_mean"]),
        "recall": lambda c: c["panel_mean"],
        "rnet":   lambda c: c["f1c"],
    }
    rng = random.Random(a.seed)
    out = {}
    for N in (20, 50, 100):
        per = {k: {} for k in list(rules) + ["oracle"]}
        for p, rs in sorted(by.items()):
            if len(rs) < N:
                continue
            acc = collections.defaultdict(list)
            for _ in range(a.draws):
                pool = rng.sample(rs, N)
                surv = [c for c in pool if c["panel_mean"] >= SCREEN]
                if not surv:
                    continue
                for name, key in rules.items():
                    pick = surv[0] if name == "first" else max(surv, key=key)
                    acc[name].append(pick["wet"])
                acc["oracle"].append(max(c["wet"] for c in surv))
            for name in per:
                if acc[name]:
                    per[name][p] = st.mean(acc[name])
        ps = sorted(set.intersection(*[set(per[k]) for k in per]))
        print(f"=== 池大小 N={N}({len(ps)} 个靶标)===")
        print(f"  {'排序规则':10s} {'实测中位':>9s} {'对 first':>9s} {'更好':>8s} {'p':>10s}")
        row = {}
        base = [per["first"][p] for p in ps]
        for name in ("first", "grade", "recall", "rnet", "oracle"):
            v = [per[name][p] for p in ps]
            if name == "first":
                print(f"  {name:10s} {st.median(v):9.2f} {'—':>9s} {'—':>8s} {'—':>10s}")
                row[name] = {"median": st.median(v)}
                continue
            d = [x - y for x, y in zip(v, base)]
            w, pv, _ = wilcoxon(d, "greater")
            print(f"  {name:10s} {st.median(v):9.2f} {st.median(d):+9.2f} "
                  f"{sum(1 for x in d if x>0):5d}/{len(d):<3d} "
                  f"{'n/a' if pv is None else f'{pv:.3g}':>10s}")
            row[name] = {"median": st.median(v), "vs_first": st.median(d), "p": pv,
                         "wins": sum(1 for x in d if x > 0), "n_targets": len(d)}
        # V1: rnet against the grade we actually ship
        d1 = [per["rnet"][p] - per["grade"][p] for p in ps]
        w, p1, _ = wilcoxon(d1, "greater")
        v1 = bool(p1 is not None and p1 < ALPHA and st.median(d1) >= 1.0)
        print(f"  V1  rnet − grade(出厂排序)= {st.median(d1):+.2f} 分,"
              f"更好 {sum(1 for x in d1 if x>0)}/{len(d1)},p = {'n/a' if p1 is None else f'{p1:.3g}'}"
              f"  -> {'达到 >=1.0 分的门槛' if v1 else '未达门槛'}")
        row["V1"] = {"diff": st.median(d1), "p": p1, "pass": v1}
        out[str(N)] = row
        print()

    print(f"=== 读 oracle 这一行时的陷阱 ===")
    print(f"  oracle 取的是【存活者里实测分最高的那条】,所以它带着胜者诅咒:在合成检验里,"
          f"即使湿实验分与任何判据都无关,")
    print(f"  oracle 仍比 first 高 +3.67 分。**oracle − first 不是可达空间,它是可达空间加上噪声取最大的膨胀。**")
    print(f"  任何排序规则的收益要和 first 比,不要和 oracle 比。\n")

    n20 = out.get("20", {})
    print(f"=== 判读(以 N=20 为准,最接近产品实际筛后存活量)===")
    if n20:
        head = n20["oracle"]["median"] - n20["first"]["median"]
        got = n20["rnet"]["vs_first"]
        print(f"  桌面上总共有 {head:+.2f} 分(oracle − first);最好的可实现规则拿到 {got:+.2f} 分,"
              f"即 {100*got/head if head else 0:.0f}%")
        print(f"  面板自己的规则:grade {n20['grade']['vs_first']:+.2f} 分,recall {n20['recall']['vs_first']:+.2f} 分")
        if not n20["V1"]["pass"]:
            print(f"  -> 换 RibonanzaNet 当选择器【达不到 1.0 分门槛】。代价(失去唯一的外部尺子、"
                  f"今后所有排行榜数字变成自证)换不回这点收益,不做。")
        else:
            print(f"  -> 达到门槛。但换选择器意味着 RNet 不能再当我们的裁判 —— 这是产品决定,不是实验结论。")

    json.dump({"screen": SCREEN, "draws": a.draws, "by_N": out}, open(a.out, "w"),
              ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
