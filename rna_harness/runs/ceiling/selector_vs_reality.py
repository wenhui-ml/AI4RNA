#!/usr/bin/env python3
"""The product does not report a correlation -- it PICKS ONE SEQUENCE. Which picker gets the best molecule?

WHY THIS AND NOT ANOTHER CORRELATION. `openknot_grade_vs_shape.py` asked whether our grade tracks the
wet-lab score and found that it does (+0.155, p=1.4e-05), that a learned model tracks it more than twice as
well (RNet +0.364), and that our residual after controlling for RNet is +0.037 at p=0.030 -- short of the
pre-registered bar. Read literally that says "stop shipping the thermodynamic panel".

But a correlation over a whole population and a CHOICE AT THE TOP OF ONE POOL are different questions. The
product generates candidates for one target and returns one of them. A signal can be weak across the range
and still decide the top of the pool, and a signal can be strong across the range and still be flat where it
matters. Nobody has measured the thing the product actually does, against reality, ever.

THE SIMULATION. For one target, draw a pool of N candidates from the designs that were really synthesised
and really probed for that target. Each selector picks one. The wet-lab OpenKnot Score of the picked design
is that selector's payoff for that draw. Repeat over draws and targets. This is the product's decision,
scored by a tube.

    ARMS
      random          pick uniformly            -- the floor; if a selector cannot beat this, it is decoration
      grade           our shipped ordinal (0 not_confirmed .. 4 certified), ties broken at random
      rnet            RNet_F1_crossed_pair alone -- the learned criterion, i.e. "replace our panel"
      grade+rnet      our grade first, RNet breaks ties -- our shipped shape with the new signal added
      rnet+grade      RNet first, our grade breaks ties -- the learned criterion with ours as a tie-break

    Ties are broken at RANDOM inside each arm before the tie-break rule applies, so no arm silently inherits
    the file's row order. Our grade has 5 levels and produces ties constantly; RNet is continuous and almost
    never ties. That asymmetry is the whole point of testing both orderings.

POPULATION. The 21,801 calibrated-topology designs already graded and stored by
`openknot_grade_vs_shape.json` -- no sequence is re-folded, so this run is minutes and $0. 48 targets,
median 338 candidates each. H-type is excluded for the same reason as before: the product adjudicates it by
a different rule.

PRE-REGISTERED, before any draw was taken. Paired within (target, draw); Wilcoxon signed-rank over the 48
targets on each target's mean payoff difference. Four tests, Bonferroni threshold 0.05/4 = 0.0125.

    P1  grade beats random.  KILL if it fails: our selection would not be doing anything real, and every
        claim about candidate selection in this product would have to be withdrawn.
    P2  rnet beats grade. Expected to hold -- H3 already showed RNet is the stronger signal. Stated so that
        P3 cannot be read as a win by omission.
    P3  grade+rnet beats rnet alone. THIS IS THE DECISION. If our grade adds nothing on top of the learned
        criterion at the point of choice, the honest product move is to hand selection to RNet and keep our
        panel only for what it uniquely provides -- the reported error rates, the ceilings and the price.
    P4  grade+rnet beats grade alone. If this fails while P3 passes, the two signals are interchangeable
        rather than complementary, and ordering is a matter of taste.

    Effect size is reported in OpenKnot Score POINTS, not just as a p-value; a significant gain of 0.2
    points on a 0-100 scale is not a product improvement and will be called one.

POOL SIZE. N = 20 is the pre-registered primary: it is the order of magnitude of the candidate pool the
solver actually produces per target inside one budget tier. N = 5 and N = 50 are reported as declared
sensitivities, because a selector's value depends on how much there is to choose between.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from m2r_pairwise import wilcoxon                                              # noqa: E402

CALIBRATED = ("kissing / chain", "higher-order", "crossing (non-H)")
ALPHA = 0.05 / 4
ARMS = ("random", "grade", "rnet", "grade+rnet", "rnet+grade")


def pick(arm, pool, rng):
    """Return the chosen design. Random tie-breaking is applied by shuffling before any max()."""
    p = list(pool)
    rng.shuffle(p)                       # so no arm inherits file order through a tie
    if arm == "random":
        return p[0]
    if arm == "grade":
        return max(p, key=lambda r: r["grade"])
    if arm == "rnet":
        return max(p, key=lambda r: r["rnet"])
    if arm == "grade+rnet":
        return max(p, key=lambda r: (r["grade"], r["rnet"]))
    if arm == "rnet+grade":
        return max(p, key=lambda r: (r["rnet"], r["grade"]))
    raise ValueError(arm)


def run(rows, n_pool, draws, seed):
    by_t = {}
    for r in rows:
        by_t.setdefault(r["puzzle"], []).append(r)
    by_t = {k: v for k, v in by_t.items() if len(v) >= n_pool}
    per_target = {a: {} for a in ARMS}
    rng = random.Random(seed)
    for t, cand in sorted(by_t.items()):
        acc = {a: [] for a in ARMS}
        for _ in range(draws):
            pool = rng.sample(cand, n_pool)
            for a in ARMS:
                acc[a].append(pick(a, pool, rng)["score"])
        for a in ARMS:
            per_target[a][t] = st.mean(acc[a])
    return per_target, len(by_t)


def compare(per_target, a, b, label):
    ts = sorted(set(per_target[a]) & set(per_target[b]))
    d = [per_target[a][t] - per_target[b][t] for t in ts]
    w, p, n = wilcoxon(d, "greater")
    pos = sum(1 for x in d if x > 0)
    verdict = "通过" if (p is not None and p < ALPHA) else "未通过"
    print(f"  {label:26s} 中位差 {st.median(d):+6.3f} 分   {pos:2d}/{len(d)} 个靶标更好   "
          f"p = {'n/a' if p is None else f'{p:.3g}':>9s}   {verdict}")
    return {"median_points": st.median(d), "mean_points": st.mean(d), "n_targets": len(d),
            "n_pos": pos, "p": p, "pass": bool(p is not None and p < ALPHA)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "openknot_grade_vs_shape.json"))
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--out", default=os.path.join(HERE, "selector_vs_reality.json"))
    a = ap.parse_args()

    rows = [r for r in json.load(open(a.src))["rows"] if r["topology"] in CALIBRATED]
    print(f"来源:{len(rows)} 条已判等级的真实设计(不重新折叠,$0)")
    print(f"每次抽签 = 从一个靶标的候选里抽 N 条组成候选池,各臂各挑一条,记它的【实测分】\n")

    out = {}
    for n_pool in (20, 5, 50):
        tag = "主分析" if n_pool == 20 else "敏感性"
        per_target, n_t = run(rows, n_pool, a.draws, a.seed + n_pool)
        print(f"=== N = {n_pool}({tag}),{n_t} 个靶标 × {a.draws} 次抽签 ===")
        for arm in ARMS:
            v = list(per_target[arm].values())
            print(f"  {arm:12s} 平均实测分 {st.mean(v):6.2f}")
        print()
        res = {"n_targets": n_t,
               "mean_score": {arm: st.mean(list(per_target[arm].values())) for arm in ARMS}}
        res["P1"] = compare(per_target, "grade", "random", "P1  等级 > 随机")
        res["P2"] = compare(per_target, "rnet", "grade", "P2  RNet > 等级")
        res["P3"] = compare(per_target, "grade+rnet", "rnet", "P3  等级+RNet > 纯 RNet")
        res["P4"] = compare(per_target, "grade+rnet", "grade", "P4  等级+RNet > 纯等级")
        res["extra_rnet_first"] = compare(per_target, "rnet+grade", "rnet",
                                          "(附)RNet+等级 > 纯 RNet")
        out[f"N={n_pool}"] = res
        print()

    p = out["N=20"]
    print("=== 预注册判决(主分析 N=20,阈值 %.4f)===" % ALPHA)
    if not p["P1"]["pass"]:
        print("  KILL —— 等级挑不过随机挑,产品关于候选选择的全部主张都必须撤回")
    else:
        print(f"  P1 通过:等级确实在挑,不是装饰(+{p['P1']['median_points']:.3f} 分)")
    print(f"  P2 {'通过' if p['P2']['pass'] else '未通过'}:RNet {'确实更强' if p['P2']['pass'] else '并没有更强'}"
          f"({p['P2']['median_points']:+.3f} 分)")
    print(f"  P3 {'通过 —— 我们的等级在【选择】上仍有增量,面板该留' if p['P3']['pass'] else '未通过 —— 在选择这件事上我们叠加不出东西,选择该交给 RNet'}"
          f"({p['P3']['median_points']:+.3f} 分)")
    print(f"  P4 {'通过' if p['P4']['pass'] else '未通过'}({p['P4']['median_points']:+.3f} 分)")
    print(f"\n  提醒:分数是 0-100 的实测分。显著但只有零点几分的差,不叫产品改进。")

    json.dump({"alpha": ALPHA, "draws": a.draws, "seed": a.seed, "results": out},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
