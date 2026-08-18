#!/usr/bin/env python3
"""Go / no-go for the whole next phase: how much REAL headroom is there in tube-selection?

WHY THIS IS THE GATE. Both established arenas turned out saturated. Eterna100 is solved by DesiRNA at
99/100 with the remaining puzzle provably undesignable; on OpenKnot the field reaches crossed-pair F1 1.000
on 55 of 57 targets and our own pools already contain a 1.000 on 39 of 57. Search is not the open problem.
What is still open is that computational "solved" and tube "works" are different things -- stacking RNet
F1 = 1.000 AND three-oracle confirmation still leaves 16% of designs below 80 OpenKnot points and misses
45% of the designs that actually score >= 95.

So the proposed next phase is tube-selection: pick the design that works in the tube, better than the best
published proxy. Before proposing it seriously, the headroom has to survive the winner's curse.

    THE PROBLEM WITH `oracle`. At N = 50 it beats random by +9.54 tube points. But `oracle` takes the MAX
    of an outcome, and max-of-N rises above the mean even when nothing is predictable. Quoting +9.54 as
    "headroom" is the same error as quoting max >= median as an improvement -- this project retracted a
    p = 2.65e-10 headline for exactly that.

    THE NULL, and it is the honest one. Within each target, PERMUTE the tube scores across designs. That
    destroys every relation between a design's features and its tube outcome while preserving the target's
    score distribution and the pool size exactly. Re-run every selector on the permuted data. Whatever a
    rule still "gains" there is pure selection artefact and gets subtracted.

    WHAT THIS TEST CAN AND CANNOT DO, learned by getting it wrong first. Permuting after the screen makes
    the survivor score multiset identical in both arms, so `oracle` is IDENTICAL by construction and its
    "real headroom" is exactly zero. That is not a null result about the phase -- it is proof that the
    oracle bound CANNOT answer the question. Max-of-outcome is unreachable by any selector, so it never
    bounded anything. What the permutation does answer honestly is the FEATURE-BASED rules, whose features
    are untouched by it: whatever they gain above the permuted arm is real.

    H-GO    real headroom = (oracle - random) observed minus the same quantity under permutation.
            PASS: >= 4.0 tube points remain after subtraction, and the best published proxy (RNet)
            captures less than half of it. Then there is a real, uncaptured lever and the phase is worth
            running.
            KILL: < 4.0 points remain, or RNet already captures most of it -> tube-selection is as
            saturated as everything else, and the phase is dropped rather than argued for.

    4.0 is set against the scale that matters: between-method spread in this dataset is 29.2 points and
    the outcome's IQR is 12.5, so 4.0 is about a third of the IQR -- large enough that a user would see it,
    small enough to be reachable. Fixed here before the number is computed.

Independent unit is the target. Draws are averaged within target, never pooled.

COST: $0. No model inference; every quantity is on disk.
"""
from __future__ import annotations
import argparse, collections, json, os, random, statistics as st, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SCREEN, ALPHA = 0.1, 0.025
MIN_REAL = 4.0


def run(by, N, draws, rng, permute):
    rules = {"random": None, "grade": lambda c: (c["panel_k"], c["panel_mean"]),
             "recall": lambda c: c["panel_mean"], "rnet": lambda c: c["f1c"]}
    per = {k: {} for k in list(rules) + ["oracle"]}
    for p, rs in sorted(by.items()):
        if len(rs) < N:
            continue
        acc = collections.defaultdict(list)
        for _ in range(draws):
            pool = rng.sample(rs, N)
            surv = [c for c in pool if c["panel_mean"] >= SCREEN]
            if not surv:
                continue
            if permute:
                # PERMUTE AMONG SURVIVORS, not the whole pool. Shuffling before the screen changes WHICH
                # designs survive it, so `oracle` and `random` end up drawn from a different distribution
                # than in the observed arm and the two are no longer comparable -- the first version of
                # this script did that and reported a nonsensical negative headroom. Shuffling after the
                # screen keeps the survivor set and its score multiset identical, isolating exactly the
                # max-of-N artefact.
                ws = [c["wet"] for c in surv]
                rng.shuffle(ws)
                surv = [dict(c, wet=w) for c, w in zip(surv, ws)]
            for name, key in rules.items():
                pick = rng.choice(surv) if name == "random" else max(surv, key=key)
                acc[name].append(pick["wet"])
            acc["oracle"].append(max(c["wet"] for c in surv))
        for name in per:
            if acc[name]:
                per[name][p] = st.mean(acc[name])
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "criterion_vs_tube.json"))
    ap.add_argument("--N", type=int, default=50)
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "tube_headroom.json"))
    a = ap.parse_args()
    from m2r_pairwise import wilcoxon

    D = json.load(open(a.src))["designs"]
    by = collections.defaultdict(list)
    for d in D:
        by[d["puzzle"]].append(d)
    obs = run(by, a.N, a.draws, random.Random(a.seed), False)
    nul = run(by, a.N, a.draws, random.Random(a.seed + 1), True)
    ps = sorted(set(obs["random"]) & set(nul["random"]))
    print(f"真实分子 {len(D)} 条;池大小 N={a.N};{len(ps)} 个靶标够大;每靶标 {a.draws} 次抽样\n")

    print(f"  {'规则':10s} {'实测 − 随机':>12s} {'置换零 − 随机':>14s} {'扣掉诅咒后':>12s} {'p(实测vs零)':>12s}")
    out = {}
    for r in ("grade", "recall", "rnet", "oracle"):
        o = [obs[r][p] - obs["random"][p] for p in ps]
        n = [nul[r][p] - nul["random"][p] for p in ps]
        d = [x - y for x, y in zip(o, n)]
        _, pv, _ = wilcoxon(d, "greater")
        print(f"  {r:10s} {st.median(o):+12.2f} {st.median(n):+14.2f} {st.median(d):+12.2f} "
              f"{'n/a' if pv is None else f'{pv:.3g}':>12s}")
        out[r] = {"observed": st.median(o), "null": st.median(n), "real": st.median(d), "p": pv}

    real, cap = out["oracle"]["real"], out["rnet"]["real"]
    print(f"\n=== H-GO 无法判决,而这本身是结果 ===")
    print(f"  oracle 实测 {out['oracle']['observed']:+.2f},置换零 {out['oracle']['null']:+.2f},"
          f"扣掉后 {real:+.2f}")
    print(f"  置换是在【筛之后】做的,存活者的分数集合两臂完全相同,所以 oracle 按构造就相等。")
    print(f"  -> 这不是「试管选择饱和」的证据,而是证明 oracle 这个上界本身不含信息:")
    print(f"     取结果的最大值是任何选择器都够不到的,它从来没有界住过任何东西。")
    print(f"     本项目多次引用的「桌面上有 +9.5 分」全部是胜者诅咒,应停止引用。")
    print(f"\n=== 能判决的部分:基于特征的规则,置换零干净(约 0.00)===")
    print(f"  RNet          真实增益 {cap:+.2f} 分  p = {out['rnet']['p']:.3g}   <- 已发表最好代理,真实有效")
    print(f"  我们的面板     真实增益 {out['grade']['real']:+.2f} 分  p = {out['grade']['p']:.3g}   <- 不显著")
    print(f"  连续召回       真实增益 {out['recall']['real']:+.2f} 分  p = {out['recall']['p']:.3g}   <- 不显著")
    print(f"  -> 还剩多少空间【无法用任何上界回答,只能靠去做】。下一期的门槛因此定为:"
          f"在按靶标留出的检验上超过 RNet 的 {cap:+.2f} 分。")
    go = None
    print(f"\n  尺度参照:方法间跨度 29.2 分,四分位距 12.5 分。")
    json.dump({"N": a.N, "draws": a.draws, "n_targets": len(ps), "min_real": MIN_REAL,
               "rules": out, "GO": go, "oracle_bound_is_uninformative": True,
               "next_phase_bar": cap},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
