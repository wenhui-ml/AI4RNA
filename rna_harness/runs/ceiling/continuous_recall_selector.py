#!/usr/bin/env python3
"""We threshold a continuous quantity at tau=1.0 and then complain our selector is coarse. Test that.

WHERE THIS COMES FROM. `selector_vs_reality.py` measured what the product actually does -- pick one sequence
out of a pool -- against wet-lab OpenKnot Scores, and our shipped grade FAILED its own kill criterion at the
pre-registered pool size (N=20: +0.322 points over random, p=0.0448 against a 0.0125 threshold). It passed
at N=5 (+0.783, p=7e-06) and did nothing at N=50 (+0.025, p=0.281).

That pattern is not random noise, and it points at a mechanism rather than at invalidity. Our grade has five
levels. In a pool of 5 that is enough resolution to order the pool; in a pool of 50 most candidates share the
top level and "pick the maximum" decays into "pick at random among the top tier". RNet is continuous and
keeps discriminating however large the pool gets.

AND WE MADE THE COARSENESS OURSELVES. `_knot_recall` computes, for each oracle, the FRACTION of the target's
crossing pairs that the oracle recovered -- a number in [0, 1] -- and then throws it away:

    full = [k for k, v in out.items() if v is not None and v >= 1.0]      # tau = 1.0, the strictest point

Everything downstream sees only how many oracles hit 1.0. A design where all three oracles recover 90% of
the crossing pairs and one where all three recover 10% are both "k=0" and are indistinguishable to the
selector. That threshold was chosen to make a VERDICT defensible, which is a different job from ranking
candidates, and it was never re-examined when the same quantity started being used to choose between them.

WHAT IS TESTED. The same tube-scored selection game, with our own signal in continuous form added as arms.
No new information is used -- `mean_recall` is computed from numbers `_knot_recall` already produces and
discards.

    grade         the shipped 5-level ordinal                                    (the incumbent)
    recall        mean over the three oracles of fractional crossing-pair recall  (same evidence, unthresholded)
    rnet          RNet_F1_crossed_pair                                            (the learned criterion)
    recall+rnet   ours first, RNet breaks ties
    rnet+recall   RNet first, ours breaks ties

PRE-REGISTERED, before any fold in this script was run. Paired within target; Wilcoxon over targets on each
target's mean payoff; primary pool size N=20 as before; N=5 and N=50 declared sensitivities. Four tests,
Bonferroni threshold 0.05/4 = 0.0125. Effect sizes in OpenKnot Score points, and a significant gain below
0.5 points will be called what it is: statistically real and practically nothing.

    Q1  recall beats random at N=20. This is the kill criterion `grade` failed. If `recall` passes it, the
        failure was our thresholding and not our evidence, and the product's selector should stop
        thresholding.
    Q2  recall beats grade at N=20. The direct measurement of what tau=1.0 costs at the point of choice.
    Q3  recall beats random at N=50. `grade` was flat here (+0.025). If a continuous score stays alive as
        the pool grows, the saturation explanation is confirmed rather than assumed.
    Q4  recall+rnet beats rnet. The integration question, re-asked with our signal at full resolution.
        `grade+rnet` LOST to plain rnet by -0.439 points, so this is not a formality.

    NOTE ON WHAT THIS CANNOT SHOW. Even if every Q passes, RNet remains the stronger single signal (P2,
    +1.216 points, already measured). Nothing here reopens that. The question is only whether our own
    evidence is worth more than we have been extracting from it.

COST: one fold of 27,362 designs, $0 API, local binaries.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics as st
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from m2r_pairwise import wilcoxon                                              # noqa: E402
from openknot_grade_vs_shape import build_population, DATA, CALIBRATED         # noqa: E402

ALPHA = 0.05 / 4
ARMS = ("random", "grade", "recall", "rnet", "recall+rnet", "rnet+recall")


def work(item):
    """Grade one design AND keep the fractional recall the shipped rule discards."""
    from rna_harness.cognitive.certify import _knot_recall, _panel_verdict
    i, seq, tgt = item
    try:
        kr = _knot_recall(seq, tgt)
        pv = _panel_verdict(seq, tgt)
    except Exception as e:                                                     # noqa: BLE001
        return {"i": i, "error": repr(e)[:160]}
    vals = [v for v in kr["per_oracle"].values() if v is not None]
    return {"i": i, "grade": 4 if pv["unanimous"] else len(kr["confirmed_by"]),
            "k": len(kr["confirmed_by"]), "per_oracle": kr["per_oracle"],
            "mean_recall": (st.mean(vals) if vals else 0.0),
            "max_recall": (max(vals) if vals else 0.0), "n_cross": kr["n_cross"]}


def pick(arm, pool, rng):
    p = list(pool)
    rng.shuffle(p)
    if arm == "random":
        return p[0]
    key = {"grade": lambda r: r["grade"],
           "recall": lambda r: r["mean_recall"],
           "rnet": lambda r: r["rnet"],
           "recall+rnet": lambda r: (r["mean_recall"], r["rnet"]),
           "rnet+recall": lambda r: (r["rnet"], r["mean_recall"])}[arm]
    return max(p, key=key)


def play(rows, n_pool, draws, seed):
    by_t = {}
    for r in rows:
        by_t.setdefault(r["puzzle"], []).append(r)
    by_t = {k: v for k, v in by_t.items() if len(v) >= n_pool}
    per = {a: {} for a in ARMS}
    rng = random.Random(seed)
    for t, cand in sorted(by_t.items()):
        acc = {a: [] for a in ARMS}
        for _ in range(draws):
            pool = rng.sample(cand, n_pool)
            for a in ARMS:
                acc[a].append(pick(a, pool, rng)["score"])
        for a in ARMS:
            per[a][t] = st.mean(acc[a])
    return per, len(by_t)


def compare(per, a, b, label):
    ts = sorted(set(per[a]) & set(per[b]))
    d = [per[a][t] - per[b][t] for t in ts]
    w, p, n = wilcoxon(d, "greater")
    pos = sum(1 for x in d if x > 0)
    print(f"  {label:28s} 中位差 {st.median(d):+6.3f} 分   {pos:2d}/{len(d)} 靶标更好   "
          f"p = {'n/a' if p is None else f'{p:.3g}':>9s}   {'通过' if (p is not None and p < ALPHA) else '未通过'}")
    return {"median_points": st.median(d), "n_targets": len(d), "n_pos": pos, "p": p,
            "pass": bool(p is not None and p < ALPHA)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--cache", default=os.path.join(HERE, "continuous_recall_folds.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "continuous_recall_selector.json"))
    a = ap.parse_args()

    pop = build_population(DATA)
    print(f"总体:{len(pop)} 条去重设计", flush=True)

    if os.path.exists(a.cache):
        cached = json.load(open(a.cache))
        print(f"  复用已折叠结果 {a.cache}({len(cached)} 条)")
        for i, p in enumerate(pop):
            c = cached.get(str(i))
            if c:
                p.update(c)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        jobs = [(i, p["seq"], p["target"]) for i, p in enumerate(pop)]
        got, errs, t1, store = 0, 0, time.time(), {}
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for f in as_completed([ex.submit(work, j) for j in jobs]):
                r = f.result()
                got += 1
                if "error" in r:
                    errs += 1
                else:
                    pop[r["i"]].update(r)
                    store[str(r["i"])] = {k: r[k] for k in ("grade", "k", "mean_recall",
                                                            "max_recall", "n_cross")}
                if got % 4000 == 0 or got == len(jobs):
                    el = (time.time() - t1) / 60
                    print(f"    {got}/{len(jobs)}  {el:.1f} min  失败 {errs}", flush=True)
        json.dump(store, open(a.cache, "w"))

    rows = [p for p in pop if "mean_recall" in p and p["topology"] in CALIBRATED]
    print(f"主分析(已标定拓扑){len(rows)} 条\n")

    print("=== 描述:连续召回率把 not_confirmed 那一档拆开了吗 ===")
    z = [r for r in rows if r["grade"] == 0]
    qs = sorted(r["mean_recall"] for r in z)
    for lab, lo, hi in (("最低 25%", 0, len(qs)//4), ("中间 50%", len(qs)//4, 3*len(qs)//4),
                        ("最高 25%", 3*len(qs)//4, len(qs))):
        cut = set(qs[lo:hi])
        s = [r["score"] for r in z if r["mean_recall"] in cut]
        if s:
            print(f"  grade=0 里 {lab}(平均召回 {st.mean(qs[lo:hi]):.3f}):"
                  f"n={len(s):5d}  实测分中位 {st.median(s):5.1f}")

    out = {}
    for n_pool in (20, 5, 50):
        per, n_t = play(rows, n_pool, a.draws, a.seed + n_pool)
        print(f"\n=== N = {n_pool}{'(主分析)' if n_pool == 20 else '(敏感性)'},{n_t} 个靶标 ===")
        for arm in ARMS:
            print(f"  {arm:14s} 平均实测分 {st.mean(list(per[arm].values())):6.2f}")
        res = {"mean_score": {arm: st.mean(list(per[arm].values())) for arm in ARMS}}
        res["Q1"] = compare(per, "recall", "random", "Q1  连续召回 > 随机")
        res["Q2"] = compare(per, "recall", "grade", "Q2  连续召回 > 现行等级")
        res["Q4"] = compare(per, "recall+rnet", "rnet", "Q4  连续召回+RNet > 纯 RNet")
        res["extra"] = compare(per, "rnet+recall", "rnet", "(附)RNet+连续召回 > 纯 RNet")
        out[f"N={n_pool}"] = res

    print(f"\n=== 预注册判决(阈值 {ALPHA:.4f})===")
    p20, p50 = out["N=20"], out["N=50"]
    print(f"  Q1 {'通过 —— 现行等级失败的原因是【我们自己的阈值】,不是证据本身' if p20['Q1']['pass'] else '未通过 —— 不是阈值的问题,证据本身就挑不出东西'}"
          f"({p20['Q1']['median_points']:+.3f} 分)")
    print(f"  Q2 {'通过 —— tau=1.0 在【选择】上确实在丢信息' if p20['Q2']['pass'] else '未通过 —— 砍成 5 档没有代价'}"
          f"({p20['Q2']['median_points']:+.3f} 分)")
    print(f"  Q3 {'通过 —— 池子变大也不饱和,饱和解释成立' if p50['Q1']['pass'] else '未通过 —— 大池子里连续量同样失效'}"
          f"(N=50 时 {p50['Q1']['median_points']:+.3f} 分)")
    print(f"  Q4 {'通过 —— 我们的证据在 RNet 之上仍有增量' if p20['Q4']['pass'] else '未通过 —— 叠不出东西,选择仍应交给 RNet'}"
          f"({p20['Q4']['median_points']:+.3f} 分)")
    print("\n  提醒:0-100 分制。显著但不足 0.5 分的差,是统计事实,不是产品改进。")

    json.dump({"alpha": ALPHA, "results": out}, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
