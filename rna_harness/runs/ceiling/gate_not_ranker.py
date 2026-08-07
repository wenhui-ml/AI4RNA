#!/usr/bin/env python3
"""Is our criterion a GATE that we have been mis-using as a RANKER? A falsifiable reading of two refutations.

HOW THIS QUESTION WAS ARRIVED AT, stated plainly because it is post-hoc. Two explanations for our selector's
failure were proposed today and both were killed by their own pre-registrations:

    "the three boundaries share one cause: the tau=1.0 conjunction"   -> A2 failed (+0.000, 6/17, p=0.38)
    "selection fails because five levels saturate"                    -> Q2 failed (-0.079, p=0.634); the
                                                                         continuous version is no better

So neither the aggregation nor the resolution is the problem. But the same run left a fact that neither
explanation predicts. Inside the single grade the certificate calls `not_confirmed`, continuous recall
separates the wet lab cleanly:

    lowest quartile of recall (0.000)   wet-lab median 80.8
    middle half            (0.232)                    83.3
    top quartile           (0.647)                    85.4

A signal that moves the outcome by 4.6 points and yet buys nothing when picking the best of a pool can only
be doing that if it is FLAT WHERE THE PICKING HAPPENS. Choosing the maximum of a pool only ever reads the
top; if several candidates sit at grade 3 or recall 1.0 and we cannot order them, our signal contributes
nothing to that choice however well it orders the bottom.

If that is right, our criterion is a FILTER, and every measurement today that treated it as a RANKER was
asking it to do a job it is not shaped for -- including yesterday's grade-max change, which is a ranking
change and which has never been scored against a wet lab.

THREE PREDICTIONS THAT CAN KILL THIS. It is a post-hoc reading, so it gets a harder bar than a
pre-registered one: it must survive all three, and each is written before the numbers were computed.

    G1  FLAT AT THE TOP. Within the top stratum (mean_recall == 1.0), our signals must NOT predict the
        wet-lab score, while RNet still must. If ours still predicts inside the top stratum, the "flat where
        it matters" story is wrong and this reading is dead.

    G2  DISCRIMINATING AT THE BOTTOM. Across the whole range our signal must predict -- already known
        (+0.155) -- and specifically it must predict INSIDE the not_confirmed stratum, where the certificate
        currently says nothing at all. That is what a filter is supposed to know.

    G3  THE GATE MUST PAY. Selection arms, tube-scored, at the product's real pool size (median 131 screened
        candidates, range 71-200 -- measured from the product's own stored n_candidates_screened, NOT the
        N=20 that an earlier script wrongly assumed):

            rnet                       rank the whole pool by RNet                         (the challenger)
            gate(k>=1) -> rnet         drop candidates our floor rejects, then rank by RNet (gate + ranker)
            gate(recall>0) -> rnet     the softest possible version of our floor
            grade -> rnet              yesterday's shape: rank by grade, RNet breaks ties  (known to LOSE)

        PASS: a gate arm beats plain rnet, paired within target, Wilcoxon one-sided, p < 0.0167 (three
        gate arms, Bonferroni). A gate that removes candidates RNet would otherwise have picked is worth
        keeping; one that does not is decoration and must be described as such.

    KILL: G1 or G3 fails -> the gate reading joins the two dead explanations, and the honest conclusion
    becomes that our criterion adds nothing to selection in any arrangement we have been able to construct.
    That is a publishable fact about this product and it will be reported as one rather than explained away
    a third time.

COST: $0. Reuses the folds already cached by continuous_recall_selector.py; no sequence is folded again.
"""
from __future__ import annotations

import collections
import json
import os
import random
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from m2r_pairwise import spearman, wilcoxon                                    # noqa: E402
from openknot_grade_vs_shape import build_population, DATA, CALIBRATED         # noqa: E402

ALPHA = 0.05 / 3
POOLS = (131, 71, 200)


def load_rows(cache):
    pop = build_population(DATA)
    c = json.load(open(cache))
    for i, p in enumerate(pop):
        v = c.get(str(i))
        if v:
            p.update(v)
    return [p for p in pop if "mean_recall" in p and p["topology"] in CALIBRATED]


def within_target(rows, pred, min_n=20):
    g = collections.defaultdict(list)
    for r in rows:
        g[r["puzzle"]].append(r)
    out = {}
    for t, v in g.items():
        if len(v) >= min_n and len({r[pred] for r in v}) >= 2:
            s = spearman([r[pred] for r in v], [r["score"] for r in v])
            if s is not None:
                out[t] = s
    return out


def report(name, rho):
    if not rho:
        print(f"  {name:34s} 无可用簇")
        return {"n": 0}
    v = list(rho.values())
    w, p, n = wilcoxon(v, "greater")
    print(f"  {name:34s} 簇 {len(v):2d}  中位 rho {st.median(v):+.3f}  "
          f"正向 {sum(1 for x in v if x>0):2d}/{len(v):2d}  p = {'n/a' if p is None else f'{p:.3g}'}")
    return {"n": len(v), "median": st.median(v), "p": p}


def main():
    cache = os.path.join(HERE, "continuous_recall_folds.json")
    rows = load_rows(cache)
    print(f"复用已缓存的折叠结果:{len(rows)} 条(已标定拓扑),$0,不重折\n")

    top = [r for r in rows if r["mean_recall"] >= 1.0 - 1e-9]
    bot = [r for r in rows if r["grade"] == 0]
    print(f"=== G1 顶层是不是平的(mean_recall == 1.0 的 {len(top)} 条)===")
    g1 = {"ours_recall": report("我们:连续召回(顶层内)", within_target(top, "mean_recall")),
          "ours_grade": report("我们:证书等级(顶层内)", within_target(top, "grade")),
          "rnet": report("RNet(顶层内)", within_target(top, "rnet"))}
    ours_p = [g1[k].get("p") for k in ("ours_recall", "ours_grade")]
    ours_flat = all(p is None or p >= 0.05 for p in ours_p)
    rnet_alive = (g1["rnet"].get("p") is not None and g1["rnet"]["p"] < 0.05)
    print(f"  -> {'通过:顶层我们确实是平的,而 RNet 还在排序' if (ours_flat and rnet_alive) else 'G1 未通过 —— 闸门解读死了'}")

    print(f"\n=== G2 底层是不是有分辨力(证书判 not_confirmed 的 {len(bot)} 条)===")
    g2 = {"recall": report("连续召回(not_confirmed 内)", within_target(bot, "mean_recall")),
          "rnet": report("RNet(not_confirmed 内)", within_target(bot, "rnet"))}
    g2_ok = (g2["recall"].get("p") is not None and g2["recall"]["p"] < 0.05)
    print(f"  -> {'通过:证书说不出话的那一档里,我们其实知道谁更好' if g2_ok else 'G2 未通过'}")

    ARMS = ("rnet", "gate_k1_rnet", "gate_soft_rnet", "grade_rnet")

    def pick(arm, pool, rng):
        q = list(pool)
        rng.shuffle(q)
        if arm == "rnet":
            return max(q, key=lambda r: r["rnet"])
        if arm == "grade_rnet":
            return max(q, key=lambda r: (r["grade"], r["rnet"]))
        keep = [r for r in q if (r["k"] >= 1 if arm == "gate_k1_rnet" else r["mean_recall"] > 0)]
        if not keep:                      # the gate rejected everything -- the product would return
            keep = q                      # not_found; here we must still pick, so fall back to the pool
        return max(keep, key=lambda r: r["rnet"])

    print(f"\n=== G3 闸门要能付账(池子取产品实测值,不是此前那个错的 N=20)===")
    g3 = {}
    for n_pool in POOLS:
        by_p = collections.defaultdict(list)
        for r in rows:
            by_p[r["puzzle"]].append(r)
        rng = random.Random(31337 + n_pool)
        per = {a: {} for a in ARMS}
        for t, cand in sorted(by_p.items()):
            if len(cand) < n_pool:
                continue
            acc = {a: [] for a in ARMS}
            for _ in range(300):
                pool = rng.sample(cand, n_pool)
                for a in ARMS:
                    acc[a].append(pick(a, pool, rng)["score"])
            for a in ARMS:
                per[a][t] = st.mean(acc[a])
        ts = sorted(per["rnet"])
        print(f"  --- N = {n_pool}({len(ts)} 个靶标)---")
        for a in ARMS:
            print(f"      {a:16s} 平均实测分 {st.mean([per[a][t] for t in ts]):6.3f}")
        row = {}
        for a in ("gate_k1_rnet", "gate_soft_rnet", "grade_rnet"):
            d = [per[a][t] - per["rnet"][t] for t in ts]
            w, p, n = wilcoxon(d, "greater")
            ok = p is not None and p < ALPHA
            print(f"      {a:16s} 对纯 RNet {st.median(d):+.3f} 分  "
                  f"{sum(1 for x in d if x>0):2d}/{len(d)}  p = {'n/a' if p is None else f'{p:.3g}':>8s}"
                  f"  {'通过' if ok else ''}")
            row[a] = {"median_points": st.median(d), "p": p, "pass": bool(ok), "n": len(d)}
        g3[f"N={n_pool}"] = row

    main_gate = g3[f"N={POOLS[0]}"]["gate_k1_rnet"]
    print(f"\n=== 判决(主池 N={POOLS[0]},阈值 {ALPHA:.4f})===")
    print(f"  G1 {'通过' if (ours_flat and rnet_alive) else '未通过'}")
    print(f"  G2 {'通过' if g2_ok else '未通过'}")
    print(f"  G3 {'通过 —— 闸门确实在付账,判据该当闸门用而不是排序器' if main_gate['pass'] else '未通过'}"
          f"({main_gate['median_points']:+.3f} 分)")
    if not ((ours_flat and rnet_alive) and main_gate["pass"]):
        print("\n  KILL:闸门解读加入今天的死亡名单。诚实结论是 —— 在我们能构造的任何一种安排下,"
              "我们的判据都没有为【选择】增加价值。这是关于本产品的一个可发表事实,照此报告,不再解释第三次。")

    json.dump({"alpha": ALPHA, "G1": g1, "G1_pass": bool(ours_flat and rnet_alive),
               "G2": g2, "G2_pass": bool(g2_ok), "G3": g3,
               "n_top": len(top), "n_bottom": len(bot)},
              open(os.path.join(HERE, "gate_not_ranker.json"), "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote gate_not_ranker.json")


if __name__ == "__main__":
    main()
