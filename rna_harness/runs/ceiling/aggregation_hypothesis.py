#!/usr/bin/env python3
"""One cause or three walls? Test whether tau=1.0 conjunction is what makes our criterion look weak.

THE RETREAT THIS SCRIPT EXISTS TO UNDO. Today produced three results that I read as three separate defeats:

    (1) M2R: per-PAIR panel agreement predicts causal rescue at p = 2.1e-28, but the per-DESIGN certificate
        grade k predicts the same thing at only p = 0.0147 -- four orders of magnitude weaker.
    (2) OpenKnot observation: our within-target correlation with wet-lab score falls from +0.213 on round 2
        to +0.033 on round 4 (the 240-mers), while RNet only falls from +0.504 to +0.227.
    (3) Selection: our grade beats random in a pool of 5 (+0.783, p=7e-06) and does not in the product's
        real pool of ~131 (+0.264, p=0.193).

I explained each locally -- wrong level of analysis, long targets are harder, five levels saturate. That is
three excuses, not an explanation. There is a single mechanism that predicts all three, and we wrote it:

    full = [k for k, v in out.items() if v is not None and v >= 1.0]      # certify.py, _knot_recall

The certificate's k is a CONJUNCTION over every crossing pair, cut at 1.0. An oracle that recovers 9 of a
target's 10 crossing pairs contributes exactly as much as one that recovers 0. That single line predicts:

    (1) per-design must be much weaker than per-pair, because the aggregation discards the per-pair evidence
    (2) longer targets have more crossing pairs, so the conjunction collapses to k=0 more often -- the
        signal must decay with length
    (3) five levels cannot order a large pool, so selection must decay with pool size

tau=1.0 was chosen to make a VERDICT defensible. It has never been re-examined for the three other jobs it
silently took on: ranking candidates, scaling to long targets, and aggregating pair evidence.

WHAT IS TESTED HERE -- prediction (1), the one whose data is already on disk. Predictions (2) and (3) need
the folds `continuous_recall_selector.py` is computing and are tested separately.

    Same 160 M2R designs, same 1,412 causally-tested crossing pairs, same outcome (mean rescue_factor of
    the design's crossing pairs). Only the predictor changes:

        k_conjunction  the shipped grade: oracles recovering ALL crossing pairs at tau=1.0   (incumbent)
        mean_recall    mean over the three oracles of the FRACTION recovered                 (no conjunction)
        max_recall     best single oracle's fraction                                          (no conjunction,
                                                                                               no averaging)

PRE-REGISTERED, before any of the three predictors was correlated with the outcome. Within-target Spearman;
Wilcoxon over targets; three tests, Bonferroni threshold 0.05/3 = 0.0167.

    A1  mean_recall predicts design-level rescue.                     (does the evidence survive at all?)
    A2  mean_recall beats k_conjunction, paired within target.        (THIS IS THE HYPOTHESIS. If the
                                                                       conjunction is what costs us, removing
                                                                       it must recover signal.)
    A3  the gain is bigger on targets with MORE crossing pairs.       (the mechanism's own signature: a
                                                                       conjunction over 3 pairs loses little,
                                                                       over 12 pairs it loses everything.
                                                                       Spearman between a target's crossing-
                                                                       pair count and the per-target gain.)

    KILL FOR THE UNIFYING STORY: A2 fails -> the conjunction is NOT the common cause, the three boundaries
    are three separate walls, and the unification I proposed is wrong and must be withdrawn as such. A3
    failing while A2 passes means the effect is real but the mechanism is not the one named -- report that
    difference rather than smoothing it.

COST: $0. 160 folds, seconds.
"""
from __future__ import annotations

import collections
import csv
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from m2r_pairwise import load, spearman, wilcoxon, DATA                        # noqa: E402

ALPHA = 0.05 / 3


def fold(item):
    from rna_harness.cognitive.certify import _knot_recall
    key, seq, tgt = item
    kr = _knot_recall(seq, tgt)
    vals = [v for v in kr["per_oracle"].values() if v is not None]
    return {"key": key, "k": len(kr["confirmed_by"]),
            "mean_recall": (st.mean(vals) if vals else 0.0),
            "max_recall": (max(vals) if vals else 0.0), "n_cross": kr["n_cross"]}


def per_target(recs, pred, out="mean_rf", min_n=4):
    g = collections.defaultdict(list)
    for r in recs:
        g[r["puzzle"]].append(r)
    rho = {}
    for t, v in g.items():
        if len(v) >= min_n and len({r[pred] for r in v}) >= 2:
            s = spearman([r[pred] for r in v], [r[out] for r in v])
            if s is not None:
                rho[t] = s
    return rho


def main():
    designs, pairs, dropped = load(DATA)
    rf = collections.defaultdict(list)
    for p in pairs:
        if p["crossed"]:
            rf[p["design"]].append(p["rf"])

    from concurrent.futures import ProcessPoolExecutor
    items = [(k, v["seq"], v["target"]) for k, v in designs.items() if k in rf]
    with ProcessPoolExecutor(max_workers=16) as ex:
        folded = list(ex.map(fold, items))
    by_key = {tuple(f["key"]): f for f in folded}

    recs = []
    for k in rf:
        f = by_key.get(k)
        if not f:
            continue
        recs.append({"puzzle": k[0], "method": k[1], "k": f["k"],
                     "mean_recall": f["mean_recall"], "max_recall": f["max_recall"],
                     "n_cross": f["n_cross"], "n_pairs_tested": len(rf[k]),
                     "mean_rf": st.mean(rf[k])})
    print(f"设计 {len(recs)} 条,靶标 {len({r['puzzle'] for r in recs})} 个,"
          f"结局 = 该设计交叉对的平均补偿恢复\n")

    print("=== 描述:三个预测量各自把 160 条设计分成几档 ===")
    for pred in ("k", "mean_recall", "max_recall"):
        vals = [r[pred] for r in recs]
        print(f"  {pred:12s} 不同取值 {len(set(vals)):3d} 个   中位 {st.median(vals):.3f}")
    zeros = [r for r in recs if r["k"] == 0]
    if zeros:
        mr = sorted(r["mean_recall"] for r in zeros)
        print(f"\n  证书判 k=0 的 {len(zeros)} 条设计,它们的连续召回率其实从 "
              f"{mr[0]:.2f} 一直到 {mr[-1]:.2f}")
        lo = [r["mean_rf"] for r in zeros if r["mean_recall"] <= mr[len(mr)//3]]
        hi = [r["mean_rf"] for r in zeros if r["mean_recall"] >= mr[2*len(mr)//3]]
        print(f"    其中召回最低的三分之一  平均 rescue 中位 {st.median(lo):+.3f}")
        print(f"    其中召回最高的三分之一  平均 rescue 中位 {st.median(hi):+.3f}"
              f"   <- 证书对这两组【完全无法区分】")

    rk = per_target(recs, "k")
    rm = per_target(recs, "mean_recall")
    rx = per_target(recs, "max_recall")
    print(f"\n=== 靶标内 Spearman(预测量, 平均补偿恢复)===")
    for name, r in (("k(合取,现行)", rk), ("mean_recall(去掉合取)", rm), ("max_recall", rx)):
        w, p, n = wilcoxon(list(r.values()), "greater")
        print(f"  {name:26s} 簇 {len(r):2d}  中位 rho {st.median(list(r.values())):+.3f}  "
              f"正向 {sum(1 for x in r.values() if x>0)}/{len(r)}  p = {p:.3g}")

    print(f"\n=== 预注册检验(阈值 {ALPHA:.4f})===")
    w, pa1, _ = wilcoxon(list(rm.values()), "greater")
    print(f"  A1 连续召回预测干预结果:{'通过' if pa1 < ALPHA else '未通过'}(p = {pa1:.3g})")

    shared = sorted(set(rk) & set(rm))
    gain = {t: rm[t] - rk[t] for t in shared}
    w, pa2, _ = wilcoxon(list(gain.values()), "greater")
    print(f"  A2 连续召回 > 合取:簇 {len(gain)},中位增益 {st.median(list(gain.values())):+.3f},"
          f"正向 {sum(1 for x in gain.values() if x>0)}/{len(gain)},p = {pa2:.3g}"
          f"  {'通过 —— 合取确实是共因' if pa2 < ALPHA else '未通过 —— 统一假设被推翻,三个边界是三堵独立的墙'}")

    ncross = {}
    for r in recs:
        ncross.setdefault(r["puzzle"], []).append(r["n_cross"])
    xs = [st.median(ncross[t]) for t in shared]
    ys = [gain[t] for t in shared]
    r3 = spearman(xs, ys)
    print(f"  A3 交叉配对越多、增益越大:rho = {r3:+.3f}(靶标交叉对数中位 "
          f"{st.median(xs):.0f},范围 {min(xs):.0f}-{max(xs):.0f})")
    print(f"     {'与机制一致' if (r3 or 0) > 0 else '与机制【不一致】—— A2 若通过则效应为真但机制不是合取'}")

    json.dump({"n_designs": len(recs), "alpha": ALPHA,
               "rho_k": rk, "rho_mean_recall": rm, "rho_max_recall": rx,
               "A1_p": pa1, "A2_median_gain": st.median(list(gain.values())), "A2_p": pa2,
               "A3_rho": r3, "recs": recs},
              open(os.path.join(HERE, "aggregation_hypothesis.json"), "w"), ensure_ascii=False, indent=1)
    print("\n  wrote aggregation_hypothesis.json")


if __name__ == "__main__":
    main()
