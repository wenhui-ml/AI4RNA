#!/usr/bin/env python3
"""Two questions the last run could not honestly answer with its own data.

ladder_truncation.py measured four budget arms over pseudobase targets 0-119 and found that (5,15,60) saves
43.2% of solver wall clock while losing nothing detectable (4 lost / 3 gained, sign p=0.5). That point looks
like the answer to goal 2 -- and it cannot be adopted from that run, because it was CHOSEN AFTER SEEING THE
CURVE while its bar had been set at 50% beforehand. Ratifying it there is retuning a threshold after seeing
the result, which this project already excluded once (share_max, G3). So it gets its own pre-registration
and its own targets.

    EXPERIMENT 1 runs on pseudobase targets 120-250 ONLY -- 131 targets never used in the run that suggested
    the point. This is a prospective replication, not a re-analysis, and nothing from targets 0-119 is
    pooled into it.

The second question is a power problem of my own making. ladder_truncation asked whether budget buys knot
grade and got +0.13, +0.06, +0.13 grades with directions 7-3, 4-1, 5-1 -- consistently positive, none
significant at n~30. The script then printed "k=1 is a property of the target", which the data does not
support: an underpowered test is not a demonstrated null, and the product behaviour it would justify
(telling a caller up front that a target cannot exceed k=1) would rest on nothing.

    EXPERIMENT 2 fixes the power two ways. It pools every pseudobase target (0-250) for the 5-vs-full
    comparison, and -- more importantly -- it scores on the ORDINAL scale the product actually uses:

        k=1  <  k=2  <  k=3  <  certified

    Restricting to targets that are knot_confirmed in BOTH arms threw away every target that climbed from
    knot_confirmed to certified, which is the largest grade improvement there is. That was a mistake in the
    outcome definition, not a tweak to reach significance, and it roughly triples the usable n.

PRE-REGISTERED, before either arm ran:

    E1-P1  On the 131 FRESH targets, (5,15,60) saves >= 30% of total solver wall clock against the full
           ladder. The bar is 30% and not the 50% used last time: 50% was set with no idea what the
           achievable ceiling was, and designability_preflight later showed a perfect oracle caps at 35.6%
           on a related quantity. Point prediction on record: ~43%.
    E1-P2  It loses no certificates detectably: directional sign test over targets, p >= 0.05, AND the
           mean ordinal grade drops by <= 0.2.
    E1-KILL Either fails -> the truncation does not ship, and the 43.2% seen on targets 0-119 is recorded
           as a single-sample observation that did not replicate.

    E2-P1  On the pooled 251 targets, ordinal grade under the full ladder exceeds grade under (5,) by
           Wilcoxon signed-rank, one-sided.
    E2-P2  Whatever the outcome, the power is reported: the n reached and the effect size the test could
           resolve. A null without that is what this run exists to correct.

    Paired by TARGET, one seed, honest n is targets. Same-config verdict flips run at 5.8% at this budget,
    so directional tests rather than net counts throughout.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from collections import Counter
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

ARMS = [("5", (5,)), ("5+15+60", (5, 15, 60)), ("full", (5, 15, 60, 180))]
FRESH_FROM = 120


def ordinal(r):
    """k=1 < k=2 < k=3 < certified. `certified` is the biggest grade jump there is and excluding it was
    the defect in the previous run's outcome definition."""
    if r["verdict"] == "certified":
        return 4
    if r["verdict"] == "knot_confirmed":
        return max(1, min(3, r["k"]))
    return 0


def work(job):
    tid, target, arm, tiers = job
    from rna_harness.cognitive.certify import certify_design
    t0 = time.time()
    try:
        c = certify_design(target, budget="standard", seed=0, tiers=tiers)
    except Exception as e:                                                     # noqa: BLE001
        return {"task_id": tid, "arm": arm, "error": repr(e)[:140]}
    k = len((c.detail or {}).get("knot", {}).get("confirmed_by") or []) if c.detail else 0
    return {"task_id": tid, "arm": arm, "verdict": c.verdict, "k": k,
            "delivered": c.verdict in ("certified", "knot_confirmed"),
            "wall_s": round(time.time() - t0, 2)}


def sign_p(up, dn):
    n = up + dn
    return sum(comb(n, i) for i in range(up, n + 1)) / 2 ** n if n else float("nan")


def load_prev():
    """Targets 0-119 from the previous run, for EXPERIMENT 2's pooling only. Never touched by E1."""
    p = os.path.join(HERE, "ladder_truncation.json")
    if not os.path.exists(p):
        return {}
    by = {}
    for r in json.load(open(p))["rows"]:
        if "error" not in r:
            by.setdefault(r["task_id"], {})[r["arm"]] = r
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default=os.path.join(HERE, "ladder_confirm.json"))
    a = ap.parse_args()

    import numpy as np
    from scipy.stats import wilcoxon
    from rna_harness.benchmarks.rnainvbench import load_pseudobase

    tasks = list(load_pseudobase())
    fresh = tasks[FRESH_FROM:]
    jobs = [(str(t.task_id), str(t.target), n, ti) for t in fresh for n, ti in ARMS]
    print(f"阶梯确认:{len(fresh)} 个【全新】靶标(pseudobase {FRESH_FROM}-{len(tasks)-1},"
          f"上一轮从未用过)× {len(ARMS)} 臂 = {len(jobs)} 次\n"
          f"  实验1 只用这批全新靶标 —— 挑出这个点的那批数据一条都不并进来\n"
          f"  实验2 会把上一轮的 0-{FRESH_FROM-1} 并进来,并明确标注是【合并】\n", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    R, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, f in enumerate(as_completed([ex.submit(work, j) for j in jobs]), 1):
            R.append(f.result())
            if n % 40 == 0 or n == len(jobs):
                print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)

    by = {}
    for r in R:
        if "error" not in r:
            by.setdefault(r["task_id"], {})[r["arm"]] = r
    F = {k: v for k, v in by.items() if all(n in v for n, _ in ARMS)}
    print(f"\n=== 实验1:全新靶标 {len(F)}/{len(fresh)} 三臂全跑通 ===")
    tot = sum(v["full"]["wall_s"] for v in F.values())
    for n, _ in ARMS:
        g = [v[n] for v in F.values()]
        print(f"  {n:>9s}  交付 {sum(1 for x in g if x['delivered']):3d}  "
              f"certified {sum(1 for x in g if x['verdict']=='certified'):3d}  "
              f"平均序数 {st.mean([ordinal(x) for x in g]):.2f}  "
              f"时间 {sum(x['wall_s'] for x in g)/60:6.1f} 分 = {sum(x['wall_s'] for x in g)/tot:5.1%}")

    arm = "5+15+60"
    save = 1 - sum(v[arm]["wall_s"] for v in F.values()) / tot
    lf = sum(1 for v in F.values() if v["full"]["delivered"] and not v[arm]["delivered"])
    la = sum(1 for v in F.values() if v[arm]["delivered"] and not v["full"]["delivered"])
    p_loss = sign_p(lf, la)
    o_full = [ordinal(v["full"]) for v in F.values()]
    o_arm = [ordinal(v[arm]) for v in F.values()]
    drop = st.mean(o_full) - st.mean(o_arm)
    e1p1 = save >= 0.30
    e1p2 = p_loss >= 0.05 and drop <= 0.20
    print(f"\n  E1-P1 省下 {save:.1%}(门槛 ≥30%,点预测 ~43%)  ->  {'通过' if e1p1 else '未通过'}")
    print(f"  E1-P2 丢证方向 {lf} 丢 / {la} 得,符号 p={p_loss:.3g};平均序数下降 {drop:+.2f}"
          f"  ->  {'通过' if e1p2 else '未通过'}")
    print(f"  判定:{'两条都过 —— 43.2% 在全新靶标上复现,截断可以进产品' if e1p1 and e1p2 else '不同时过 —— 不进产品,上一轮的 43.2% 记为未复现的单次观测'}")

    # ---------------- EXPERIMENT 2: pooled, ordinal ----------------------------------------------------
    prev = load_prev()
    pool = dict(prev)
    for k, v in by.items():
        pool.setdefault(k, {}).update(v)
    P = [(ordinal(v["5"]), ordinal(v["full"])) for v in pool.values() if "5" in v and "full" in v]
    print(f"\n=== 实验2(合并 0-{len(tasks)-1}):预算买不买得到等级,序数尺度 k1<k2<k3<certified ===")
    print(f"  配对靶标 {len(P)}(上一轮同一问题只有 ~30,因为把 knot_confirmed->certified 的靶标扔掉了)")
    d = np.array([b - a_ for a_, b in P])
    up, dn, tie = int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())
    if (d != 0).sum() >= 8:
        w = float(wilcoxon([p[1] for p in P], [p[0] for p in P], alternative="greater")[1])
    else:
        w = float("nan")
    se = float(d.std(ddof=1)) / len(d) ** 0.5
    mde = 2.8 * se               # roughly the effect an 80%-power two-sided test at alpha=.05 could resolve
    print(f"  full 更高 {up}   更低 {dn}   持平 {tie}")
    print(f"  平均差 {d.mean():+.3f} 级  95% [{d.mean()-1.96*se:+.3f}, {d.mean()+1.96*se:+.3f}]"
          f"   Wilcoxon 单边 p={w:.3g}")
    print(f"  E2-P2 检验力陈述:n={len(P)},本设计能分辨的最小效应约 {mde:.3f} 级;"
          f"上一轮 n~30 时约 {mde*(len(P)/30)**0.5:.3f} 级 —— 那个 +0.13 当时【本来就看不见】")
    e2 = w < 0.05 and d.mean() > 0
    print(f"  E2-P1  ->  {'通过:预算确实买得到等级,所以截断的代价不止证书数' if e2 else '未通过'}"
          f"{'' if e2 else f',而这次是有检验力的 null(能分辨 {mde:.3f} 级,实测 {d.mean():+.3f})'}")

    json.dump({"E1": {"n_fresh": len(F), "save": save, "lost": lf, "gained": la, "p_loss": p_loss,
                      "grade_drop": drop, "P1": e1p1, "P2": e1p2, "ships": bool(e1p1 and e1p2)},
               "E2": {"n_paired": len(P), "up": up, "down": dn, "tie": tie, "mean_gain": float(d.mean()),
                      "p": w, "mde": mde, "passes": bool(e2)},
               "rows": R}, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
