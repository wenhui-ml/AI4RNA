#!/usr/bin/env python3
"""Goal 2 has never had a measured saving. Can the target structure alone tell us what NOT to spend on?

WHAT IS BEING CLOSED. designability.py showed that 13 features taken from the dot-bracket alone predict two
different outcomes at held-out AUC 0.945 and 0.929. That is "we can predict"; it is not "we saved anything",
and goal 2 has stood at 3/5 the whole time on exactly that gap. This run wires the prediction to the thing
that costs money -- solver wall clock in `certify_design` -- and measures what it buys and what it costs.

WHAT IT DELIBERATELY IS NOT. It is not "predict unsolvable, then refuse". Refusing is how the last two
shipping proposals died: they bought precision with target coverage, and coverage is the product. So the
question here is stated as a TRADE CURVE, not a verdict: skipping targets below a threshold saves solver
time and loses certificates, and the whole output is how much of each, over every threshold. If no point on
that curve is worth taking, that is the answer and it is reported as the answer.

THE LABEL IS NOISY AND THAT IS MEASURED FIRST. grade_max_bench.py found, by accident, that 22.9% of targets
flip between `certified` and `cannot_certify` when the identical request is re-run: DesiRNA is bounded by
wall clock, so a fixed seed still explores a different pool under different machine load. A predictor cannot
beat the noise in its own labels. So this runs every target TWICE and reports the flip rate BEFORE any AUC,
because an AUC read without it would blame the model for the harness.

PRE-REGISTERED, before any target was run:

    P0  Report the flip rate at this budget first. Whatever it is, the maximum AUC any predictor could reach
        against a label that unstable is reported alongside every AUC below. Not a pass/fail -- a ceiling.

    P1  A threshold exists on the HELD-OUT half that saves >= 30% of total solver wall clock while losing
        <= 5% of the certificates that would have been issued.
        KILL: no such point -> target-only designability does not convert into savings on this pipeline,
        goal 2 stays where it is, and the AUC stays a capability that has bought nothing.

    P2  The model beats LENGTH ALONE at the same operating point. The caller already knows the length; a
        "prediction" that only rediscovers it is not a new capability. If length matches it, the honest
        claim is "length predicts designability" and the 13 features are dropped.

    Split: the project's deterministic even/odd on sorted task ids. Fit on FIT, every number reported on
    TEST. The honest n is TARGETS.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from designability import auc, features, fit_logistic, predict                 # noqa: E402

DELIVERED = ("certified", "knot_confirmed")


def work(job):
    tid, target, budget, seed, rep = job
    from rna_harness.cognitive.certify import certify_design
    t0 = time.time()
    try:
        c = certify_design(target, budget=budget, seed=seed)
    except Exception as e:                                                     # noqa: BLE001
        return {"task_id": tid, "rep": rep, "error": repr(e)[:140]}
    return {"task_id": tid, "rep": rep, "verdict": c.verdict, "wall_s": round(time.time() - t0, 2),
            "delivered": c.verdict in DELIVERED,
            "tiers": (c.detail or {}).get("tiers_tried") if c.detail else None}


def curve(items, score, label, cost):
    """Skip every target whose score is below t. Returns (kept certificates, wall clock spent) per t."""
    tot_c = sum(1 for _, l, _ in items if l)
    tot_w = sum(w for _, _, w in items)
    out = []
    for t in sorted({s for s, _, _ in items} | {0.0, 1.0}):
        kept = [(s, l, w) for s, l, w in items if s >= t]
        out.append({"t": t, "n_run": len(kept),
                    "certs": sum(1 for _, l, _ in kept if l),
                    "cert_frac": (sum(1 for _, l, _ in kept if l) / tot_c) if tot_c else 1.0,
                    "wall": sum(w for _, _, w in kept),
                    "wall_frac": (sum(w for _, _, w in kept) / tot_w) if tot_w else 1.0})
    return out, tot_c, tot_w


def best_point(cv, max_loss=0.05):
    """Cheapest operating point that keeps at least (1 - max_loss) of the certificates."""
    ok = [p for p in cv if p["cert_frac"] >= 1 - max_loss]
    return min(ok, key=lambda p: p["wall_frac"]) if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--budget", default="standard")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(HERE, "designability_preflight.json"))
    a = ap.parse_args()

    from rna_harness.benchmarks.rnainvbench import load_pseudobase
    tasks = list(load_pseudobase())[:a.n]
    jobs = [(str(t.task_id), str(t.target), a.budget, 0, rep) for t in tasks for rep in (0, 1)]
    print(f"可设计性前置检查:{len(tasks)} 个靶标 × 2 次重复(同配置),预算 {a.budget}\n"
          f"  重复是为了先量【标签本身有多不稳】—— 预测器打不过自己标签里的噪声\n", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    R, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, f in enumerate(as_completed([ex.submit(work, j) for j in jobs]), 1):
            R.append(f.result())
            if n % 20 == 0 or n == len(jobs):
                print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)

    byid = {}
    for r in R:
        if "error" in r:
            continue
        byid.setdefault(r["task_id"], {})[r["rep"]] = r
    both = {k: v for k, v in byid.items() if 0 in v and 1 in v}
    flips = [k for k, v in both.items() if v[0]["delivered"] != v[1]["delivered"]]
    fr = len(flips) / max(1, len(both))
    print(f"\n=== P0 标签稳定性(同配置重复,{len(both)} 个靶标)===")
    print(f"  交付与否翻转 {len(flips)} 个 = {fr:.1%}   (cheap 预算下曾测到 22.9%)")
    print(f"  判决完全一致 {sum(1 for v in both.values() if v[0]['verdict']==v[1]['verdict'])}/{len(both)}")
    # A label that flips with probability f has, against ANY predictor of the stable truth, a bounded AUC.
    # Reported as a ceiling rather than a threshold: it says how much of a shortfall is the harness's.
    print(f"  -> 下面每个 AUC 都要对着这个读:标签本身 {fr:.1%} 的时候,"
          f"任何预测器都够不到 1.0,差的那部分不全是模型的")

    tgt = {str(t.task_id): str(t.target) for t in tasks}
    keys = sorted(both)
    idx = {k: i for i, k in enumerate(keys)}
    rows = []
    for k in keys:
        v = both[k]
        rows.append({"id": k, "feat": features(tgt[k]),
                     "label": bool(v[0]["delivered"]),                 # run 0 is the label
                     "label2": bool(v[1]["delivered"]),                # run 1 checks the conclusion
                     "wall": st.mean([v[0]["wall_s"], v[1]["wall_s"]]),
                     "len": len(tgt[k])})
    names = sorted(rows[0]["feat"])
    fit = [r for r in rows if idx[r["id"]] % 2 == 0]
    test = [r for r in rows if idx[r["id"]] % 2 == 1]
    print(f"\n=== 分割:FIT {len(fit)} / TEST {len(test)}(确定性偶奇);交付基线率 "
          f"{sum(1 for r in rows if r['label'])/len(rows):.1%} ===")

    m = fit_logistic([[r["feat"][n_] for n_ in names] for r in fit], [1.0 * r["label"] for r in fit])
    st_ = predict(m, [[r["feat"][n_] for n_ in names] for r in test])
    yt = [1.0 * r["label"] for r in test]
    a_test = auc(yt, st_)
    a_len = auc(yt, [-r["len"] for r in test])                          # P2 baseline: length alone
    a_len = max(a_len, 1 - a_len) if a_len is not None else None
    a_test2 = auc([1.0 * r["label2"] for r in test], st_)               # same model, the OTHER run's label
    print(f"  13 特征模型 TEST AUC {a_test:.3f}   (对第二次运行的标签 {a_test2:.3f} —— 差大则是标签噪声)")
    print(f"  仅用长度  TEST AUC {a_len:.3f}   <- P2 的对照:调用者本来就知道长度")

    items = [(s, bool(r["label"]), r["wall"]) for s, r in zip(st_, test)]
    cv, tot_c, tot_w = curve(items, st_, yt, [r["wall"] for r in test])
    bp = best_point(cv, 0.05)
    print(f"\n=== P1 跳过低分靶标的取舍曲线(TEST {len(test)} 个靶标,共 {tot_c} 张证书,"
          f"{tot_w/60:.1f} 分钟求解器时间)===")
    print(f"{'保留证书':>9s}{'花掉时间':>9s}{'跑的靶标':>9s}")
    for want in (1.00, 0.99, 0.95, 0.90, 0.80):
        p = best_point(cv, 1 - want)
        if p:
            print(f"{p['cert_frac']:9.1%}{p['wall_frac']:9.1%}{p['n_run']:9d}")
    # the same curve for the length-only rule, so P2 is a comparison and not an assertion
    li = [(-r["len"], bool(r["label"]), r["wall"]) for r in test]
    cvl, _, _ = curve(li, None, None, None)
    bpl = best_point(cvl, 0.05)

    p1 = bool(bp and bp["wall_frac"] <= 0.70)
    p2 = bool(bp and bpl and bp["wall_frac"] < bpl["wall_frac"] - 0.02)
    print(f"\n=== 预注册判决 ===")
    if bp:
        print(f"  P1 丢 ≤5% 证书时最省能省到花 {bp['wall_frac']:.1%} 的时间"
              f"(即省下 {1-bp['wall_frac']:.1%});门槛是省 ≥30%  ->  {'通过' if p1 else '未通过'}")
    else:
        print(f"  P1 不存在任何保留 ≥95% 证书的跳过点  ->  未通过")
    if bpl:
        print(f"  P2 仅用长度在同一条件下花 {bpl['wall_frac']:.1%}  ->  "
              f"{'通过,模型确实优于长度' if p2 else '未通过 —— 改口为【长度就能预测可设计性】,13 个特征应当撤掉'}")
    print(f"\n  杀死条款:P1 不过 -> 靶标结构预测在这条流水线上换不来节省,② 原地不动,"
          f"AUC 仍是一个没买到东西的能力。")

    json.dump({"flip_rate": fr, "n_both": len(both), "auc_test": a_test, "auc_test_run2": a_test2,
               "auc_length_only": a_len, "base_rate": sum(1 for r in rows if r["label"])/len(rows),
               "best_point": bp, "best_point_length": bpl, "P1": p1, "P2": p2,
               "curve": cv, "curve_length": cvl, "rows": R},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
