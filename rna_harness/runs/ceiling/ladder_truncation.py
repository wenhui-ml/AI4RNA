#!/usr/bin/env python3
"""90.2% of solver time is spent by targets that run the ladder to the end. What does the end of it buy?

WHERE THIS QUESTION CAME FROM. designability_preflight.py killed the idea that a target-only predictor can
save money here -- a PERFECT oracle would save 35.6%, because that is all the time non-delivering targets
consume, and the ten most expensive targets are 7/10 successes. But it located the cost exactly:

    certified        69 targets   median  25s    9.8% of the clock   (returns the moment the panel agrees)
    knot_confirmed   30 targets   median 609s   54.6%                (return sits OUTSIDE the tier loop)
    not delivered    21 targets   median 582s   35.6%

and of the certified ones, 57 of 69 finish in the FIRST 5s tier while exactly ONE needs the 180s tier. So
the ladder is cheap for the targets it works on and enormous for the targets it does not, and nobody has
measured what the upper tiers return.

THIS ALSO ANSWERS A SEPARATE OPEN QUESTION, which is why it is one run and not two. `k=1` is 57% of what we
ship, and the third outside judge recognises those designs at 0.10x its own native ceiling. Whether that is
a property of the TARGET (no better sequence exists) or of the SEARCH (we did not look long enough) decides
a product behaviour: if budget does not move the grade, we should TELL the caller this target cannot be
certified above k=1 instead of spending the ladder finding out.

PRE-REGISTERED, before any arm ran:

    P1  (goal 2) Truncating to (5,15) saves >= 50% of total solver wall clock across the population.
    P2  (goal 4, the cost) That truncation loses <= 5% of certificates AND drops the mean knot grade by
        <= 0.2. FAILING P2 DOES NOT MEAN SHIPPING IT ANYWAY -- the trade curve is reported and no
        truncation ships without a point that clears both.
    P3  Does budget buy grade at all? If mean k is flat across the four arms, k=1 is a property of the
        target and the product should say so up front. If it rises, the ladder is doing real work for
        goal 4 even when it is not changing the verdict, and P1's saving has a price that must be named.

    Paired by TARGET across arms, one seed. The honest n is targets. Same-config verdict flips run at 5.8%
    at this budget (measured, designability_preflight P0), so a difference of a few targets is noise and
    directional sign tests are used rather than raw counts.
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

ARMS = [("5", (5,)), ("5+15", (5, 15)), ("5+15+60", (5, 15, 60)), ("full", (5, 15, 60, 180))]
DELIVERED = ("certified", "knot_confirmed")


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
            "delivered": c.verdict in DELIVERED, "wall_s": round(time.time() - t0, 2)}


def sign_p(up, dn):
    n = up + dn
    return sum(comb(n, i) for i in range(up, n + 1)) / 2 ** n if n else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default=os.path.join(HERE, "ladder_truncation.json"))
    a = ap.parse_args()

    from rna_harness.benchmarks.rnainvbench import load_pseudobase
    tasks = list(load_pseudobase())[:a.n]
    jobs = [(str(t.task_id), str(t.target), name, tiers)
            for t in tasks for name, tiers in ARMS]
    print(f"阶梯截断:{len(tasks)} 个靶标 × {len(ARMS)} 个预算臂 = {len(jobs)} 次\n"
          f"  臂:{[n for n, _ in ARMS]}\n"
          f"  按靶标配对;同配置翻转率在本预算下实测 5.8%,所以用方向符号检验而不是原始计数\n", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    R, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, f in enumerate(as_completed([ex.submit(work, j) for j in jobs]), 1):
            R.append(f.result())
            if n % 40 == 0 or n == len(jobs):
                print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)

    by = {}
    for r in R:
        if "error" in r:
            continue
        by.setdefault(r["task_id"], {})[r["arm"]] = r
    names = [n for n, _ in ARMS]
    full = {k: v for k, v in by.items() if all(n in v for n in names)}
    print(f"\n=== 四个臂都跑通的靶标 {len(full)}/{len(by)} ===")

    print(f"\n{'臂':>10s}{'交付':>7s}{'certified':>11s}{'平均 k':>8s}{'总时间(分)':>12s}{'占 full':>9s}")
    tot_full = sum(v["full"]["wall_s"] for v in full.values())
    S = {}
    for n in names:
        g = [v[n] for v in full.values()]
        d = [x for x in g if x["delivered"]]
        w = sum(x["wall_s"] for x in g)
        S[n] = {"delivered": len(d), "certified": sum(1 for x in g if x["verdict"] == "certified"),
                "mean_k": st.mean([x["k"] for x in d if x["verdict"] == "knot_confirmed"] or [0]),
                "wall_min": w / 60, "wall_frac": w / tot_full}
        print(f"{n:>10s}{len(d):7d}{S[n]['certified']:11d}{S[n]['mean_k']:8.2f}"
              f"{w/60:12.1f}{w/tot_full:9.1%}")

    print(f"\n=== P3 预算买不买得到等级(只看两臂都发 knot_confirmed 的靶标)===")
    P3 = {}
    for n in names[:-1]:
        pair = [(v[n]["k"], v["full"]["k"]) for v in full.values()
                if v[n]["verdict"] == v["full"]["verdict"] == "knot_confirmed"]
        if len(pair) < 8:
            print(f"  {n:>10s} vs full:配对靶标 {len(pair)} < 8,不检验")
            continue
        up = sum(1 for x, y in pair if y > x)
        dn = sum(1 for x, y in pair if y < x)
        p = sign_p(up, dn)
        P3[n] = {"n": len(pair), "up": up, "down": dn, "p": p,
                 "mean_gain": st.mean([y - x for x, y in pair])}
        print(f"  {n:>10s} -> full:{len(pair):3d} 个靶标  full 更高 {up}  更低 {dn}  "
              f"平均 {st.mean([y-x for x,y in pair]):+.2f} 级  符号检验 p={p:.3g}")

    print(f"\n=== P1/P2 截断到 (5,15) 的取舍 ===")
    t = S["5+15"]
    f_ = S["full"]
    save = 1 - t["wall_frac"]
    cert_loss = 1 - (t["delivered"] / f_["delivered"]) if f_["delivered"] else 1.0
    k_drop = f_["mean_k"] - t["mean_k"]
    print(f"  省下时间 {save:.1%}(门槛 ≥50%)")
    print(f"  丢证书  {cert_loss:.1%}(门槛 ≤5%)   {f_['delivered']} -> {t['delivered']} 张")
    print(f"  平均等级下降 {k_drop:+.2f}(门槛 ≤0.20)")
    p1 = save >= 0.50
    p2 = cert_loss <= 0.05 and k_drop <= 0.20
    print(f"\n=== 预注册判决 ===")
    print(f"  P1 省 ≥50% : {'通过' if p1 else '未通过'}")
    print(f"  P2 代价可接受 : {'通过' if p2 else '未通过 —— 只报取舍曲线,不截断出厂'}")
    if P3:
        anyk = any(v["p"] < 0.05 and v["mean_gain"] > 0 for v in P3.values())
        pos = sum(1 for v in P3.values() if v["mean_gain"] > 0)
        # A NULL IS NOT A DEMONSTRATION OF NO EFFECT, and an earlier version of this line said it was.
        # With ~30 paired targets per comparison the test cannot see a small gain, so "not significant"
        # here means UNDERPOWERED, not "budget buys nothing". The direction is reported alongside.
        print(f"  P3 {'预算确实买得到等级' if anyk else '未确立'} —— "
              + ("截断的代价不只是证书数,还有等级,必须一起报价" if anyk else
                 f"{len(P3)} 组比较里 {pos} 组方向为正(平均 "
                 f"{st.mean([v['mean_gain'] for v in P3.values()]):+.2f} 级),但 n≈"
                 f"{int(st.mean([v['n'] for v in P3.values()]))} 都不显著。"
                 f"这是【检验力不足】,不是【预算买不到等级】;不得据此宣称 k=1 是靶标的性质"))

    json.dump({"arms": S, "P3": P3, "save": save, "cert_loss": cert_loss, "k_drop": k_drop,
               "P1": p1, "P2": p2, "n_targets": len(full), "rows": R},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
