#!/usr/bin/env python3
"""Does keeping the BEST candidate instead of the FIRST one actually buy grade, and does it cost anything?

WHAT CHANGED. `_certify_design_impl` used to stop grading candidates at the first sequence that cleared the
shipped floor (`knot_confirmed`, k>=1). It now grades every candidate in the pool and keeps the highest.
Three non-redundant outside judges (RNAPKplex, Knotty, IPknot; pairwise kappa 0.341/0.109/0.113) agree that
the panel's evidence lives almost entirely in the last vote -- k=2 -> k=3 gains +64.6/+36.4/+29.0 points
against +0.6/+2.1/+5.5 for k=1 -> k=2 -- so shipping k=1 while a k=3 sat unexamined in the same pool was a
real loss, not a cosmetic one.

WHY IT SHOULD BE FREE, and why that has to be measured rather than asserted. Both grading questions -- does
the whole structure match (`_panel_verdict`) and are the crossing stems reproduced (`_knot_recall`) -- are
questions about ONE prediction per oracle per sequence, and the panel verdict already folds every candidate.
Those folds are now computed once (`oracles_pk.oracle_pairs`, LRU) and shared, so grading the rest of the
pool costs set intersections rather than subprocesses. A counter on the actual subprocess folds
(`oracle_call_counts`) is what turns "free" into something falsifiable.

PRE-REGISTERED, before either arm ran:

    P1  Grade improves: over targets that get a knot_confirmed certificate under both arms, the shipped
        grade is strictly higher on at least 30% of them.
    P2  Nothing regresses: ZERO targets end up with a lower grade, and zero lose a certificate they had.
        A single regression kills the change -- there is no version of this worth a lost certificate.
    P3  It is free: oracle subprocess folds PER CANDIDATE SCREENED do not INCREASE. If they do, this is no
        longer a free win but a purchase, and it must be reported as "grade bought with oracle time" at
        that price, never as an improvement.

    The honest n is TARGETS.

    P3 IS PER CANDIDATE, NOT PER TARGET, AND THAT IS NOT A CONVENIENCE. The two arms share a seed, but
    DesiRNA is limited by WALL CLOCK, so under different machine load the same seed yields a different
    number of candidates -- the smoke run made the new arm look 84 folds CHEAPER per target, which the
    change cannot possibly cause. Folds per target measures how busy the machine was; folds per candidate
    measures the policy. Both are reported, and only the second is read as evidence.

    Reported either way, including the outcome where the pool rarely holds anything better and the whole
    change is worth almost nothing in practice -- the 61.6% headroom measured on judge_paired.json came from
    a different generator setting (6 replicas, fixed 20s) than the shipped ladder uses.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)


def work(job):
    tid, target, budget, seed = job
    from rna_harness.cognitive.certify import certify_design
    from rna_harness.oracles_pk import oracle_call_counts, reset_oracle_call_counts
    out = {"task_id": tid}
    for arm, gm in (("old", False), ("new", True)):
        # The LRU is per process and would leak folds from the first arm into the second, making the
        # second look free by construction. Cleared between arms so each arm pays its own way.
        from rna_harness.oracles_pk import _oracle_pairs_cached
        _oracle_pairs_cached.cache_clear()
        reset_oracle_call_counts()
        t0 = time.time()
        try:
            c = certify_design(target, budget=budget, seed=seed, grade_max=gm)
        except Exception as e:                                                 # noqa: BLE001
            out[arm] = {"error": repr(e)[:140]}
            continue
        k = len((c.detail or {}).get("knot", {}).get("confirmed_by") or []) if c.detail else 0
        scr = (c.detail or {}).get("n_candidates_screened") or 0
        folds = sum(oracle_call_counts().values())
        out[arm] = {"verdict": c.verdict, "k": k, "wall_s": round(time.time() - t0, 1),
                    "folds": folds, "screened": scr,
                    "folds_per_cand": (folds / scr) if scr else None,
                    "strictness": c.strictness}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--budget", default="cheap")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(HERE, "grade_max_bench.json"))
    a = ap.parse_args()

    from rna_harness.benchmarks.rnainvbench import load_pseudobase
    tasks = list(load_pseudobase())[:a.n]
    jobs = [(str(t.task_id), str(t.target), a.budget, a.seed) for t in tasks]
    print(f"等级最大化 成对基准:{len(jobs)} 个靶标 × 2 臂(旧=取第一个过线 / 新=取池内最好),"
          f"预算 {a.budget},种子 {a.seed}\n  两臂共用种子,所以看到的是同一条候选流;诚实的 n 是靶标数\n",
          flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    R, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, f in enumerate(as_completed([ex.submit(work, j) for j in jobs]), 1):
            R.append(f.result())
            if n % 10 == 0 or n == len(jobs):
                print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)

    ok = [r for r in R if "error" not in r.get("old", {}) and "error" not in r.get("new", {})]
    kc = [r for r in ok if r["old"]["verdict"] == "knot_confirmed" and r["new"]["verdict"] == "knot_confirmed"]
    print(f"\n=== 两臂都跑通 {len(ok)}/{len(R)};两臂都发 knot_confirmed 的 {len(kc)} 个靶标 ===")
    print(f"  判决分布 旧: {dict(Counter(r['old']['verdict'] for r in ok))}")
    print(f"           新: {dict(Counter(r['new']['verdict'] for r in ok))}")

    if not kc:
        print("\n  !! 没有一个靶标在两臂都拿到 knot_confirmed —— 这个预算下测不了,报不可测,不报 null")
        json.dump({"insufficient": True, "n_ok": len(ok), "rows": R},
                  open(a.out, "w"), ensure_ascii=False, indent=1)
        return

    up = [r for r in kc if r["new"]["k"] > r["old"]["k"]]
    down = [r for r in kc if r["new"]["k"] < r["old"]["k"]]
    lost = [r for r in ok if r["old"]["verdict"] == "knot_confirmed" and r["new"]["verdict"] != "knot_confirmed"]
    print(f"\n=== P1/P2 等级 ===")
    print(f"  旧等级分布 {dict(sorted(Counter(r['old']['k'] for r in kc).items()))}"
          f"   平均 {st.mean([r['old']['k'] for r in kc]):.2f}")
    print(f"  新等级分布 {dict(sorted(Counter(r['new']['k'] for r in kc).items()))}"
          f"   平均 {st.mean([r['new']['k'] for r in kc]):.2f}")
    print(f"  提升 {len(up)} 个 = {len(up)/len(kc):.1%}   下降 {len(down)} 个   丢证书 {len(lost)} 个")

    fo = [r["old"]["folds"] for r in ok]
    fn = [r["new"]["folds"] for r in ok]
    pc_o = [r["old"]["folds_per_cand"] for r in ok if r["old"].get("folds_per_cand")]
    pc_n = [r["new"]["folds_per_cand"] for r in ok if r["new"].get("folds_per_cand")]
    print(f"\n=== P3 代价 ===")
    print(f"  【每候选】折叠次数(策略可归因):旧 {st.mean(pc_o):.3f}  新 {st.mean(pc_n):.3f}  "
          f"差 {st.mean(pc_n)-st.mean(pc_o):+.3f}   n={len(pc_o)}/{len(pc_n)}")
    print(f"  (理论值应为 3.000 —— 每条候选三家神谕各折一次,knot_recall 全部命中缓存)")
    print(f"  每靶标折叠次数(受负载影响,仅供参考):旧 {st.mean(fo):.1f}  新 {st.mean(fn):.1f}")
    print(f"  每靶标候选数(两臂应相近,差大说明限时求解器受负载扰动):"
          f"旧 {st.mean([r['old']['screened'] for r in ok]):.1f}  新 {st.mean([r['new']['screened'] for r in ok]):.1f}")

    p1 = len(up) / len(kc) >= 0.30
    p2 = len(down) == 0 and len(lost) == 0
    p3 = st.mean(pc_n) <= st.mean(pc_o) + 1e-9
    print(f"\n=== 预注册判决 ===")
    print(f"  P1 等级提升 ≥30% : {'通过' if p1 else '未通过'}({len(up)/len(kc):.1%})")
    print(f"  P2 零下降零丢证 : {'通过' if p2 else '未通过 —— 这条不过就整改回退,没有值得丢一张证书的版本'}")
    print(f"  P3 每候选折叠不增 : {'通过,是白捡的' if p3 else '未通过 —— 改口为【用神谕时间买等级】,并按这个口径报价'}")

    json.dump({"n_ok": len(ok), "n_both_kc": len(kc), "n_up": len(up), "n_down": len(down),
               "n_lost": len(lost), "k_old": st.mean([r["old"]["k"] for r in kc]),
               "k_new": st.mean([r["new"]["k"] for r in kc]),
               "folds_old": st.mean(fo), "folds_new": st.mean(fn),
               "folds_per_cand_old": st.mean(pc_o), "folds_per_cand_new": st.mean(pc_n),
               "P1": p1, "P2": p2, "P3": p3, "rows": R},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
