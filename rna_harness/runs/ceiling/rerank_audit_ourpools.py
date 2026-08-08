#!/usr/bin/env python3
"""Which regime are OUR pools in -- the one where ranking by RibonanzaNet works, or the one where it does not?

RUNS IN THE RibonanzaNet venv.

`rerank_audit.py` measured, on real designs with wet-lab scores, what ranking a homogeneous pool by RNet
crossed-pair F1 actually buys. The answer split cleanly in two, and the split is by SATURATION:

    Eterna pools   77.4% of the pool already sits at F1 = 1.000, a median of 78 candidates tie at the top,
                   and the tube says the argmax is -0.30 points WORSE than the pool median (21/56, p=0.79).
                   Where the metric saturates, argmax is an arbitrary draw from the tied set and buys nothing.

    M2 pools       0% saturation, a median of 2 tied, and the argmax is +0.55 points better (21/35, p=0.022)
                   -- real, one-sixth of the pre-registered 1.0-point bar, and so a KILL on V1 anyway.

Yesterday's headline (+0.250 F1, 51/57, p = 2.6e-10) cannot distinguish these two worlds, because it is
measured in the same units it selected on and is positive by construction in both. This script asks the one
question that does distinguish them, about our own pools rather than theirs:

    O1  SATURATION. What share of each of our 57 pools already scores F1 = 1.000, and how many candidates
        tie at the top? Compared directly against the Eterna and M2 numbers above.

    O2  DISCRIMINATION. Over the whole pool, is RNet's score spread out or piled at the ceiling? A pool whose
        top is a large tied plateau cannot be ranked, whatever the headline gain says.

    O3  THE RESCUES. The 10 targets the product returned not_found for were credited with a median 0.982.
        If those pools are saturated too, that number is an arbitrary pick from a plateau of candidates our
        own panel rejected, and "coverage 47/57 -> 57/57" is not a capability.

    READ-OFF, fixed before the numbers are seen, so the answer cannot be chosen afterwards:
        tie mass like Eterna's (tens of candidates tied at 1.000)  -> expected real gain is ~0 or negative;
                                                                      the rerank claim is withdrawn entirely.
        tie mass like M2's (a handful)                             -> expected real gain ~ +0.55 wet-lab
                                                                      points; quotable only in those units.

COST: $0. Local CPU.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pool_rerank_score import _score_chunk                                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default=os.path.join(HERE, "pool_rerank_pools.json"))
    ap.add_argument("--shipped", default=os.path.join(HERE, "leaderboard_ours.json"))
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(HERE, "rerank_audit_ourpools.json"))
    a = ap.parse_args()

    P = json.load(open(a.pools))
    T = P["targets"]
    pools = {r["puzzle"]: r["pool"] for r in P["rows"] if "pool" in r}
    S = json.load(open(a.shipped))
    delivered = {r["puzzle"] for r in S["rows"] if r.get("sequence")}

    jobs = [((p, i), c["seq"], T[p]["target"])
            for p in sorted(pools) for i, c in enumerate(pools[p])]
    print(f"{len(pools)} 个池,合计 {len(jobs)} 条候选,{a.workers} 进程\n", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    CH = max(1, len(jobs) // (a.workers * 8))
    chunks = [jobs[i:i + CH] for i in range(0, len(jobs), CH)]
    sc, t0, done = {}, time.time(), 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for res in ex.map(_score_chunk, chunks):
            for k, v in res:
                sc[k] = v
            done += 1
            if done % 40 == 0 or done == len(chunks):
                print(f"    {done}/{len(chunks)}  {(time.time()-t0)/60:.1f} min", flush=True)

    rows = {}
    for p in sorted(pools):
        v = sorted((sc[(p, i)] for i in range(len(pools[p])) if (p, i) in sc), reverse=True)
        if not v:
            continue
        top = v[0]
        rows[p] = {"n": len(v), "top": top, "median": st.median(v),
                   "frac_at_1": sum(1 for x in v if x >= 0.999) / len(v),
                   "n_tied_at_top": sum(1 for x in v if x >= top - 1e-12),
                   "delivered": p in delivered}

    def show(label, keys):
        if not keys:
            return
        fr = [rows[k]["frac_at_1"] for k in keys]
        ti = [rows[k]["n_tied_at_top"] for k in keys]
        print(f"  {label:22s} {len(keys):3d} 个池  满分占比中位 {st.median(fr):6.1%}"
              f"  顶端并列数中位 {st.median(ti):5.0f}  (范围 {min(ti)}-{max(ti)})")

    print(f"\n=== O1/O2 我们自己的池:饱和程度 ===")
    print(f"  {'对照 Eterna':22s}  56 个池  满分占比中位  77.4%  顶端并列数中位    78   -> 湿实验 -0.30 分")
    print(f"  {'对照 M2':22s}  35 个池  满分占比中位   0.0%  顶端并列数中位     2   -> 湿实验 +0.55 分")
    show("我们(全部)", sorted(rows))
    show("我们(已交付)", [k for k in sorted(rows) if rows[k]["delivered"]])
    show("我们(not_found 救回)", [k for k in sorted(rows) if not rows[k]["delivered"]])

    ours = st.median([rows[k]["n_tied_at_top"] for k in rows])
    verdict = ("像 Eterna —— 顶端是一个大平台,argmax 只是在平台上随机取一条,预期真实增益 ~0 或为负,"
               "重排结论应整体撤回"
               if ours >= 20 else
               "像 M2 —— 顶端并列很少,RNet 排序确有信号,但真实增益上限约 +0.55 湿实验分,"
               "绝不可用 +0.250 F1 的口径")
    print(f"\n  判读:{verdict}")

    json.dump({"per_target": rows, "median_tie_at_top": ours, "verdict": verdict,
               "reference": {"Eterna": {"frac_at_1": 0.774, "ties": 78, "wetlab_gain": -0.30},
                             "M2": {"frac_at_1": 0.0, "ties": 2, "wetlab_gain": +0.55}}},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
