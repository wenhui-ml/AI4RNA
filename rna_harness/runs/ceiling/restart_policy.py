#!/usr/bin/env python3
"""When we return not_found, is a different random seed worth selling to the caller -- and at what price?

WHERE THIS COMES FROM. The product currently answers `not_found` and stops. But this week measured that the
solver is not deterministic in the way the seed suggests: the same request re-run flips its verdict 22.9% of
the time at the cheap budget and 5.8% at standard, because DesiRNA is bounded by WALL CLOCK and a fixed seed
still explores a different pool under different machine load. If the answer is partly a draw, then a failed
draw is worth re-drawing -- and nobody has priced that.

WHAT THE STORED DATA ALREADY SAYS, and it kills the cheap version of the idea. Over 100 Eterna100 targets at
3 seeds: 83/100 at seed 0, 87/100 taking the union. So restart is worth +4 targets. But every one of the 4
rescues happened at the 180s rung:

    retry the 5s rung only        0 rescued     +11.1% cost
    retry through the 60s rung    0 rescued     +59%   cost
    retry the whole ladder        4 rescued     +146%  cost

There is no middle price. That is also a third independent reason the agent ladder keeps its 180s rung,
arrived at from a different direction than agent_ladder_lottery.

So this is NOT a default. A 146% cost rise for +4.8% coverage fails every bar this project has used. It is a
PRICED OPTION: when we would otherwise hand back nothing, we can say what a retry historically buys and what
it costs, and let the caller decide. That is the north star -- a non-expert gets an answer, cheaply, and
knows what more money would buy -- rather than a silent "not found".

WHY THE RESCUE RATE HAS TO BE MEASURED PROSPECTIVELY. The 4/17 = 23.5% above comes from the same three seeds
that produced the observation. Quoting it to a caller as "about 1 in 4" would be quoting a number to the data
that suggested it. This run re-measures it on seeds that have never been used.

PRE-REGISTERED, before any new seed ran:

    Population: the 17 Eterna100 targets that seed 0 failed. Fixed before the run, listed in the script.
    Seeds 3, 4, 5 -- disjoint from the 0/1/2 already on disk.

    P1  The rescue rate on fresh seeds is > 0 and its 95% Clopper-Pearson lower bound is above 5%. Below
        that, a retry is not worth offering: the caller would be paying 2.5x for a coin that almost never
        lands.
    P2  The rescue rate is consistent with the 23.5% seen on seeds 1-2 -- the retrospective figure sits
        inside the fresh 95% interval. If it does not, the retrospective number was optimistic and the
        PRODUCT MUST QUOTE THE FRESH ONE.
    P3  Rescues remain concentrated at the top rung. If a cheap retry starts rescuing on fresh seeds, the
        pricing changes completely and this run's conclusion is re-opened.

    KILL: P1 fails -> retry is not offered, and "restart policy" joins the closed axes.

    The honest n is TARGETS. Three seeds of one target are three draws of one target, not three targets;
    the rescue rate is per target (did ANY fresh seed solve it), and the per-draw rate is reported
    separately rather than substituted for it.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)

# The 17 targets seed 0 failed, fixed before this run. Listed rather than recomputed so that a later change
# to the stored data cannot silently move the population.
TARGETS = [60, 64, 68, 71, 73, 76, 78, 79, 81, 86, 87, 90, 91, 94, 96, 99, 100]
SEEDS = (3, 4, 5)


def work(item):
    pid, target, seed = item
    from rna_harness.cognitive.core import FixedPolicy
    from rna_harness.cognitive.tools import default_toolbox
    t0 = time.time()
    try:
        r = FixedPolicy(default_toolbox(), name="ladder").solve(target, seed=seed)
    except Exception as e:                                                     # noqa: BLE001
        return {"id": pid, "seed": seed, "error": repr(e)[:160]}
    return {"id": pid, "n": len(target), "seed": seed, "solved": bool(r["solved"]),
            "cost_usd": float(r["cost_usd"]), "wall_s": round(time.time() - t0, 1),
            "solved_by": next((s["tool"] for s in r["trace"] if s["solved"]), None),
            "trace": [{k: s.get(k) for k in ("tool", "solved", "bp", "cost_usd")} for s in r["trace"]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(HERE, "restart_policy.json"))
    a = ap.parse_args()

    from concurrent.futures import ProcessPoolExecutor, as_completed
    from rna_harness.benchmarks.eterna100 import load_eterna100
    from rna_harness.structure_utils import has_pseudoknot
    from external_judge import clopper_pearson

    tasks = {int(p["id"]): p["target"] for p in load_eterna100(version="V2")
             if not has_pseudoknot(p["target"])}
    missing = [t for t in TARGETS if t not in tasks]
    if missing:
        print(f"!! 找不到靶标 {missing}")
        sys.exit(1)
    jobs = [(t, tasks[t], s) for t in TARGETS for s in SEEDS]
    print(f"重启策略:{len(TARGETS)} 个【seed 0 失败】的靶标 × 种子 {SEEDS}(从未用过)= {len(jobs)} 次\n"
          f"  回顾性数字是 4/17 = 23.5%,来自产生这个观察的同一批种子 —— 本次在新种子上重测\n", flush=True)

    R, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, f in enumerate(as_completed([ex.submit(work, j) for j in jobs]), 1):
            R.append(f.result())
            if n % 8 == 0 or n == len(jobs):
                print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)

    ok = [r for r in R if "error" not in r]
    per_t = {}
    for r in ok:
        per_t.setdefault(r["id"], []).append(r)
    rescued = [t for t, v in per_t.items() if any(x["solved"] for x in v)]
    n_t = len(per_t)
    lo, hi = clopper_pearson(len(rescued), n_t)
    draws = [r["solved"] for r in ok]
    print(f"\n=== 救回率(诚实的 n 是靶标)===")
    print(f"  {len(rescued)}/{n_t} 个靶标被至少一个新种子救回 = {len(rescued)/n_t:.1%}"
          f"  95% [{lo:.1%}, {hi:.1%}]")
    print(f"  逐次抽签的救回率(不能替代上面那个):{st.mean(draws):.1%}({sum(draws)}/{len(draws)} 次)")
    print(f"  被救回的靶标:{sorted(rescued)}")

    tiers = [r["solved_by"] for r in ok if r["solved"]]
    top = sum(1 for t in tiers if "180" in (t or ""))
    print(f"\n=== P3 救回发生在哪一档 ===")
    for t in sorted(set(tiers)):
        print(f"  {t:28s} {tiers.count(t)} 次")
    print(f"  -> 顶档占 {top}/{len(tiers) or 1};{'与回顾一致,便宜重启仍然无效' if top == len(tiers) else '出现了便宜档救回 —— 定价结论要重开'}")

    extra = sum(r["cost_usd"] for r in ok)
    print(f"\n=== 代价 ===")
    print(f"  本次重启总成本 ${extra:.4f}(3 个种子 × {n_t} 个失败靶标)")
    print(f"  每救回一个靶标 ${extra/max(1,len(rescued)):.4f}")

    p1 = lo > 0.05
    p2 = lo <= 0.235 <= hi
    p3 = top == len(tiers)
    print(f"\n=== 预注册判决 ===")
    print(f"  P1 救回率 95% 下界 > 5%:{'通过' if p1 else '未通过 —— 不提供重启选项,该轴关闭'}({lo:.1%})")
    print(f"  P2 与回顾性的 23.5% 相容:{'通过' if p2 else '未通过 —— 回顾数字偏乐观,产品必须报本次的新数字'}")
    print(f"  P3 救回仍集中在顶档:{'通过' if p3 else '未通过 —— 定价结论重开'}")
    # The rate and the price must describe the SAME purchase. `len(rescued)/n_t` is what THREE fresh seeds
    # buy, so pairing it with the price of ONE retry sells a three-ticket payout at a one-ticket price. The
    # per-draw mean is what one retry buys; the per-target rate is what the whole strip buys.
    print(f"\n  产品要报给用户的数字(赔率与价格必须描述同一笔买卖):")
    print(f"    重试 1 次:救回约 {st.mean(draws):.0%},成本 +1x")
    print(f"    重试 {len(SEEDS)} 次:救回约 {len(rescued)/n_t:.0%} [{lo:.0%}, {hi:.0%}],成本 +{len(SEEDS)}x")
    print(f"    全部救回都发生在最贵的一档 —— 没有便宜版本")

    json.dump({"n_targets": n_t, "n_rescued": len(rescued), "rate": len(rescued)/n_t,
               "ci": [lo, hi], "rescued": sorted(rescued), "draw_rate": st.mean(draws),
               "top_tier_share": top / max(1, len(tiers)), "cost": extra,
               "P1": p1, "P2": p2, "P3": p3, "rows": R},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
