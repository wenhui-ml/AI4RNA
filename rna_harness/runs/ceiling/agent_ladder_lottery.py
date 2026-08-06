#!/usr/bin/env python3
"""Is the agent ladder's last rung a capability, or six lottery tickets? Asked prospectively, on new seeds.

WHAT HAPPENED AND WHY THIS RUN EXISTS. agent_ladder_truncation.py replayed 400 stored cascades exactly and
found that dropping desirna@180s from the agent ladder saves 47.3% of cost on Eterna100 while losing 2.4% of
solves. Its pre-registered P2 then FAILED on a directional test -- 6 targets lost, 0 gained, p=0.0156 -- and
the verdict was "keep the rung".

That test was MIS-SPECIFIED and I am not going to pretend otherwise. In an exact replay of a
stop-at-first-success cascade, removing a rung can only lose solves and can never gain one, so "0 gained" is
true by construction and the sign test has no random mechanism to test. It reduces to "were there at least 5
losses", which is not what a directional test is for. Writing a criterion that cannot fail in one direction
is the defect instrument.assert_control_can_fail exists to catch, and it got past me.

Looking at the 6 targets afterwards showed only ONE is solved by all three seeds; the other five solve in
exactly 1 of 3, and four of those are 286-400 nt. That looks like luck rather than capability -- but it is
an analysis run AFTER a pre-registered test failed, which is exactly the move this project refuses. So it
decides nothing. It becomes a hypothesis, and this run tests it prospectively on seeds never used.

PRE-REGISTERED, before any new seed ran:

    On 10 FRESH seeds per target (seeds 100-109, disjoint from the 0/1/2 already on disk), for each of the
    six targets, measure how often the full ladder solves it and how often the truncated ladder does.

    P1  At least 4 of the 6 are LOTTERY TICKETS: the full ladder solves them in <= 3 of 10 seeds.
        A rung whose contribution is a one-in-three chance on four-fifths of the targets it "buys" is not
        delivering a capability, and the 47.3% it costs is not being earned.
    P2  Target 52 (the only one stable at 3/3) reproduces: full ladder solves it in >= 8 of 10 seeds AND
        the truncated ladder in strictly fewer. If it does not reproduce, then the rung buys NOTHING
        reliable at all and the case against it is stronger, not weaker.

    DECISION DECLARED NOW: P1 passing is what would justify dropping the rung from tools.py. P1 failing
    means the rung stays and the 47.3% is a price we have measured and chosen to pay. Either way the
    per-solve price is reported ($0.2604 per extra Eterna100 solve as measured).

    The cluster is the TARGET; 10 seeds of one target are 10 repeats, not 10 observations. Rates are
    per target and the six targets are reported individually, not averaged into a single number.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)

TARGETS = [52, 60, 73, 78, 96, 97]          # the six the last rung "buys" on Eterna100
SEEDS = list(range(100, 110))               # disjoint from the 0/1/2 already on disk
DROP = "desirna@180s"


def work(item):
    pid, target, seed = item
    from rna_harness.cognitive.core import FixedPolicy
    from rna_harness.cognitive.tools import default_toolbox
    t0 = time.time()
    try:
        r = FixedPolicy(default_toolbox(), name="ladder").solve(target, seed=seed)
        tr = [{k: s.get(k) for k in ("tool", "solved", "bp", "cost_usd")} for s in r["trace"]]
    except Exception as e:                                                     # noqa: BLE001
        return {"id": pid, "seed": seed, "error": repr(e)[:160]}
    return {"id": pid, "n": len(target), "seed": seed, "trace": tr,
            "wall_s": round(time.time() - t0, 1)}


def solved(trace, drop_last: bool):
    for t in trace:
        if drop_last and t["tool"].startswith(DROP):
            continue
        if t.get("solved"):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(HERE, "agent_ladder_lottery.json"))
    a = ap.parse_args()

    # Same loader call and same pseudoknot filter as verify_ladder.py, so the ids here index exactly the
    # rows those 400 stored cascades came from. Using a different filter would silently measure other targets.
    from rna_harness.benchmarks.eterna100 import load_eterna100
    from rna_harness.structure_utils import has_pseudoknot
    tasks = {int(p["id"]): p["target"] for p in load_eterna100(version="V2")
             if not has_pseudoknot(p["target"])}
    missing = [p for p in TARGETS if p not in tasks]
    if missing:
        print(f"!! 找不到靶标 {missing};可用 id 例:{sorted(tasks)[:8]}")
        sys.exit(1)
    jobs = [(p, tasks[p], s) for p in TARGETS for s in SEEDS]
    print(f"彩票检验:{len(TARGETS)} 个靶标 × {len(SEEDS)} 个【全新】种子 {SEEDS[0]}-{SEEDS[-1]} = {len(jobs)} 次\n"
          f"  这些种子与磁盘上已有的 0/1/2 不相交 —— 是前瞻检验,不是重新分析\n", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    R, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, f in enumerate(as_completed([ex.submit(work, j) for j in jobs]), 1):
            R.append(f.result())
            if n % 10 == 0 or n == len(jobs):
                print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)

    print(f"\n{'靶标':>6s}{'长度':>7s}{'完整阶梯':>10s}{'去掉@180':>10s}{'仅@180 才解':>12s}   判定")
    rows, lottery = {}, 0
    for p in TARGETS:
        g = [r for r in R if r.get("id") == p and "error" not in r]
        if not g:
            print(f"{p:6d}   全部失败")
            continue
        fu = sum(1 for r in g if solved(r["trace"], False))
        tr = sum(1 for r in g if solved(r["trace"], True))
        lot = fu <= 3
        lottery += lot
        rows[p] = {"n": g[0]["n"], "seeds": len(g), "full": fu, "trunc": tr, "only180": fu - tr,
                   "lottery": lot}
        print(f"{p:6d}{g[0]['n']:7d}{fu:>7d}/{len(g)}{tr:>7d}/{len(g)}{fu-tr:12d}   "
              f"{'彩票(≤3/10)' if lot else '稳定'}")

    p52 = rows.get(52)
    P1 = lottery >= 4
    P2 = bool(p52 and p52["full"] >= 8 and p52["trunc"] < p52["full"])
    print(f"\n=== 预注册判决 ===")
    print(f"  P1 六个里 ≥4 个是彩票:{lottery}/6  ->  {'通过' if P1 else '未通过'}")
    if p52:
        print(f"  P2 靶标 52 复现(完整 ≥8/10 且截断更少):{p52['full']}/10 对 {p52['trunc']}/10"
              f"  ->  {'通过' if P2 else '未通过 —— 那这一档连一个可靠的靶标都没买到,反对它的理由更强'}")
    print(f"\n  决定:{'把 desirna@180s 从 tools.py 的阶梯里去掉' if P1 else '保留 @180 —— 47.3% 是我们量过并选择支付的价格'}")
    print(f"  已量的单价:Eterna100 上每多解一个靶标 $0.2604")

    json.dump({"targets": rows, "n_lottery": lottery, "P1": P1, "P2": P2,
               "seeds": SEEDS, "rows": R}, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
