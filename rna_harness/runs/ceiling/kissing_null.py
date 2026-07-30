#!/usr/bin/env python3
"""同一条判据,在同一批拓扑上,喂随机序列 —— 这条规则在这个亚群上还成立吗?

WHY THIS RUNS BEFORE THE RESULT IS QUOTED. The calibrated pseudoknot rule (crossing stems fully reproduced by
at least one of three oracles) was measured at 84.0% sensitivity and 0.4% random false accept on 250
Pseudobase natives -- as ONE population. It was never stratified by topology. Re-adjudicating the archived
candidates under it gives 12/12 on kissing targets where the strict rule gives 0, and a number that good is
exactly when this project's history says to check whether the rule has quietly become vacuous on the
subgroup: if kissing targets carry few crossing pairs, or pairs these predictors recall for free, then random
sequences will "confirm" too.

MATCHED NULL: for each target, the same number of draws as the real pool, drawn uniformly from ACGU at the
same length, judged by the identical rule. If the null accepts at a rate anywhere near the real one, the
result is an artefact of the subgroup and must not be reported as a capability.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)


def work(job):
    tid, target, n_draw, seed = job
    from rna_harness.cognitive.certify import _knot_recall
    rng = random.Random(hash((tid, seed)) & 0xFFFF)
    hits = 0
    for _ in range(n_draw):
        s = "".join(rng.choice("ACGU") for _ in range(len(target)))
        if _knot_recall(s, target).get("knot_confirmed"):
            hits += 1
    from rna_harness.benchmarks.rnainvbench import _round_only, pk_pairs
    return {"task_id": tid, "n_draw": n_draw, "hits": hits,
            "n_cross": len(pk_pairs(target) - pk_pairs(_round_only(target)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--per-group", type=int, default=12)
    ap.add_argument("--draws", type=int, default=25)
    a = ap.parse_args()
    top = {r["task_id"]: r for r in json.load(open(os.path.join(HERE, "pk_topology.json")))["rows"]}
    pk = {r["task_id"]: r for r in
          json.load(open(os.path.join(HERE, "agent", "pk_desirna.json")))["rows"]}
    groups = ["kissing / chain", "higher-order", "H-type"]
    jobs, keep = [], {g: 0 for g in groups}
    for tid, t in top.items():
        g = t["topology"]
        if g in groups and tid in pk and keep[g] < a.per_group:
            keep[g] += 1
            jobs.append((tid, pk[tid]["target"], a.draws, 1))
    print(f"匹配零假设:{len(jobs)} 个靶标 x {a.draws} 条随机序列,判据与真实结果完全相同\n", flush=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    R = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, f in enumerate(as_completed([ex.submit(work, j) for j in jobs]), 1):
            R.append(f.result())
            if n % 8 == 0:
                print(f"    {n}/{len(jobs)}", flush=True)
    by = {r["task_id"]: r for r in R}
    real = json.load(open(os.path.join(HERE, "kissing_deliver.json")))
    got = {r["task_id"]: r for r in real["rows"]}

    print(f"\n=== 判据在每个拓扑亚群上还成立吗 ===")
    print(f"  {'拓扑类':<18} {'条数':>5} {'交叉对(中位)':>13} {'真实交付':>10} "
          f"{'随机也被确认的靶标':>18} {'每次抽取的误收率':>16}")
    out = {}
    for g in groups:
        ids = [t for t in top if top[t]["topology"] == g and t in by]
        if not ids:
            continue
        nc = sorted(by[t]["n_cross"] for t in ids)
        anyhit = sum(1 for t in ids if by[t]["hits"] > 0)
        rate = sum(by[t]["hits"] for t in ids) / sum(by[t]["n_draw"] for t in ids)
        deliv = sum(1 for t in ids if got.get(t, {}).get("delivered"))
        print(f"  {g:<18} {len(ids):>5} {nc[len(nc)//2]:>13} {deliv:>10} {anyhit:>18} {rate:>16.1%}")
        out[g] = {"n": len(ids), "median_cross_pairs": nc[len(nc) // 2], "delivered": deliv,
                  "targets_with_any_random_hit": anyhit, "per_draw_false_accept": round(rate, 4)}
    print("\n  读法:每次抽取的误收率若接近真实交付率,这条判据在该亚群上就是空的,不能当能力报。")
    json.dump({"by_topology": out, "rows": R},
              open(os.path.join(HERE, "kissing_null.json"), "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote rna_harness/runs/ceiling/kissing_null.json")


if __name__ == "__main__":
    main()
