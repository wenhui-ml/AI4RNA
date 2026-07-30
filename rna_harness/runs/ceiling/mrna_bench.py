#!/usr/bin/env python3
"""领域 B:密码子约束 mRNA 设计,三臂,全部精确判定。

  A  工具自己的答案          DesiRNA + 它能表达的最接近的逐位 IUPAC 约束(它在这个领域的最佳可用机制)
  A2 筛选工具的候选池        我们的既有机制,用在它的池上
  B  同义密码子池 + 筛选     我们自己的提议分布 —— 采样同义密码子是平凡且精确的,不是"更好的搜索"

判据对三臂完全相同,而且都是精确的:**翻译必须等于所要求的肽,且 MFE 必须等于靶结构。**
靶结构取自某个同义变体的 MFE,所以【至少存在一个解】,失败绝不会是"无解"。

预期(跑前声明):A 会折叠对但**翻译错**,因为它的约束多放进 35-40% 编码错误氨基酸的三联体;
A2 取决于它的池里有没有恰好翻译正确的;B 按构造一定翻译正确,问题只在折叠。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)


def run_one(item, timelimit, seed, sample_budget_s):
    from rna_harness.benchmarks.mrna_suite import CODONS, translate
    from rna_harness.experiments.rssd_constrained import ConstrainedDesiRNA
    from rna_harness.task import DesignTask, OracleConfig
    import RNA
    target, pep = item["target"], item["peptide"]

    def ok(s):
        return bool(s) and translate(s) == pep and RNA.fold(s)[0] == target

    # ---- A / A2: the tool, given the closest restraint it can express -----------------------------
    t0 = time.time()
    res = ConstrainedDesiRNA(replicas=10, timelimit_s=timelimit, results_number=50).design(
        DesignTask(task_id="mrna", target=target,
                   oracle=OracleConfig(param_set="turner2004", temperature=37.0, dangles=2),
                   meta={"seq_restr": item["seq_restr"]}), seed=seed)
    tool_wall = time.time() - t0
    pool, seen = [], set()
    for s in ([res.best_sequence] if res.best_sequence else []) + list(res.candidates or []):
        if s and len(s) == len(target) and s not in seen:
            seen.add(s)
            pool.append(s)
    tool = res.best_sequence
    a2 = next((s for s in pool if ok(s)), None)

    # ---- B: our own proposal distribution -- synonymous codons, sampled ---------------------------
    t1 = time.time()
    rng = random.Random(1000 + seed)
    b, tried = None, 0
    while time.time() - t1 < sample_budget_s:
        cand = "".join(rng.choice(CODONS[aa]) for aa in pep)
        tried += 1
        if RNA.fold(cand)[0] == target:      # translation is correct BY CONSTRUCTION here
            b = cand
            break
    b_wall = time.time() - t1

    return {"id": item["id"], "n": item["n"], "seed": seed, "leak": item["leak"]["leak"],
            "pool": len(pool), "tool_wall_s": round(tool_wall, 1), "b_wall_s": round(b_wall, 1),
            "b_sampled": tried,
            "tool_sequence": tool,
            "tool_folds": bool(tool) and RNA.fold(tool)[0] == target,
            "tool_translates": bool(tool) and translate(tool) == pep,
            "tool_protein": translate(tool) if tool else None,
            "tool_ok": ok(tool),
            "a2_ok": bool(a2), "b_ok": bool(b),
            "pool_translating": sum(1 for s in pool if translate(s) == pep),
            "pool_qualifying": sum(1 for s in pool if ok(s))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timelimit", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--sample-budget", type=float, default=15.0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    from rna_harness.benchmarks.mrna_suite import build
    S = build()
    jobs = [(it, a.timelimit, s, a.sample_budget) for it in S for s in range(a.seeds)]
    print(f"mRNA 密码子约束基准: {len(S)} 个肽 x {a.seeds} 种子 = {len(jobs)} 次\n")
    for it in S:
        print(f"  {it['id']:<12} {it['n']:>3} nt  逐位约束泄漏 {it['leak']['leak']:.0%}")
    print(flush=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    R, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, f in enumerate(as_completed([ex.submit(run_one, *j) for j in jobs]), 1):
            R.append(f.result())
            print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)

    nA = sum(1 for r in R if r["tool_ok"])
    nA2 = sum(1 for r in R if r["a2_ok"])
    nB = sum(1 for r in R if r["b_ok"])
    print(f"\n=== {len(R)} 次,判据相同且精确:翻译正确 AND 折叠等于靶结构 ===")
    print(f"  A  工具自己的答案                {nA}/{len(R)}")
    print(f"  A2 筛选工具的池                  {nA2}/{len(R)}")
    print(f"  B  同义密码子池 + 筛选           {nB}/{len(R)}")
    print(f"\n  拆开看工具为什么失败:")
    print(f"    它的答案折叠正确               {sum(1 for r in R if r['tool_folds'])}/{len(R)}")
    print(f"    它的答案**翻译正确**            {sum(1 for r in R if r['tool_translates'])}/{len(R)}   <- 关键")
    pt = [r["pool_translating"] for r in R]
    print(f"    它整个池里翻译正确的条数(中位) {sorted(pt)[len(pt)//2]} / {sorted(r['pool'] for r in R)[len(R)//2]}")
    tw = sorted(r["tool_wall_s"] for r in R)
    bw = sorted(r["b_wall_s"] for r in R)
    print(f"\n  耗时中位:工具 {tw[len(tw)//2]:.1f}s   同义采样 {bw[len(bw)//2]:.1f}s "
          f"(采样条数中位 {sorted(r['b_sampled'] for r in R)[len(R)//2]})")
    b = sum(1 for r in R if r["b_ok"] and not r["tool_ok"])
    c = sum(1 for r in R if r["tool_ok"] and not r["b_ok"])
    n = b + c
    p = (sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n * 2) if n else 1.0
    print(f"  McNemar (B vs A): {b}-{c}, p={p:.2e}")
    print(f"\n  {'肽':<12} {'A':>4} {'A2':>4} {'B':>4}  {'工具交付的蛋白(取一例)':<24}")
    for it in S:
        sub = [r for r in R if r["id"] == it["id"]]
        wrong = next((r["tool_protein"] for r in sub if r["tool_protein"] != it["peptide"]), None)
        print(f"  {it['id']:<12} {sum(1 for r in sub if r['tool_ok']):>4} "
              f"{sum(1 for r in sub if r['a2_ok']):>4} {sum(1 for r in sub if r['b_ok']):>4}  "
              f"{str(wrong)[:24]:<24} (要求 {it['peptide']})")
    json.dump({"n": len(R), "A": nA, "A2": nA2, "B": nB, "mcnemar": {"b": b, "c": c, "p": p},
               "rows": R}, open(os.path.join(HERE, "mrna_bench.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"\n  wrote rna_harness/runs/ceiling/mrna_bench.json")


if __name__ == "__main__":
    main()
