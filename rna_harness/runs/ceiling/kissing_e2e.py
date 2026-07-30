#!/usr/bin/env python3
"""端到端:agent 从一句话开始,在整个领域几乎交付不了的 64 条靶标上,真的跑一遍。

WHY THIS AND NOT THE RE-ADJUDICATION. The 51-of-55 figure came from re-scoring sequences that were ALREADY in
the archive: it is a statement about the criterion, and it is silent about whether the shipped pipeline
actually produces such a sequence when someone asks. This runs the thing a user would run.

TWO ARMS, same targets, same budget, same verdict:
    agent   run() on a request that contains the structure -- adopt, check, design, certify
    direct  certify_design() on the same target -- the baseline without the loop

BUDGET IS `cheap` ON BOTH, DECLARED: one 5-second tier. Every delivery rate below is therefore a LOWER bound,
and the comparison between arms is what the run is for.

THE VERDICT IS THE SHIPPED ONE and carries its operating point: `knot_confirmed` means the CROSSING STEMS are
fully reproduced by at least one of three independent predictors (84.0% sensitivity on 250 natural
structures, 0.4% random false accept, and 0.0% on this very subgroup by the matched null). It is never a
claim about the whole structure.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
OUT = os.path.join(HERE, "kissing_e2e")


def work(job):
    tid, target, topology, arm, budget = job
    dest = os.path.join(OUT, f"{tid.replace(':', '_')}-{arm}.json")
    if os.path.exists(dest):
        return dest
    t0 = time.time()
    try:
        if arm == "agent":
            from rna_harness.cognitive.agent import run
            sess = run(f"我有一个结构 {target} ,请给我一条能折叠成它的 RNA 序列",
                       workspace=os.path.join(OUT, "_ws", f"{tid.replace(':', '_')}.json"))
            cert = sess.certificate or {}
            rec = {"verdict": cert.get("verdict") or ("refused" if sess.done else "ran-out"),
                   "sequence": cert.get("sequence"), "cost_usd": sess.total_cost_usd,
                   "llm_cost_usd": sess.llm_cost_usd, "n_steps": len(sess.steps),
                   "outcome": sess.outcome}
        else:
            from rna_harness.cognitive.certify import certify_design
            c = certify_design(target, budget=budget)
            rec = {"verdict": c.verdict, "sequence": c.sequence, "cost_usd": c.cost_usd,
                   "llm_cost_usd": 0.0, "n_steps": 0, "outcome": c.reason[:80]}
    except Exception as e:                                                    # noqa: BLE001
        rec = {"verdict": "error", "sequence": None, "cost_usd": 0.0, "llm_cost_usd": 0.0,
               "outcome": repr(e)[:160]}
    rec.update({"task_id": tid, "arm": arm, "topology": topology, "n": len(target),
                "delivered": rec["verdict"] in ("certified", "knot_confirmed"),
                "wall_s": round(time.time() - t0, 1)})
    with open(dest, "w") as f:
        json.dump(rec, f, ensure_ascii=False)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", default="cheap")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--arms", default="agent,direct")
    ap.add_argument("--analyse-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.join(OUT, "_ws"), exist_ok=True)
    top = {r["task_id"]: r for r in json.load(open(os.path.join(HERE, "pk_topology.json")))["rows"]}
    pk = {r["task_id"]: r for r in
          json.load(open(os.path.join(HERE, "agent", "pk_desirna.json")))["rows"]}
    sel = [t for t in top if top[t]["topology"] in ("kissing / chain", "higher-order") and t in pk]
    jobs = [(t, pk[t]["target"], top[t]["topology"], arm, a.budget)
            for t in sel for arm in a.arms.split(",")]
    print(f"端到端:{len(sel)} 条靶标(kissing + higher-order)x {len(a.arms.split(','))} 臂,"
          f"预算 {a.budget}\n", flush=True)
    if not a.analyse_only:
        todo = [j for j in jobs
                if not os.path.exists(os.path.join(OUT, f"{j[0].replace(':', '_')}-{j[3]}.json"))]
        print(f"  {len(todo)} 个待跑", flush=True)
        from concurrent.futures import ProcessPoolExecutor, as_completed
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for n, f in enumerate(as_completed([ex.submit(work, j) for j in todo]), 1):
                f.result()
                if n % 10 == 0:
                    print(f"    {n}/{len(todo)}  {(time.time()-t0)/60:.1f} min", flush=True)

    R = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, "*.json")))]
    print(f"\n=== {len(R)} 次运行,判据为出厂的已校准假结判据 ===")
    print(f"  {'臂':<8} {'拓扑类':<18} {'交付':>9} {'$求解器':>10} {'$思考':>9} {'中位耗时':>9}")
    out = {}
    for arm in a.arms.split(","):
        for g in ("kissing / chain", "higher-order"):
            sub = [r for r in R if r["arm"] == arm and r["topology"] == g]
            if not sub:
                continue
            d = sum(1 for r in sub if r["delivered"])
            solver = sum((r.get("cost_usd") or 0) - (r.get("llm_cost_usd") or 0) for r in sub)
            think = sum(r.get("llm_cost_usd") or 0 for r in sub)
            w = sorted(r.get("wall_s") or 0 for r in sub)
            print(f"  {arm:<8} {g:<18} {str(d) + '/' + str(len(sub)):>9} {solver:>10.4f} "
                  f"{think:>9.4f} {w[len(w)//2]:>8.0f}s")
            out[f"{arm}:{g}"] = {"delivered": d, "n": len(sub), "solver_usd": solver,
                                 "think_usd": think, "median_wall_s": w[len(w) // 2]}
    verd = {}
    for r in R:
        verd.setdefault(r["arm"], {}).setdefault(r["verdict"], 0)
        verd[r["arm"]][r["verdict"]] += 1
    print(f"\n  判决分布:{json.dumps(verd, ensure_ascii=False)}")
    json.dump({"budget": a.budget, "summary": out, "verdicts": verd, "rows": R},
              open(os.path.join(HERE, "kissing_e2e.json"), "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote rna_harness/runs/ceiling/kissing_e2e.json")


if __name__ == "__main__":
    main()
