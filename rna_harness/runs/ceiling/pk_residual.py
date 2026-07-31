#!/usr/bin/env python3
"""The ones the calibrated criterion still cannot deliver -- is that a SEARCH problem or a JUDGMENT problem?

WHY THIS MATTERS MORE THAN ITS SIZE. The shipped criterion turns 1/55 into 51/55 on kissing topologies and
0/9 into 5/9 on higher-order ones. The 8 that remain decide whether section 4.1 has headroom inside its own
domain or has hit its ceiling, and those are opposite conclusions with opposite next steps:

    SEARCH-LIMITED   a longer run does find a sequence the criterion accepts
                     -> the residual is bought with money, and the ladder already knows how
    JUDGMENT-LIMITED the pool contains sequences whose crossing stems no predictor will reproduce
                     -> more money buys nothing; only a better-calibrated or additional oracle moves it
    UNREACHABLE      nothing folds there at all, at any budget
                     -> correct behaviour is to refuse, and refusing is already implemented

THREE MEASUREMENTS, no interpretation beyond them. For each residual target the solver is re-run at a budget
an order of magnitude above the archived one and the whole returned pool is scored, so the answer separates
"we did not look hard enough" from "there is nothing there to accept". Every per-oracle result is kept: a
target where one predictor reproduces 90% of the crossing pairs and stops short is a different animal from one
where all three return nothing, and lumping them would hide exactly the signal worth having.

This is offline arithmetic over solver output. It spends CPU, not model calls.
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


def triage(job):
    tid, target, timelimit, seed = job
    from rna_harness.cognitive.certify import certify_sequence
    from rna_harness.generators.desirna import DesiRNAGenerator
    from rna_harness.task import DesignTask, OracleConfig
    t0 = time.time()
    oc = OracleConfig(param_set="turner2004", temperature=37.0, dangles=2)
    try:
        res = DesiRNAGenerator(replicas=10, timelimit_s=timelimit, results_number=50).design(
            DesignTask(task_id="resid", target=target, oracle=oc), seed=seed)
    except Exception as e:                                                    # noqa: BLE001
        return {"task_id": tid, "n": len(target), "error": repr(e)[:160]}

    pool, seen = [], set()
    for s in ([res.best_sequence] if res.best_sequence else []) + list(res.candidates or []):
        if s and len(s) == len(target) and s not in seen:
            seen.add(s)
            pool.append(s)

    accepted, best = None, None
    for s in pool:
        cert = certify_sequence(s, target)
        d = cert.detail or {}
        # HOW CLOSE THE PANEL CAME, per oracle -- the number that separates the two failure modes.
        # certify_sequence reports it at detail["knot"]["per_oracle"] as a per-oracle crossing-pair recall.
        recalls = ((d.get("knot") or {}).get("per_oracle")) or {}
        top = max(recalls.values()) if recalls else 0.0
        if cert.verdict in ("certified", "knot_confirmed"):
            accepted = {"sequence": s, "verdict": cert.verdict, "cross_recall": recalls}
            break
        if best is None or top > best["best_cross_recall"]:
            best = {"sequence": s, "verdict": cert.verdict, "best_cross_recall": top,
                    "cross_recall": recalls, "best_bp": d.get("best_bp")}

    if accepted:
        klass = "search-limited"
    elif best and best["best_cross_recall"] > 0.0:
        klass = "judgment-limited"
    else:
        klass = "unreachable-or-no-signal"
    return {"task_id": tid, "n": len(target), "pool": len(pool), "timelimit_s": timelimit,
            "class": klass, "accepted": accepted, "closest": best,
            "wall_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timelimit", type=int, default=180, help="an order of magnitude above the archive")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(HERE, "pk_residual.json"))
    a = ap.parse_args()

    deliv = json.load(open(os.path.join(HERE, "kissing_deliver.json")))
    top = {r["task_id"]: r for r in json.load(open(os.path.join(HERE, "pk_topology.json")))["rows"]}
    pk = {r["task_id"]: r for r in
          json.load(open(os.path.join(HERE, "agent", "pk_desirna.json")))["rows"]}
    resid = [r["task_id"] for r in deliv["rows"]
             if not r.get("delivered")
             and top.get(r["task_id"], {}).get("topology") in ("kissing / chain", "higher-order")]
    jobs = [(t, pk[t]["target"], a.timelimit, 0) for t in resid if t in pk]
    print(f"残差三分诊:{len(jobs)} 条(标定判据下仍未交付),每条 {a.timelimit}s\n", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    R, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, f in enumerate(as_completed([ex.submit(triage, j) for j in jobs]), 1):
            R.append(f.result())
            print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)

    from collections import Counter
    c = Counter(r.get("class", "error") for r in R)
    print(f"\n=== 分类 ===")
    for k, v in c.most_common():
        print(f"  {k:<26} {v}")
    for r in sorted(R, key=lambda x: x.get("task_id", "")):
        cl = r.get("closest") or {}
        print(f"  {r.get('task_id'):<16} n={r.get('n')}  {r.get('class', r.get('error', '?')):<26}"
              f" 最好交叉复现率 {cl.get('best_cross_recall', '-')}")
    json.dump({"timelimit_s": a.timelimit, "classes": dict(c), "rows": R},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
