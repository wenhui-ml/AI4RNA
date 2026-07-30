#!/usr/bin/env python3
"""在整个领域几乎交付不了任何东西的地方,我们能交付什么?

WHERE THIS COMES FROM, and it is one step rather than a leap. Re-reporting the archived pseudoknot results by
TOPOLOGY (computed from the target's own pair table, not annotated) produced a domain-shaped fact:

    topology         n     single-oracle claims   all-three-exact confirms
    H-type         187                      179                        58
    kissing/chain   55                        1                         1
    higher-order     9                        0                         0

64 targets, one claim between them. That is what "the field cannot get accurate results here" looks like, and
it is the same task the agent already does -- given a target structure, return a verified sequence.

WHAT IS ACTUALLY BEING ASKED. The confirms column above uses the STRICT rule: all three predictors reproduce
the whole structure exactly. That rule has 0.8% sensitivity on 250 NATURAL structures -- it rejects most of
nature's own answers -- so "1 of 55" says almost nothing about whether a design is good. The rule this project
actually ships for pseudoknots is different and CALIBRATED: are the CROSSING STEMS -- the pairs that make the
target a pseudoknot at all -- fully reproduced by at least one of the three (k=1, tau=1.0), which measures
84.0% sensitivity on those same natives at 0.4% false accept on random sequences.

So the question is not "can we search better". It is: on the topologies where the strict rule delivers ~1 of
64, what does the calibrated rule deliver from the SAME archived candidates? Zero solver cost -- this is
re-adjudication of sequences that were already generated and stored.

The answer is reported per topology, with the operating point attached, because `knot_confirmed` is a claim
about the crossing stems only and must never be read as a claim about the whole structure.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)


def work(job):
    tid, target, cands = job
    from rna_harness.cognitive.certify import KNOT_OPERATING_POINT, _knot_recall
    best = None
    for seq in cands:
        if not seq or len(seq) != len(target):
            continue
        kr = _knot_recall(seq, target)
        ran = [v for v in kr["per_oracle"].values() if v is not None] if "per_oracle" in kr else []
        if kr.get("knot_confirmed"):
            return {"task_id": tid, "delivered": True, "sequence": seq,
                    "confirmed_by": kr.get("confirmed_by"), "n_cross": kr.get("n_cross")}
        if best is None or (ran and max(ran) > best[1]):
            best = (seq, max(ran) if ran else 0.0)
    return {"task_id": tid, "delivered": False, "best_recall": round(best[1], 3) if best else None,
            "note": KNOT_OPERATING_POINT["rule"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    a = ap.parse_args()
    top = {r["task_id"]: r for r in json.load(open(os.path.join(HERE, "pk_topology.json")))["rows"]}
    pk = {r["task_id"]: r for r in
          json.load(open(os.path.join(HERE, "agent", "pk_desirna.json")))["rows"]}
    groups = ["kissing / chain", "higher-order", "H-type"]
    jobs = []
    for tid, t in top.items():
        if t["topology"] not in groups or tid not in pk:
            continue
        row = pk[tid]
        cands = ([row["best_pk_seq"]] if row.get("best_pk_seq") else []) + list(row.get("candidates") or [])
        if cands:
            jobs.append((tid, row["target"], cands))
    if a.limit:
        keep = {g: 0 for g in groups}
        sel = []
        for j in jobs:
            g = top[j[0]]["topology"]
            if keep[g] < a.limit:
                keep[g] += 1
                sel.append(j)
        jobs = sel
    print(f"重新裁决 {len(jobs)} 个靶标的归档候选(零求解成本)\n", flush=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    R = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, f in enumerate(as_completed([ex.submit(work, j) for j in jobs]), 1):
            R.append(f.result())
            if n % 20 == 0:
                print(f"    {n}/{len(jobs)}", flush=True)
    by = {r["task_id"]: r for r in R}

    print(f"\n=== 出厂的已校准判据 vs 严格判据,同一批归档候选 ===")
    print(f"  {'拓扑类':<18} {'条数':>5} {'严格(三家全对)':>16} {'已校准(交叉茎,k=1)':>20}")
    out = {}
    for g in groups:
        ids = [tid for tid in top if top[tid]["topology"] == g and tid in by]
        strict = sum(1 for tid in ids if top[tid]["panel_confirms"])
        calib = sum(1 for tid in ids if by[tid]["delivered"])
        print(f"  {g:<18} {len(ids):>5} {strict:>16} {calib:>20}")
        out[g] = {"n": len(ids), "strict": strict, "calibrated": calib}
    print("\n  已校准判据的工作点必须随行:交叉茎被 >=1 家完全复现;")
    print("  250 个天然结构上灵敏度 84.0%,随机序列误收 0.4%;它只说【交叉茎】,不说整条结构。")
    json.dump({"by_topology": out, "rows": R},
              open(os.path.join(HERE, "kissing_deliver.json"), "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote rna_harness/runs/ceiling/kissing_deliver.json")


if __name__ == "__main__":
    main()
