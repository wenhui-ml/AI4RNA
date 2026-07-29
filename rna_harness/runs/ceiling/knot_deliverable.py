#!/usr/bin/env python3
"""What the calibrated criterion is worth in deliverable answers, on the same 125 held-out crossing targets.

The exact-and-unanimous rule yields 16/125 and has 0.8% sensitivity on known natural structures -- it refuses
almost everything, including things that are right. Calibration found one rule that separates real structures
from noise: the CROSSING STEMS fully reproduced by at least one of the three predictors, 84.0% sensitivity at
0.4% false accept on random sequences.

This measures what that buys a user. Same stored pools, no new search. Reported against both the strict rule
and the population that a single-oracle tool would have handed over, so the three are directly comparable.

The claim being counted is deliberately narrower than the strict one: `knot_confirmed` says the pseudoknot
stems are reproduced, NOT that the whole structure is. Composing it with an exact nested-skeleton check was
tried and refuted -- ViennaRNA recovers the nested reduction from only 20 of 186 native sequences -- so the
narrower claim is the honest one to report, and it is labelled as such everywhere it appears.
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
AGENT = os.path.join(HERE, "agent")
OUT = os.path.join(HERE, "knot_deliverable")


def work(item):
    tid, target, pool = item
    dest = os.path.join(OUT, f"t{tid}.json")
    if os.path.exists(dest):
        return dest
    from rna_harness.cognitive.certify import _knot_recall, _panel_verdict
    t0 = time.time()
    knot, strict, ref_only = None, None, None
    for c in pool:
        v = _panel_verdict(c, target)
        if strict is None and v["unanimous"]:
            strict = c
        if ref_only is None and v["reference_only"]:
            ref_only = c
        if knot is None:
            kr = _knot_recall(c, target)
            if kr["knot_confirmed"]:
                knot = (c, kr["confirmed_by"])
        if strict and knot:
            break
    rec = {"id": tid, "n": len(target), "n_pool": len(pool),
           "strict": strict is not None, "knot": knot is not None,
           "knot_by": (knot[1] if knot else []), "ref_only": ref_only is not None,
           "wall_s": round(time.time() - t0, 2)}
    with open(dest, "w") as f:
        json.dump(rec, f)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-cands", type=int, default=50)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--analyse-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    rows = {int(r["task_id"].split(":")[1]): r
            for r in json.load(open(os.path.join(AGENT, "pk_desirna.json")))["rows"]}
    held = sorted(i for i in rows if i % 2 == 1)
    items = []
    for i in held:
        t = rows[i]["target"]
        seen, uniq = set(), []
        for c in [rows[i].get("best_pk_seq")] + list(rows[i].get("candidates") or []):
            if c and len(c) == len(t) and c not in seen:
                seen.add(c)
                uniq.append(c)
        items.append((i, t, uniq[:a.max_cands]))
    print(f"calibrated criterion on {len(held)} held-out crossing targets, stored pools\n", flush=True)

    if not a.analyse_only:
        todo = [it for it in items if not os.path.exists(os.path.join(OUT, f"t{it[0]}.json"))]
        print(f"  {len(todo)} targets, {a.workers} workers", flush=True)
        from concurrent.futures import ProcessPoolExecutor, as_completed
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(work, it) for it in todo]
            for n, _f in enumerate(as_completed(futs), 1):
                if n % 25 == 0 or n == len(todo):
                    print(f"    {n}/{len(todo)}  {(time.time()-t0)/60:.1f} min", flush=True)

    R = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, "*.json")))]
    R = [r for r in R if r["id"] in set(held)]
    n = len(R)
    strict = sum(r["strict"] for r in R)
    knot = sum(r["knot"] for r in R)
    both = sum(1 for r in R if r["strict"] and r["knot"])
    neither = sum(1 for r in R if not r["strict"] and not r["knot"])

    print(f"\n=== what each criterion can hand a user, {n} targets ===")
    print(f"  {'criterion':<44} {'targets':>8} {'sensitivity on natives':>24} {'random FA':>11}")
    print(f"  {'certified (whole structure, all 3 exact)':<44} {strict:>8} {'0.8%':>24} {'0.0%':>11}")
    print(f"  {'knot_confirmed (crossing stems, >=1 oracle)':<44} {knot:>8} {'84.0%':>24} {'0.4%':>11}")
    print(f"\n  strict is a subset of knot: {both}/{strict}   neither: {neither}/{n}")
    print(f"  -> the calibrated rule answers {knot/max(n,1):.0%} of requests against "
          f"{strict/max(n,1):.0%}, a {knot/max(strict,1):.1f}x increase in answered requests")

    from collections import Counter
    by = Counter(o for r in R for o in r["knot_by"])
    print(f"\n=== which predictor confirms the knot ===")
    for o, c in by.most_common():
        print(f"  {o:<12} {c:>4}")
    print(f"  (a target counts once per confirming oracle, so these sum above {knot})")

    print(f"\n=== by length ===")
    for lo, hi, lbl in ((0, 40, "<=40nt"), (41, 70, "41-70"), (71, 10 ** 9, ">70")):
        sub = [r for r in R if lo <= r["n"] <= hi]
        if sub:
            print(f"  {lbl:>7}: certified {sum(r['strict'] for r in sub):>3}/{len(sub):<3}   "
                  f"knot_confirmed {sum(r['knot'] for r in sub):>3}/{len(sub)}")

    json.dump({"n": n, "certified": strict, "knot_confirmed": knot, "both": both, "neither": neither,
               "confirming_oracle": dict(by), "per_target": R},
              open(os.path.join(HERE, "knot_deliverable.json"), "w"), indent=1)
    print(f"\n  wrote rna_harness/runs/ceiling/knot_deliverable.json")


if __name__ == "__main__":
    main()
