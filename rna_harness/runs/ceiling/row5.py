#!/usr/bin/env python3
"""Row 5 of the product table, under the criterion that is actually usable.

The three-arm acceptance was run under the STRICT rule (whole structure, all three oracles exact) and gave a
huge separation: the standard tool issues 96 certificates the panel confirms 0 of, we issue 16 and refuse 80.
But that rule has 0.8% sensitivity on known natural structures -- it refuses nearly everything, including
correct designs -- so a separation measured under it says little about what a user gets.

The calibrated rule (crossing stems fully reproduced by >=1 of 3 oracles; 84.0% sensitivity on natives,
3.1% [0.1, 16.2] applied-policy false accept) answers 124/125 for us. The open question, and the reason row 5
carries no tick yet, is what the STANDARD TOOL scores under that same rule. pKiss alone confirms the knot on
110/125 targets somewhere in the pool, so "124 vs 0" is certainly wrong and the real margin is unmeasured.

Two definitions of "what the standard tool hands you", because they differ and both are real:
  A1  best_pk_seq      -- the sequence DesiRNA itself reports as best
  A2  first pKiss hit  -- what DesirnaTool did before the panel: scan candidates, stop at the first the
                          reference oracle accepts, return it
B is ours: any candidate in the same pool that is knot_confirmed.
"""
from __future__ import annotations

import argparse, glob, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
AGENT = os.path.join(HERE, "agent")
OUT = os.path.join(HERE, "row5")


def work(item):
    tid, target, reported, pool = item
    dest = os.path.join(OUT, f"t{tid}.json")
    if os.path.exists(dest):
        return dest
    from rna_harness.cognitive.certify import _knot_recall
    from rna_harness.oracles_pk import forms_target, pkiss_pairs
    t0 = time.time()
    a1 = _knot_recall(reported, target)["knot_confirmed"] if reported else False
    first = next((c for c in pool if forms_target(pkiss_pairs(c), target) is True), None)
    a2 = _knot_recall(first, target)["knot_confirmed"] if first else False
    rec = {"id": tid, "n": len(target), "A1_reported": bool(a1), "A2_first_hit": bool(a2),
           "had_first_hit": first is not None, "wall_s": round(time.time() - t0, 2)}
    with open(dest, "w") as f:
        json.dump(rec, f)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-cands", type=int, default=50)
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
                seen.add(c); uniq.append(c)
        items.append((i, t, rows[i].get("best_pk_seq"), uniq[:a.max_cands]))
    todo = [it for it in items if not os.path.exists(os.path.join(OUT, f"t{it[0]}.json"))]
    print(f"row 5 under the calibrated rule: {len(held)} targets, {len(todo)} to run", flush=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, f in enumerate(as_completed([ex.submit(work, it) for it in todo]), 1):
            f.result()
            if n % 25 == 0:
                print(f"    {n}/{len(todo)}", flush=True)

    R = [json.load(open(f)) for f in glob.glob(os.path.join(OUT, "*.json"))]
    R = [r for r in R if r["id"] in set(held)]
    kd = {r["id"]: r for r in json.load(open(os.path.join(HERE, "knot_deliverable.json")))["per_target"]}
    B = sum(1 for r in R if kd.get(r["id"], {}).get("knot"))
    a1 = sum(r["A1_reported"] for r in R)
    a2 = sum(r["A2_first_hit"] for r in R)
    n = len(R)
    print(f"\n=== knot_confirmed, same {n} targets, same pools ===")
    print(f"  {'arm':<46} {'confirmed':>10}")
    print(f"  {'A1  the sequence DesiRNA reports as best':<46} {a1:>10}")
    print(f"  {'A2  first candidate the reference oracle accepts':<46} {a2:>10}")
    print(f"  {'B   any candidate in the pool (ours)':<46} {B:>10}")
    print(f"\n  margin over A1: {B - a1:+d}   over A2: {B - a2:+d}")
    only = [r["id"] for r in R if kd.get(r["id"], {}).get("knot") and not r["A1_reported"]]
    print(f"  targets where WE confirm the knot and the tool's own output does not: {len(only)}")
    from math import comb
    b, c = len(only), sum(1 for r in R if r["A1_reported"] and not kd.get(r["id"], {}).get("knot"))
    m = b + c
    p = min(1.0, 2.0 * sum(comb(m, i) for i in range(min(b, c) + 1)) / 2.0 ** m) if m else 1.0
    print(f"  exact McNemar vs A1: B-only {b}, A1-only {c}, p = {p:.3g}")
    json.dump({"n": n, "A1": a1, "A2": a2, "B": B, "B_only": only, "mcnemar_p": p,
               "per_target": R}, open(os.path.join(HERE, "row5.json"), "w"), indent=1)
    print(f"\n  wrote rna_harness/runs/ceiling/row5.json")


if __name__ == "__main__":
    main()
