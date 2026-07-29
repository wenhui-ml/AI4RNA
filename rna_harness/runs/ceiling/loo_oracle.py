#!/usr/bin/env python3
"""Is "16/125 deliverable" a property of the CONJUNCTION, or of ShapeKnots alone?

The acceptance result rests on a three-oracle panel that WE chose, and the rejection is not evenly shared.
Over 2783 candidate screenings the first rejection came from ShapeKnots 60.6% of the time, ProbKnot 37.9%,
and pKiss 1.5%. So one predictor holds most of the veto. If ShapeKnots is simply a conservative or poorly
calibrated predictor at these lengths, then:

  * "16/125 oracle-robust designs" is really "designs ShapeKnots happens to like", and
  * "96 false certificates" is really "96 sequences ShapeKnots disagrees with",

and both headline numbers are about one tool rather than about robustness. That is the single most dangerous
unexamined assumption in the acceptance table, so it gets measured before anything is built on it.

METHOD. Re-screen the same stored pools with NO short-circuit: evaluate all three oracles on every candidate
and store the three verdicts. Every leave-one-out and single-oracle count then falls out of re-analysis, with
one scan. `None` (the oracle failed to run) is kept distinct from False throughout -- conflating them is the
bug that once made the tool report zero solves with no diagnostic.

WHAT WOULD FALSIFY THE PANEL. If dropping ShapeKnots multiplies the deliverable count several-fold while
dropping ProbKnot or pKiss barely moves it, the panel is a ShapeKnots filter wearing a conjunction's clothes.
If all three leave-one-out counts stay in the same range, the conjunction is doing the work.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
AGENT = os.path.join(HERE, "agent")
OUT = os.path.join(HERE, "loo_oracle")
ORACLES = ("pkiss", "probknot", "shapeknots")


def work(item):
    tid, target, pool = item
    dest = os.path.join(OUT, f"t{tid}.json")
    if os.path.exists(dest):
        return dest
    from rna_harness.oracles_pk import forms_target, pkiss_pairs, probknot_pairs, shapeknots_pairs
    fns = {"pkiss": pkiss_pairs, "probknot": probknot_pairs, "shapeknots": shapeknots_pairs}
    t0 = time.time()
    cands = []
    for c in pool:
        v = {k: forms_target(fns[k](c), target) for k in ORACLES}   # NO short-circuit: all three, always
        cands.append({"seq": c, **{k: v[k] for k in ORACLES}})
    rec = {"id": tid, "n": len(target), "n_cands": len(cands), "cands": cands,
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
    print(f"leave-one-oracle-out on {len(held)} held-out crossing targets, no short-circuit\n", flush=True)

    if not a.analyse_only:
        todo = [it for it in items if not os.path.exists(os.path.join(OUT, f"t{it[0]}.json"))]
        print(f"  {len(todo)} targets, {a.workers} workers", flush=True)
        from concurrent.futures import ProcessPoolExecutor, as_completed
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(work, it) for it in todo]
            for n, _f in enumerate(as_completed(futs), 1):
                if n % 25 == 0 or n == len(todo):
                    el = time.time() - t0
                    print(f"    {n}/{len(todo)}  {el/60:.1f} min  "
                          f"eta {el/max(n,1)*(len(todo)-n)/60:.1f} min", flush=True)

    R = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, "*.json")))]
    R = [r for r in R if r["id"] in set(held)]

    def deliverable(subset):
        """targets where SOME single candidate satisfies every oracle in `subset` (None never counts)."""
        return [r["id"] for r in R
                if any(all(c.get(k) is True for k in subset) for c in r["cands"])]

    print(f"\n=== deliverable count under every oracle subset ({len(R)} targets) ===")
    print(f"  {'criterion':<34} {'deliverable':>12} {'vs full panel':>14}")
    full = deliverable(ORACLES)
    rows_out = {}
    for size in (3, 2, 1):
        for sub in itertools.combinations(ORACLES, size):
            d = deliverable(sub)
            rows_out[" AND ".join(sub)] = len(d)
            tag = ""
            if size == 2:
                dropped = [o for o in ORACLES if o not in sub][0]
                tag = f"(drop {dropped})"
            print(f"  {' AND '.join(sub) + ' ' + tag:<34} {len(d):>12} "
                  f"{(len(d)/max(len(full),1)):>13.2f}x")

    print(f"\n=== the falsifier ===")
    drops = {o: len(deliverable(tuple(x for x in ORACLES if x != o))) for o in ORACLES}
    worst = max(drops, key=lambda k: drops[k])
    print(f"  dropping each oracle in turn: " + ", ".join(f"{k} -> {v}" for k, v in drops.items()))
    ratio = drops[worst] / max(len(full), 1)
    print(f"  largest effect: dropping {worst} takes {len(full)} -> {drops[worst]} ({ratio:.2f}x)")
    if ratio >= 3.0:
        print(f"  VERDICT: the panel is largely a {worst} filter -- the headline is about ONE predictor.")
    elif ratio >= 1.8:
        print(f"  VERDICT: {worst} carries a lot of the veto, but the conjunction is still doing work.")
    else:
        print(f"  VERDICT: no single oracle dominates; the conjunction is what produces the number.")

    print(f"\n=== agreement structure over all screened candidates ===")
    tot = sum(len(r["cands"]) for r in R)
    for o in ORACLES:
        acc = sum(1 for r in R for c in r["cands"] if c.get(o) is True)
        na = sum(1 for r in R for c in r["cands"] if c.get(o) is None)
        print(f"  {o:<12} accepts {acc:>5}/{tot} ({100*acc/max(tot,1):>4.1f}%)   failed to run {na}")
    unan = sum(1 for r in R for c in r["cands"] if all(c.get(k) is True for k in ORACLES))
    anyone = sum(1 for r in R for c in r["cands"] if any(c.get(k) is True for k in ORACLES))
    print(f"  unanimous candidates {unan}/{tot} ({100*unan/max(tot,1):.1f}%); "
          f"accepted by at least one {anyone}/{tot} ({100*anyone/max(tot,1):.1f}%)")

    json.dump({"n_targets": len(R), "subset_deliverable": rows_out, "drops": drops,
               "full_panel": len(full), "largest_drop_ratio": ratio,
               "n_candidates": tot, "unanimous_candidates": unan},
              open(os.path.join(HERE, "loo_oracle.json"), "w"), indent=1)
    print(f"\n  wrote rna_harness/runs/ceiling/loo_oracle.json")


if __name__ == "__main__":
    main()
