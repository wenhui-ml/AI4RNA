#!/usr/bin/env python3
"""How many crossing targets can we actually HAND a user a design for? Correcting a number of my own.

I have been quoting "26/125 panel-unanimous" as the count of oracle-robust designs the pool+conjunction
mechanism produces. Building certify() showed that number is an upper bound, not a count. The archived panel
column (runs/axis1_oracle_panel) evaluates forms_T PER ORACLE over the candidate pool, so a target is scored
"unanimous" when pKiss accepts candidate A, ProbKnot accepts candidate B and ShapeKnots accepts candidate C.
Three different sequences. You cannot hand that to anyone.

A DELIVERABLE design is ONE sequence that satisfies all three independent predictors. That is what certify()
requires and it is the only thing a certificate can be issued for. Spot check on 6 archived "unanimous"
targets found 4 with such a sequence and 2 without, so the correction is real and worth measuring properly.

This scans the archived candidate pools -- no new search, the sequences were stored -- and counts, per target,
whether any single candidate is unanimous. Reported against the archived per-oracle number on the same split.

Short-circuit order is cost-ordered by length, from the fitted curves in runs/ceiling/oracle_cost.py:
pKiss 146 ms at n=30 exponent 0.14, ProbKnot 95 ms exponent 0.26, ShapeKnots 60 ms exponent 1.90. ShapeKnots
is cheapest only below about 45 nt and the most expensive above it, so a fixed order is wrong at some length.
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
OUT = os.path.join(HERE, "deliverable")


def _order(n):
    from rna_harness.oracles_pk import pkiss_pairs, probknot_pairs, shapeknots_pairs
    ms = lambda a, b: a * (max(n, 1) / 30.0) ** b
    return tuple(x[1:] for x in sorted(
        ((ms(59.9, 1.90), "shapeknots", shapeknots_pairs),
         (ms(94.7, 0.26), "probknot", probknot_pairs),
         (ms(146.0, 0.14), "pkiss", pkiss_pairs))))


def work(item):
    tid, target, pool = item
    dest = os.path.join(OUT, f"t{tid}.json")
    if os.path.exists(dest):
        return dest
    from rna_harness.oracles_pk import forms_target
    ORDER = _order(len(target))
    t0 = time.time()
    win, n_scr, per_oracle_any, first_fail = None, 0, {k: False for k, _ in ORDER}, {}
    ref_only = None
    for c in pool:
        n_scr += 1
        verdicts = {}
        ok = True
        for k, fn in ORDER:
            v = forms_target(fn(c), target)
            verdicts[k] = v
            if v is True:
                per_oracle_any[k] = True
            if v is not True:
                ok = False
                first_fail[k] = first_fail.get(k, 0) + 1
                break
        if ok:
            win = c
            break
        # a candidate the REFERENCE oracle alone accepts is what a single-oracle tool would have returned
        if ref_only is None and verdicts.get("pkiss") is True:
            ref_only = c
    rec = {"id": tid, "n": len(target), "n_pool": len(pool), "n_screened": n_scr,
           "deliverable": win is not None, "seq": win, "ref_only_seq": ref_only,
           "any_oracle_accepts": per_oracle_any, "rejected_first_by": first_fail,
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
    panel = {int(r["task_id"].split(":")[1]): r
             for r in json.load(open(os.path.join(AGENT, "desirna_panel_pool.json")))["rows"]}
    held = sorted(i for i in rows if i % 2 == 1)                 # the same parity split used all along

    items = []
    for i in held:
        t = rows[i]["target"]
        pool = [rows[i].get("best_pk_seq")] + list(rows[i].get("candidates") or [])
        seen, uniq = set(), []
        for c in pool:
            if c and len(c) == len(t) and c not in seen:
                seen.add(c)
                uniq.append(c)
        items.append((i, t, uniq[:a.max_cands]))

    arch = sum(1 for i in held
               if panel.get(i, {}).get("forms_T_pkiss") and panel[i].get("forms_T_probknot")
               and panel[i].get("forms_T_shapeknots"))
    print(f"deliverable designs on the {len(held)} held-out crossing targets")
    print(f"  archived per-ORACLE 'panel-unanimous' on this split: {arch}/{len(held)}  <- the number I quoted")
    print(f"  measuring: does ANY SINGLE candidate satisfy all three?  "
          f"(pools capped at {a.max_cands})\n", flush=True)

    if not a.analyse_only:
        todo = [it for it in items if not os.path.exists(os.path.join(OUT, f"t{it[0]}.json"))]
        print(f"  {len(todo)} targets to scan, {a.workers} workers", flush=True)
        from concurrent.futures import ProcessPoolExecutor, as_completed
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(work, it) for it in todo]
            for n, _f in enumerate(as_completed(futs), 1):
                if n % 20 == 0 or n == len(todo):
                    el = time.time() - t0
                    print(f"    {n}/{len(todo)}  {el/60:.1f} min  "
                          f"eta {el/max(n,1)*(len(todo)-n)/60:.1f} min", flush=True)

    recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, "*.json")))]
    recs = [r for r in recs if r["id"] in set(held)]
    dl = [r for r in recs if r["deliverable"]]
    ro = [r for r in recs if not r["deliverable"] and r.get("ref_only_seq")]
    print(f"\n=== the number that can actually be handed to a user ===")
    print(f"  DELIVERABLE (one sequence, all three oracles): {len(dl)}/{len(recs)}")
    print(f"  archived per-oracle upper bound on this split: {arch}/{len(held)}")
    if arch:
        print(f"  -> the quoted figure was {arch/max(len(dl),1):.2f}x the deliverable count")
    print(f"  would-be FALSE certificates (reference oracle alone accepts, panel does not): "
          f"{len(ro)}/{len(recs)}")
    print(f"  neither: {len(recs)-len(dl)-len(ro)}/{len(recs)}")

    print(f"\n=== which oracle does the rejecting (first-fail counts over all screened candidates) ===")
    agg = {}
    for r in recs:
        for k, v in (r.get("rejected_first_by") or {}).items():
            agg[k] = agg.get(k, 0) + v
    tot = sum(agg.values()) or 1
    for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<12} rejected first on {v:>6} candidate screenings  ({100*v/tot:.1f}%)")

    print(f"\n=== by length ===")
    for lo, hi, lbl in ((0, 40, "<=40nt"), (41, 70, "41-70"), (71, 10 ** 9, ">70")):
        sub = [r for r in recs if lo <= r["n"] <= hi]
        if sub:
            print(f"  {lbl:>7}: deliverable {sum(1 for r in sub if r['deliverable']):>3}/{len(sub):<3}")

    json.dump({"held": held, "archived_per_oracle": arch, "deliverable": len(dl),
               "would_be_false": len(ro), "n": len(recs),
               "rejected_first_by": agg, "per_target": recs},
              open(os.path.join(HERE, "deliverable.json"), "w"), indent=1)
    print(f"\n  wrote rna_harness/runs/ceiling/deliverable.json")


if __name__ == "__main__":
    main()
