#!/usr/bin/env python3
"""The 16 Eterna100 targets no budget tier ever solved, attacked by decomposition instead of by money.

PAIRED, SAME CODE PATH, SAME ORACLE. Arm A is the shipped cascade on the whole target. Arm B decomposes,
designs each unit on its own, PINS only the units that verifiably fold, and hands the seams to the same
constrained solver. Both arms report their real wall clock, because arm B pays for a unit stage that arm A
does not and a win bought with extra seconds is not a win.

The control is re-run rather than quoted. These targets are known to fail at <= 180 s -- that is why they are
here -- but a number lifted from an older run measured an older code path.

DECLARED BEFORE THE RUN: 2 of the 16 (60, 71) decompose into a single unit, so arm B cannot differ from arm A
on them by construction; they are reported and excluded from the paired comparison rather than silently
dropped. Teslagon (100) has 0 of 9 units individually designable, so arm B has nothing to pin -- it is kept,
because an honest negative is part of the measurement.
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
OUT = os.path.join(HERE, "extreme_bench")
HOPELESS = [60, 68, 71, 73, 76, 79, 81, 86, 87, 90, 91, 94, 96, 97, 99, 100]


def work(job):
    tid, arm, budget = job
    dest = os.path.join(OUT, f"{tid}-{arm}.json")
    if os.path.exists(dest):
        return dest
    from rna_harness.benchmarks.eterna100 import load_eterna100
    row = {int(r["id"]): r for r in load_eterna100()}[tid]
    t0 = time.time()
    try:
        if arm == "whole":
            from rna_harness.cognitive.certify import certify_design
            c = certify_design(row["target"], budget=budget)
            rec = {"verdict": c.verdict, "sequence": c.sequence, "cost_usd": c.cost_usd}
        else:
            from rna_harness.cognitive.parts import design_by_parts
            o = design_by_parts(row["target"], repair_budget=budget)
            rec = {"verdict": o.get("verdict") or ("error" if not o.get("ok") else "?"),
                   "sequence": o.get("sequence"), "cost_usd": o.get("cost_usd") or 0.0,
                   "n_units": o.get("n_units"), "units_solved": o.get("units_solved"),
                   "seed_coverage": o.get("seed_coverage"), "bp_away": o.get("bp_away"),
                   "constraint_held": o.get("constraint_held"),
                   "note": o.get("note") or o.get("error")}
    except Exception as e:                                                # noqa: BLE001
        # One target that raises must not take the other 31 jobs with it: the pool re-raises from
        # f.result() and the whole comparison is lost, including the arms that had already finished.
        rec = {"verdict": "error", "sequence": None, "cost_usd": 0.0, "note": repr(e)[:200]}
    rec.update({"id": tid, "arm": arm, "name": row.get("name"), "n": len(row["target"]),
                "solved": rec["verdict"] in ("certified", "knot_confirmed"),
                "wall_s": round(time.time() - t0, 1)})
    with open(dest, "w") as f:
        json.dump(rec, f, ensure_ascii=False)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", default="standard")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--analyse-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    jobs = [(t, arm, a.budget) for t in HOPELESS for arm in ("whole", "parts")]
    print(f"extreme benchmark: {len(HOPELESS)} targets x 2 arms at budget={a.budget}", flush=True)
    if not a.analyse_only:
        todo = [j for j in jobs if not os.path.exists(os.path.join(OUT, f"{j[0]}-{j[1]}.json"))]
        print(f"  {len(todo)} to run", flush=True)
        from concurrent.futures import ProcessPoolExecutor, as_completed
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for n, f in enumerate(as_completed([ex.submit(work, j) for j in todo]), 1):
                f.result()
                print(f"    {n}/{len(todo)}  {(time.time()-t0)/60:.1f} min", flush=True)

    R = {}
    for f in sorted(glob.glob(os.path.join(OUT, "*.json"))):
        r = json.load(open(f))
        R.setdefault(r["id"], {})[r["arm"]] = r
    single = [t for t in R if (R[t].get("parts") or {}).get("n_units", 9) < 2]
    paired = [t for t in sorted(R) if t not in single and "whole" in R[t] and "parts" in R[t]]

    print(f"\n=== paired, {len(paired)} decomposable targets "
          f"({len(single)} excluded: a single unit, so the arms cannot differ) ===")
    print(f"  {'id':>4} {'n':>4} {'units':>7} {'cover':>6} {'whole':>7} {'parts':>7} "
          f"{'bp':>4} {'whole s':>8} {'parts s':>8}  name")
    w_ok = p_ok = 0
    for t in paired:
        w, p = R[t]["whole"], R[t]["parts"]
        w_ok += bool(w["solved"])
        p_ok += bool(p["solved"])
        mark = "  <-- RECOVERED" if p["solved"] and not w["solved"] else (
            "  <-- LOST" if w["solved"] and not p["solved"] else "")
        print(f"  {t:>4} {w['n']:>4} {str(p.get('units_solved')) + '/' + str(p.get('n_units')):>7} "
              f"{(p.get('seed_coverage') or 0):>6.0%} {str(w['solved']):>7} {str(p['solved']):>7} "
              f"{str(p.get('bp_away')):>4} {w['wall_s']:>8.0f} {p['wall_s']:>8.0f}  "
              f"{str(p.get('name'))[:26]}{mark}")
    print(f"\n  whole-target cascade   {w_ok}/{len(paired)}   "
          f"{sum(R[t]['whole']['wall_s'] for t in paired)/60:.0f} min total")
    print(f"  decompose + repair     {p_ok}/{len(paired)}   "
          f"{sum(R[t]['parts']['wall_s'] for t in paired)/60:.0f} min total")
    for t in single:
        print(f"\n  excluded {t}: {(R[t].get('parts') or {}).get('note') or 'single unit'}")
    json.dump({"budget": a.budget, "paired": paired, "excluded_single_unit": single,
               "whole_solved": w_ok, "parts_solved": p_ok, "rows": R},
              open(os.path.join(HERE, "extreme_bench.json"), "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote rna_harness/runs/ceiling/extreme_bench.json")


if __name__ == "__main__":
    main()
