#!/usr/bin/env python3
"""Where the remaining money on the S(B) plane actually goes: DesiRNA's FIXED per-invocation cost.

Measured directly (n=400, -t 5): the DesiRNA subprocess took 40.1 s while our own _verify_best took 3.2 s,
i.e. 7% -- and 200 ViennaRNA folds at 400 nt cost only 1.7 s, so candidate verification is not the bottleneck.
The overhead is DesiRNA's own start-up, and it is INDEPENDENT of -t. A cascade pays it ONCE PER RUNG.

That reframes the efficiency lever. Early-stopping inside a run reclaims UNUSED SEARCH time; it cannot
reclaim start-up. On a long target where start-up is ~35 s, a t=5 rung spends 7x more on starting than on
searching, so the cheap rungs are nearly pure waste there.

This recovers start-up per target from the four measured tiers -- wall_s = startup(n) + t, so
startup_i = mean_t (wall_s(i,t) - t) -- then simulates LENGTH-ADAPTIVE ladders against the flat one on the
per-target outcomes already measured. No new solver calls.
"""
from __future__ import annotations

import collections
import glob
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
USD_PER_CORE_S = 0.05 / 3600.0


def main():
    by = collections.defaultdict(dict)
    for f in glob.glob(os.path.join(HERE, "sota_curve", "*.json")):
        d = json.load(open(f))
        by[d["t"]][d["id"]] = d
    B = sorted(by)
    ids = sorted(by[B[0]])
    REPL = by[B[0]][ids[0]]["replicas"]

    start = {i: st.median([by[t][i]["wall_s"] - t for t in B]) for i in ids}
    n_of = {i: by[B[0]][i]["n"] for i in ids}
    print(f"per-invocation START-UP recovered from wall_s = startup(n) + t, {len(B)} tiers per target")
    print(f"  {'length band':<12} {'n':>4} {'startup s (median)':>19} {'$ per invocation':>17} "
          f"{'= how many t=5 searches':>24}")
    bands = ((0, 100, "<=100nt"), (101, 250, "101-250"), (251, 10 ** 9, ">250"))
    for lo, hi, lbl in bands:
        sub = [i for i in ids if lo <= n_of[i] <= hi]
        s = st.median([start[i] for i in sub])
        print(f"  {lbl:<12} {len(sub):>4} {s:>19.1f} {s*REPL*USD_PER_CORE_S:>17.5f} "
              f"{s/5.0:>23.1f}x")

    def ladder_cost(rungs_for):
        """Cost and solves when each target runs its own rung list, stopping at the first solve.
        Per-rung cost is the MEASURED cost_usd of that (target, tier) cell."""
        cost = solved = 0
        cost = 0.0
        for i in ids:
            for t in rungs_for(n_of[i]):
                cost += by[t][i]["cost_usd"]
                if by[t][i].get("solved"):
                    solved += 1
                    break
        return cost, solved

    FULL = [5, 15, 60, 180]
    LADDERS = {
        "flat [5,15,60,180] (shipped)": lambda n: FULL,
        "skip t=5 above 250nt": lambda n: ([15, 60, 180] if n > 250 else FULL),
        "only [60,180] above 250nt": lambda n: ([60, 180] if n > 250 else FULL),
        "only [180] above 250nt": lambda n: ([180] if n > 250 else FULL),
        "only [60,180] above 150nt": lambda n: ([60, 180] if n > 150 else FULL),
        "two rungs everywhere [5,180]": lambda n: [5, 180],
    }
    print(f"\n  {'ladder':<32} {'solved':>7} {'$ total':>9} {'vs shipped':>11} {'$/solve':>9}")
    base = None
    out = {}
    for name, fn in LADDERS.items():
        c, s = ladder_cost(fn)
        if base is None:
            base = c
        print(f"  {name:<32} {s:>7} {c:>9.4f} {base/c:>10.3f}x {c/max(s,1):>9.5f}")
        out[name] = {"solved": s, "cost": c, "vs_shipped": base / c}

    print(f"\n=== how much of the shipped bill is start-up rather than search ===")
    tot = fired = 0.0
    n_inv = collections.Counter()
    for i in ids:
        for t in FULL:
            n_inv[i] += 1
            tot += by[t][i]["cost_usd"]
            if by[t][i].get("solved"):
                break
    su = sum(n_inv[i] * start[i] * REPL * USD_PER_CORE_S for i in ids)
    print(f"  total invocations fired: {sum(n_inv.values())}")
    print(f"  start-up share of the ${tot:.4f} bill: ${su:.4f} = {100*su/tot:.0f}%")
    for lo, hi, lbl in bands:
        sub = [i for i in ids if lo <= n_of[i] <= hi]
        c = sum(by[t][i]["cost_usd"] for i in sub for t in FULL
                if True) and 0.0
        cc = 0.0
        for i in sub:
            for t in FULL:
                cc += by[t][i]["cost_usd"]
                if by[t][i].get("solved"):
                    break
        s2 = sum(n_inv[i] * start[i] * REPL * USD_PER_CORE_S for i in sub)
        print(f"  {lbl:>8}: ${cc:.4f} of which ${s2:.4f} start-up = {100*s2/max(cc,1e-9):.0f}%")

    # Exact upper bound on early-stop. It can only reclaim search time paid AFTER a solution exists, so:
    #   * nothing on a target that never solves -- there is no moment to stop at;
    #   * nothing on an EARLIER tier of a target that escalates -- that search genuinely found nothing;
    #   * at most the FULL search budget of the tier that solves (the limit where the solution appears at t=0).
    solving_tier = {}
    for i in ids:
        for t in FULL:
            if by[t][i].get("solved"):
                solving_tier[i] = t
                break
    reclaim = sum(solving_tier[i] * REPL * USD_PER_CORE_S for i in solving_tier)
    hopeless = [i for i in ids if i not in solving_tier]
    dead = sum(sum(FULL) * REPL * USD_PER_CORE_S for _ in hopeless)
    print(f"\n=== the three parts of the shipped ${tot:.4f} bill ===")
    print(f"  start-up (invocation count x startup(n))   ${su:.4f}  {100*su/tot:>4.0f}%  "
          f"early-stop CANNOT touch this")
    print(f"  search on the {len(hopeless)} targets that never solve   ${dead:.4f}  {100*dead/tot:>4.0f}%  "
          f"early-stop CANNOT touch this -- nothing to stop at")
    print(f"  search on the {len(solving_tier)} targets that do solve      ${reclaim:.4f}  "
          f"{100*reclaim/tot:>4.0f}%  the ONLY part early-stop can reclaim")
    print(f"\n  perfect early-stop ceiling: ${tot:.4f} -> ${tot-reclaim:.4f} = "
          f"{tot/(tot-reclaim):.3f}x")
    print(f"  best ladder reshaping (measured above)    = {max(v['vs_shipped'] for v in out.values()):.3f}x")
    print(f"  abstention on the {len(hopeless)} hopeless targets      = "
          f"{tot/(tot-dead):.3f}x   <- the only lever with real size")
    print(f"  all knowledge at once (frontier.py ORACLE) = 3.357x")

    json.dump({"startup_s": start, "n": n_of, "replicas": REPL, "ladders": out,
               "shipped_cost": tot, "startup_cost": su, "startup_share": su / tot},
              open(os.path.join(HERE, "startup.json"), "w"), indent=1)
    print(f"\n  wrote rna_harness/runs/ceiling/startup.json")


if __name__ == "__main__":
    main()
