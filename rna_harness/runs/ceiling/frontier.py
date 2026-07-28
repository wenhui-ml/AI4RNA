#!/usr/bin/env python3
"""Three frontiers on the measured DesiRNA curve: FLAT, CASCADE, ORACLE. Free -- pure re-analysis of
sota_curve's 400 cells, no solver calls.

WHY THIS RUNS BEFORE ANY POLICY IS BUILT. Every efficiency claim on this branch divides by a FLAT baseline
(run every target at one budget). No scheduler would do that: the obvious zero-knowledge policy is a CASCADE
-- run the cheap tier on everything, escalate only what is still unsolved. Whatever the cascade saves is free
and needs no learning, so a learned policy that beats FLAT by less than the cascade does has shown nothing.
Earlier today a bp-ranking heuristic was withdrawn for exactly this reason: random ranking matched it in the
cheap region where the saving lived.

    FLAT(t)   every target at budget t
    CASCADE   t ascending; a target that solves is retired; the unsolved pay the next tier too
    ORACLE    each target pays ONLY its minimum solving tier; targets that never solve pay the cheapest
              probe once. Requires outcome knowledge -> an UPPER BOUND on any policy, not a policy.

ORACLE / CASCADE is therefore the entire prize available to routing, abstention, or any learned schedule on
this benchmark. Reporting it first means we know the size of the prize before deciding whether to build the
machine that chases it.

The script also decomposes the cascade bill by target class, because where the money goes determines what a
policy would have to predict.
"""
from __future__ import annotations

import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
CELLS = os.path.join(HERE, "sota_curve")


def main():
    by = collections.defaultdict(dict)
    for f in glob.glob(os.path.join(CELLS, "*.json")):
        d = json.load(open(f))
        by[d["t"]][d["id"]] = d
    budgets = sorted(by)
    ids = sorted(by[budgets[0]])
    full = [t for t in budgets if len(by[t]) == len(ids)]
    if full != budgets:
        print(f"  NOTE: incomplete budgets excluded: {sorted(set(budgets) - set(full))}")
        budgets = full
    cost = {t: sum(by[t][i].get("cost_usd", 0.0) for i in ids) / len(ids) for t in budgets}
    print(f"DesiRNA on Eterna100-V2, {len(ids)} targets, exact `match`; measured $/target per tier:")
    print("  " + "  ".join(f"t={t}s ${cost[t]:.5f}" for t in budgets) + "\n")

    print(f"  {'frontier':<28} {'solved':>7} {'$ total':>9} {'vs FLAT(max t)':>15}")
    flat_top = cost[budgets[-1]] * len(ids)
    for t in budgets:
        s = sum(1 for i in ids if by[t][i].get("solved"))
        print(f"  {'FLAT t=' + str(t) + 's':<28} {s:>7} {cost[t]*len(ids):>9.3f} "
              f"{flat_top/(cost[t]*len(ids)):>14.2f}x")

    unsolved, casc = set(ids), 0.0
    for t in budgets:
        casc += cost[t] * len(unsolved)
        unsolved -= {i for i in unsolved if by[t][i].get("solved")}
    solved = len(ids) - len(unsolved)
    print(f"  {'CASCADE (zero knowledge)':<28} {solved:>7} {casc:>9.3f} {flat_top/casc:>14.2f}x")

    need = {}
    for i in ids:
        for t in budgets:
            if by[t][i].get("solved"):
                need[i] = t
                break
    orac = sum(cost[need[i]] for i in need) + cost[budgets[0]] * (len(ids) - len(need))
    print(f"  {'ORACLE (upper bound)':<28} {len(need):>7} {orac:>9.3f} {flat_top/orac:>14.2f}x")

    print("\n=== the prize ===")
    print(f"  FLAT(t={budgets[-1]}s) -> CASCADE : {flat_top/casc:.2f}x   FREE, no learning, any scheduler")
    print(f"  CASCADE     -> ORACLE  : {casc/orac:.2f}x   the ENTIRE value of routing/abstention/learning")
    print(f"  a learned policy that beats FLAT by less than {flat_top/casc:.2f}x has shown nothing")

    print("\n=== minimum solving tier (what the oracle knows) ===")
    d = collections.Counter(need.values())
    for t in budgets:
        print(f"  t={t:>4}s solves it first : {d.get(t, 0):>3}")
    hopeless = [i for i in ids if i not in need]
    print(f"  never solves at t<={budgets[-1]}s : {len(hopeless):>3}")

    print("\n=== where the cascade's money actually goes ===")
    per_target_full = sum(cost[t] for t in budgets)
    hope_cost = len(hopeless) * per_target_full
    print(f"  the {len(hopeless)} never-solving targets pay EVERY tier: "
          f"{len(hopeless)} x ${per_target_full:.5f} = ${hope_cost:.3f}")
    print(f"  that is {100.0*hope_cost/casc:.0f}% of the ${casc:.3f} cascade bill")
    print(f"  -> the prize is not smarter search, it is ABSTAINING on the {len(hopeless)} hopeless targets")

    nlen = {i: by[budgets[0]][i]["n"] for i in ids}
    print(f"\n=== the {len(hopeless)} hopeless targets by length ===")
    for lo, hi, lbl in ((0, 100, "<=100nt"), (101, 250, "101-250"), (251, 10 ** 9, ">250")):
        sub = [i for i in hopeless if lo <= nlen[i] <= hi]
        allsub = [i for i in ids if lo <= nlen[i] <= hi]
        print(f"  {lbl:>8}: {len(sub):>3} of {len(allsub):>3} targets")

    json.dump({"budgets": budgets, "cost_per_target": cost, "n": len(ids),
               "flat": {t: sum(1 for i in ids if by[t][i].get("solved")) for t in budgets},
               "flat_cost": {t: cost[t] * len(ids) for t in budgets},
               "cascade": {"solved": solved, "cost": casc},
               "oracle": {"solved": len(need), "cost": orac},
               "free_gain": flat_top / casc, "learning_prize": casc / orac,
               "min_tier_hist": {str(t): d.get(t, 0) for t in budgets},
               "hopeless": hopeless, "hopeless_share_of_cascade": hope_cost / casc,
               "need": {str(k): v for k, v in need.items()}},
              open(os.path.join(HERE, "frontier.json"), "w"), indent=1)
    print(f"\n  wrote {os.path.relpath(os.path.join(HERE, 'frontier.json'), REPO)}")


if __name__ == "__main__":
    main()
