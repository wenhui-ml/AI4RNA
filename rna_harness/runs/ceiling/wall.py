#!/usr/bin/env python3
"""Are the 16 targets I called "hopeless" a DESIGNABILITY wall or merely a BUDGET wall?

The label was earned at t<=180s only, and a lot rests on it: the abstention experiment, the 2.148x prize, the
claim that 67% of the cascade bill goes to targets that can never solve -- and, in the product, the sentence
`not_found` says to a user ("this tier could not solve it; a larger budget is available"). If those targets are
a designability wall, that sentence sells the user a budget they cannot spend usefully, which is a
misleading answer rather than an honest one.

The published curve disagrees with my label: Wirecki et al. report 95/100 at 1 h per target and 97/100 at 24 h,
so most of a 16-target residual at 180 s should reappear as solvable given more time.

Extends the ladder to t=600 and t=1800 on exactly those 16 (cells already computed by
sota_curve.py --budgets 600,1800 --only-unsolved).

DECLARED BEFORE THE RUN, in sota_curve.py's --only-unsolved help text:
    >= 8 of 16 solving by t<=1800s  ->  BUDGET wall
    <= 3 of 16                      ->  DESIGNABILITY wall
    in between                      ->  inconclusive, reported as such, not rounded to either side
"""
from __future__ import annotations

import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
CELLS = os.path.join(HERE, "sota_curve")
HOPELESS = [60, 68, 71, 73, 76, 79, 81, 86, 87, 90, 91, 94, 96, 97, 99, 100]


def main():
    by = collections.defaultdict(dict)
    for f in glob.glob(os.path.join(CELLS, "*.json")):
        d = json.load(open(f))
        by[d["t"]][d["id"]] = d
    new = [t for t in (600, 1800) if by.get(t)]
    if not new:
        raise SystemExit("no t=600/1800 cells found; run sota_curve.py --budgets 600,1800 --only-unsolved")
    print(f"the {len(HOPELESS)} targets unsolved at t<=180s, extended to {new}\n")
    print(f"  {'id':>4} {'n':>5} {'bp@180':>7} " + "".join(f"{'bp@' + str(t):>9}" for t in new) + "   solved")
    solved = {t: [] for t in new}
    for i in HOPELESS:
        row = f"  {i:>4} {by[180][i]['n']:>5} {by[180][i]['bp']:>7} "
        tags = []
        for t in new:
            c = by[t].get(i)
            row += f"{(c['bp'] if c else '-'):>9}"
            if c and c.get("solved"):
                solved[t].append(i)
                tags.append(f"t={t}")
        print(row + "   " + ",".join(tags))

    union = sorted(set().union(*[set(v) for v in solved.values()]))
    print()
    for t in new:
        print(f"  t={t:<5} solved {len(solved[t]):>2}/16  {solved[t]}")
    print(f"  union   solved {len(union):>2}/16  {union}")

    n = len(union)
    verdict = "BUDGET WALL" if n >= 8 else ("DESIGNABILITY WALL" if n <= 3 else "INCONCLUSIVE")
    print(f"\n=== declared rule: >=8 budget | <=3 designability | else inconclusive ===")
    print(f"  measured {n}/16  ->  {verdict}")

    spend = {t: sum(by[t][i].get("cost_usd", 0.0) for i in by[t]) for t in new}
    print(f"  spend: " + " + ".join(f"t={t} ${spend[t]:.3f}" for t in new)
          + f" = ${sum(spend.values()):.3f}")

    print(f"\n=== what this means for what the product TELLS a user ===")
    if verdict == "BUDGET WALL":
        print(f"  `not_found` may honestly offer a larger budget: {n} of 16 do become solvable.")
    elif verdict == "DESIGNABILITY WALL":
        print(f"  `not_found` must NOT offer a larger budget as if it would help -- only {n} of 16 become")
        print(f"  solvable even at t=1800s. Selling more compute here would be misleading.")
    else:
        print(f"  {n} of 16 become solvable. Neither 'just pay more' nor 'give up' is supported; the honest")
        print(f"  wording is that more budget helps SOME targets and the caller cannot be told which.")

    if union:
        c = {t: sum(by[t][i].get("cost_usd", 0.0) for i in by[t]) / max(len(by[t]), 1) for t in by}
        print(f"\n  marginal price of these {n} solves: "
              + ", ".join(f"t={t} ${c[t]:.4f}/target" for t in new))
        got600 = len(solved.get(600, []))
        if got600:
            print(f"  t=600 alone: +{got600} solves at ${16*c[600]:.4f} for the 16 "
                  f"= ${16*c[600]/got600:.4f}/solve")
        print(f"  published slope in this region (85->95, $0.83->$50) is about $4.92/solve")

    json.dump({"hopeless": HOPELESS, "new_tiers": new,
               "solved": {str(t): solved[t] for t in new}, "union": union,
               "n_union": n, "verdict": verdict, "spend": {str(t): spend[t] for t in new}},
              open(os.path.join(HERE, "wall.json"), "w"), indent=1)
    print(f"\n  wrote rna_harness/runs/ceiling/wall.json")


if __name__ == "__main__":
    main()
