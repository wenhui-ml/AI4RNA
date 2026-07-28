#!/usr/bin/env python3
"""Which rungs should the product's cascade actually contain? Simulated on measured per-target outcomes.
Zero new solver calls.

WHY THIS IS THE ENGINEERING QUESTION. cognitive/core.py's solve loop is ALREADY a cascade: it fires the
router's plan in order and returns on the first oracle-verified solve. So nothing needs inventing. What decides
the product's cost and capability is the LADDER -- which (tool, budget) rungs are in default_toolbox() and in
what order. Today's measurement makes that decidable for the first time, because we now have every rung's
per-target outcome on all 100 Eterna targets:

    rung            solved   $/target      source
    desirna@5s       72/100   0.00211      runs/ceiling/sota_curve (new today)
    remc@5s          58/100   0.00271      runs/learning_vs_cold/table.json (archived)
    desirna@15s      77/100   0.00341      new today
    remc@30s         59/100   0.00371      archived
    remc@120s        62/100   0.00654      archived
    desirna@60s      81/100   0.00986      new today
    desirna@180s     84/100   0.02670      new today

All three RemcTool rungs are STRICTLY DOMINATED by desirna@5s: more expensive per target AND fewer solves.
They currently sit at the HEAD of every cascade because router._cold_order sorts by nominal_cost_usd() =
cores x budget_s, which for remc@5s (8 x 5) is below desirna@5s (10 x 5). The nominal ordering is wrong
because RemcTool overruns its wall budget on long targets -- measured here: declared 5 s, actual 34.6 s at
n=400 (6.9x), 8.1 s at n=108, 0.03 s at n=67. replica_exchange only checks the clock between rounds, and one
round on a 400 nt target takes ~35 s. So the cheap head of the cascade is most expensive exactly where it is
most useless.

Seeds differ between sources (archived desirna@60s/180s = 80/83, today's = 81/84). The remc columns are
archived, the desirna columns are today's. The comparison is therefore not single-seed-clean, and the
conclusion does not depend on 1-2 targets either way.
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
ARCH = os.path.join(HERE, "audit", "lvc_table.json")


def load():
    rows = collections.defaultdict(dict)
    if os.path.exists(ARCH):
        for k, v in json.load(open(ARCH)).items():
            pid, tool = k.split(":", 1)
            rows[tool][int(pid)] = (bool(v["solved"]), float(v["cost"]))
    for f in glob.glob(os.path.join(HERE, "sota_curve", "*.json")):
        x = json.load(open(f))
        rows[f"desirna@{x['t']}s"][x["id"]] = (bool(x.get("solved")), float(x.get("cost_usd", 0.0)))
    return rows


def run_ladder(rungs, rows, ids):
    """Fire rungs in order per target, pay each, stop at the first verified solve (core.solve's semantics)."""
    cost = solved = 0.0, 0
    cost, solved, first = 0.0, 0, collections.Counter()
    for i in ids:
        for r in rungs:
            ok, c = rows[r][i]
            cost += c
            if ok:
                solved += 1
                first[r] += 1
                break
    return cost, solved, first


def main():
    rows = load()
    have = sorted(rows, key=lambda t: sum(c for _, c in rows[t].values()) / len(rows[t]))
    ids = sorted(set.intersection(*[set(rows[t]) for t in have]))
    print(f"rungs available: {have}\n  {len(ids)} targets with every rung measured\n")

    LADDERS = {
        "CURRENT default_toolbox": ["remc@5s", "remc@30s", "remc@120s", "desirna@60s", "desirna@180s"],
        "drop the dominated remc rungs": ["desirna@60s", "desirna@180s"],
        "add the cheap desirna rungs": ["desirna@5s", "desirna@15s", "desirna@60s", "desirna@180s"],
        "cheap desirna only": ["desirna@5s", "desirna@15s"],
        "desirna@5s alone": ["desirna@5s"],
        "always-SOTA (desirna@180s)": ["desirna@180s"],
        "keep remc AFTER cheap desirna": ["desirna@5s", "desirna@15s", "remc@120s",
                                          "desirna@60s", "desirna@180s"],
    }
    print(f"  {'ladder':<32} {'solved':>7} {'$ total':>9} {'$/solve':>9} {'vs CURRENT':>11}")
    base = None
    out = {}
    for name, rungs in LADDERS.items():
        if any(r not in rows for r in rungs):
            continue
        c, s, first = run_ladder(rungs, rows, ids)
        if base is None:
            base = (c, s)
        print(f"  {name:<32} {s:>7} {c:>9.4f} {c/max(s,1):>9.5f} "
              f"{(base[0]/c if c else float('nan')):>10.2f}x")
        out[name] = {"rungs": rungs, "solved": s, "cost": c, "cost_per_solve": c / max(s, 1),
                     "first_solver": dict(first)}

    cur, new = out["CURRENT default_toolbox"], out["add the cheap desirna rungs"]
    print(f"\n=== the change worth making ===")
    print(f"  CURRENT : {cur['solved']}/{len(ids)} at ${cur['cost']:.4f}   "
          f"first solver = {cur['first_solver']}")
    print(f"  PROPOSED: {new['solved']}/{len(ids)} at ${new['cost']:.4f}   "
          f"first solver = {new['first_solver']}")
    print(f"  -> {new['solved'] - cur['solved']:+d} solves and "
          f"{cur['cost']/new['cost']:.2f}x cheaper, from reordering the ladder alone")
    print(f"  the three remc rungs currently solve {sum(v for k, v in cur['first_solver'].items() if 'remc' in k)}"
          f" targets first, and every target pays all three before reaching DesiRNA")

    json.dump({"n": len(ids), "rung_summary": {t: {"solved": sum(1 for s, _ in rows[t].values() if s),
                                                   "cost_per_target": sum(c for _, c in rows[t].values()) / len(rows[t])}
                                               for t in have},
               "ladders": out}, open(os.path.join(HERE, "ladder.json"), "w"), indent=1)
    print(f"\n  wrote rna_harness/runs/ceiling/ladder.json")


if __name__ == "__main__":
    main()
