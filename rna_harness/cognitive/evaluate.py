"""Cost-at-quality evaluation: the cognitive core vs fixed-policy baselines on a puzzle set.

Records per-puzzle (solved, cost_usd, n_actions) for each policy and aggregates solve-count + total
cost. The falsifiable claim: the cognitive core reaches EQUAL solve quality at LOWER total cost($)
than always-SOTA, because it solves the easy majority with cheap tools and pays for DesiRNA only when
needed. The cognitive router learns across the stream, so its later-puzzle cost should also fall.
Every number is cost-tagged; nothing here touches a nucleotide (domain-agnostic).
"""
from __future__ import annotations

import argparse
import json
import os
import time

from ..benchmarks.eterna100 import load_eterna100
from .core import CognitiveCore, FixedPolicy
from .tools import DesirnaTool, RemcTool, default_toolbox

OUT_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "runs", "cognitive_eval")


def build_policies(which):
    pol = {}
    if "cognitive" in which:
        pol["cognitive"] = CognitiveCore(default_toolbox(), learn=True)   # router auto-built on same tools
    if "always_cheap" in which:
        pol["always_cheap"] = FixedPolicy([RemcTool(wall_s=120)], name="always_cheap")
    if "always_sota" in which:
        pol["always_sota"] = FixedPolicy([DesirnaTool(timelimit_s=180)], name="always_sota")
    return pol


def run(ids, which, outdir, seed=0):
    os.makedirs(outdir, exist_ok=True)
    puz = {p["id"]: p for p in load_eterna100(version="V2")}
    ids = [i for i in ids if i in puz]
    pols = build_policies(which)
    print(f"cognitive-eval on {len(ids)} puzzles; policies={list(pols)}", flush=True)
    rows, agg = [], {name: {"solved": 0, "cost_usd": 0.0} for name in pols}
    for pid in ids:
        t = puz[pid]["target"]
        rec = {"pid": pid, "len": len(t), "by_policy": {}}
        for name, pol in pols.items():
            r = pol.solve(t, seed=seed)
            rec["by_policy"][name] = {"solved": r["solved"], "cost_usd": r["cost_usd"],
                                      "n_actions": r["n_actions"]}
            agg[name]["solved"] += int(r["solved"])
            agg[name]["cost_usd"] += r["cost_usd"]
        rows.append(rec)
        print(f"  pid {pid:>3} n={len(t):>4} | " +
              " ".join(f"{nm}:{v['solved']}/${v['cost_usd']:.2e}" for nm, v in rec["by_policy"].items()),
              flush=True)
        json.dump({"ids": ids, "agg": agg, "rows": rows}, open(os.path.join(outdir, "eval.json"), "w"), indent=1)
    print("\n=== AGGREGATE (solve-count, total $, mean $/solve) ===", flush=True)
    for name, a in agg.items():
        n_solved = a["solved"]
        mean = a["cost_usd"] / n_solved if n_solved else float("nan")
        print(f"  {name:14} solved={n_solved}/{len(ids)}  total_cost=${a['cost_usd']:.3e}  "
              f"mean_$/solve=${mean:.3e}", flush=True)
    print("thesis: cognitive should match always_sota's solved at a LOWER total_cost.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, nargs="+", default=None)
    ap.add_argument("--n", type=int, default=None, help="first N puzzles by id if --ids not given")
    ap.add_argument("--policies", nargs="+", default=["cognitive", "always_cheap", "always_sota"])
    ap.add_argument("--outdir", default=OUT_DEFAULT)
    a = ap.parse_args()
    puz = sorted(p["id"] for p in load_eterna100(version="V2"))
    ids = a.ids if a.ids else (puz[: a.n] if a.n else puz)
    run(ids, a.policies, a.outdir)
