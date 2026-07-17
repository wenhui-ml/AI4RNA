"""Learning vs no-learning: is the LEARNED router doing real work, or is the cognitive core just a
good fixed cheap-first cascade (a "wrapper")?

Controlled design (isolate the ONLY thing learning changes -- tool ORDER):
  1. BUILD a fixed (puzzle x tool) outcome table ONCE. Each tool is run a single time per puzzle and
     its oracle-verified (solved, cost, wall) is recorded. Resumable.
  2. SIMULATE two ordering policies on that SAME table, so tool stochasticity and CPU-contention noise
     are held constant across arms and the only difference is the ORDER tools are fired in:
        cold  : cheapest-first cascade for every puzzle (no learning; a sensible fixed policy)
        learn : Router accumulates (feature -> tool -> solved,cost) across the stream and re-orders by
                expected cost-to-solve, so predicted-hard puzzles skip the doomed cheap tier.
Because plan() only REORDERS the toolbox (never drops a tool), both arms solve EXACTLY the same set --
learning can change only cost. Falsifiable claim: over passes, learn's total cost drops below cold's at
equal solves, and the saving concentrates on the hard puzzles. If the saving is ~0, the core is a
wrapper. Domain-agnostic; every number is cost-tagged.

Caveats recorded honestly: (a) RemcTool's wall bound is soft (checked once per epoch) and the box is
shared, so absolute costs are a loaded-box upper bound -- but the arms share one table, so the DELTA is
robust to both. (b) Tools are stochastic and non-monotonic in budget; the table is a single realization.
"""
from __future__ import annotations

import argparse
import json
import os

from ..benchmarks.eterna100 import load_eterna100
from ..cognitive.features import features as extract_features
from ..cognitive.router import Router
from ..cognitive.tools import default_toolbox

OUT_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "runs", "learning_vs_cold")
# an easy-heavy mix with a real hard tail, continuous with the cogeval run
DEFAULT_IDS = [8, 1, 23, 45, 18, 59, 95, 49, 37, 16, 36, 6, 66, 57, 72, 77, 60, 50]


def _table_path(outdir):
    return os.path.join(outdir, "table.json")


def build_table(ids, outdir, seed=0):
    """Run each tool once per puzzle; record (solved, cost, wall, bp). Resumable: existing keys skipped."""
    os.makedirs(outdir, exist_ok=True)
    puz = {p["id"]: p for p in load_eterna100(version="V2")}
    ids = [i for i in ids if i in puz]
    tools = default_toolbox()
    path = _table_path(outdir)
    table = json.load(open(path)) if os.path.exists(path) else {}
    print(f"build_table: {len(ids)} puzzles x {len(tools)} tools; {len(table)} entries cached", flush=True)
    for pid in ids:
        t = puz[pid]["target"]
        for tool in tools:
            key = f"{pid}:{tool.name}"
            if key in table:
                continue
            r = tool.attempt(t, seed=seed)
            table[key] = {"pid": pid, "tool": tool.name, "solved": bool(r.solved),
                          "cost": float(r.cost_usd), "wall": float(r.wall_s), "bp": int(r.bp),
                          "available": bool(r.available)}
            json.dump(table, open(path, "w"), indent=0)
            print(f"  pid {pid:>3} {tool.name:12} solved={int(r.solved)} "
                  f"cost=${r.cost_usd:.2e} wall={r.wall_s:.1f}s bp={r.bp}", flush=True)
    print(f"build_table done: {len(table)} entries -> {path}", flush=True)
    return table


def _cold_order(tools):
    return sorted(tools, key=lambda t: t.nominal_cost_usd())


def simulate(ids, table, mode, passes=1):
    """Walk each puzzle's ordered toolbox, paying table costs until one solves. mode in {cold, learn}."""
    puz = {p["id"]: p for p in load_eterna100(version="V2")}
    ids = [i for i in ids if i in puz]
    tools = default_toolbox()
    feats = {pid: extract_features(puz[pid]["target"]) for pid in ids}
    router = Router(tools)
    per_pass = []
    for p in range(passes):
        total, solved, rows = 0.0, 0, []
        for pid in ids:
            feat = feats[pid]
            order = _cold_order(tools) if mode == "cold" else router.plan(feat)
            cost, ok, used = 0.0, False, []
            for tool in order:
                e = table[f"{pid}:{tool.name}"]
                cost += e["cost"]
                used.append(tool.name)
                if mode == "learn":
                    router.record(feat, tool.name, e["solved"], e["cost"])
                if e["solved"]:
                    ok = True
                    break
            total += cost
            solved += int(ok)
            rows.append({"pid": pid, "cost": round(cost, 8), "solved": ok, "n_actions": len(used),
                         "first_tool": used[0] if used else None, "used": used})
        per_pass.append({"pass": p, "total_cost": round(total, 8), "solved": solved, "rows": rows})
    return per_pass


def _hard_ids(ids, table, tools):
    """Puzzles where the cheapest (remc) tools all fail -- the ones a learned router should route past."""
    remc = [t.name for t in tools if t.name.startswith("remc@")]
    hard = []
    for pid in ids:
        if all(not table[f"{pid}:{n}"]["solved"] for n in remc):
            hard.append(pid)
    return hard


def run_sim(ids, outdir):
    path = _table_path(outdir)
    if not os.path.exists(path):
        raise SystemExit(f"no table at {path}; run --build first")
    table = json.load(open(path))
    ids = [i for i in ids if all(f"{i}:{t.name}" in table for t in default_toolbox())]
    tools = default_toolbox()
    cold = simulate(ids, table, "cold", passes=1)[0]
    learn = simulate(ids, table, "learn", passes=4)
    hard = _hard_ids(ids, table, tools)

    print(f"\n=== learning vs no-learning on {len(ids)} puzzles ({len(hard)} hard: {hard}) ===", flush=True)
    print(f"  {'cold (no-learn)':22} solved={cold['solved']}/{len(ids)}  total=${cold['total_cost']:.4f}", flush=True)
    for lp in learn:
        print(f"  {'learn pass '+str(lp['pass']):22} solved={lp['solved']}/{len(ids)}  "
              f"total=${lp['total_cost']:.4f}", flush=True)
    last = learn[-1]
    save = cold["total_cost"] - last["total_cost"]
    pct = 100.0 * save / cold["total_cost"] if cold["total_cost"] else 0.0
    print(f"\n  converged saving = ${save:.4f}  ({pct:.1f}% of cold)  | solves equal: "
          f"{cold['solved']==last['solved']}", flush=True)

    # where the saving lands: per-hard-puzzle cold vs converged-learn, and did learn skip the cheap tier?
    cmap = {r["pid"]: r for r in cold["rows"]}
    lmap = {r["pid"]: r for r in last["rows"]}
    print("\n  hard-puzzle attribution (cold cost -> learn cost | learn's first tool):", flush=True)
    hsave = 0.0
    for pid in hard:
        c, l = cmap[pid], lmap[pid]
        hsave += c["cost"] - l["cost"]
        skipped = l["first_tool"] and not l["first_tool"].startswith("remc@")
        print(f"    pid {pid:>3}: ${c['cost']:.4f} -> ${l['cost']:.4f}  first={l['first_tool']:12} "
              f"{'SKIPPED cheap tier' if skipped else 'still cheap-first'}", flush=True)
    print(f"  saving on hard puzzles alone = ${hsave:.4f} "
          f"({100.0*hsave/save if save else float('nan'):.0f}% of total saving)", flush=True)

    out = {"ids": ids, "hard": hard, "cold": cold, "learn": learn,
           "saving_usd": round(save, 8), "saving_pct": round(pct, 2)}
    json.dump(out, open(os.path.join(outdir, "sim.json"), "w"), indent=1)
    print(f"\n  wrote {os.path.join(outdir, 'sim.json')}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="build/extend the fixed outcome table")
    ap.add_argument("--sim", action="store_true", help="simulate cold vs learn on the table")
    ap.add_argument("--ids", type=int, nargs="+", default=DEFAULT_IDS)
    ap.add_argument("--outdir", default=OUT_DEFAULT)
    a = ap.parse_args()
    if a.build:
        build_table(a.ids, a.outdir)
    if a.sim:
        run_sim(a.ids, a.outdir)
    if not (a.build or a.sim):
        ap.error("pass --build and/or --sim")
