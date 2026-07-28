#!/usr/bin/env python3
"""End-to-end check that the new default_toolbox() actually delivers what ladder.py simulated.

ladder.py predicted, by replaying per-target outcomes: 84/100 at $1.0811 for
[desirna@5s, desirna@15s, desirna@60s, desirna@180s], against 84/100 at $2.1900 for the old
[remc@5s, remc@30s, remc@120s, desirna@60s, desirna@180s].

A replay is not a run. It assumes every rung's outcome is independent of the cascade that reached it, reuses
outcomes measured under different seeds, and cannot see integration faults (the panel screen now being on, the
new DesiRNA discovery path, oracle-verification overhead inside the cascade). This fires the REAL
FixedPolicy over the REAL toolbox on all 100 targets and reports the same two numbers so the claim is
measured rather than modelled.

FixedPolicy, not CognitiveCore(learn=True): the declared ladder order is what was simulated, and a learning
router would reorder mid-run and make the comparison meaningless. The learned arm is a separate question.

Per-target incremental persistence, and the sequence is stored.
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
OUT = os.path.join(HERE, "verify_ladder")


def work(item):
    pid, target = item
    dest = os.path.join(OUT, f"p{pid}.json")
    if os.path.exists(dest):
        return dest
    from rna_harness.cognitive.core import FixedPolicy
    from rna_harness.cognitive.tools import default_toolbox
    tb = default_toolbox()
    t0 = time.time()
    try:
        r = FixedPolicy(tb, name="ladder").solve(target, seed=0)
        rec = {"id": pid, "n": len(target), "solved": bool(r["solved"]),
               "cost_usd": float(r["cost_usd"]), "n_actions": int(r["n_actions"]),
               "wall_s": round(time.time() - t0, 1), "seq": r.get("seq"),
               "first_solver": next((s["tool"] for s in r["trace"] if s["solved"]), None),
               "trace": [{k: s.get(k) for k in ("tool", "solved", "bp", "cost_usd")} for s in r["trace"]]}
    except Exception as e:                                          # noqa: BLE001
        rec = {"id": pid, "n": len(target), "error": repr(e)[:200],
               "wall_s": round(time.time() - t0, 1)}
    with open(dest, "w") as f:
        json.dump(rec, f)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--analyse-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    from rna_harness.benchmarks.eterna100 import load_eterna100
    from rna_harness.cognitive.tools import default_toolbox
    from rna_harness.structure_utils import has_pseudoknot
    tb = default_toolbox()
    T = [(p["id"], p["target"]) for p in load_eterna100(version="V2") if not has_pseudoknot(p["target"])]
    print(f"end-to-end verification of default_toolbox() on {len(T)} Eterna100-V2 targets")
    print(f"  ladder: {[t.name for t in tb]}")
    print(f"  simulated target (ladder.py): 84/100 at $1.0811; old ladder was 84/100 at $2.1900\n",
          flush=True)

    if not a.analyse_only:
        todo = [it for it in T if not os.path.exists(os.path.join(OUT, f"p{it[0]}.json"))]
        print(f"  {len(todo)} to run, {a.workers} workers", flush=True)
        from concurrent.futures import ProcessPoolExecutor, as_completed
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(work, it) for it in todo]
            for n, _f in enumerate(as_completed(futs), 1):
                if n % 10 == 0 or n == len(todo):
                    el = time.time() - t0
                    print(f"    {n}/{len(todo)}  {el/60:.1f} min  eta {el/n*(len(todo)-n)/60:.1f} min",
                          flush=True)

    recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, "*.json")))]
    ok = [r for r in recs if "error" not in r]
    err = [r for r in recs if "error" in r]
    solved = sum(1 for r in ok if r.get("solved"))
    cost = sum(r.get("cost_usd", 0.0) for r in ok)
    print(f"\n=== measured ===")
    print(f"  {solved}/{len(ok)} solved, ${cost:.4f} total, ${cost/max(solved,1):.5f}/solve, "
          f"{len(err)} errors")
    print(f"  simulated: 84/100 at $1.0811  ->  "
          f"{'CONSISTENT' if abs(solved-84) <= 2 and cost < 1.6 else 'DIVERGES -- the replay was wrong'}")
    print(f"  vs the old ladder ($2.1900): {2.1900/cost:.2f}x cheaper at "
          f"{solved - 84:+d} solves")
    import collections
    fs = collections.Counter(r.get("first_solver") for r in ok if r.get("solved"))
    print(f"\n  first solver: {dict(fs)}")
    acts = collections.Counter(r.get("n_actions") for r in ok)
    print(f"  rungs fired per target: {dict(sorted(acts.items()))}")
    for lo, hi, lbl in ((0, 100, "<=100nt"), (101, 250, "101-250"), (251, 10 ** 9, ">250")):
        sub = [r for r in ok if lo <= r["n"] <= hi]
        print(f"  {lbl:>8}: {sum(1 for r in sub if r.get('solved')):>3}/{len(sub):<3} "
              f"${sum(r.get('cost_usd',0) for r in sub):.4f}")
    if err:
        print(f"\n  ERRORS ({len(err)}): {[(r['id'], r['error'][:80]) for r in err[:5]]}")
    json.dump({"ladder": [t.name for t in tb], "solved": solved, "n": len(ok), "cost_usd": cost,
               "errors": len(err), "simulated_solved": 84, "simulated_cost": 1.0811,
               "old_ladder_cost": 2.1900, "per_target": recs},
              open(os.path.join(HERE, "verify_ladder.json"), "w"), indent=1)
    print(f"\n  wrote rna_harness/runs/ceiling/verify_ladder.json")


if __name__ == "__main__":
    main()
