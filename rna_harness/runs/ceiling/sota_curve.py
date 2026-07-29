#!/usr/bin/env python3
"""Steps 1 and 2 as ONE experiment: DesiRNA's S(B) curve on the full Eterna100 under the exact `match` oracle.

WHY THESE ARE ONE EXPERIMENT. Step 1 wanted the missing SOTA column on Eterna100 (this repo has never run
DesiRNA on all 100 -- grep of every branch's runs/*.json returns nothing). Step 2 wanted DesiRNA's own
cost-vs-capability curve, because every efficiency claim on this branch (1.54x, 3.51x, the nested 25x) is
computed against ONE time budget of an over-provisioned baseline. The t=180 column IS the Step-1 answer, so
sweeping t gives both for the price of the sweep.

THE PUBLISHED ANCHOR, which makes this a calibration and not just a measurement. generators/desirna.py cites
Wirecki et al., NAR 2025: DesiRNA reaches **85/100 at under 1 min per target and 97/100 at 24 h per target**
on Eterna100-V2. So:

  * t = 60 s is a REPRODUCTION CHECK on our own adapter. If our DesiRNA wrapper does not land near 85/100
    there, then every DesiRNA number this project has reported is suspect, and that is a more important
    finding than the curve.
  * 97/100 at 24 h/target is the field ceiling and it costs ~24 h x 10 replicas x 100 targets = 24,000
    core-hours = ~$1,200 at our $0.05/core-h. Our harness sits at 64/100. So the effect axis on this
    benchmark is NOT closed -- but the interesting quantity is not "beat 97", it is "how far up that curve
    can be reached for how little", which is exactly S(B).

BUDGET AXIS. DesiRNA's own `-t` in seconds, converted to USD by the repo's cost model
(solver_cost_usd(wall, replicas)). Seconds of wall x cores is a valid cross-length axis; fold-calls are not,
because one ViennaRNA fold costs 1 ms at 30 nt and 179 ms at 400 nt.

DISCIPLINE ADDED HERE. Every returned sequence is stored. The three archived agent runs stored only counts,
which is why today's re-adjudication had to re-run them; that will not happen to this run.

Adjudication is ours, never DesiRNA's self-report: DesiRNAGenerator._verify_best refolds every candidate with
the primary ViennaRNA oracle (turner2004/dangles=2) and `solved` means MFE == target exactly. The adapter also
passes `-p 2004` so DesiRNA OPTIMISES the same energy model we VERIFY with.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
os.environ.setdefault("DESIRNA_CMD",
                      "/home/ubuntu/miniconda3/envs/rna/bin/python /data/wenhui/DesiRNA/DesiRNA.py")

OUT = os.path.join(HERE, "sota_curve")
# Wirecki et al., NAR 2025 (PMC11744100), verbatim for the Turner2004 / Eterna100-V2 setting we use:
#   "With the Turner2004 model, DesiRNA solved 97 out of 100 puzzles within 24 hours (85 under a minute,
#    95 under 1 h)."   Default 10 replicas; "each simulation run utilized 10 threads."
# At $0.05/core-h that published curve is: $0.83 -> 85, $50 -> 95, $1200 -> 97. The last 12 solves cost
# 1400x the first 85. That shape, not the endpoint 97, is what makes this benchmark an efficiency problem.
PUBLISHED = {60: 85, 3600: 95, 86400: 97}


def work(item):
    """One (target, budget, seed) cell -> one json file. Written immediately so a kill loses nothing."""
    tid, name, target, tl, repl, nres, seed = item
    dest = os.path.join(OUT, f"p{tid}_t{tl}_s{seed}.json")
    if os.path.exists(dest):
        return dest
    from rna_harness.cognitive.cost import solver_cost_usd
    from rna_harness.generators.desirna import DesiRNAGenerator, desirna_available
    from rna_harness.task import DesignTask, OracleConfig
    rec = {"id": tid, "name": name, "n": len(target), "t": tl, "replicas": repl, "seed": seed}
    if not desirna_available():
        rec["error"] = "desirna unavailable"
    else:
        t0 = time.time()
        try:
            res = DesiRNAGenerator(replicas=repl, timelimit_s=tl, results_number=nres).design(
                DesignTask(task_id=f"eterna100:{tid}", target=target,
                           oracle=OracleConfig(param_set="turner2004", temperature=37.0, dangles=2)),
                seed=seed)
            wall = time.time() - t0
            rec.update({"solved": bool(res.solved), "bp": int(res.best_bp_distance),
                        "n_cands": int(res.n_evals), "wall_s": round(wall, 1),
                        "cost_usd": solver_cost_usd(wall, repl),
                        "seq": res.best_sequence or None,          # STORED -- see module docstring
                        "meta_error": (res.meta or {}).get("error")})
        except Exception as e:                                     # noqa: BLE001
            rec.update({"error": repr(e)[:200], "wall_s": round(time.time() - t0, 1)})
    with open(dest, "w") as f:
        json.dump(rec, f)
    return dest


def summarise(budgets, seeds):
    recs = []
    for f in sorted(os.listdir(OUT)):
        if f.endswith(".json"):
            recs.append(json.load(open(os.path.join(OUT, f))))
    by = {}
    for r in recs:
        by.setdefault((r["t"], r["seed"]), {})[r["id"]] = r
    print(f"\n=== DesiRNA S(B) on Eterna100-V2, exact `match` oracle (our verification, not its report) ===")
    print(f"  {'-t (s)':>7} {'solved/seed':>18} {'union':>6} {'$/target':>9} {'$ total':>8} "
          f"{'errors':>7} {'published':>10}")
    rows = []
    for t in budgets:
        per, union, costs, errs = [], set(), [], 0
        for s in range(seeds):
            d = by.get((t, s), {})
            if not d:
                continue
            sv = [i for i, r in d.items() if r.get("solved")]
            per.append(len(sv)); union |= set(sv)
            costs += [r.get("cost_usd", 0.0) for r in d.values()]
            errs += sum(1 for r in d.values() if r.get("error") or r.get("meta_error"))
        if not per:
            continue
        pub = PUBLISHED.get(t)
        rows.append({"t": t, "per_seed": per, "union": len(union),
                     "cost_per_target": (sum(costs) / len(costs) if costs else 0.0),
                     "cost_total": sum(costs) / max(seeds, 1), "errors": errs, "published": pub})
        print(f"  {t:>7} {str(per):>18} {len(union):>6} {rows[-1]['cost_per_target']:>9.4f} "
              f"{rows[-1]['cost_total']:>8.3f} {errs:>7} {('  ' + str(pub)) if pub else '        --':>10}")

    anchor = next((r for r in rows if r["t"] == 60), None)
    if anchor:
        print(f"\n=== calibration against the published anchor (85/100 at <1 min) ===")
        got = max(anchor["per_seed"])
        print(f"  our adapter at -t 60: {got}/100 against published 85/100  "
              f"-> {'CONSISTENT' if got >= 78 else 'ADAPTER UNDER-PERFORMS -- treat every DesiRNA number on this repo as suspect'}")

    # length regimes: this branch measured our own harness at 53/75 below 250 nt and 9/25 above
    print(f"\n=== by length, at each budget (our harness: 53/75 below 250 nt, 9/25 above) ===")
    print(f"  {'-t (s)':>7} {'<=100':>8} {'101-250':>9} {'>250':>8}")
    for t in budgets:
        d = {}
        for s in range(seeds):
            d.update(by.get((t, s), {}))
        if not d:
            continue
        bands = [("<=100", lambda n: n <= 100), ("101-250", lambda n: 100 < n <= 250),
                 (">250", lambda n: n > 250)]
        cells = []
        for _lbl, f in bands:
            sub = [r for r in d.values() if f(r["n"])]
            cells.append(f"{sum(1 for r in sub if r.get('solved'))}/{len(sub)}")
        print(f"  {t:>7} {cells[0]:>8} {cells[1]:>9} {cells[2]:>8}")

    # the efficiency question this was built to answer
    if len(rows) >= 2:
        print(f"\n=== what this does to our efficiency claims ===")
        top = max(rows, key=lambda r: r["t"])
        print(f"  every archived saving on this branch (Eterna 1.54x, nested 25x) divides by the cost of")
        print(f"  DesiRNA at ONE budget. On this curve the same capability is available at:")
        for r in rows:
            if r["union"] and top["union"]:
                print(f"    -t {r['t']:>4}s -> {r['union']:>3}/100 at ${r['cost_per_target']:.4f}/target"
                      f"   ({100.0*r['union']/top['union']:.0f}% of the -t {top['t']}s capability at "
                      f"{r['cost_per_target']/top['cost_per_target']:.3f}x the price)")
    json.dump({"published_anchor": PUBLISHED, "budgets": budgets, "seeds": seeds, "rows": rows,
               "per_cell": recs}, open(os.path.join(HERE, "sota_curve.json"), "w"), indent=1)
    print(f"\n  wrote {os.path.relpath(os.path.join(HERE, 'sota_curve.json'), REPO)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="5,15,60,180")   # 60 = the published <1 min anchor
    ap.add_argument("--replicas", type=int, default=10)
    ap.add_argument("--results", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--workers", type=int, default=16)    # x10 replicas = 160 of 256 cores
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--only-unsolved", action="store_true",
                    help="restrict to targets unsolved at EVERY tier already on disk. Used to ask whether the "
                         "16 targets called 'hopeless' at t<=180s are a DESIGNABILITY wall or merely a BUDGET "
                         "wall -- the published curve reaches 95/100 at 1 h/target, so most of them should "
                         "flip if it is budget. Declared before the run: >=8 of 16 solving by t<=1800s means "
                         "budget wall, <=3 means designability wall.")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    budgets = [int(x) for x in a.budgets.split(",")]

    from rna_harness.benchmarks.eterna100 import load_eterna100
    from rna_harness.structure_utils import has_pseudoknot
    P = load_eterna100(version="V2")
    T = [(p["id"], p.get("name", ""), p["target"]) for p in P if not has_pseudoknot(p["target"])]
    if a.only_unsolved:
        seen, solved_any = set(), set()
        for f in os.listdir(OUT) if os.path.isdir(OUT) else []:
            if f.endswith(".json"):
                d = json.load(open(os.path.join(OUT, f)))
                seen.add(d["id"])
                if d.get("solved"):
                    solved_any.add(d["id"])
        T = [t for t in T if t[0] in seen and t[0] not in solved_any]
        print(f"  --only-unsolved: {len(T)} targets unsolved at every tier already measured -> "
              f"{[t[0] for t in T]}")
    print(f"DesiRNA S(B) on Eterna100-V2: {len(T)} nested targets x {len(budgets)} budgets x {a.seeds} seed(s)")
    print(f"  budgets -t {budgets} s, -R {a.replicas}, -r {a.results}, -p 2004 (matches our oracle)")
    print(f"  published anchor: 85/100 at <1 min, 97/100 at 24 h (Wirecki 2025); our harness: 64/100")
    est = sum(budgets) * len(T) * a.seeds * a.replicas
    print(f"  upper-bound compute if every run uses its full -t: {est/3600:.1f} core-h "
          f"= ${est/3600*0.05:.2f}\n", flush=True)

    if not a.analyse_only:
        items = [(i, nm, tg, t, a.replicas, a.results, s)
                 for s in range(a.seeds) for t in budgets for i, nm, tg in T]
        todo = [it for it in items
                if not os.path.exists(os.path.join(OUT, f"p{it[0]}_t{it[3]}_s{it[6]}.json"))]
        print(f"  {len(items)} cells, {len(todo)} to run, {a.workers} workers", flush=True)
        from concurrent.futures import ProcessPoolExecutor, as_completed
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(work, it) for it in todo]
            for n, _f in enumerate(as_completed(futs), 1):
                if n % 20 == 0 or n == len(todo):
                    el = time.time() - t0
                    print(f"    {n}/{len(todo)}  {el/60:.1f} min  eta {el/n*(len(todo)-n)/60:.1f} min",
                          flush=True)
    summarise(budgets, a.seeds)


if __name__ == "__main__":
    main()
