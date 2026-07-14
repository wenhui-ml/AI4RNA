"""Frozen-search-budget attribution experiment (driver).

Question: at a FIXED refinement budget, does retrieved experience (memory) reach the target at LOWER
cost than a generic prior or fresh per-unit search? The discriminator is COST -- how small a refine
budget still solves -- because a solvable target solves at a large budget under any seed. So we SWEEP
a small budget ladder and, for each mode, record best bp-distance (PRIMARY, continuous) and solved
(SECONDARY) at each budget. A mode that reaches low bp / solved at a SMALLER budget wins on cost.

Modes (all share the identical refine ladder; only the SEED differs):
  priors       no memory, no search  (GC-stiff canonical assembly)
  cold_search  no memory, uses search (per-unit inverse_fold builds the seed)   [run twice: variance]
  memory       memory,    no search  (retrieval warm-start)
  memory_clamp memory,    no search  (retrieval + GC-clamped unit boundaries)

Library = the base model's OWN 58 verified Eterna solves (self-generated experience; oracle-gated on
ingest; NEVER sample solutions). Eval = the 42 model-UNSOLVED puzzles, forbidden from the library
(leakage guard). Staged by length: `--max-len` runs the tractable set first; deferred targets are
LOGGED, never silently dropped. Polite CPU: small pools (restarts<=4), set RNA_MAX_CORES low.
"""
from __future__ import annotations

import argparse
import json
import os
import time

from ..benchmarks.eterna100 import load_eterna100
from .retrieval import RetrievalLibrary
from .frozen_budget import build_seed, _refine

BASE_SUMMARY = os.path.join(os.path.dirname(__file__), "..", "runs", "glm5.2_eterna100", "summary.json")
# Order matters under the per-target wall cap: run the CHEAP key arms (priors/memory) FIRST so they
# always complete; the EXPENSIVE cold_search (per-unit search) runs LAST and absorbs any truncation.
MODES = ["priors", "memory", "memory_clamp", "cold_search"]


def load_split():
    puz = {p["id"]: p for p in load_eterna100(version="V2")}
    summ = json.load(open(BASE_SUMMARY))
    solved, unsolved, seqs = [], [], {}
    for r in summ["records"]:
        pid = int(r["task_id"].split(":")[1])
        (solved if r["solved"] else unsolved).append(pid)
        seqs[pid] = r.get("final_sequence") or ""
    return puz, solved, unsolved, seqs


def build_library(puz, solved_ids, unsolved_ids, seqs, save_path):
    lib = RetrievalLibrary()
    lib.forbid([puz[i]["target"] for i in unsolved_ids])   # leakage guard on eval set
    ingested = 0
    for pid in solved_ids:
        if lib.add_solution(puz[pid]["target"], seqs[pid], source="agent-train")["added"]:
            ingested += 1
    lib.save(save_path)
    return lib, ingested


def run(max_len, budgets, restarts, close_cap, pur, tries, cold_repeats, wall_cap_s, outdir):
    os.makedirs(outdir, exist_ok=True)
    puz, solved_ids, unsolved_ids, seqs = load_split()
    lib, ingested = build_library(puz, solved_ids, unsolved_ids, seqs, os.path.join(outdir, "library.json"))
    print(f"library: ingested={ingested}/{len(solved_ids)}  {lib.stats()}", flush=True)

    eval_ids = sorted(unsolved_ids, key=lambda i: len(puz[i]["target"]))
    todo = [i for i in eval_ids if len(puz[i]["target"]) <= max_len]
    deferred = [i for i in eval_ids if len(puz[i]["target"]) > max_len]
    print(f"eval: {len(todo)} targets <= {max_len}nt this stage; "
          f"DEFERRED (>{max_len}nt): {len(deferred)} -> {deferred}", flush=True)

    rows = []
    for pid in todo:
        t = puz[pid]["target"]
        n = len(t)
        tgt0 = time.time()
        rec = {"pid": pid, "name": puz[pid]["name"], "len": n, "modes": {}}
        for mode in MODES:
            reps = cold_repeats if mode == "cold_search" else 1
            mode_runs = []
            for rep in range(reps):
                seed, tel = build_seed(t, mode, lib=lib, pur=pur, tries=tries)
                ladder = {}
                for b in budgets:
                    if time.time() - tgt0 > wall_cap_s:
                        ladder[b] = {"bp": None, "solved": None, "wall_capped": True}
                        continue
                    r0 = time.time()
                    # pure-hillclimb bp@budget: close_pair OFF so the SEED's convergence is what we measure
                    _, bp, solved = _refine(seed, t, restarts=restarts, iters=b,
                                            close_cap=close_cap, use_close=False)
                    ladder[b] = {"bp": bp, "solved": bool(solved), "s": round(time.time() - r0, 2)}
                mode_runs.append({"tel": tel, "ladder": ladder})
            rec["modes"][mode] = mode_runs
        rows.append(rec)
        # compact per-target line: best bp at the SMALLEST budget where each mode is measured
        b0 = budgets[0]
        def _bp0(m):
            runs = rec["modes"][m]
            vals = [rn["ladder"].get(b0, {}).get("bp") for rn in runs if rn["ladder"].get(b0, {}).get("bp") is not None]
            return min(vals) if vals else None
        cov = rec["modes"]["memory"][0]["tel"]["coverage"]
        print(f"  pid {pid:>3} n={n:>3} cov={cov if cov is None else round(cov,2)} "
              f"| bp@{b0}: priors={_bp0('priors')} cold={_bp0('cold_search')} "
              f"mem={_bp0('memory')} clamp={_bp0('memory_clamp')}  ({time.time()-tgt0:.0f}s)", flush=True)
        with open(os.path.join(outdir, "results.json"), "w") as f:
            json.dump({"config": {"max_len": max_len, "budgets": budgets, "restarts": restarts,
                                  "close_cap": close_cap, "cold_repeats": cold_repeats},
                       "deferred": deferred, "rows": rows}, f, indent=1)

    _aggregate(rows, budgets)
    print(f"\nwrote {os.path.join(outdir, 'results.json')}", flush=True)


def _lget(ladder, b):
    """Fetch ladder[b] tolerating int keys (in-memory) OR str keys (after JSON round-trip)."""
    return ladder.get(b, ladder.get(str(b), {}))


def _aggregate(rows, budgets):
    print("\n=== AGGREGATE: solve-rate and mean bp by mode x budget ===", flush=True)
    print(f"{'mode':13} " + " ".join(f"b={b:<6}" for b in budgets), flush=True)
    for mode in MODES:
        cells = []
        for b in budgets:
            bps, solves, k = [], 0, 0
            for rec in rows:
                runs = rec["modes"][mode]
                vals = [_lget(rn["ladder"], b) for rn in runs]
                vbp = [v["bp"] for v in vals if v.get("bp") is not None]
                if not vbp:
                    continue
                k += 1
                bps.append(min(vbp))
                if any(v.get("solved") for v in vals):
                    solves += 1
            if k:
                cells.append(f"{solves}/{k},{sum(bps)/k:.1f}")
            else:
                cells.append("-")
        print(f"{mode:13} " + " ".join(f"{c:<8}" for c in cells), flush=True)
    print("(cell = solved/measured, mean_bp)  lower budget where memory already solves = cost win", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, default=200)
    ap.add_argument("--budgets", type=int, nargs="+", default=[50, 200, 1000, 4000])
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--close-cap", type=int, default=2000)
    ap.add_argument("--pur", type=int, default=6)
    ap.add_argument("--tries", type=int, default=30)
    ap.add_argument("--cold-repeats", type=int, default=2)
    ap.add_argument("--wall-cap-s", type=float, default=120.0)
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "..", "runs", "frozen_budget_s1"))
    a = ap.parse_args()
    run(a.max_len, a.budgets, a.restarts, a.close_cap, a.pur, a.tries,
        a.cold_repeats, a.wall_cap_s, a.outdir)
