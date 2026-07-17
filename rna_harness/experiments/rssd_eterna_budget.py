"""E5 -- Eterna 86->97 budget curve: is the gap to SOTA (97) really "just budget"?

The project claims the 86->97 Eterna gap is pure per-target compute (DesiRNA gets 97 at up to 24h/target; we
stopped at minutes). E5 tests it: take a HARD Eterna residual set and run DesiRNA at escalating time budgets
(-t 60/600/1800s), MFE-verify (Eterna is nested), and report how many solve at each budget -- the ceiling
climbing with compute. Targets still unsolved at 1800s are the genuinely-hard tail (the #97/#100 that need
hours). Nested MFE oracle (no pKiss). DesiRNA-bound -- queued after E1 to avoid core contention."""
from __future__ import annotations

import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import RNA

from ..benchmarks.eterna100 import load_eterna100
from ..generators.desirna import DesiRNAGenerator, desirna_available
from ..task import DesignTask, OracleConfig

OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "rssd_eterna_budget")
_T2004 = OracleConfig(param_set="turner2004", temperature=37.0, dangles=2)


def _hard_ids():
    """Our best memory-on agent's UNSOLVED Eterna V2 ids = the hard residual to attack with budget."""
    p = "rna_harness/runs/glm52_eterna_memON/summary.json"
    recs = json.load(open(p)).get("records", [])
    ids = []
    for r in recs:
        if not r.get("solved"):
            tid = r.get("task_id", "")
            try:
                ids.append(int(str(tid).split(":")[1]) if ":" in str(tid) else int(tid))
            except Exception:
                pass
    return sorted(set(ids))


def _one(idx, target, tl, seed, replicas):
    gen = DesiRNAGenerator(replicas=replicas, timelimit_s=tl, results_number=20)
    res = gen.design(DesignTask(task_id=f"eterna:{idx}", target=target, oracle=_T2004), seed=seed)
    best = res.best_sequence or None
    solved = bool(best) and RNA.fold(best)[0] == target
    if not solved:
        for c in (getattr(res, "candidates", None) or []):
            if RNA.fold(c)[0] == target:
                solved = True; break
    return idx, tl, solved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", type=int, nargs="+", default=[60, 600, 1800])
    ap.add_argument("--max-targets", type=int, default=20, help="cap the hard set (longest-first) for cost")
    ap.add_argument("--replicas", type=int, default=10)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=OUT)
    a = ap.parse_args()
    if not desirna_available():
        raise SystemExit("DesiRNA unavailable")
    os.makedirs(a.outdir, exist_ok=True)
    puz = {p["id"]: p["target"] for p in load_eterna100(version="V2")}
    hard = [i for i in _hard_ids() if i in puz]
    hard.sort(key=lambda i: -len(puz[i]))                 # longest (hardest) first
    hard = hard[: a.max_targets]
    print(f"E5 Eterna budget curve: {len(hard)} hard residual targets x budgets {a.budgets}s (DesiRNA, MFE-verify)",
          flush=True)

    solved_at = {tl: set() for tl in a.budgets}
    for tl in a.budgets:                                   # escalate budget tier by tier
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(_one, i, puz[i], tl, a.seed, a.replicas): i for i in hard}
            for fut in as_completed(futs):
                idx, t, ok = fut.result()
                if ok:
                    solved_at[tl].add(idx)
        print(f"  -t {tl:>4}s : solved {len(solved_at[tl])}/{len(hard)} of the hard set", flush=True)
        json.dump({tl2: sorted(solved_at[tl2]) for tl2 in a.budgets},
                  open(os.path.join(a.outdir, "budget_curve.json"), "w"), indent=1)

    cumulative = set()
    print(f"\n=== E5 ETERNA 86->97 BUDGET CURVE (hard residual, n={len(hard)}) ===", flush=True)
    for tl in a.budgets:
        cumulative |= solved_at[tl]
        print(f"  by -t {tl}s (cumulative): {len(cumulative)}/{len(hard)} hard residual solved", flush=True)
    still = [i for i in hard if i not in cumulative]
    print(f"  still unsolved at {max(a.budgets)}s: {len(still)}/{len(hard)}  {still}", flush=True)
    print(f"  interpretation: if solves rise steadily with -t, the 86->97 gap is BUDGET; the {len(still)} that "
          f"resist even {max(a.budgets)}s are the genuinely-hard tail (need hours, per DesiRNA's own 24h #97/#100).",
          flush=True)
    json.dump({"config": vars(a), "hard": hard, "solved_at": {tl: sorted(solved_at[tl]) for tl in a.budgets},
               "still_unsolved": still}, open(os.path.join(a.outdir, "budget_curve.json"), "w"), indent=1)
    print(f"  wrote {os.path.join(a.outdir, 'budget_curve.json')}", flush=True)


if __name__ == "__main__":
    main()
