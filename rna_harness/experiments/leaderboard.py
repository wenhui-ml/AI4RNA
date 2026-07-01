"""Unified scorecard for the openclaude_rna harness — three panels:

  Panel 1 (standard leaderboard, credibility/comparability): Eterna100-V2 full 100 puzzles,
    per classical generator: solved/100 (MFE match), uMFE/100, mean NED (ensemble defect),
    mean bp-distance, runtime -- under a stated compute budget.
  Panel 2 (hybrid-constraint seam, differentiation): success on structure+GC/motif/coding/
    forbidden combinations.
  Panel 3 (agent capability, the headline): actuation ablation (A / A_model / A_harness) and
    meta-learning arms (A_memory, oracle-calls-to-solve). Live-agent A is added separately by
    agent_runner / meta_runner --executor agent.

Writes scorecard.json (rendered by the dashboard "Scorecard" tab). Honest by construction:
classical baselines are reported alongside, and the compute budget is recorded -- the agent's
value is on Panel 3's new axis, not on beating saturated Panel 1 solved-count.

Red line: all metrics are design success under the ViennaRNA computational oracle; no in-vivo
claims.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from statistics import mean
from typing import Dict, List, Optional

from ..benchmarks import eterna100_tasks, synthesize_hybrid_tasks
from ..task import DesignTask
from ..verifier import verify
from .run import _parallel


def _budget_for(n: int) -> Dict[str, int]:
    """Length-scaled budgets so the full-100 run (12-400 nt) stays BOUNDED (baseline
    reference under a fixed, modest budget -- not a max-effort SOTA attempt)."""
    return {
        "rnainverse": max(4, min(20, 2000 // max(n, 1))),
        "local_search": min(60 * n, 4000),
    }


def _panel1_unit(payload):
    from ..generators import LocalSearchGenerator, RNAinverseGenerator

    task = DesignTask.from_dict(payload["task"])
    method = payload["method"]
    seed = payload["seed"]
    b = _budget_for(task.n)[method]
    gen = RNAinverseGenerator() if method == "rnainverse" else LocalSearchGenerator()
    t0 = time.time()
    res = gen.design(task, budget=b, seed=seed)
    v = verify(task, res.best_sequence, compute_cross_oracle=False)
    ned = (v.ensemble or {}).get("ensemble_defect")
    return {
        "task_id": task.task_id, "puzzle_id": task.meta.get("puzzle_id"), "n": task.n,
        "method": method, "solved": v.solved, "umfe": v.is_umfe,
        "ned": ned, "bp_distance": (v.ensemble or {}).get("bp_distance_to_target"),
        "secs": round(time.time() - t0, 2),
    }


def panel1_standard(workers: int, methods: List[str], seed: int = 0) -> Dict:
    tasks = eterna100_tasks(version="V2")
    payloads = [{"task": t.to_dict(), "method": m, "seed": seed} for t in tasks for m in methods]
    rows = _parallel(_panel1_unit, payloads, workers)
    out = {"benchmark": "Eterna100-V2", "n_puzzles": len(tasks), "budget": "length-scaled (see _budget_for)", "by_method": {}}
    for m in methods:
        mr = [r for r in rows if r["method"] == m]
        solved = [r for r in mr if r["solved"]]
        neds = [r["ned"] for r in mr if r["ned"] is not None]
        neds_solved = [r["ned"] for r in solved if r["ned"] is not None]
        out["by_method"][m] = {
            "solved": len(solved), "n": len(mr),
            "solved_rate": round(len(solved) / max(len(mr), 1), 4),
            "umfe": sum(1 for r in mr if r["umfe"]),
            "mean_NED_all": round(mean(neds), 4) if neds else None,
            "mean_NED_solved": round(mean(neds_solved), 4) if neds_solved else None,
            "mean_secs": round(mean(r["secs"] for r in mr), 2) if mr else None,
        }
    out["rows"] = rows
    return out


def _panel2_unit(payload):
    from ..generators import LocalSearchGenerator

    task = DesignTask.from_dict(payload["task"])
    res = LocalSearchGenerator().design(task, budget=payload["budget"], seed=payload["seed"])
    v = verify(task, res.best_sequence, compute_cross_oracle=False)
    return {"task_id": task.task_id, "family": task.family, "solved": v.solved,
            "mfe_match": v.mfe_match, "constraints_ok": (v.constraints or {}).get("all_satisfied"),
            "profile": task.meta.get("hybrid_profile")}


def panel2_hybrid(workers: int, n_base: int = 15, budget: int = 4000, seed: int = 0) -> Dict:
    base = sorted(eterna100_tasks(max_len=60), key=lambda t: t.n)[:n_base]
    tasks: List[DesignTask] = []
    for b in base:
        tasks += synthesize_hybrid_tasks(b)
    payloads = [{"task": t.to_dict(), "budget": budget, "seed": seed} for t in tasks]
    rows = _parallel(_panel2_unit, payloads, workers)
    by_profile: Dict[str, List] = {}
    for r in rows:
        by_profile.setdefault(r["profile"], []).append(r)
    out = {"n_base": len(base), "n_tasks": len(tasks), "by_profile": {}}
    for prof, rs in sorted(by_profile.items()):
        out["by_profile"][prof] = {
            "n": len(rs),
            "solved_rate": round(mean(r["solved"] for r in rs), 4),
            "struct_match_rate": round(mean(r["mfe_match"] for r in rs), 4),
            "constraints_ok_rate": round(mean(bool(r["constraints_ok"]) for r in rs), 4),
        }
    return out


def panel3_agent(workers: int, ablation_max_len: int = 80, ablation_seeds: int = 3, ablation_steps: int = 70) -> Dict:
    from ..meta.meta_runner import hairpin_family, run_arm_classical
    from .run import run_ablation

    tasks = eterna100_tasks(max_len=ablation_max_len)
    abl = run_ablation(tasks, ablation_seeds, ablation_steps, workers)["summary"]
    fam = hairpin_family(12, seed=0)
    meta_arms = {a: {k: v for k, v in run_arm_classical(fam, a, 4000, 0).items() if k != "rows"}
                 for a in ("with_memory", "no_memory", "permuted_memory")}
    return {"actuation_ablation": abl, "n_ablation_tasks": len(tasks), "meta_arms": meta_arms,
            "note": "classical (zero-credit) instruments; live-agent A via agent_runner/meta_runner --executor agent"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", nargs="*", default=["1", "2", "3"])
    ap.add_argument("--methods", nargs="*", default=["rnainverse", "local_search"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--out", default="rna_harness/runs/scorecard")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    scorecard: Dict[str, object] = {"panels": {}, "red_line": "design success under ViennaRNA oracle; no in-vivo claims"}
    t0 = time.time()

    if "1" in args.panels:
        print(f"[panel1] Eterna100-V2 full 100 x {args.methods} (workers={args.workers}) ...", flush=True)
        scorecard["panels"]["standard_leaderboard"] = panel1_standard(args.workers, args.methods)
        print("  ", json.dumps(scorecard["panels"]["standard_leaderboard"]["by_method"]), flush=True)
    if "2" in args.panels:
        print("[panel2] hybrid-constraint seam ...", flush=True)
        scorecard["panels"]["hybrid_seam"] = panel2_hybrid(args.workers)
        print("  ", json.dumps(scorecard["panels"]["hybrid_seam"]["by_profile"]), flush=True)
    if "3" in args.panels:
        print("[panel3] agent-capability (actuation + meta) ...", flush=True)
        scorecard["panels"]["agent_capability"] = panel3_agent(args.workers)
        print("  actuation:", json.dumps(scorecard["panels"]["agent_capability"]["actuation_ablation"]), flush=True)

    scorecard["wall_secs"] = round(time.time() - t0, 1)
    path = os.path.join(args.out, "scorecard.json")
    with open(path, "w") as f:
        json.dump(scorecard, f, indent=2)
    print(f"\n[scorecard] wrote {path} ({scorecard['wall_secs']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
