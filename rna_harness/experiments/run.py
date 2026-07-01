"""Reproducible experiment runner (zero-credit; classical generators).

Experiments:
  * ablation  -- actuation-fidelity ablation: repair episodes under feedback modes
                 {diagnosis, random, silent_fallback}. Reports solve-rate + A (A_model,
                 A_harness) stratified. The headline test: solve-rate must fall and A must
                 drop when the loop is broken (else the A->0 null signature).
  * baselines -- rnainverse & local_search solve-rate over the benchmark.
  * collect   -- GRPO groups + RFT/GRPO/full JSONL export (the RLVR dataset).
  * coords    -- four-coordinate (D_sym/R_det/T_trace) report over sampled sequences.
  * all       -- everything.

Parallelized across CPU cores (ViennaRNA is CPU-bound; each worker process is independent,
so global param-set state is safe).

Usage:
  python -m rna_harness.experiments.run --experiment all --max-len 50 --seeds 3 \
      --out runs/zero_credit --workers 32
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from statistics import mean
from typing import Dict, List, Optional

from ..benchmarks import eterna100_tasks, synthesize_hybrid_tasks
from ..task import DesignTask


# --------------------------------------------------------------------------- workers
def _ablation_unit(payload):
    from ..rl.collect import repair_episode

    task = DesignTask.from_dict(payload["task"])
    mode = payload["mode"]
    seed = payload["seed"]
    max_steps = payload["max_steps"]
    samples, act, solved = repair_episode(task, max_steps=max_steps, seed=seed, feedback_mode=mode)
    return {
        "task_id": task.task_id,
        "family": task.family,
        "n": task.n,
        "mode": mode,
        "seed": seed,
        "solved": solved,
        "actuation": act,
        "n_samples": len(samples),
    }


def _baseline_unit(payload):
    from ..generators import LocalSearchGenerator, RNAinverseGenerator
    from ..verifier import verify

    task = DesignTask.from_dict(payload["task"])
    method = payload["method"]
    budget = payload["budget"]
    seed = payload["seed"]
    gen = RNAinverseGenerator() if method == "rnainverse" else LocalSearchGenerator()
    t0 = time.time()
    res = gen.design(task, budget=budget, seed=seed)
    v = verify(task, res.best_sequence, compute_cross_oracle=False)
    return {
        "task_id": task.task_id,
        "family": task.family,
        "n": task.n,
        "method": method,
        "seed": seed,
        "solved": v.solved,
        "bp_distance": res.best_bp_distance,
        "reward": v.reward,
        "n_evals": res.n_evals,
        "secs": round(time.time() - t0, 2),
    }


def _collect_unit(payload):
    from ..generators import RNAinverseGenerator
    from ..rl.collect import collect_group

    task = DesignTask.from_dict(payload["task"])
    k = payload["k"]
    budget = payload["budget"]
    seed = payload["seed"]
    samples = collect_group(task, RNAinverseGenerator(), k=k, budget=budget, seed=seed,
                            compute_cross_oracle=payload.get("cross_oracle", False))
    return [s.to_dict() for s in samples]


# --------------------------------------------------------------------------- helpers
def _parallel(fn, payloads, workers):
    out = []
    if workers <= 1:
        for p in payloads:
            out.append(fn(p))
        return out
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fn, p) for p in payloads]
        for f in as_completed(futs):
            out.append(f.result())
    return out


def _agg(rows, key_fields, value_field="solved"):
    """Aggregate mean(value_field) grouped by the key_fields tuple."""
    groups: Dict[tuple, List] = {}
    for r in rows:
        k = tuple(r[f] for f in key_fields)
        groups.setdefault(k, []).append(r[value_field])
    return {"/".join(map(str, k)): round(mean(v), 4) for k, v in sorted(groups.items())}


# --------------------------------------------------------------------------- experiments
def run_ablation(tasks, seeds, max_steps, workers) -> Dict:
    modes = ["diagnosis", "random", "silent_fallback"]
    payloads = [
        {"task": t.to_dict(), "mode": m, "seed": s, "max_steps": max_steps}
        for t in tasks for m in modes for s in range(seeds)
    ]
    rows = _parallel(_ablation_unit, payloads, workers)
    summary = {}
    for m in modes:
        mrows = [r for r in rows if r["mode"] == m]
        solve = mean(r["solved"] for r in mrows) if mrows else 0.0
        As = [r["actuation"]["A"] for r in mrows if r["actuation"]["A"] is not None]
        Am = [r["actuation"]["A_model"] for r in mrows if r["actuation"]["A_model"] is not None]
        Ah = [r["actuation"]["A_harness"] for r in mrows if r["actuation"]["A_harness"] is not None]
        summary[m] = {
            "solve_rate": round(solve, 4),
            "A": round(mean(As), 4) if As else None,
            "A_model": round(mean(Am), 4) if Am else None,
            "A_harness": round(mean(Ah), 4) if Ah else None,
            "n_episodes": len(mrows),
        }
    return {"summary": summary, "rows": rows}


def run_baselines(tasks, seeds, workers, ls_budget=3000, inv_budget=30) -> Dict:
    payloads = []
    for t in tasks:
        for s in range(seeds):
            payloads.append({"task": t.to_dict(), "method": "rnainverse", "budget": inv_budget, "seed": s})
            payloads.append({"task": t.to_dict(), "method": "local_search", "budget": ls_budget, "seed": s})
    rows = _parallel(_baseline_unit, payloads, workers)
    # best-of-seeds solve per (task, method)
    best: Dict[tuple, bool] = {}
    for r in rows:
        k = (r["task_id"], r["method"])
        best[k] = best.get(k, False) or r["solved"]
    summary = {}
    for method in ("rnainverse", "local_search"):
        ks = [k for k in best if k[1] == method]
        summary[method] = {
            "solve_rate_best_of_seeds": round(mean(best[k] for k in ks), 4) if ks else 0.0,
            "n_tasks": len(ks),
        }
    return {"summary": summary, "rows": rows}


def run_collect(tasks, k, budget, seed, workers, out_dir) -> Dict:
    from ..rl.collect import export_full, export_grpo, export_rft
    from ..rl.schema import RLSample

    payloads = [{"task": t.to_dict(), "k": k, "budget": budget, "seed": seed} for t in tasks]
    nested = _parallel(_collect_unit, payloads, workers)
    flat_dicts = [d for group in nested for d in group]
    samples = [RLSample(**{kk: vv for kk, vv in d.items() if kk in RLSample.__dataclass_fields__}) for d in flat_dicts]
    os.makedirs(out_dir, exist_ok=True)
    n_rft = export_rft(samples, os.path.join(out_dir, "rft.jsonl"))
    n_grpo = export_grpo(samples, os.path.join(out_dir, "grpo.jsonl"))
    n_full = export_full(samples, os.path.join(out_dir, "trajectories.jsonl"))
    n_solved = sum(1 for s in samples if s.solved)
    return {
        "n_samples": len(samples),
        "n_solved": n_solved,
        "solve_rate": round(n_solved / max(len(samples), 1), 4),
        "rft_rows": n_rft,
        "grpo_rows": n_grpo,
        "full_rows": n_full,
        "out_dir": out_dir,
    }


def run_coords(tasks, workers) -> Dict:
    """Four-coordinate report averaged over tasks (D_sym/R_det/T_trace)."""
    from ..determinism import four_coordinate_report
    from ..generators import RNAinverseGenerator, LocalSearchGenerator

    reports = []
    for t in tasks:
        # a mix of solved + unsolved seqs so T_trace has failures to localize
        inv = RNAinverseGenerator().design(t, budget=8, seed=0).candidates[:4]
        ls = LocalSearchGenerator(restarts=2).design(t, budget=200, seed=1).candidates[:4]
        seqs = list({s for s in (inv + ls) if s})[:8]
        if seqs:
            reports.append(four_coordinate_report(t, seqs))
    def _avg(name):
        vals = [r["coordinates"][name]["value"] for r in reports if name in r["coordinates"]]
        return round(mean(vals), 4) if vals else None
    return {"D_sym": _avg("D_sym"), "R_det": _avg("R_det"), "T_trace": _avg("T_trace"), "n_tasks": len(reports)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="all", choices=["ablation", "baselines", "collect", "coords", "all"])
    ap.add_argument("--max-len", type=int, default=50)
    ap.add_argument("--ids", type=int, nargs="*", default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--k", type=int, default=8, help="GRPO group size for collect")
    ap.add_argument("--collect-budget", type=int, default=20)
    ap.add_argument("--hybrid", action="store_true", help="also include hybrid-constraint variants")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--out", default="runs/zero_credit")
    args = ap.parse_args(argv)

    tasks = eterna100_tasks(ids=args.ids, max_len=args.max_len)
    if args.hybrid:
        extra = []
        for t in tasks:
            extra += synthesize_hybrid_tasks(t)
        tasks = tasks + extra
    os.makedirs(args.out, exist_ok=True)
    result: Dict[str, object] = {
        "config": vars(args),
        "n_tasks": len(tasks),
        "task_ids": [t.task_id for t in tasks],
    }
    t0 = time.time()

    if args.experiment in ("ablation", "all"):
        print(f"[ablation] {len(tasks)} tasks x {args.seeds} seeds x 3 modes ...", flush=True)
        result["ablation"] = run_ablation(tasks, args.seeds, args.max_steps, args.workers)["summary"]
        print("  ", json.dumps(result["ablation"]), flush=True)
    if args.experiment in ("baselines", "all"):
        print(f"[baselines] {len(tasks)} tasks ...", flush=True)
        result["baselines"] = run_baselines(tasks, args.seeds, args.workers)["summary"]
        print("  ", json.dumps(result["baselines"]), flush=True)
    if args.experiment in ("coords", "all"):
        print("[coords] four-coordinate report ...", flush=True)
        result["coords"] = run_coords(tasks, args.workers)
        print("  ", json.dumps(result["coords"]), flush=True)
    if args.experiment in ("collect", "all"):
        print(f"[collect] GRPO/RFT dataset (k={args.k}) ...", flush=True)
        result["collect"] = run_collect(tasks, args.k, args.collect_budget, 0, args.workers, args.out)
        print("  ", json.dumps(result["collect"]), flush=True)

    result["wall_secs"] = round(time.time() - t0, 1)
    summary_path = os.path.join(args.out, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWROTE {summary_path}  ({result['wall_secs']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
