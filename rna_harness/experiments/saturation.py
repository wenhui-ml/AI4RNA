"""Clean saturation curve: run the parallel campaign over MULTIPLE SEEDS and average.

The single-seed campaign metric (oracle-calls-to-solve) is high-variance on a small task
set. Averaging independent seeds smooths it into a publishable saturation curve that isolates
the memory effect: with_memory should drop across epochs while no_memory stays flat/high and
permuted_memory (wrong playbook) does not help.

Writes averaged campaign_summary.json (dashboard-compatible) + saturation_summary.json (with
std) + saturation.png.
"""
from __future__ import annotations

import argparse
import json
import os
from statistics import mean, pstdev
from typing import Dict, List

from ..benchmarks import eterna100_tasks
from .campaign import run_campaign


def _avg_curves(per_seed: List[Dict], arms: List[str], epochs: int) -> Dict:
    """Average per-(arm,epoch) across seeds. per_seed = list of run_campaign summaries."""
    out_curves: Dict[str, List[Dict]] = {}
    for arm in arms:
        pts = []
        for epoch in range(epochs):
            ocs, srs, ams = [], [], []
            for summ in per_seed:
                for c in summ["curves"].get(arm, []):
                    if c["epoch"] == epoch:
                        if c["mean_oracle_calls_solved"] is not None:
                            ocs.append(c["mean_oracle_calls_solved"])
                        srs.append(c["solve_rate"])
                        if c.get("A_memory") is not None:
                            ams.append(c["A_memory"])
            if not srs:
                continue
            pts.append({
                "epoch": epoch,
                "solve_rate": round(mean(srs), 4),
                "mean_oracle_calls_solved": round(mean(ocs), 2) if ocs else None,
                "oracle_calls_std": round(pstdev(ocs), 2) if len(ocs) > 1 else 0.0,
                "A_memory": round(mean(ams), 4) if ams else None,
                "n_seeds": len(srs),
            })
        out_curves[arm] = pts
    return out_curves


def _figure(curves: Dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"with_memory": "#27AE60", "no_memory": "#7F8C8D", "permuted_memory": "#C0392B"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    for arm, pts in curves.items():
        ep = [p["epoch"] for p in pts]
        oc = [p["mean_oracle_calls_solved"] for p in pts]
        sd = [p.get("oracle_calls_std", 0) for p in pts]
        col = colors.get(arm, "#1B6CA8")
        ax1.errorbar(ep, oc, yerr=sd, marker="o", capsize=3, color=col, label=arm)
        ax2.plot(ep, [p["solve_rate"] for p in pts], marker="s", color=col, label=arm)
    ax1.set_xlabel("epoch (accumulated experience)"); ax1.set_ylabel("oracle-calls-to-solve (mean±std)")
    ax1.set_title("Self-improvement: oracle cost drops with learned memory"); ax1.legend()
    ax2.set_xlabel("epoch"); ax2.set_ylabel("solve rate"); ax2.set_title("Solve rate"); ax2.legend()
    fig.suptitle("RNA meta-learning saturation (classical, multi-seed avg)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, default=80)
    ap.add_argument("--n-tasks", type=int, default=35)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--arms", nargs="*", default=["with_memory", "no_memory", "permuted_memory"])
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--max-workers", type=int, default=48)
    ap.add_argument("--out", default="rna_harness/runs/saturation")
    args = ap.parse_args(argv)

    tasks = sorted(eterna100_tasks(max_len=args.max_len), key=lambda t: t.n)[: args.n_tasks]
    os.makedirs(args.out, exist_ok=True)
    print(f"[saturation] {len(tasks)} tasks x {args.epochs} epochs x {args.seeds} seeds x {len(args.arms)} arms", flush=True)

    per_seed = []
    for s in range(args.seeds):
        summ = run_campaign(
            tasks, executor="classical", arms=args.arms, epochs=args.epochs,
            out_dir=os.path.join(args.out, f"seed_{s}"), classical_budget=args.budget,
            resume=False, seed=s, max_workers=args.max_workers,
        )
        per_seed.append(summ)
        print(f"[saturation] seed {s} done", flush=True)

    curves = _avg_curves(per_seed, args.arms, args.epochs)
    # dashboard-compatible summary
    with open(os.path.join(args.out, "campaign_summary.json"), "w") as f:
        json.dump({"n_episodes": sum(x["n_episodes"] for x in per_seed),
                   "curves": {a: [{**p, "n_strategies": None} for p in pts] for a, pts in curves.items()}}, f, indent=2)
    with open(os.path.join(args.out, "saturation_summary.json"), "w") as f:
        json.dump({"config": vars(args), "curves": curves}, f, indent=2)
    _figure(curves, os.path.join(args.out, "saturation.png"))

    print("\n=== SATURATION (mean oracle-calls-to-solve ± std, over seeds) ===", flush=True)
    for arm, pts in curves.items():
        line = " | ".join(f"e{p['epoch']}: {p['mean_oracle_calls_solved']}±{p['oracle_calls_std']} (solve {p['solve_rate']})" for p in pts)
        print(f"  {arm:16s}: {line}", flush=True)
    print(f"[saturation] wrote {args.out}/saturation.png + summaries", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
