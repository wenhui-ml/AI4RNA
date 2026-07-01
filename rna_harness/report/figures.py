"""Generate figures from an experiment summary.json.

Figures:
  1. actuation.png   -- solve-rate + A across feedback modes (the headline result)
  2. baselines.png   -- classical generator solve rates
  3. coords.png      -- four-coordinate determinism (D_sym/R_det/T_trace) = 1.0

Usage:
  python -m rna_harness.report.figures --summary runs/zero_credit_main/summary.json --out runs/zero_credit_main/figures
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fig_actuation(ablation: dict, path: str) -> None:
    modes = ["diagnosis", "random", "silent_fallback"]
    modes = [m for m in modes if m in ablation]
    solve = [ablation[m]["solve_rate"] for m in modes]
    A = [ablation[m].get("A") or 0.0 for m in modes]
    Am = [ablation[m].get("A_model") or 0.0 for m in modes]
    Ah = [ablation[m].get("A_harness") or 0.0 for m in modes]
    x = range(len(modes))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.bar(x, solve, color=["#27AE60", "#E67E22", "#C0392B"])
    ax1.set_xticks(list(x)); ax1.set_xticklabels(modes, rotation=10)
    ax1.set_ylabel("solve rate"); ax1.set_title("Solve rate vs feedback mode")
    for i, v in enumerate(solve):
        ax1.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    w = 0.25
    ax2.bar([i - w for i in x], A, w, label="A", color="#1B6CA8")
    ax2.bar([i for i in x], Am, w, label="A_model", color="#5DADE2")
    ax2.bar([i + w for i in x], Ah, w, label="A_harness", color="#85C1E9")
    ax2.set_xticks(list(x)); ax2.set_xticklabels(modes, rotation=10)
    ax2.set_ylabel("actuation fidelity"); ax2.set_title("Actuation fidelity (A = A_model x A_harness)")
    ax2.legend()
    fig.suptitle("Diagnosis-Actuation Gap on RNA (loop closure lifts solve-rate and A)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_baselines(baselines: dict, path: str) -> None:
    names = list(baselines.keys())
    vals = [baselines[n].get("solve_rate_best_of_seeds", 0.0) for n in names]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(names, vals, color="#7F8C8D")
    ax.set_ylabel("solve rate (best of seeds)"); ax.set_title("Classical baseline floor (Eterna100 subset)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f"{v:.2f}", ha="center")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_coords(coords: dict, path: str) -> None:
    names = ["D_sym", "R_det", "T_trace"]
    vals = [coords.get(n) or 0.0 for n in names]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(names, vals, color="#27AE60")
    ax.set_ylim(0, 1.1); ax.set_ylabel("coordinate value")
    ax.set_title("Four-coordinate determinism (RNA = high-alpha_f positive control)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    with open(args.summary) as f:
        d = json.load(f)
    out = args.out or os.path.join(os.path.dirname(args.summary), "figures")
    os.makedirs(out, exist_ok=True)
    made = []
    if "ablation" in d:
        fig_actuation(d["ablation"], os.path.join(out, "actuation.png")); made.append("actuation.png")
    if "baselines" in d:
        fig_baselines(d["baselines"], os.path.join(out, "baselines.png")); made.append("baselines.png")
    if "coords" in d:
        fig_coords(d["coords"], os.path.join(out, "coords.png")); made.append("coords.png")
    print(f"wrote {made} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
