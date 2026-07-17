"""E2' Goodhart, multi-seed with bootstrap CIs. Wraps the (unchanged) single-seed goodhart_pressure.run
over K independent solver seeds, then aggregates robust-acceptance per (arm, budget) with a bootstrap 95%
CI over (seed x target) units. Hardens the carrying leg of the cross-domain Goodhart claim: it shows the
cheap-vs-strong gap and its budget trend are stable to solver-seed noise, not a single lucky run.

Uses goodhart_pressure's own Metropolis kernel (no replica_exchange / coordinated_endgame), so there is
no large-n endgame hang; targets are short (n<=max_n). Model-relative throughout (P(T) under ViennaRNA)."""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics as st

from ..benchmarks.eterna100 import load_eterna100
from .goodhart_pressure import run as gh_run

OUT_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "runs", "goodhart_multiseed")


def _bootstrap_ci(units, stat, B=5000, seed=0):
    rng = random.Random(seed)
    m = len(units)
    vals = sorted(stat([units[rng.randrange(m)] for _ in range(m)]) for _ in range(B))
    return vals[int(0.025 * B)], vals[int(0.975 * B)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="N shortest Eterna targets (n<=max_n)")
    ap.add_argument("--budgets", type=int, nargs="+", default=[200, 1000, 5000, 20000])
    ap.add_argument("--theta", type=float, default=0.9)
    ap.add_argument("--max-n", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--outdir", default=OUT_DEFAULT)
    a = ap.parse_args()

    puz = load_eterna100(version="V2")
    cand = sorted((p for p in puz if len(p["target"]) <= a.max_n), key=lambda p: len(p["target"]))
    ids = [p["id"] for p in cand[: a.n]]
    print(f"E2' Goodhart multi-seed: {len(ids)} targets (n<={a.max_n}) x {a.seeds} seeds; "
          f"budgets={a.budgets}; theta={a.theta}", flush=True)
    os.makedirs(a.outdir, exist_ok=True)

    # 1) run K independent seeds (each writes its own goodhart.json)
    all_rows = []
    for s in range(a.seeds):
        sdir = os.path.join(a.outdir, f"s{s}")
        print(f"\n=== seed {s} ===", flush=True)
        gh_run(ids, a.budgets, sdir, seed=s * 1000, theta=a.theta, max_n=a.max_n)
        d = json.load(open(os.path.join(sdir, "goodhart.json")))
        for r in d["rows"]:
            all_rows.append({**r, "seed": s})

    # 2) aggregate: robust-accept (P(T)>=theta) per (arm, budget), bootstrap CI over (seed,target) units
    print(f"\n=== AGGREGATE over {a.seeds} seeds (bootstrap 95% CI over seed x target) ===", flush=True)
    agg = {}
    for arm in ("cheap", "strong"):
        agg[arm] = {}
        for B in a.budgets:
            units = [r for r in all_rows if r["arm"] == arm and r["budget"] == B]  # one per (seed,target)
            ra = 100 * sum(u["p_target"] >= a.theta for u in units) / len(units)
            ci = _bootstrap_ci(units, lambda U: 100 * sum(u["p_target"] >= a.theta for u in U) / len(U))
            med_pT = st.median([u["p_target"] for u in units])
            agg[arm][B] = {"robust_accept": round(ra, 2), "ci": [round(ci[0], 2), round(ci[1], 2)],
                           "median_pT": round(med_pT, 4), "n_units": len(units)}
            print(f"  {arm:6} B={B:>6} robust-accept={ra:5.1f}%  95%CI [{ci[0]:5.1f},{ci[1]:5.1f}]  "
                  f"median P(T)={med_pT:.3f}", flush=True)

    # 3) the Goodhart signature: cheap arm's robust-accept TREND vs budget (should fall), strong's (rise)
    lo, hi = a.budgets[0], a.budgets[-1]
    print(f"\n  cheap  {lo}->{hi}: {agg['cheap'][lo]['robust_accept']}% -> {agg['cheap'][hi]['robust_accept']}%  "
          f"(Goodhart: optimizing the cheap proxy DESTROYS robustness)", flush=True)
    print(f"  strong {lo}->{hi}: {agg['strong'][lo]['robust_accept']}% -> {agg['strong'][hi]['robust_accept']}%  "
          f"(optimizing the strong predicate BUILDS robustness)", flush=True)
    json.dump({"agg": agg, "rows": all_rows, "budgets": a.budgets, "theta": a.theta, "seeds": a.seeds,
               "ids": ids}, open(os.path.join(a.outdir, "goodhart_multiseed.json"), "w"), indent=1)
    print(f"\n  wrote {os.path.join(a.outdir, 'goodhart_multiseed.json')}", flush=True)


if __name__ == "__main__":
    main()
