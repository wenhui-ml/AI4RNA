"""F4 -- ONLINE streaming compounding (a stronger form of the cost 真门槛 than the held-out split).

The held-out study (rssd_compounding) proves the learned abstention TRANSFERS. This asks the harder,
temporal question: with NO train/test split, if targets arrive one-by-one and the router learns ONLINE
(cold start, no pre-training), does the cost-per-solve DECREASE over the stream? i.e. does the agent get
CHEAPER as it accrues experience -- compounding in TIME, not just generalising to a held-out set.

Mechanism (pure simulation on the cached real-cost tables, so instant + reproducible):
  * stream = a deterministic shuffle of all 251 PK targets.
  * per target: fire the cheap skill (pkdecomp). If it solves -> done. Else consult the ONLINE k-NN
    predictor P(DesiRNA solves | features) built from everything seen SO FAR:
      - too few same-feature neighbours (cold) -> FIRE DesiRNA (explore + observe its outcome), or
      - enough data -> fire only if P>=tau, else ABSTAIN (skip the expensive doomed call).
    Firing DesiRNA always records its true outcome (that is how the predictor learns).
  * report cost-per-solve per stream WINDOW -> the curve should fall as the predictor sharpens, and the
    cumulative cost should approach always-SOTA's from far below. No silent caps; abstention losses reported.
Reuses rssd_compounding's real-cost columns (pkdecomp measured, DesiRNA cached + measured cost)."""
from __future__ import annotations

import argparse
import json
import os
import random

from ..benchmarks.rnainvbench import load_pseudobase
from ..cognitive.features import FEATURE_NAMES, features as extract_features
from ..cognitive.router import Router
from ..cognitive.tools import DesirnaTool
from .rssd_compounding import _desirna_column

OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "rssd_stream")
TABLE = os.path.join(os.path.dirname(__file__), "..", "runs", "rssd_compounding", "pkdecomp_table.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.5, help="abstain from SOTA when learned P(solve)<tau")
    ap.add_argument("--knn-k", type=int, default=7)
    ap.add_argument("--windows", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=20, help="stream orderings to average over (variance control)")
    ap.add_argument("--outdir", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    pk = json.load(open(TABLE))
    des_solved, des_cost, des_src = _desirna_column()
    tasks = load_pseudobase()
    ids = [int(t.task_id.split(":")[1]) for t in tasks if str(int(t.task_id.split(":")[1])) in pk]
    tgt = {int(t.task_id.split(":")[1]): t.target for t in tasks}
    feats = {i: extract_features(tgt[i]) for i in ids}
    des_name = DesirnaTool(timelimit_s=300).name

    def skill_solved(i): return pk[str(i)]["solved"]
    def skill_cost(i):   return pk[str(i)]["cost"]

    n_win = a.windows
    win_size = max(1, len(ids) // n_win)
    # accumulate per-window cost & solves across many stream orderings, then average
    win_cost = [0.0] * n_win
    win_solved = [0] * n_win
    win_n = [0] * n_win
    tot_cost_stream = tot_cost_sota = tot_lost = 0.0

    for s in range(a.seeds):
        order = list(ids)
        random.Random(1000 + s).shuffle(order)
        router = Router([DesirnaTool(timelimit_s=300)])           # online, cold start
        for pos, i in enumerate(order):
            w = min(pos // win_size, n_win - 1)
            win_n[w] += 1
            c = skill_cost(i)
            solved = skill_solved(i)
            if not solved:
                x = [feats[i][k] for k in FEATURE_NAMES]
                est = router._knn(x, des_name, k=a.knn_k)          # what we know SO FAR
                fire = (est is None) or (est[0] >= a.tau)          # cold -> explore; else exploit
                if fire:
                    c += des_cost
                    ds = des_solved.get(i, False)
                    solved = ds
                    router.record(feats[i], des_name, ds, des_cost)  # observe outcome -> learn
                else:
                    if des_solved.get(i, False):
                        tot_lost += 1                              # abstained on a real solve
            win_cost[w] += c
            win_solved[w] += int(solved)
            tot_cost_stream += c
            tot_cost_sota += des_cost                              # always-SOTA would fire DesiRNA on every target

    # per-window average cost-per-solve
    print(f"\n=== F4 ONLINE STREAMING COMPOUNDING ({len(ids)} PK targets, {a.seeds} stream orders, "
          f"tau={a.tau}; DesiRNA cost={des_src}) ===", flush=True)
    print(f"  cost-per-solve by stream window (cold-start router learns ONLINE; should FALL):", flush=True)
    cps = []
    for w in range(n_win):
        c = win_cost[w] / max(1, a.seeds)
        sv = win_solved[w] / max(1, a.seeds)
        v = c / max(sv, 1e-9)
        cps.append(v)
        print(f"    window {w+1}/{n_win} (~targets {w*win_size+1}-{(w+1)*win_size}): "
              f"cost/solve=${v:.4f}  (avg cost ${c:.4f}, solves {sv:.1f})", flush=True)
    drop = 100 * (cps[0] - cps[-1]) / cps[0] if cps[0] else 0.0
    print(f"  -> cost-per-solve fell {drop:.1f}% from first to last window "
          f"({'COMPOUNDS over time' if drop > 5 else 'flat -- no temporal compounding'})", flush=True)
    print(f"  cumulative: online stream ${tot_cost_stream/a.seeds:.3f} vs always-SOTA ${tot_cost_sota/a.seeds:.3f} "
          f"-> {tot_cost_sota/max(tot_cost_stream,1e-9):.1f}x cheaper; real solves lost/stream={tot_lost/a.seeds:.1f}", flush=True)
    json.dump({"cost_per_solve_by_window": cps, "drop_pct": drop,
               "cum_stream": tot_cost_stream / a.seeds, "cum_sota": tot_cost_sota / a.seeds,
               "lost_per_stream": tot_lost / a.seeds, "config": vars(a)},
              open(os.path.join(a.outdir, "stream.json"), "w"), indent=1)
    print(f"  wrote {os.path.join(a.outdir, 'stream.json')}", flush=True)


if __name__ == "__main__":
    main()
