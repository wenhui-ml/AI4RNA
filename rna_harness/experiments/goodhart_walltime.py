"""R2-3.8: matched-WALL-TIME Goodhart control.

The step-budget study (goodhart_pressure.py) matches the two arms on PROPOSAL count, but a strong-arm step
(MFE + partition function) costs more CPU than a cheap-arm step (MFE only), so at matched steps the strong arm
spends more compute. A reviewer notes the arm x budget interaction is therefore an UPPER BOUND on the strong
arm's advantage. This script re-runs the identical kernel/reward but matches the arms on WALL-TIME instead of
step count: each arm optimizes until a wall budget elapses (cooling by elapsed fraction), so both arms get
EQUAL COMPUTE. If the strong arm still improves robustness with budget while the cheap arm does not, the
failure-to-transfer is not an artifact of the compute asymmetry. We also log mean steps/arm (the strong arm
should do fewer steps per wall-second)."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import RNA

from ..benchmarks.eterna100 import load_eterna100
from .goodhart_pressure import _cheap_score, _strong_score, _p_target, _init_seq, _mfe, _BASES

OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "goodhart_walltime")
_PAIRSET = [("G", "C"), ("C", "G"), ("A", "U"), ("U", "A"), ("G", "U"), ("U", "G")]


def optimize_wt(target, arm, wall_s, seed):
    """Identical Metropolis kernel to goodhart_pressure.optimize, but looped on WALL-TIME (cooling by
    elapsed fraction). Returns (best_seq, n_steps)."""
    rng = random.Random(seed)
    seq, pt = _init_seq(target, rng)
    score_fn = _cheap_score if arm == "cheap" else _strong_score
    cur, _ = score_fn("".join(seq), target)
    best_seq, best = list(seq), cur
    n = len(target)
    t0 = time.perf_counter()
    steps = 0
    while True:
        el = time.perf_counter() - t0
        if el >= wall_s:
            break
        T = max(0.03, 0.5 * (1.0 - el / wall_s))          # cooling by elapsed fraction (matches step-schedule shape)
        cand = list(seq)
        if pt and rng.random() < 0.5:
            i, j = rng.choice(pt)
            a, b = rng.choice(_PAIRSET)
            cand[i], cand[j] = a, b
        else:
            k = rng.randrange(n)
            cand[k] = rng.choice(_BASES)
        cs, _ = score_fn("".join(cand), target)
        if cs <= cur or rng.random() < math.exp(-(cs - cur) / T):
            seq, cur = cand, cs
            if cs < best:
                best_seq, best = list(cand), cs
        steps += 1
    return "".join(best_seq), steps


def _one(args):
    pid, target, arm, wall_s, seed = args
    s, steps = optimize_wt(target, arm, wall_s, seed)
    st, _ = _mfe(s)
    bp = RNA.bp_distance(st, target)
    pT = _p_target(s, target)
    return {"pid": pid, "arm": arm, "wall_s": wall_s, "seed": seed,
            "bp": bp, "p_target": round(pT, 4), "steps": steps}


def run(ids, walls, seeds, theta, max_n, workers, outdir):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    os.makedirs(outdir, exist_ok=True)
    puz = {p["id"]: p for p in load_eterna100(version="V2")}
    targets = [(i, puz[i]["target"]) for i in ids if i in puz and len(puz[i]["target"]) <= max_n]
    jobs = [(pid, t, arm, w, sd) for (pid, t) in targets for arm in ("cheap", "strong")
            for w in walls for sd in range(seeds)]
    print(f"matched-wall-time Goodhart: {len(targets)} targets x {seeds} seeds x {len(walls)} walls x 2 arms "
          f"= {len(jobs)} runs; theta={theta}", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs)):
            rows.append(f.result())
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(jobs)}", flush=True)

    # aggregate per (arm, wall): cheap-accept (bp==0), robust-accept (pT>=theta), mean steps
    m = len(targets) * seeds
    agg = {}
    for arm in ("cheap", "strong"):
        for w in walls:
            cell = [r for r in rows if r["arm"] == arm and r["wall_s"] == w]
            ca = sum(int(r["bp"] == 0) for r in cell)
            ra = sum(int(r["p_target"] >= theta) for r in cell)
            steps = sorted(r["steps"] for r in cell)
            agg[f"{arm}@{w}"] = {"cheap_accept": round(100 * ca / m, 1), "robust_accept": round(100 * ra / m, 1),
                                 "median_steps": steps[len(steps) // 2] if steps else 0, "n": len(cell)}
    json.dump({"agg": agg, "rows": rows, "walls": walls, "seeds": seeds, "theta": theta, "n_targets": len(targets)},
              open(os.path.join(outdir, "goodhart_walltime.json"), "w"), indent=1)

    print("\n=== MATCHED-WALL-TIME Goodhart (18 targets x 5 seeds) ===", flush=True)
    print(f"{'wall(s)':>8} | {'cheap: proxy  robust  steps':>34} | {'strong: robust  steps':>26}", flush=True)
    for w in walls:
        c = agg[f"cheap@{w}"]; s = agg[f"strong@{w}"]
        print(f"{w:>8} | proxy={c['cheap_accept']:5.1f}% robust={c['robust_accept']:5.1f}% steps={c['median_steps']:>5} "
              f"| robust={s['robust_accept']:5.1f}% steps={s['median_steps']:>5}", flush=True)
    cr = [agg[f"cheap@{w}"]["robust_accept"] for w in walls]
    sr = [agg[f"strong@{w}"]["robust_accept"] for w in walls]
    print(f"\n  cheap-arm robustness across wall-budget:  {cr}", flush=True)
    print(f"  strong-arm robustness across wall-budget: {sr}", flush=True)
    print(f"  DiD (strong gain - cheap gain, first->last): "
          f"{(sr[-1]-sr[0]) - (cr[-1]-cr[0]):+.1f} pts", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=18)
    ap.add_argument("--walls", type=float, nargs="+", default=[0.25, 1.0, 4.0, 16.0])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--theta", type=float, default=0.9)
    ap.add_argument("--max-n", type=int, default=50)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--outdir", default=OUT)
    a = ap.parse_args()
    puz = load_eterna100(version="V2")
    ids = [p["id"] for p in sorted(puz, key=lambda p: len(p["target"])) if len(p["target"]) <= a.max_n][: a.n]
    run(ids, a.walls, a.seeds, a.theta, a.max_n, a.workers, a.outdir)
