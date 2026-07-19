"""R3-3.4: E(x)-free cheap-reward control for the Goodhart study.

Reviewer concern: the cheap arm's reward $C_{cheap}=1000\,d(MFE,T)+E(x)$ adds a continuous absolute-MFE-energy
tail after exact match, which might bias toward GC-rich low-energy-but-low-ensemble sequences and thus
manufacture the failure-to-transfer. Control: an E(x)-FREE cheap arm whose reward is $d(MFE,T)$ ONLY (a plain
exact-match objective; after d=0 the kernel random-walks among exact-match sequences, no energy pressure). If
the E(x)-free arm ALSO fails to transfer (robustness flat as budget grows), the energy tail is not the cause.
Same fixed Metropolis kernel/schedule as goodhart_pressure; three arms (cheap-energy, cheap-plain, strong)."""
from __future__ import annotations

import argparse
import json
import math
import os
import random

import RNA

from ..benchmarks.eterna100 import load_eterna100
from .goodhart_pressure import _cheap_score, _strong_score, _p_target, _init_seq, _mfe, _BASES

OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "goodhart_ablation")
_PAIRSET = [("G", "C"), ("C", "G"), ("A", "U"), ("U", "A"), ("G", "U"), ("U", "G")]


def _cheap_plain(seq, target):
    """E(x)-free proxy: base-pair distance ONLY (no energy tail); ties among exact-match are neutral."""
    st, _ = _mfe(seq)
    bp = RNA.bp_distance(st, target)
    return float(bp), bp


def _score(arm):
    return {"cheap": _cheap_score, "plain": _cheap_plain, "strong": _strong_score}[arm]


def optimize(target, arm, budget, seed):
    rng = random.Random(seed)
    seq, pt = _init_seq(target, rng)
    fn = _score(arm)
    cur, _ = fn("".join(seq), target)
    best_seq, best = list(seq), cur
    n = len(target)
    for step in range(budget):
        T = max(0.03, 0.5 * (1.0 - step / budget))
        cand = list(seq)
        if pt and rng.random() < 0.5:
            i, j = rng.choice(pt); a, b = rng.choice(_PAIRSET); cand[i], cand[j] = a, b
        else:
            k = rng.randrange(n); cand[k] = rng.choice(_BASES)
        cs, _ = fn("".join(cand), target)
        if cs <= cur or rng.random() < math.exp(-(cs - cur) / T):
            seq, cur = cand, cs
            if cs < best:
                best_seq, best = list(cand), cs
    return "".join(best_seq)


def _one(args):
    pid, target, arm, budget, seed = args
    s = optimize(target, arm, budget, seed)
    st, _ = _mfe(s)
    return {"pid": pid, "arm": arm, "budget": budget, "seed": seed,
            "bp": RNA.bp_distance(st, target), "p_target": round(_p_target(s, target), 4)}


def run(ids, budgets, seeds, theta, max_n, workers, outdir):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    os.makedirs(outdir, exist_ok=True)
    puz = {p["id"]: p for p in load_eterna100(version="V2")}
    targets = [(i, puz[i]["target"]) for i in ids if i in puz and len(puz[i]["target"]) <= max_n]
    jobs = [(pid, t, arm, b, sd) for (pid, t) in targets for arm in ("cheap", "plain", "strong")
            for b in budgets for sd in range(seeds)]
    print(f"Goodhart ablation: {len(targets)} targets x {seeds} seeds x {len(budgets)} budgets x 3 arms "
          f"= {len(jobs)} runs", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs)):
            rows.append(f.result())
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(jobs)}", flush=True)
    m = len(targets) * seeds
    agg = {}
    for arm in ("cheap", "plain", "strong"):
        for b in budgets:
            cell = [r for r in rows if r["arm"] == arm and r["budget"] == b]
            agg[f"{arm}@{b}"] = {"proxy_exact": round(100 * sum(int(r["bp"] == 0) for r in cell) / m, 1),
                                 "robust": round(100 * sum(int(r["p_target"] >= theta) for r in cell) / m, 1)}
    json.dump({"agg": agg, "rows": rows, "budgets": budgets, "theta": theta, "n_targets": len(targets)},
              open(os.path.join(outdir, "ablation.json"), "w"), indent=1)
    print("\n=== E(x)-FREE CONTROL (robust P(T)>=theta by arm x budget) ===", flush=True)
    print(f"{'budget':>8} | {'cheap-energy':>12} | {'cheap-plain':>11} | {'strong':>7}", flush=True)
    for b in budgets:
        print(f"{b:>8} | {agg[f'cheap@{b}']['robust']:>11.1f}% | {agg[f'plain@{b}']['robust']:>10.1f}% | "
              f"{agg[f'strong@{b}']['robust']:>6.1f}%", flush=True)
    ce = [agg[f"cheap@{b}"]["robust"] for b in budgets]
    pl = [agg[f"plain@{b}"]["robust"] for b in budgets]
    print(f"\n  cheap-energy robustness: {ce}", flush=True)
    print(f"  cheap-plain  robustness: {pl}  (if also flat -> E(x) tail is NOT the cause)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=18)
    ap.add_argument("--budgets", type=int, nargs="+", default=[200, 1000, 5000, 20000])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--theta", type=float, default=0.9)
    ap.add_argument("--max-n", type=int, default=50)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--outdir", default=OUT)
    a = ap.parse_args()
    puz = load_eterna100(version="V2")
    ids = [p["id"] for p in sorted(puz, key=lambda p: len(p["target"])) if len(p["target"]) <= a.max_n][: a.n]
    run(ids, a.budgets, a.seeds, a.theta, a.max_n, a.workers, a.outdir)
