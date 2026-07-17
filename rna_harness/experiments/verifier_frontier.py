"""Paper D central experiment: the VERIFIER cost-robustness frontier + the gaming curve (MFE domain).

A cost-optimal solver reaches EXACT-MATCH (fold==target) -- the CHEAP verifier it optimises. We re-judge
that SAME solution under stricter DECLARED predicates and record survival + per-predicate cost:
    match  fold(seq)==target                          (cheap; what the solver optimised)
    umfe   target is the UNIQUE MFE                    (closes the degenerate co-optimal loophole)
    pfe    P(target) in the Boltzmann ensemble >= th   (robust fold, not a fragile 1-of-many)
The drop match->umfe->pfe is the FALSE-POSITIVE MASS a cheap-verifier-optimiser emits (Goodhart at the
optimisation level, no RL). We test whether it GROWS with target hardness. Continuous companions:
P(target), ensemble defect. Cost = verifier WALL per level. This is the nested/MFE strictness AXIS; the
pseudoknot ORACLE axis (MFE-skeleton vs pKiss) is in track_b_pk.fp_demo. Red line: solved is under the
declared oracle only, never in-vivo.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import RNA

from ..benchmarks.eterna100 import load_eterna100
from ..cognitive.tools import verify

OUT_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "runs", "verifier_frontier")
LEVELS = ["match", "umfe", "pfe"]


def solve_exact(target, seed=0, wall_s=30.0, max_calls=300000):
    """Cost-optimal solver = REMC optimising the exact-match (cheap) verifier. Returns a bp==0 sequence
    or None. This is the 'agent that games the cheap verifier'."""
    from .anneal import replica_exchange
    r = replica_exchange(target, n_replicas=8, wall_s=wall_s, max_calls=max_calls, seed=seed)
    seq = r.get("best_seq")
    if seq and RNA.fold(seq)[0] == target:
        return seq
    return None


def _continuous(seq, target):
    """P(target) in the Boltzmann ensemble + ensemble defect (the continuous robustness companions)."""
    fc = RNA.fold_compound(seq)
    _, mfe_e = fc.mfe()
    fc.exp_params_rescale(mfe_e)
    fc.pf()
    return float(fc.pr_structure(target)), float(fc.ensemble_defect(target))


def ladder_eval(seq, target):
    """Judge one exact-match solution under each strictness level; time each verifier call (its cost)."""
    out = {}
    for lv in LEVELS:
        t0 = time.time()
        solved, _bp, _oracle = verify(seq, target, strictness=lv)
        out[lv] = {"solved": bool(solved), "wall_s": round(time.time() - t0, 5)}
    pT, ed = _continuous(seq, target)
    out["p_target"] = round(pT, 4)
    out["ensemble_defect"] = round(ed, 4)
    return out


def _bin(n):
    return "<50" if n < 50 else "50-99" if n < 100 else "100-199" if n < 200 else ">=200"


def _bootstrap_ci(rows, stat, B=5000, seed=0):
    import random as _r
    rng = _r.Random(seed)
    n = len(rows)
    vals = sorted(stat([rows[rng.randrange(n)] for _ in range(n)]) for _ in range(B))
    return vals[int(0.025 * B)], vals[int(0.975 * B)]


def run(ids, outdir, seed=0, wall_s=30.0, seeds=1):
    os.makedirs(outdir, exist_ok=True)
    puz = {p["id"]: p for p in load_eterna100(version="V2")}
    ids = [i for i in ids if i in puz]
    print(f"verifier-frontier on {len(ids)} targets x {seeds} seed(s); solver=REMC(exact-match); "
          f"levels={LEVELS}", flush=True)
    rows = []
    for pid in ids:
        t = puz[pid]["target"]
        solved_any = 0
        for s in range(seeds):
            seq = solve_exact(t, seed=seed + s, wall_s=wall_s)   # each seed = an independent solution
            if seq is None:
                continue
            ev = ladder_eval(seq, t)
            rows.append({"pid": pid, "seed": s, "n": len(t), "bin": _bin(len(t)), **ev})
            solved_any += 1
        print(f"  pid {pid:>3} n={len(t):>4} | solved {solved_any}/{seeds} seeds", flush=True)
        json.dump({"rows": rows}, open(os.path.join(outdir, "frontier.json"), "w"), indent=1)

    if not rows:
        print("no solved targets; nothing to aggregate", flush=True)
        return
    import statistics as _st
    n = len(rows)
    # survival = fraction of exact-match solutions that also pass the stricter predicate
    surv = {lv: sum(r[lv]["solved"] for r in rows) / n for lv in LEVELS}
    cost = {lv: sum(r[lv]["wall_s"] for r in rows) / n for lv in LEVELS}
    ci = {lv: _bootstrap_ci(rows, lambda s, lv=lv: 100 * sum(r[lv]["solved"] for r in s) / len(s)) for lv in LEVELS}
    pt_ci = _bootstrap_ci(rows, lambda s: _st.median([r["p_target"] for r in s]))
    med_pt = _st.median([r["p_target"] for r in rows])
    print(f"\n=== FRONTIER over {n} solutions ({len(ids)} targets x seeds); bootstrap 95% CI ===", flush=True)
    for lv in LEVELS:
        print(f"  {lv:6} survival={surv[lv]*100:5.1f}%  95%CI [{ci[lv][0]:.1f}, {ci[lv][1]:.1f}]  "
              f"mean_verify_cost={cost[lv]*1e3:8.3f} ms", flush=True)
    print(f"  median P(T)={med_pt:.3f}  95%CI [{pt_ci[0]:.3f}, {pt_ci[1]:.3f}]", flush=True)
    print(f"  false-positive mass (match not umfe) = {(surv['match']-surv['umfe'])*100:.1f}%  "
          f"(match not pfe) = {(surv['match']-surv['pfe'])*100:.1f}%", flush=True)

    # gaming vs hardness: false-positive rate (match pass but pfe fail) per length bin
    print("\n=== gaming vs hardness (exact-match but NOT pfe-robust, by length) ===", flush=True)
    bins = {}
    for r in rows:
        b = bins.setdefault(r["bin"], {"n": 0, "fp": 0, "pT": []})
        b["n"] += 1
        b["fp"] += int(r["match"]["solved"] and not r["pfe"]["solved"])
        b["pT"].append(r["p_target"])
    for b in ["<50", "50-99", "100-199", ">=200"]:
        if b in bins:
            v = bins[b]
            mp = sum(v["pT"]) / len(v["pT"])
            print(f"  len {b:>7}: n={v['n']:>3}  fp_rate={100*v['fp']/v['n']:5.1f}%  mean_P(T)={mp:.3f}", flush=True)

    agg = {"n": n, "survival": surv, "survival_ci": ci, "median_pT": med_pt, "median_pT_ci": pt_ci,
           "mean_verify_cost_s": cost,
           "bins": {b: {"n": v["n"], "fp": v["fp"], "mean_pT": sum(v["pT"]) / len(v["pT"])}
                    for b, v in bins.items()}}
    json.dump({"agg": agg, "rows": rows}, open(os.path.join(outdir, "frontier.json"), "w"), indent=1)
    print(f"\n  wrote {os.path.join(outdir, 'frontier.json')}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, nargs="+", default=None)
    ap.add_argument("--n", type=int, default=None, help="first N puzzle ids if --ids not given")
    ap.add_argument("--wall-s", type=float, default=30.0, help="per-target REMC budget")
    ap.add_argument("--seeds", type=int, default=1, help="independent solver seeds per target (for CIs)")
    ap.add_argument("--max-n", type=int, default=None,
                    help="skip targets longer than this (avoids the coordinated_endgame wall-guard bug on large n)")
    ap.add_argument("--outdir", default=OUT_DEFAULT)
    a = ap.parse_args()
    puz_all = {p["id"]: p for p in load_eterna100(version="V2")}
    all_ids = sorted(puz_all)
    ids = a.ids if a.ids else (all_ids[: a.n] if a.n else all_ids)
    if a.max_n:
        kept = [i for i in ids if len(puz_all[i]["target"]) <= a.max_n]
        print(f"--max-n {a.max_n}: kept {len(kept)}/{len(ids)} targets (dropped "
              f"{len(ids) - len(kept)} longer ones)", flush=True)
        ids = kept
    run(ids, a.outdir, wall_s=a.wall_s, seeds=a.seeds)
