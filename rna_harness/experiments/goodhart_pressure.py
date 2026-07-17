"""E2': controlled Goodhart measurement -- does optimizing a CHEAP reward widen the gap to a STRONG one?

Addresses the review's two catches:
 (1) exact-match is BINARY and saturates -> "more optimization pressure" is ill-defined. So the cheap
     objective is CONTINUOUS and does not saturate: minimize (bp_to_target, then MFE free energy) -- it
     keeps pulling toward ever-more-stable exact-match folds, blind to the ensemble.
 (2) weak-accept/strong-reject is only Goodhart if the PROPOSER is fixed and only the REWARD changes.
     Both arms use the IDENTICAL Metropolis point-mutation kernel + schedule + seed; only score() differs:
        cheap  : score = bp*BIG + mfe_energy         (ViennaRNA MFE only; the proxy)
        strong : score = normalized ensemble defect   (Boltzmann ensemble; the intended objective)
Optimization pressure = evaluation budget B (swept). We report, per arm and B, the CHEAP-accept rate
(bp==0) and the STRONG-accept rate (P(T)>=theta, theta>1/2 so it nests inside uMFE). The Goodhart
signature: the cheap arm's cheap-accept -> 100% while its strong-accept stays low (the gap WIDENS with
B); the strong arm shows no such split. Model-relative (one ViennaRNA model); no biophysical-truth claim.
Positioned as a deterministic-domain extension of extensional-vs-isomorphic verifier gaming (2604.15149).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random

import RNA

from ..benchmarks.eterna100 import load_eterna100

OUT_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "runs", "goodhart_pressure")
_BIG = 1000.0
_BASES = "ACGU"


def _mfe(seq):
    st, e = RNA.fold(seq)
    return st, e


def _cheap_score(seq, target):
    """Proxy reward: exact-match first, then MFE stability (continuous, ensemble-blind)."""
    st, e = _mfe(seq)
    bp = RNA.bp_distance(st, target)
    return bp * _BIG + e, bp                       # minimize; bp dominates, energy is the non-saturating tail


def _strong_score(seq, target):
    """Intended reward: normalized ensemble defect (sees the whole Boltzmann ensemble)."""
    fc = RNA.fold_compound(seq)
    _, mfe_e = fc.mfe()
    fc.exp_params_rescale(mfe_e)
    fc.pf()
    ned = fc.ensemble_defect(target)
    return ned, ned                                 # minimize


def _p_target(seq, target):
    fc = RNA.fold_compound(seq)
    _, mfe_e = fc.mfe()
    fc.exp_params_rescale(mfe_e)
    fc.pf()
    return fc.pr_structure(target)


def _init_seq(target, rng):
    """Random paired-consistent start: pair positions get a random canonical pair, others random."""
    n = len(target)
    seq = [rng.choice(_BASES) for _ in range(n)]
    pt = _pairs(target)
    pairset = [("G", "C"), ("C", "G"), ("A", "U"), ("U", "A"), ("G", "U"), ("U", "G")]
    for i, j in pt:
        a, b = rng.choice(pairset)
        seq[i], seq[j] = a, b
    return seq, pt


def _pairs(target):
    st, out = [], []
    for i, c in enumerate(target):
        if c == "(":
            st.append(i)
        elif c == ")":
            out.append((st.pop(), i))
    return out


def optimize(target, arm, budget, seed):
    """Fixed Metropolis point-mutation kernel; only the scoring `arm` differs. Returns best sequence."""
    rng = random.Random(seed)
    seq, pt = _init_seq(target, rng)
    pairset = [("G", "C"), ("C", "G"), ("A", "U"), ("U", "A"), ("G", "U"), ("U", "G")]
    score_fn = _cheap_score if arm == "cheap" else _strong_score
    cur, _aux = score_fn("".join(seq), target)
    best_seq, best = list(seq), cur
    n = len(target)
    for step in range(budget):
        T = max(0.03, 0.5 * (1.0 - step / budget))     # identical cooling for both arms
        cand = list(seq)
        if pt and rng.random() < 0.5:                  # mutate a paired position (keep it a canonical pair)
            i, j = rng.choice(pt)
            a, b = rng.choice(pairset)
            cand[i], cand[j] = a, b
        else:                                          # mutate a single position
            k = rng.randrange(n)
            cand[k] = rng.choice(_BASES)
        cs, _ = score_fn("".join(cand), target)
        if cs <= cur or rng.random() < math.exp(-(cs - cur) / T):
            seq, cur = cand, cs
            if cs < best:
                best_seq, best = list(cand), cs
    return "".join(best_seq)


def run(ids, budgets, outdir, seed=0, theta=0.9, max_n=50):
    os.makedirs(outdir, exist_ok=True)
    puz = {p["id"]: p for p in load_eterna100(version="V2")}
    targets = [(i, puz[i]["target"]) for i in ids if i in puz and len(puz[i]["target"]) <= max_n]
    print(f"E2' Goodhart: {len(targets)} targets (n<={max_n}); budgets={budgets}; theta={theta}", flush=True)
    rows = []
    for B in budgets:
        for arm in ("cheap", "strong"):
            cheap_ok = strong_ok = 0
            for pid, t in targets:
                s = optimize(t, arm, B, seed=seed + pid)
                st, _ = _mfe(s)
                bp = RNA.bp_distance(st, t)
                pT = _p_target(s, t)
                cheap_ok += int(bp == 0)
                strong_ok += int(pT >= theta)
                rows.append({"budget": B, "arm": arm, "pid": pid, "bp": bp, "p_target": round(pT, 4)})
            m = len(targets)
            print(f"  B={B:>6} arm={arm:6} | cheap-accept(bp=0)={100*cheap_ok/m:5.1f}%  "
                  f"strong-accept(P>= {theta})={100*strong_ok/m:5.1f}%  gap={100*(cheap_ok-strong_ok)/m:5.1f}pp",
                  flush=True)
            json.dump({"rows": rows, "budgets": budgets, "theta": theta}, open(os.path.join(outdir, "goodhart.json"), "w"), indent=1)
    print("\nGoodhart signature = cheap arm's gap WIDENS with B while strong arm's stays ~0.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, nargs="+", default=None)
    ap.add_argument("--n", type=int, default=20, help="use the N shortest Eterna targets if --ids not given")
    ap.add_argument("--budgets", type=int, nargs="+", default=[200, 1000, 5000, 20000])
    ap.add_argument("--theta", type=float, default=0.9)
    ap.add_argument("--max-n", type=int, default=50)
    ap.add_argument("--outdir", default=OUT_DEFAULT)
    a = ap.parse_args()
    puz = load_eterna100(version="V2")
    if a.ids:
        ids = a.ids
    else:
        ids = [p["id"] for p in sorted(puz, key=lambda p: len(p["target"])) if len(p["target"]) <= a.max_n][: a.n]
    run(ids, a.budgets, a.outdir, theta=a.theta, max_n=a.max_n)
