"""R3.6 + R3.4 -- decompose the RNA verification cost into STANDALONE / INCREMENTAL / CASCADE, and measure
the CROSS-ORACLE cost (pKiss / ProbKnot vs the within-domain pfe rung), all on one declared machine.

A reviewer notes the paper's ~4x figure (match/uMFE/pfe at 98/301/384 ms) is a STANDALONE cost -- each rung
folds from scratch -- and asks for (a) the INCREMENTAL cost of the strong rung given the cheap oracle's
computation is already in hand (the reuse the paper appeals to), (b) the CASCADE cost on a stream (cheap for
all, strong only for survivors), and (c) the cross-oracle switch cost, which the paper had left to future
work. We build ONE fold_compound per candidate and time each marginal step on the SAME DP object, so the
reuse is measured, not asserted:

  match  verdict  <- fc.mfe()                       (shared base work)
  uMFE   verdict  <- + fc.subopt(0)                 (enumerate co-optimal ground states)
  pfe    verdict  <- + fc.pf() + fc.pr_structure(T) (partition function + target probability)

Standalone(rung)   = build_fc + all steps up to that rung, from scratch.
Incremental(rung)  = marginal steps only, given match's fc.mfe() already ran (the cascade reuse).
Cascade/candidate  = Standalone(match) + survivor_fraction * Incremental(pfe).

Cross-oracle: pKiss and ProbKnot wall time at matched lengths vs pfe. Report medians (heavy-tailed);
hardware is recorded in the output. Run on an otherwise-idle machine for clean timing."""
from __future__ import annotations

import argparse
import json
import os
import platform
import time

import numpy as np
import RNA


def _rand_seq(n, rng):
    return "".join(rng.choice(list("ACGU"), size=n))


def _rand_struct(seq):
    """A realistic target = the MFE fold of the sequence (so pr_structure(T) is meaningful)."""
    st, _ = RNA.fold(seq)
    return st


def _time(fn, reps):
    """median wall seconds over reps (plus one warmup)."""
    fn()
    xs = []
    for _ in range(reps):
        w = time.perf_counter(); fn(); xs.append(time.perf_counter() - w)
    return float(np.median(xs))


def _decompose_one(n, rng, reps):
    seq = _rand_seq(n, rng); tgt = _rand_struct(seq)
    # standalone timings (each from scratch)
    def match_sa():
        fc = RNA.fold_compound(seq); fc.mfe()
    def umfe_sa():
        fc = RNA.fold_compound(seq); fc.mfe(); fc.subopt(0)
    def pfe_sa():
        fc = RNA.fold_compound(seq); fc.mfe(); fc.pf(); fc.pr_structure(tgt)
    t_match = _time(match_sa, reps)
    t_umfe_sa = _time(umfe_sa, reps)
    t_pfe_sa = _time(pfe_sa, reps)
    # incremental: build once, mfe once, then time only the marginal steps
    def umfe_inc():
        fc = RNA.fold_compound(seq); fc.mfe()  # shared
        w = time.perf_counter(); fc.subopt(0); return time.perf_counter() - w
    def pfe_inc():
        fc = RNA.fold_compound(seq); fc.mfe()  # shared
        w = time.perf_counter(); fc.pf(); fc.pr_structure(tgt); return time.perf_counter() - w
    umfe_inc(); pfe_inc()
    t_umfe_inc = float(np.median([umfe_inc() for _ in range(reps)]))
    t_pfe_inc = float(np.median([pfe_inc() for _ in range(reps)]))
    return {"n": n, "match": t_match, "umfe_sa": t_umfe_sa, "pfe_sa": t_pfe_sa,
            "umfe_inc": t_umfe_inc, "pfe_inc": t_pfe_inc}


def _cross_oracle(n, rng, reps):
    """pKiss and ProbKnot wall time at length n (medians)."""
    from ..benchmarks.rnainvbench import pkiss_fold
    from ..oracles_pk import probknot_pairs
    seq = _rand_seq(n, rng)
    def pk(): pkiss_fold(seq, timeout_s=60.0)
    def prob(): probknot_pairs(seq, timeout_s=60.0)
    out = {"n": n}
    try:
        pk(); out["pkiss"] = float(np.median([_t(pk) for _ in range(reps)]))
    except Exception:
        out["pkiss"] = None
    try:
        prob(); out["probknot"] = float(np.median([_t(prob) for _ in range(reps)]))
    except Exception:
        out["probknot"] = None
    return out


def _t(fn):
    w = time.perf_counter(); fn(); return time.perf_counter() - w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[30, 50, 80, 120, 160, 200, 260, 300])
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--cross-lengths", type=int, nargs="+", default=[30, 50, 80, 120, 150])
    ap.add_argument("--cross-reps", type=int, default=8)
    ap.add_argument("--survivor", type=float, default=0.082, help="fraction passing match->pfe (paper: pfe 8.2%)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "..", "runs", "cost_decompose"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    hw = {"cpu": platform.processor() or "unknown", "machine": platform.machine(),
          "python": platform.python_version(), "note": "single-thread timing; run on idle host"}
    try:
        with open("/proc/cpuinfo") as f:
            for ln in f:
                if ln.startswith("model name"):
                    hw["cpu_model"] = ln.split(":", 1)[1].strip(); break
    except OSError:
        pass

    dec = [_decompose_one(n, rng, a.reps) for n in a.lengths]
    cross = [_cross_oracle(n, rng, a.cross_reps) for n in a.cross_lengths]

    print(f"=== R3.6 COST DECOMPOSITION (median ms; {hw.get('cpu_model','?')}) ===")
    print(f"  {'n':>4} | {'match':>7} {'uMFE_sa':>8} {'pfe_sa':>8} | {'uMFE_inc':>9} {'pfe_inc':>8} | "
          f"{'casc/cand':>9} {'vs pfe_sa':>9}")
    for d in dec:
        casc = d["match"] + a.survivor * d["pfe_inc"]
        ratio = casc / d["pfe_sa"] if d["pfe_sa"] else float("nan")
        d["cascade_per_cand"] = casc; d["cascade_vs_pfe_sa"] = ratio
        print(f"  {d['n']:>4} | {d['match']*1e3:7.2f} {d['umfe_sa']*1e3:8.2f} {d['pfe_sa']*1e3:8.2f} | "
              f"{d['umfe_inc']*1e3:9.2f} {d['pfe_inc']*1e3:8.2f} | {casc*1e3:9.2f} {ratio:8.2f}x")
    dmax = dec[-1]
    print(f"\n  at n={dmax['n']}: standalone match/uMFE/pfe = "
          f"{dmax['match']*1e3:.0f}/{dmax['umfe_sa']*1e3:.0f}/{dmax['pfe_sa']*1e3:.0f} ms; "
          f"INCREMENTAL pfe (reuse) = {dmax['pfe_inc']*1e3:.0f} ms "
          f"({100*dmax['pfe_inc']/dmax['pfe_sa']:.0f}% of standalone); "
          f"cascade/candidate at survivor={a.survivor:.3f} = {dmax['cascade_per_cand']*1e3:.0f} ms "
          f"({dmax['cascade_vs_pfe_sa']:.2f}x always-pfe).")
    print("\n  cross-oracle wall time (median ms):")
    print(f"  {'n':>4} | {'pkiss':>8} {'probknot':>9}")
    for c in cross:
        pk = f"{c['pkiss']*1e3:8.1f}" if c.get("pkiss") is not None else "     n/a"
        pb = f"{c['probknot']*1e3:9.1f}" if c.get("probknot") is not None else "      n/a"
        print(f"  {c['n']:>4} | {pk} {pb}")

    json.dump({"hardware": hw, "survivor_fraction": a.survivor, "decompose": dec, "cross_oracle": cross},
              open(os.path.join(a.outdir, "cost_decompose.json"), "w"), indent=1)
    print(f"\n  wrote {os.path.join(a.outdir, 'cost_decompose.json')}")


if __name__ == "__main__":
    main()
