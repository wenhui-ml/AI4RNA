#!/usr/bin/env python3
"""M68 arm C, the measurement the arm turned into.

inverse_pf_fold accepted 148 of 157 attempts -- it folds to the target almost every time -- and
produced ZERO molecules the placement statistic can be computed on. The reason is not a failure of
the optimiser and not a defect in the harness: a maximiser of the target's ensemble probability puts
G--C at essentially every paired position, so there is no A--U arm to contrast against. The gap is a
contrast between where G--C sits and where A--U sits, and it is undefined when one arm is empty.

The admission rule is binary, so leaning on it would hide the size of the effect. This measures the
thing directly instead: how many pairs of each type each designer spends, on the SAME targets and
the SAME nested-core pair sets, with a target-clustered bootstrap. No admission rule is applied.

COST: $0, offline, CPU only.
"""
from __future__ import annotations
import json, os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import numpy as np
import RNA

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m63_undesigned_placement as M63

SEED = 20260902
NBOOT = 2000


def pair_budget(seq, P):
    """The pair-type budget of one molecule: how many pairs of each type it spends."""
    c = {"CG": 0, "AU": 0, "GU": 0, "other": 0}
    for a, b in P.items():
        if a >= b:
            continue
        t = "".join(sorted({seq[a], seq[b]}))
        c[t if t in c else "other"] += 1
    n = sum(c.values())
    return c, n


def boot(rows, key, rng):
    """rows: (cluster, value). Cluster bootstrap over targets, the unit used everywhere here."""
    by = {}
    for cl, v in rows:
        by.setdefault(cl, []).append(v)
    ks = sorted(by)
    if not ks:
        return None
    pt = float(np.mean([v for k in ks for v in by[k]]))
    bs = []
    for _ in range(NBOOT):
        pick = rng.integers(0, len(ks), len(ks))
        vals = [v for i in pick for v in by[ks[i]]]
        bs.append(float(np.mean(vals)))
    return {"mean": round(pt, 4), "ci": [round(float(np.quantile(bs, .025)), 4),
                                         round(float(np.quantile(bs, .975)), 4)],
            "n": sum(len(by[k]) for k in ks), "n_clusters": len(ks)}


def main():
    rng = np.random.default_rng(SEED)
    des = M63.designed_rows()
    targets, designed = {}, []
    for (pz, _g, _a, _b, _c, _gc, _np, meth, db, seq, win) in des:
        Q, sub, lo = M63.nested_core(db, win)
        if len(Q) // 2 < 8:
            continue
        targets.setdefault(pz, (sub, lo))
        P = M63.pairs_from_db(sub)
        designed.append((pz, seq[lo:lo + len(sub)], P))
    print(f"targets {len(targets)}   designed molecules {len(designed)}", flush=True)

    arms = {"designed": designed}
    for label, fn, req in (("inverse_fold", RNA.inverse_fold, True),
                           ("inverse_pf_fold", RNA.inverse_pf_fold, False)):
        got = []
        t_all = time.time()
        for pz, (sub, _lo) in targets.items():
            P = M63.pairs_from_db(sub)
            t0 = time.time()
            for _ in range(M63.RESTARTS):
                if time.time() - t0 > M63.BUDGET_S:
                    break
                start = "".join("ACGU"[int(x)] for x in rng.integers(0, 4, len(sub)))
                try:
                    s, d = fn(start, sub)
                except Exception:
                    continue
                if req and d > 0:
                    continue
                got.append((pz, s.upper().replace("T", "U"), P))
        arms[label] = got
        print(f"  {label}: {len(got)} molecules, {time.time()-t_all:.0f}s", flush=True)

    OUT = {"seed": SEED, "nboot": NBOOT, "targets": len(targets),
           "note": "no admission rule applied; every molecule produced is counted"}
    print(f"\n  {'arm':<16}{'n':>5}{'AU pairs':>11}{'95% CI':>20}{'AU share':>11}"
          f"{'95% CI':>20}{'zero-AU':>10}", flush=True)
    for label, rows in arms.items():
        au, sh, zero = [], [], []
        for pz, s, P in rows:
            c, n = pair_budget(s, P)
            if not n:
                continue
            au.append((pz, c["AU"])); sh.append((pz, c["AU"] / n))
            zero.append((pz, 1.0 if c["AU"] == 0 else 0.0))
        A, S, Z = boot(au, None, rng), boot(sh, None, rng), boot(zero, None, rng)
        OUT[label] = {"n": len(rows), "au_pairs": A, "au_share": S, "frac_zero_au": Z,
                      "cg_share": boot([(pz, pair_budget(s, P)[0]["CG"] / max(pair_budget(s, P)[1], 1))
                                        for pz, s, P in rows], None, rng)}
        print(f"  {label:<16}{len(rows):>5}{A['mean']:>11.2f}   [{A['ci'][0]:.2f}, {A['ci'][1]:.2f}]"
              f"{S['mean']:>11.3f}   [{S['ci'][0]:.3f}, {S['ci'][1]:.3f}]{100*Z['mean']:>9.1f}%",
              flush=True)

    d, o = OUT["designed"]["au_share"], OUT["inverse_pf_fold"]["au_share"]
    OUT["designed_minus_optimiser_au_share"] = round(d["mean"] - o["mean"], 4)
    OUT["optimiser_zero_au_fraction"] = OUT["inverse_pf_fold"]["frac_zero_au"]["mean"]
    print(f"\ndesigned spend {d['mean']:.3f} of their pairs on A--U; the ensemble optimiser spends "
          f"{o['mean']:.3f}, and {100*OUT['optimiser_zero_au_fraction']:.1f}% of its molecules "
          f"spend none at all.", flush=True)
    json.dump(OUT, open(os.path.join(HERE, "m68f_pair_type_budget.json"), "w"), indent=1)
    print("\nwrote m68f_pair_type_budget.json", flush=True)


if __name__ == "__main__":
    main()
