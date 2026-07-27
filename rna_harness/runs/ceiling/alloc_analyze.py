#!/usr/bin/env python3
"""Step 2b: score the measured frontier against PREREG.md, prediction by prediction.

Written while alloc_run.py was still executing, so the scoring rule could not be chosen after seeing the
answer. Every comparison is paired WITHIN execution cell: the project's own execution-variance result
forbids comparing counts from separately-run executions.

The metric is panel-unanimous solves (pKiss AND ProbKnot AND ShapeKnots agree). In-loop pKiss solves are
reported alongside, because step 0b found the two dissociate: Delta at large N predicts the in-loop oracle
almost perfectly (AUC 0.951) and the honest metric worse (0.711).

Statistics. 15 cells is few, so: exact sign test on paired per-cell differences (binomial, ties dropped),
a cell-level bootstrap for ratios, and Clopper-Pearson on pooled solve counts. Power is limited by design
and is reported rather than hidden.
"""
import json, math, os, sys
from collections import defaultdict
import numpy as np
from scipy.stats import binom, beta

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "alloc")
ARMS = ["uniform", "delta", "random", "length", "oracle"]


def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


def sign_test(d):
    """Exact two-sided sign test on paired differences; ties dropped."""
    pos = int(sum(1 for x in d if x > 0)); neg = int(sum(1 for x in d if x < 0))
    n = pos + neg
    if n == 0:
        return pos, neg, 1.0
    p = min(1.0, 2 * binom.sf(max(pos, neg) - 1, n, 0.5))
    return pos, neg, p


def main():
    recs = []
    for f in sorted(os.listdir(OUT)):
        if f.endswith(".json"):
            recs.append(json.load(open(os.path.join(OUT, f))))
    plan = json.load(open(os.path.join(HERE, "alloc_plan.json")))
    cure = json.load(open(os.path.join(HERE, "cure.json")))
    B0, fracs, ncells = plan["B0"], plan["fracs"], plan["ncells"]
    errs = [r for r in recs if "error" in r]
    print(f"{len(recs)} design cells on disk, {len(errs)} errors")
    if errs:
        from collections import Counter
        for e, c in Counter(r["error"][:70] for r in errs).most_common(5):
            print(f"   {c:4d}x  {e}")

    # per (arm, frac, cell): realised calls and both solve counts
    agg = defaultdict(lambda: {"calls": 0, "unan": 0, "pkiss": 0, "n": 0})
    for r in recs:
        if "error" in r:
            continue
        k = (r["arm"], round(r["frac"], 2), r["cell"])
        a = agg[k]
        a["calls"] += r.get("n_pkiss", 0)
        a["n"] += 1
        a["pkiss"] += int(bool(r.get("solved")))
        a["unan"] += int(bool((r.get("panel") or {}).get("unanimous")))

    def series(arm, frac, key):
        return np.array([agg[(arm, frac, c)][key] for c in range(ncells)], float)

    print("\n=== MEASURED FRONTIER, ids 30-59, 15 cells, panel-unanimous ===")
    print(f"  {'arm':>8} {'B/B0':>5} {'m':>3} {'calls/cell':>11} {'%B0':>5} "
          f"{'unan/cell':>10} {'95% CI (pooled)':>18} {'pkiss/cell':>11}")
    tab = {}
    for frac in fracs:
        for arm in ARMS:
            c = series(arm, frac, "calls"); u = series(arm, frac, "unan"); p = series(arm, frac, "pkiss")
            m = plan["planned"][f"0|{arm}|{frac}"]["m"]
            k, n = int(u.sum()), int(m * ncells)
            lo, hi = cp(k, max(n, 1))
            tab[(arm, frac)] = {"calls": c, "unan": u, "pkiss": p}
            print(f"  {arm:>8} {frac:5.2f} {m:3d} {c.mean():11.0f} {100*c.mean()/B0:5.1f} "
                  f"{u.mean():10.2f} [{lo*m:6.2f},{hi*m:6.2f}] {p.mean():11.2f}")
        print()

    print("=== P1/P2/P3/P4: policy vs uniform, paired within cell ===")
    print(f"  {'B/B0':>5} {'delta':>7} {'uniform':>8} {'ratio':>7} {'95% CI':>16} "
          f"{'+/-/p (sign)':>16} {'calls d/u':>12}")
    for frac in fracs:
        du, uu = tab[("delta", frac)]["unan"], tab[("uniform", frac)]["unan"]
        dc, uc = tab[("delta", frac)]["calls"], tab[("uniform", frac)]["calls"]
        pos, neg, p = sign_test(du - uu)
        rng = np.random.default_rng(0)
        bs = [du[i].sum() / max(uu[i].sum(), 1e-9)
              for i in (rng.integers(0, ncells, (4000, ncells)))]
        lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
        print(f"  {frac:5.2f} {du.mean():7.2f} {uu.mean():8.2f} "
              f"{du.mean()/max(uu.mean(),1e-9):7.2f} [{lo:6.2f},{hi:6.2f}] "
              f"{pos:3d}/{neg:3d}/{p:7.4f} {dc.mean()/max(uc.mean(),1e-9):12.2f}")
    pre = {0.10: 3.04, 0.20: 2.25, 0.50: 1.39, 1.00: 1.06}
    print("  pre-registered ratios: " + ", ".join(f"{f}->{pre[f]:.2f}" for f in sorted(pre)))

    print("\n=== P5/P6: is it Delta, or just abstaining? (same m, same budget) ===")
    for ctrl in ("random", "length"):
        print(f"  delta vs {ctrl}:")
        for frac in fracs:
            du, cu = tab[("delta", frac)]["unan"], tab[(ctrl, frac)]["unan"]
            pos, neg, p = sign_test(du - cu)
            print(f"    {frac:5.2f}  delta {du.mean():5.2f}  {ctrl} {cu.mean():5.2f}  "
                  f"diff {du.mean()-cu.mean():+5.2f}  {pos:3d}/{neg:3d}  p={p:.4f}")

    print("\n=== bracket: does the policy sit strictly inside [uniform, oracle]? ===")
    for frac in fracs:
        u, d, o = (tab[(a, frac)]["unan"].mean() for a in ("uniform", "delta", "oracle"))
        ok = "yes" if u - 1e-9 <= d <= o + 1e-9 else "NO -- bug, not a finding"
        print(f"  {frac:5.2f}  uniform {u:5.2f} <= delta {d:5.2f} <= oracle {o:5.2f}   {ok}")

    print("\n=== equal-solves cost, by interpolation on the measured frontier ===")
    fu = sorted(fracs)
    yu = [tab[("uniform", f)]["unan"].mean() for f in fu]
    yd = [tab[("delta", f)]["unan"].mean() for f in fu]
    cu = [tab[("uniform", f)]["calls"].mean() for f in fu]
    cd = [tab[("delta", f)]["calls"].mean() for f in fu]
    ndraw, nheld = cure["ndraw"], len(cure["held"])
    for i, f in enumerate(fu):
        t = yu[i]
        if t <= 0 or t > max(yd):
            print(f"  uniform at B/B0={f}: {t:.2f} solves -- outside the delta arm's measured range")
            continue
        j = int(np.searchsorted(yd, t))
        if j == 0:
            need = cd[0] * t / max(yd[0], 1e-9)
        else:
            w = (t - yd[j - 1]) / max(yd[j] - yd[j - 1], 1e-9)
            need = cd[j - 1] + w * (cd[j] - cd[j - 1])
        dc = ndraw * nheld
        print(f"  uniform {t:5.2f} solves at {cu[i]:6.0f} calls  ->  delta needs {need:6.0f}  "
              f"= {cu[i]/max(need,1e-9):5.2f}x   "
              f"({cu[i]/(need+dc):.2f}x charging Delta's {dc} calls, "
              f"{cu[i]/(need+dc/ncells):.2f}x amortised)")

    print("\n=== P7: equal-split-among-funded was the deployed rule, so this is a model check ===")
    print("  (the delta arm IS equal-split; cure.json predicted the unequal-funding gain at <5%)")
    print("\n=== P8: level calibration ===")
    for frac, mrow in zip(fracs, [r for r in cure["frontier"] if round(r[0], 2) in fracs]):
        pass
    fmap = {round(r[0], 2): r for r in cure["frontier"]}
    for frac in fracs:
        r = fmap.get(round(frac, 2))
        if not r:
            continue
        obs = tab[("delta", frac)]["unan"].mean()
        print(f"  B/B0={frac:4.2f}  model raw {r[2]:5.2f}  x0.64 = {r[2]*0.64:5.2f}  observed {obs:5.2f}"
              f"   obs/raw {obs/max(r[2],1e-9):5.2f}")

    print("\n=== power note ===")
    print(f"  {ncells} cells, {len(cure['held'])} held-out targets, uniform baseline "
          f"{tab[('uniform',1.0)]['unan'].mean():.2f} solves/cell. A sign test on 15 pairs cannot reach "
          f"p<0.05 with fewer than 12/15 concordant differences; read the ratios and their bootstrap "
          f"intervals, not the p-values alone.")

    json.dump({(f"{a}|{f}"): {"calls": tab[(a, f)]["calls"].tolist(),
                              "unan": tab[(a, f)]["unan"].tolist(),
                              "pkiss": tab[(a, f)]["pkiss"].tolist()}
               for a in ARMS for f in fracs},
              open(os.path.join(HERE, "alloc_frontier.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
