#!/usr/bin/env python3
"""How many INDEPENDENT voices does the three-predictor panel actually have?

The paper asserts that pKiss, ProbKnot and de-novo ShapeKnots "share nearest-neighbour thermodynamic
parameters, so their errors can correlate", and its carry-over analysis finds agreement tracking shared model
family. That is an argument from provenance, not a measurement. This script measures it, on THIS paper's own
calibration data, so the claim becomes a number.

Method (the idea is borrowed; every number here is computed from cc_ascent's own logs, nothing is imported):
pairwise Cohen's kappa between predictors over the 80 native bpRNA pseudoknots of native_calib.json, where
each predictor's call is "did folding this native's OWN sequence recover its known crossing". Kappa, not raw
agreement, because at ~1-20% positive rates two predictors that both mostly say no agree ~80% of the time by
construction; kappa removes exactly that chance floor.

Why it matters for r_mix. The primary metric is unanimity over three predictors. If two of the three are
strongly correlated and the third is the odd one out, then "all three agree" is nearer a TWO-vote conjunction
than a three-vote one, and the paper's own caution ("cross-predictor consistency, not independent physical
confirmation") is quantified rather than asserted.

Offline; no oracle calls, no episodes.
"""
import json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
PRED = ["pkiss", "probknot", "shapeknots"]


def kappa(a, b):
    """Cohen's kappa for two binary raters over the same items."""
    n = len(a)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    both = sum(1 for x, y in zip(a, b) if x and y)
    neither = sum(1 for x, y in zip(a, b) if (not x) and (not y))
    po = (both + neither) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    k = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    return k, po, pe


def kappa_ci(a, b, B=20000, seed=20260801):
    """Percentile bootstrap over ITEMS (the natives), which is the sampling unit here."""
    import random
    rng = random.Random(seed)
    n = len(a)
    ks = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        k, _, _ = kappa([a[i] for i in idx], [b[i] for i in idx])
        if not math.isnan(k):
            ks.append(k)
    if not ks:
        return float("nan"), float("nan")
    ks.sort()
    return ks[int(0.025 * len(ks))], ks[int(0.975 * len(ks))]


def main():
    d = json.load(open(os.path.join(HERE, "native_calib.json")))
    rows = d["rows"]
    n = len(rows)
    calls = {p: [bool(r[p]) for r in rows] for p in PRED}
    print(f"panel agreement on {n} native bpRNA pseudoknots (each predictor folds the native's OWN sequence;\n"
          f"call = its known crossing recovered). Marginal recall:")
    for p in PRED:
        print(f"  {p:11} {sum(calls[p]):3d}/{n} = {sum(calls[p])/n:.3f}")
    unan = sum(1 for i in range(n) if all(calls[p][i] for p in PRED))
    print(f"  {'all three':11} {unan:3d}/{n} = {unan/n:.3f}")

    print("\npairwise Cohen's kappa (chance-corrected; raw agreement shown for contrast):")
    ks = {}
    for i in range(len(PRED)):
        for j in range(i + 1, len(PRED)):
            p, q = PRED[i], PRED[j]
            k, po, pe = kappa(calls[p], calls[q])
            lo, hi = kappa_ci(calls[p], calls[q])
            ks[(p, q)] = k
            print(f"  {p:11}~{q:11} kappa={k:+.3f} [{lo:+.3f},{hi:+.3f}]   raw agreement={po:.3f} "
                  f"(chance {pe:.3f})")

    # Which pair is the tight one, and is one predictor the odd voice out?
    tight = max(ks, key=lambda kk: ks[kk])
    loose = min(ks, key=lambda kk: ks[kk])
    odd = [p for p in PRED if p not in tight][0]
    print(f"\n  tightest pair: {tight[0]}~{tight[1]} (kappa {ks[tight]:+.3f})")
    print(f"  loosest pair:  {loose[0]}~{loose[1]} (kappa {ks[loose]:+.3f})")
    print(f"  odd voice out: {odd}")
    spread = ks[tight] - ks[loose]
    print(f"  kappa spread across the three pairs: {spread:.3f}")
    if ks[tight] > 2 * max(ks[loose], 1e-9) and ks[tight] > 0.4:
        print(f"  -> the panel is structured, not exchangeable: {tight[0]}+{tight[1]} behave as a correlated\n"
              f"     block and {odd} carries the independent signal, so unanimity over three predictors is\n"
              f"     nearer a TWO-voice conjunction than a three-voice one.")
    else:
        print("  -> no pair dominates; the three behave closer to exchangeable on this set.")

    out = {"n": n, "marginal": {p: sum(calls[p]) for p in PRED}, "unanimous": unan,
           "kappa": {f"{a}~{b}": round(v, 4) for (a, b), v in ks.items()},
           "tightest": f"{tight[0]}~{tight[1]}", "odd_voice": odd}
    json.dump(out, open(os.path.join(HERE, "panel_agreement.json"), "w"), indent=1)
    print("\n-> panel_agreement.json")


if __name__ == "__main__":
    main()
