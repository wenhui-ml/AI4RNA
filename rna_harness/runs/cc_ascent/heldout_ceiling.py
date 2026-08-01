#!/usr/bin/env python3
"""Read the held-out judge's confirmation rate against the judge's OWN ceiling, not against 100%.

The problem with r_ho as currently reported. The paper scores designs by whether de-novo ShapeKnots recovers
the target crossing, and calls the result "weak" because it is a few percent. But a few percent against WHAT?
ShapeKnots recovers only 9 of 80 NATIVE pseudoknots from their own natural sequences. A judge that confirms
11% of ground truth cannot confirm 90% of anything, so reading our rate against an implicit 100% understates
it by construction. The reference class has to be the judge's own rate on ground truth, measured with the same
predicate and the same tool configuration.

The confound this must not walk into. The judge's accuracy depends on topology, so an aggregate comparison of
our designs against natives compares one topology mixture against another. We therefore stratify by crossing
stem count -- 1 crossing stem = H-type, 2 = kissing, >=3 = complex, the same classification the dataset table
uses -- and report the aggregate only alongside the strata, never on its own.

What this does and does not establish. It does NOT create new capability evidence; the designs are unchanged.
It removes an unfair denominator from how the existing evidence is read. Any conclusion here is about the
FAIRNESS OF THE COMPARISON, not about a new discovery.

Offline over native_calib.json and bigpool_s{0,1,2}/sweep.json; no oracle calls.
"""
import json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1, 2]
STRICT = range(126, 251)          # the strictly-untouched pool


def clopper_pearson(k, n, alpha=0.05):
    """Exact binomial interval; the rates here are small and n is modest, so normal approximations lie."""
    if n == 0:
        return float("nan"), float("nan")
    def beta_q(p, a, b):
        # bisection on the regularized incomplete beta via its continued-fraction-free integral
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if _betainc(a, b, mid) < p:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    lo = 0.0 if k == 0 else beta_q(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta_q(1 - alpha / 2, k + 1, n - k)
    return lo, hi


def _betainc(a, b, x):
    """Regularized incomplete beta by series; adequate for the small integer a,b used here."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    # continued fraction (Lentz)
    def cf(a, b, x):
        tiny = 1e-30
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, d = 1.0, 1.0 - qab * x / qap
        d = tiny if abs(d) < tiny else d
        d = 1.0 / d
        h = d
        for m in range(1, 300):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            d = tiny if abs(d) < tiny else d
            c = 1.0 + aa / c
            c = tiny if abs(c) < tiny else c
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            d = tiny if abs(d) < tiny else d
            c = 1.0 + aa / c
            c = tiny if abs(c) < tiny else c
            d = 1.0 / d
            de = d * c
            h *= de
            if abs(de - 1.0) < 3e-16:
                break
        return h
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) / a
    if x < (a + 1) / (a + b + 2):
        return front * cf(a, b, x)
    return 1.0 - math.exp(b * math.log(1 - x) + a * math.log(x) - lbeta) / b * cf(b, a, 1 - x)


def n_cross_stems(db):
    """Crossing-stem count: how many DISTINCT non-round bracket classes the structure uses. 1 = H-type,
    2 = kissing, >=3 = complex -- the classification the dataset table reports."""
    return len({c.lower() for c in db if c in "[]{}<>"} & {"[", "{", "<"}) or \
        len({c for c in db if c in "[{<"})


def topo(db):
    k = len({c for c in db if c in "[{<"})
    return "H-type" if k <= 1 else ("kissing" if k == 2 else "complex")


def main():
    # --- the judge's own ceiling, on ground truth, same predicate and configuration ---
    nat = json.load(open(os.path.join(HERE, "native_calib.json")))["rows"]
    print("=== de-novo ShapeKnots on NATIVE pseudoknots, folding each native's own sequence ===")
    print("    (this is the judge's own ceiling: the reference class for any design rate)")
    by_t = {}
    for r in nat:
        # natives carry no dot-bracket here; stratify on the crossing-pair count we do have
        b = "H-ish (<=5 crossing pairs)" if (r.get("n_cross") or 0) <= 5 else "richer (>5)"
        by_t.setdefault(b, []).append(bool(r["shapeknots"]))
    allv = [bool(r["shapeknots"]) for r in nat]
    k, n = sum(allv), len(allv)
    lo, hi = clopper_pearson(k, n)
    print(f"  aggregate          {k:3d}/{n:3d} = {k/n:.3f}  95% CP [{lo:.3f},{hi:.3f}]")
    for b, v in sorted(by_t.items()):
        k2, n2 = sum(v), len(v)
        l2, h2 = clopper_pearson(k2, n2)
        print(f"  {b:26} {k2:3d}/{n2:3d} = {k2/n2:.3f}  95% CP [{l2:.3f},{h2:.3f}]")

    # --- our designs, same judge, same predicate, strictly-untouched pool ---
    print("\n=== de-novo ShapeKnots on OUR DESIGNS, strictly-untouched pool (ids 126-250) ===")
    rows = {}
    for s in SEEDS:
        p = os.path.join(HERE, f"bigpool_s{s}", "sweep.json")
        if not os.path.exists(p):
            print(f"  (bigpool_s{s} missing)"); continue
        for r in json.load(open(p))["rows"]:
            idx = int(r["id"].split(":")[1])
            if idx not in STRICT:
                continue
            for arm in ("panel_auto", "autonomous"):
                a = r["arms"].get(arm) or {}
                sk = ((a.get("panel") or {}) or {}).get("shapeknots")
                rows.setdefault((arm, s), []).append((topo(r.get("target", "")), bool(sk)))
    for arm, label in (("panel_auto", "panel operator"), ("autonomous", "single-oracle gamer")):
        per = [(s, rows[(arm, s)]) for s in SEEDS if (arm, s) in rows]
        if not per:
            continue
        tot = [x for _, v in per for x in v]
        k, n = sum(b for _, b in tot), len(tot)
        lo, hi = clopper_pearson(k, n)
        print(f"  {label:20} pooled {k:3d}/{n:3d} = {k/n:.3f}  95% CP [{lo:.3f},{hi:.3f}]"
              f"   per-seed {[sum(b for _, b in v) for _, v in per]}")
        byt = {}
        for t, b in tot:
            byt.setdefault(t, []).append(b)
        for t, v in sorted(byt.items()):
            k2, n2 = sum(v), len(v)
            l2, h2 = clopper_pearson(k2, n2)
            print(f"    {t:12} {k2:3d}/{n2:4d} = {k2/n2:.3f}  95% CP [{l2:.3f},{h2:.3f}]")

    print("\n=== the reading this licenses ===")
    print("  r_ho must be read against the judge's own ceiling, not against 100%. Report both, stratified,")
    print("  and state plainly that natives and designs are DIFFERENT POPULATIONS, so the ratio of the two")
    print("  rates is an orientation, not a corrected estimate.")


if __name__ == "__main__":
    main()
