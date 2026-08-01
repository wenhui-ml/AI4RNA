#!/usr/bin/env python3
"""R10 (reviewer W1/W4c): the unbounded family at scale -- the one place magnitude is estimable.

Why this run exists. The binding limitation of this study is that Pseudobase++ holds 251 crossing targets in
60 structural clusters, only 24 of them disjoint from our development, so every magnitude on it is
unestimable and no reanalysis fixes that. resolution.py puts a number on the consequence: at 60 targets x 5
seeds the design reaches 80% power at an observed gap of ~6/60, and Delta_gamer is 4/60 -- our headline sits
at the resolution limit. The parameterized H-type family T_{n,m} has no such ceiling: it is non-context-free
for every m >= 1, it is the family the class-level reading is actually about, and it is unbounded, so power
here is a budget decision rather than a property of a benchmark.

What this can and cannot buy, stated before the numbers.
  CAN: an estimable magnitude for Delta_gamer, on targets sharing no structural cluster with anything we
       developed on, and a success-vs-size curve -- the only evidence in this paper that bears on whether a
       single finite-description procedure covers a growing family rather than a fixed set.
  CANNOT: generalization. This is one synthetic distribution with a uniform topology and no natural sequence
       constraints, so its rates are NOT comparable to the Pseudobase++ rates and must never be quoted beside
       them as if they were. It is also the topology the operator's decompose-compose scaffold was built for,
       which is a home-field advantage we name rather than discount.

Offline over family_grid/grid_s{0..4}.json; no oracle calls.
"""
import json, math, os, random

random.seed(20260801)
HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ["always_cheap", "autonomous", "panel_auto"]
LABEL = {"always_cheap": "CF floor (REMC, pk-free oracle)",
         "autonomous": "gamer (pKiss only)",
         "panel_auto": "panel operator (pKiss AND ProbKnot)"}


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    m = min(b, c)
    return min(2 * sum(math.comb(n, k) * 0.5 ** n for k in range(m + 1)), 1.0)


def cluster_ci(per_target, B=20000):
    """Bootstrap the paired difference resampling TARGETS, carrying each target's seeds together."""
    ids = list(per_target)
    if not ids:
        return float("nan"), float("nan")
    out = []
    for _ in range(B):
        d = n = 0
        for _ in ids:
            db, k = per_target[ids[random.randrange(len(ids))]]
            d += db; n += k
        out.append(d / n if n else 0.0)
    out.sort()
    return out[int(0.025 * B)], out[int(0.975 * B)]


def load():
    data = {}
    for s in range(5):
        p = os.path.join(HERE, "family_grid", f"grid_s{s}.json")
        if os.path.exists(p):
            d = json.load(open(p))
            if len(d.get("rows") or []) >= 100:
                data[s] = d["rows"]
    return data


def main():
    data = load()
    if not data:
        raise SystemExit("no family grid found")
    seeds = sorted(data)
    n_t = len(data[seeds[0]])
    print(f"unbounded H-type family T_(n,m), n,m in 3..12 = {n_t} targets, seeds {seeds}, deterministic\n")

    print(f"  {'arm':38} {'pKiss':>14} {'panel-unanimous':>20}")
    for a in ARMS:
        pk = un = n = 0
        for s in seeds:
            for r in data[s]:
                v = r["arms"].get(a) or {}
                if v.get("error"):
                    continue
                p_ = v.get("panel") or {}
                n += 1; pk += bool(p_.get("pkiss")); un += bool(p_.get("unanimous"))
        print(f"  {LABEL[a]:38} {pk:5d}/{n:<5d}={pk/max(1,n):.3f} {un:6d}/{n:<5d}={un/max(1,n):.3f}")

    # Delta_gamer, paired within (target, seed), clustered on target.
    per_t, b, c, npair = {}, 0, 0, 0
    for s in seeds:
        for r in data[s]:
            A = (r["arms"].get("autonomous") or {}).get("panel") or {}
            B = (r["arms"].get("panel_auto") or {}).get("panel") or {}
            x, y = bool(A.get("unanimous")), bool(B.get("unanimous"))
            npair += 1
            b += int(y and not x); c += int(x and not y)
            d0, k0 = per_t.get((r["n"], r["m"]), (0, 0))
            per_t[(r["n"], r["m"])] = (d0 + int(y) - int(x), k0 + 1)
    lo, hi = cluster_ci(per_t)
    print(f"\n  Delta_gamer on {npair} paired (target, seed) cells:")
    print(f"    b={b} (operator only), c={c} (gamer only), exact McNemar p={mcnemar(b, c):.4g}")
    print(f"    paired difference {(b-c)/npair:+.4f}, target-clustered 95% CI [{lo:+.4f},{hi:+.4f}] "
          f"({len(per_t)} targets)")
    print(f"    for contrast, the same quantity on Pseudobase++ is +0.067 [+0.023,+0.120] at its "
          f"resolution limit")

    # Success vs size: the uniform-procedure question. Does a fixed-description operator hold up as the
    # family parameter grows, or does it fall off a cliff at some size?
    print("\n  success vs family parameter (panel-unanimous, pooled over seeds; the uniform-procedure curve)")
    print(f"    {'total stem n+m':>15} {'targets':>8} {'operator':>10} {'gamer':>8} {'CF floor':>9}")
    by_size = {}
    for s in seeds:
        for r in data[s]:
            k = r["n"] + r["m"]
            e = by_size.setdefault(k, {a: [0, 0] for a in ARMS})
            for a in ARMS:
                v = r["arms"].get(a) or {}
                if v.get("error"):
                    continue
                e[a][1] += 1
                e[a][0] += bool((v.get("panel") or {}).get("unanimous"))
    for k in sorted(by_size):
        e = by_size[k]
        tot = e["panel_auto"][1]
        print(f"    {k:15d} {tot:8d} {e['panel_auto'][0]/max(1,tot):10.2f} "
              f"{e['autonomous'][0]/max(1,e['autonomous'][1]):8.2f} "
              f"{e['always_cheap'][0]/max(1,e['always_cheap'][1]):9.2f}")

    lo_k = [k for k in sorted(by_size) if k <= 12]
    hi_k = [k for k in sorted(by_size) if k >= 18]
    def rate(ks, a):
        num = sum(by_size[k][a][0] for k in ks); den = sum(by_size[k][a][1] for k in ks)
        return num / max(1, den)
    print(f"\n    operator, smallest sizes (n+m<=12): {rate(lo_k,'panel_auto'):.3f}   "
          f"largest (n+m>=18): {rate(hi_k,'panel_auto'):.3f}")
    print("    The curve is the evidence about a UNIFORM procedure: a fixed-description operator that holds")
    print("    up as the parameter grows is what a memorized table cannot be and a CF generator provably")
    print("    cannot supply. A curve that decays to zero would instead say the operator has a size limit")
    print("    and the family-level reading is unsupported. We report the curve rather than a headline rate.")
    print("\n    Scope, restated because these numbers are large: one synthetic distribution, uniform")
    print("    topology, no natural sequence constraints, and the exact topology the operator's")
    print("    decompose-compose scaffold was built for. Not comparable to the Pseudobase++ rates.")


if __name__ == "__main__":
    main()
