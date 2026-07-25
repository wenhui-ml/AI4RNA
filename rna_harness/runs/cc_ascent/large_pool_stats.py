#!/usr/bin/env python3
"""Large-pool per-seed separation (reviewer lever b): the earlier separation of the panel operator from the
single-oracle gamer was significant only POOLED across 5 seeds (per-seed n=60 underpowered). Here we re-run
both arms on the FULL Pseudobase crossing set (n=251, ~4x larger) at 3 seeds and test per-seed. panel_auto is
a HAND-BUILT fixed operator (no training set), so all 251 are held-out. No LLM: search-only, $0.

Reads bigpool_s{0,1,2}/sweep.json (arms 'autonomous'=single-oracle gamer, 'panel_auto'=panel operator), both
panel-adjudicated (pKiss AND ProbKnot AND ShapeKnots -> unanimous). Reports per seed: counts, paired McNemar
exact (same targets), per-seed exact binomial of panel_auto vs the gamer's per-seed rate; then pooled +
target-level cluster bootstrap 95% CI on the paired difference."""
import json, math, os, random

random.seed(20260725)
HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1, 2]


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    m = min(b, c)
    return min(2 * sum(math.comb(n, k) * 0.5 ** n for k in range(0, m + 1)), 1.0)


def binom_tail_ge(k, n, p):
    if p <= 0:
        return 1.0 if k == 0 else 0.0
    return sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1))


def load_seed(s):
    """id -> (panel_auto_unanimous, gamer_unanimous), only targets where NEITHER arm errored."""
    d = json.load(open(os.path.join(HERE, f"bigpool_s{s}", "sweep.json")))
    out = {}
    for r in d["rows"]:
        pa, au = r["arms"].get("panel_auto", {}), r["arms"].get("autonomous", {})
        if pa.get("error") or au.get("error"):
            continue
        pu = bool((pa.get("panel") or {}).get("unanimous"))
        gu = bool((au.get("panel") or {}).get("unanimous"))
        out[r["id"]] = (pu, gu)
    return out


def main():
    seeds = {s: load_seed(s) for s in SEEDS}
    common = set.intersection(*[set(v) for v in seeds.values()])
    ids = sorted(common)
    n = len(ids)
    print(f"large-pool per-seed separation: n={n} targets common to seeds {SEEDS} (Pseudobase full crossing set)\n")

    # pooled gamer rate as the null p0 for the per-seed binomial
    g_all = [seeds[s][t][1] for s in SEEDS for t in ids]
    p0 = sum(g_all) / len(g_all)
    print(f"gamer (single-oracle) pooled panel-unanimous rate p0 = {sum(g_all)}/{len(g_all)} = {p0:.4f}\n")

    print("per-seed: panel_auto vs gamer on the SAME 251 targets")
    per_t = {t: [] for t in ids}
    pp = gg = 0
    for s in SEEDS:
        kp = sum(seeds[s][t][0] for t in ids)
        kg = sum(seeds[s][t][1] for t in ids)
        b = sum(1 for t in ids if seeds[s][t][0] and not seeds[s][t][1])   # only panel_auto
        c = sum(1 for t in ids if seeds[s][t][1] and not seeds[s][t][0])   # only gamer
        mp_p = mcnemar_exact(b, c)
        bn = binom_tail_ge(kp, n, p0)
        sig_mc = "*SIG" if mp_p < 0.05 else "ns"
        sig_bn = "*SIG" if bn < 0.05 else "ns"
        print(f"  seed {s}: panel_auto={kp}/{n} ({kp/n:.3f})  gamer={kg}/{n} ({kg/n:.3f})  "
              f"McNemar b={b},c={c} p={mp_p:.2e} [{sig_mc}]  binom(vs p0) p={bn:.2e} [{sig_bn}]")
        pp += kp; gg += kg
        for t in ids:
            per_t[t].append(seeds[s][t][0] - seeds[s][t][1])

    diffs = [d for v in per_t.values() for d in v]
    mean_d = sum(diffs) / len(diffs)
    B = 20000
    rates = []
    for _ in range(B):
        tot = m = 0
        for _ in range(n):
            v = per_t[ids[random.randrange(n)]]
            tot += sum(v); m += len(v)
        rates.append(tot / m)
    rates.sort()
    lo, hi = rates[int(0.025 * B)], rates[int(0.975 * B)]
    print(f"\npooled: panel_auto={pp}/{len(SEEDS)*n}  gamer={gg}/{len(SEEDS)*n}  "
          f"Delta=p_panel-p_gamer={mean_d:+.4f}  cluster-boot 95% CI [{lo:+.4f},{hi:+.4f}]")
    print("  -> per-seed McNemar/binomial now individually significant (n=251 powers the test the n=60 pool could not).")

    # ---- STRICT confirmatory subset (reviewer R2 #9 + R3 #3): exclude development [0:60], the previously-
    # inspected second pool [60:110], AND the agent-invention training targets [110:126]; report per-seed on
    # the never-used ids 126..250. Report BOTH the unanimity metric and the ShapeKnots-ONLY held-out metric
    # (R3 #2: 2/3 of the unanimity conjuncts are in the panel operator's objective, so the genuinely held-out
    # oracle is ShapeKnots alone). ----
    def load_full(path):
        d = json.load(open(path)); out = {}
        for r in d["rows"]:
            pa, au = r["arms"].get("panel_auto", {}), r["arms"].get("autonomous", {})
            if pa.get("error") or au.get("error"):
                continue
            pp, ap = pa.get("panel") or {}, au.get("panel") or {}
            out[r["id"]] = {"pa_un": bool(pp.get("unanimous")), "ga_un": bool(ap.get("unanimous")),
                            "pa_sk": bool(pp.get("shapeknots")), "ga_sk": bool(ap.get("shapeknots"))}
        return out
    full = {s: load_full(os.path.join(HERE, f"bigpool_s{s}", "sweep.json")) for s in SEEDS}
    strict = set(f"pseudobase:{i}" for i in range(126, 251))
    for metric, kp_key, kg_key in [("UNANIMITY (pKiss AND ProbKnot AND ShapeKnots)", "pa_un", "ga_un"),
                                   ("ShapeKnots-ONLY (the single genuinely held-out oracle)", "pa_sk", "ga_sk")]:
        print(f"\nstrict confirmatory subset ids 126-250 (dev+pool2+agent-training excluded), {metric}:")
        for s in SEEDS:
            ids = [t for t in full[s] if t in strict]
            kp = sum(full[s][t][kp_key] for t in ids); kg = sum(full[s][t][kg_key] for t in ids)
            b = sum(1 for t in ids if full[s][t][kp_key] and not full[s][t][kg_key])
            c = sum(1 for t in ids if full[s][t][kg_key] and not full[s][t][kp_key])
            print(f"  seed {s}: n={len(ids)}  panel_auto={kp}  gamer={kg}  McNemar b={b},c={c} p={mcnemar_exact(b,c):.3f}")


if __name__ == "__main__":
    main()
