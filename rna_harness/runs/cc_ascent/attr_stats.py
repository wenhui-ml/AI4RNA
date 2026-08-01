#!/usr/bin/env python3
"""R8, owed experiment (4): attribute the agent-written operators' carry-over advantage to a mechanism.

The quantity. Carry-over is P(de-novo ShapeKnots confirms | the arm's OWN pKiss AND ProbKnot predicate
confirmed) -- conditional on each arm's own success, so no arm's objective can flatter it. That is the
statistic on which the frozen agent operators reach 0.302 [0.275,0.330] and every hand-built variant sits at
0.09-0.16. This script reads it for the transplant chain.

The chain, each step one mechanism (see attribution.sh):
    panel_auto -> pv_ungated -> pv_scalar1 -> pv_scalar2 -> pv_anneal        + pv_bigpk as compute control
A step that moves carry-over identifies a mechanism. If only pv_bigpk moves, the agents' advantage was
compute, and the honest conclusion is search rather than a transferable design principle.

Two things this must not get wrong.
  * Carry-over has a DIFFERENT DENOMINATOR per arm (each arm's own successes), so it is not a paired
    per-target quantity and a paired bootstrap over targets is not available for it. We therefore report an
    unpaired interval on the conditional rate, clustered by target, and separately report the count that
    generates it -- an arm that solves almost nothing can post a high carry-over on a handful of designs.
  * A carry-over gain is worthless if it is bought by collapsing the denominator. We therefore always print
    r_in beside it: a mechanism only counts if carry-over rises WITHOUT the success count collapsing.

Offline over attr_s{0..4}/sweep.json; no oracle calls.
"""
import json, math, os, random

random.seed(20260801)
HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1, 2, 3, 4]
CHAIN = ["panel_auto", "pv_ungated", "pv_scalar1", "pv_scalar2", "pv_anneal"]
ARMS = CHAIN + ["pv_bigpk"]
LABEL = {"panel_auto": "baseline (hand-built)", "pv_ungated": "+ ungated ProbKnot",
         "pv_scalar1": "+ scalar cost (sym)", "pv_scalar2": "+ scalar, FP x2",
         "pv_anneal": "+ annealing", "pv_bigpk": "3x pKiss allowance [compute control]"}
AGENT = (333, 1104, 0.302, 0.275, 0.330)     # arep_stats.txt, for reference only


def clopper(k, n, upper):
    if n == 0:
        return float("nan")
    try:
        from scipy.stats import beta
    except ImportError:
        return float("nan")
    if upper:
        return 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))


def load(s, need=60):
    p = os.path.join(HERE, f"attr_s{s}", "sweep.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    if len(d.get("rows") or []) < need:
        print(f"  (skipping incomplete attr_s{s}: {len(d.get('rows') or [])}/{need})")
        return None
    return d["rows"]


def cluster_ci(per_target, B=20000):
    """Bootstrap the conditional rate by resampling TARGETS (the clustering unit), carrying each drawn
    target's (successes, carried) pair together. Respects that one target contributes several seeds."""
    ids = list(per_target)
    if not ids:
        return float("nan"), float("nan")
    out = []
    for _ in range(B):
        num = den = 0
        for _ in ids:
            c, s_ = per_target[ids[random.randrange(len(ids))]]
            num += c; den += s_
        out.append(num / den if den else 0.0)
    out.sort()
    return out[int(0.025 * B)], out[int(0.975 * B)]


def main():
    data = {s: r for s in SEEDS if (r := load(s)) is not None}
    if not data:
        raise SystemExit("no attribution sweeps found")
    seeds = sorted(data)
    print(f"attribution chain, deterministic regime, seeds {seeds}, n=60 targets\n")
    print(f"  {'arm':38} {'r_in':>12} {'carry-over':>22} {'per-seed r_in':>20}")
    base = None
    for a in ARMS:
        per_t, k, n, rin_seed = {}, 0, 0, []
        for s in seeds:
            cnt = 0
            for r in data[s]:
                v = r["arms"].get(a) or {}
                if v.get("error"):
                    continue
                if v.get("solved") and v.get("seq"):
                    cnt += 1
                    carried = bool((v.get("panel") or {}).get("shapeknots"))
                    c0, s0 = per_t.get(r["id"], (0, 0))
                    per_t[r["id"]] = (c0 + int(carried), s0 + 1)
                    k += int(carried); n += 1
            rin_seed.append(cnt)
        lo, hi = cluster_ci(per_t)
        rate = k / n if n else float("nan")
        mark = ""
        if base is not None and not math.isnan(rate):
            mark = "  <-- moves" if lo > base[1] else ""
        if a == "panel_auto":
            base = (rate, hi)
        print(f"  {LABEL[a]:38} {sum(rin_seed)/len(rin_seed):6.1f}/60  "
              f"{k:4d}/{n:4d} = {rate:.3f} [{lo:.3f},{hi:.3f}]  {rin_seed}{mark}")
    print(f"\n  {'frozen agent operators (reference)':38} {'---':>10}  "
          f"{AGENT[0]:4d}/{AGENT[1]:4d} = {AGENT[2]:.3f} [{AGENT[3]:.3f},{AGENT[4]:.3f}]")
    print("\n  A mechanism counts only if carry-over rises AND r_in does not collapse: a high conditional")
    print("  rate on a handful of surviving designs is not an improvement. Read the two columns together.")

    # Realized oracle use, so the paper can state that each mechanism ENGAGED rather than assuming it. The
    # ungated variants should show ProbKnot calls rising to meet pKiss; the compute control should show pKiss
    # rising alone. A transplant that changed nothing measurable would be a bug, not a null.
    print("\n=== realized oracle use per target (mean over all cells) -- did each mechanism engage? ===")
    print(f"  {'arm':38} {'pKiss':>8} {'ProbKnot':>9} {'pb/pk':>7}")
    for a_ in ARMS:
        pk, pb = [], []
        for s in seeds:
            for r in data[s]:
                st = ((r["arms"].get(a_) or {}).get("stats")) or {}
                if st.get("n_pkiss") is not None:
                    pk.append(st["n_pkiss"]); pb.append(st.get("n_probknot") or 0)
        if pk:
            print(f"  {LABEL[a_]:38} {sum(pk)/len(pk):8.1f} {sum(pb)/len(pb):9.1f} "
                  f"{sum(pb)/max(1e-9,sum(pk)):7.2f}")


if __name__ == "__main__":
    main()
