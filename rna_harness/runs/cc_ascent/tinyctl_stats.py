#!/usr/bin/env python3
"""R9: the compute control run the OTHER way -- does a TIGHTER oracle budget raise carry-over?

Why. agent_budget.py measured the frozen agent-written operators at ~28 and ~13 realized pKiss calls per
target against the hand-built operator's 128 in the same regime, and kimi-k3's source sets an explicit
`budget = [30]`. That kills "the agents searched harder" and raises its inverse: a search forced to commit
early may accept candidates from a different region of sequence space, one the held-out predictor likes.
pv_tinypk is the hand-built operator at a 30-call pKiss allowance and nothing else changed.

Judged by the same rule as every other transplant, stated before the numbers were seen: a mechanism counts
only if carry-over RISES and r_in does NOT collapse. A tight budget will certainly cut r_in; if carry-over
rises only because the denominator fell, that is the artifact the rule exists to reject, not a mechanism.

Offline over tiny_s{0..4}/sweep.json; no oracle calls.
"""
import json, math, os, random

random.seed(20260801)
HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1, 2, 3, 4]
ARMS = ["panel_auto", "pv_tinypk", "pv_bigpk"]
LABEL = {"panel_auto": "baseline (150-call allowance)",
         "pv_tinypk": "30-call allowance [inverse control]",
         "pv_bigpk": "450-call allowance [compute control]"}
AGENT = (333, 1104, 0.302, 0.275, 0.330)


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


def cluster_ci(per_target, B=20000):
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
    data = {}
    for s in SEEDS:
        p = os.path.join(HERE, f"tiny_s{s}", "sweep.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        if len(d.get("rows") or []) >= 60:
            data[s] = d["rows"]
    if not data:
        raise SystemExit("no tinypk sweeps found")
    seeds = sorted(data)
    print(f"inverse compute control, deterministic regime, seeds {seeds}, n=60 targets\n")
    print(f"  {'arm':38} {'r_in':>10} {'carry-over':>22} {'real pKiss':>11}")
    base = None
    for a in ARMS:
        per_t, k, n, rin, pk = {}, 0, 0, [], []
        for s in seeds:
            cnt = 0
            for r in data[s]:
                v = r["arms"].get(a) or {}
                if v.get("error"):
                    continue
                st = v.get("stats") or {}
                if st.get("n_pkiss") is not None:
                    pk.append(st["n_pkiss"])
                if v.get("solved") and v.get("seq"):
                    cnt += 1
                    carried = bool((v.get("panel") or {}).get("shapeknots"))
                    c0, s0 = per_t.get(r["id"], (0, 0))
                    per_t[r["id"]] = (c0 + int(carried), s0 + 1)
                    k += int(carried); n += 1
            rin.append(cnt)
        lo, hi = cluster_ci(per_t)
        rate = k / n if n else float("nan")
        mark = ""
        if base is not None and not math.isnan(rate) and lo > base[1]:
            mark = "  <-- carry-over MOVES"
        if a == "panel_auto":
            base = (rate, hi, sum(rin) / len(rin))
        print(f"  {LABEL[a]:38} {sum(rin)/len(rin):6.1f}/60  {k:3d}/{n:4d} = {rate:.3f} "
              f"[{lo:.3f},{hi:.3f}]  {sum(pk)/max(1,len(pk)):9.1f}{mark}")
    print(f"\n  {'frozen agent operators (the target)':38} {'---':>10}  "
          f"{AGENT[0]:3d}/{AGENT[1]:4d} = {AGENT[2]:.3f} [{AGENT[3]:.3f},{AGENT[4]:.3f}]")
    if base:
        print(f"\n  Rule, fixed before the numbers: a mechanism counts only if carry-over rises AND r_in holds")
        print(f"  near the baseline's {base[2]:.1f}/60. A conditional rate bought by collapsing its own")
        print("  denominator is the artifact this column exists to catch, not a finding.")


if __name__ == "__main__":
    main()
