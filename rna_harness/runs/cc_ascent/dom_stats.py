#!/usr/bin/env python3
"""R11, D3 seventh test, STAGE 2: does biasing toward ensemble dominance reproduce the gap?

Stage 1 (d3_structural.py) pre-stated six structural features of accepted designs and two gates -- (a) the
feature differs between the frozen agent-written arm and ours, (b) it predicts held-out carry-over WITHIN the
hand-built arm -- both adjusted for overall GC after the negative control showed the unadjusted gate admits
anything GC-correlated. Exactly one feature passed both: `decoy`, the number of off-target windows
complementary to the crossing stem's 5' arm (agent designs 0.003 per design, ours 0.060).

pv_dominant adds that count as a tiebreak in the acceptance rule and changes nothing else. It calls no oracle,
so a compute explanation is structurally impossible here, unlike every arm matched only on allowance.

THE RULE, WRITTEN BEFORE THE RUN AND IDENTICAL TO THE ONE EVERY EARLIER TRANSPLANT FACED:
    carry-over must RISE, and r_in must NOT collapse.
A conditional rate can always be raised by refusing to solve anything hard, so the two columns are read
together or not at all. Passing Stage 1's observational gate earned this arm a test; only the test decides.

Offline over ndc2_s{0..4}/sweep.json; no oracle calls.
"""
import json, math, os, random

random.seed(20260801)
HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1, 2, 3, 4]
ARMS = ["panel_auto", "pv_dominant"]
LABEL = {"panel_auto": "baseline (hand-built)",
         "pv_dominant": "+ ensemble-dominance bias [stage-2]"}
AGENT = (333, 1104, 0.302, 0.275, 0.330)


def cluster_ci(per_target, B=20000):
    """Percentile bootstrap of the conditional rate, resampling TARGETS -- the clustering unit, and the same
    unit the paper's transplant table uses, so this arm is graded on the standard the others were."""
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


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    m = min(b, c)
    return min(2 * sum(math.comb(n, k) * 0.5 ** n for k in range(m + 1)), 1.0)


def main():
    data = {}
    for s in SEEDS:
        p = os.path.join(HERE, f"dom_s{s}", "sweep.json")
        if os.path.exists(p):
            d = json.load(open(p))
            if len(d.get("rows") or []) >= 60:
                data[s] = d["rows"]
    if not data:
        raise SystemExit("no nodecoy sweeps found")
    seeds = sorted(data)
    print(f"D3 stage 2: ensemble-dominance transplant, deterministic regime, seeds {seeds}, n=60 targets\n")
    print(f"  {'arm':42} {'r_in':>10} {'carry-over':>24} {'real pKiss':>11}")
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
        if base is not None and not math.isnan(rate):
            mark = "  <-- MOVES" if lo > base[1] else "  (interval contains the baseline)"
        if a == "panel_auto":
            base = (rate, hi, sum(rin) / len(rin))
        print(f"  {LABEL[a]:42} {sum(rin)/len(rin):6.1f}/60  {k:3d}/{n:4d} = {rate:.3f} "
              f"[{lo:.3f},{hi:.3f}]  {sum(pk)/max(1,len(pk)):9.1f}{mark}")

    # Paired on the (target, seed) cells BOTH arms solved -- the same difficulty-matched design used for the
    # positive control, so the transplant is graded exactly as the thing it is trying to reproduce.
    nb = nc = npair = 0
    for s in seeds:
        for r in data[s]:
            A = r["arms"].get("panel_auto") or {}
            B = r["arms"].get("pv_dominant") or {}
            if not (A.get("solved") and A.get("seq") and B.get("solved") and B.get("seq")):
                continue
            a_sk = bool((A.get("panel") or {}).get("shapeknots"))
            b_sk = bool((B.get("panel") or {}).get("shapeknots"))
            npair += 1
            nb += int(b_sk and not a_sk)
            nc += int(a_sk and not b_sk)
    if npair:
        print(f"\n  paired on the {npair} (target, seed) cells both arms solved: "
              f"b={nb} (transplant only), c={nc} (baseline only), exact p={mcnemar(nb, nc):.4g}, "
              f"difference {(nb - nc)/npair:+.4f}")

    print(f"\n  {'frozen agent operators (the target to reproduce)':42} {'---':>10}  "
          f"{AGENT[0]:3d}/{AGENT[1]:4d} = {AGENT[2]:.3f} [{AGENT[3]:.3f},{AGENT[4]:.3f}]")
    if base:
        print(f"\n  Rule, fixed before the run: carry-over must RISE and r_in must hold near the baseline's")
        print(f"  {base[2]:.1f}/60. Stage 1 earned this arm a test; the test is what decides.")


if __name__ == "__main__":
    main()
