"""E5' -- risk-constrained verifier routing on the NESTED ladder, with guarantees (elevates C5 from
"adaptive routing exists" to a theorem-backed result).

Setup: a nested, priced ladder of decidable verifiers  A_pfe(theta>1/2) subset A_uMFE subset A_match,
gold = the strongest rung we certify against (pfe). A workload of candidate (seq, target) pairs arrives;
for each we must output the GOLD verdict (accept/reject) as cheaply as possible.

THEOREM (soundness of the cheapest-first cascade, from nesting): run rungs cheap->gold; REJECT at the
first rung that rejects; escalate ACCEPTs. Because A_gold subset ... subset A_match, a cheap-rung REJECT
is a CERTAIN gold REJECT, so the cascade's verdict EQUALS gold's on every input (epsilon = 0), while
paying the gold cost only on candidates that pass all cheaper rungs. Saving vs always-gold = the gold
cost avoided on early-REJECTED candidates. (We also assert this empirically: 0 cases of cheap-reject &
gold-accept.)

RISK-CALIBRATED STOPPING (epsilon >= 0): to save more you may STOP-AND-ACCEPT at a cheaper rung k without
escalating, incurring a false-positive whenever gold would reject. We pick, on a CALIBRATION split, the
cheapest rung whose false-positive rate has a Clopper-Pearson UPPER bound <= epsilon at confidence 1-delta,
then apply it to the TEST split -- a finite-sample guarantee. The resulting E[cost]-vs-epsilon curve is the
verifier cost-risk frontier; when cheap rungs are highly gameable (RNA: ~90% FP) small-epsilon accept-
stopping is impossible, so robustness certification MUST pay the strong rung -- a quantified cost-of-
robustness, not a routing free lunch. Model-relative; the gold verdict is defined under the declared oracle.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import RNA

from ..benchmarks.eterna100 import load_eterna100
from ..cognitive.tools import verify

OUT_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "runs", "verifier_routing")
RUNGS = ["match", "umfe", "pfe"]          # cheap -> gold; gold = last
_BASES = "ACGU"


def _clopper_pearson_upper(k, n, delta=0.05):
    """Upper (1-delta) confidence bound on a binomial rate k/n. Uses scipy if available, else Hoeffding."""
    if n == 0:
        return 1.0
    try:
        from scipy.stats import beta
        return 1.0 if k == n else float(beta.ppf(1 - delta, k + 1, n - k))
    except Exception:
        import math
        return min(1.0, k / n + math.sqrt(math.log(1 / delta) / (2 * n)))    # Hoeffding fallback


def _solver_seq(target, rng, budget=4000):
    """A cheap local search to an exact-match (bp==0) solution, else best found."""
    n = len(target)
    pt, st = [], []
    for i, c in enumerate(target):
        if c == "(":
            st.append(i)
        elif c == ")":
            pt.append((st.pop(), i))
    x = [rng.choice(_BASES) for _ in range(n)]
    ps = [("G", "C"), ("C", "G"), ("A", "U"), ("U", "A"), ("G", "U"), ("U", "G")]
    for i, j in pt:
        a, b = rng.choice(ps)
        x[i], x[j] = a, b
    best, bestbp = "".join(x), RNA.bp_distance(RNA.fold("".join(x))[0], target)
    for _ in range(budget):
        c = list(x)
        if pt and rng.random() < 0.5:
            i, j = rng.choice(pt); a, b = rng.choice(ps); c[i], c[j] = a, b
        else:
            c[rng.randrange(n)] = rng.choice(_BASES)
        bp = RNA.bp_distance(RNA.fold("".join(c))[0], target)
        if bp <= bestbp:
            x, best, bestbp = c, "".join(c), bp
            if bp == 0:
                break
    return best


def build_workload(ids, rng, rand_per=8, sols_per=1, seed=0):
    """Per target: `sols_per` cheap-solver solutions (accept-side, some fragile) + rand_per random seqs
    (reject-side). Each candidate labelled with its (match,uMFE,pfe) verdict and per-rung verify wall cost.
    rand_per=0, sols_per>1 => accept-heavy workload for a non-degenerate risk frontier."""
    puz = {p["id"]: p for p in load_eterna100(version="V2")}
    W = []
    for pid in ids:
        t = puz[pid]["target"]
        n = len(t)
        cands = [_solver_seq(t, rng) for _ in range(sols_per)] + \
                ["".join(rng.choice(_BASES) for _ in range(n)) for _ in range(rand_per)]
        for s in cands:
            verd, cost = {}, {}
            for r in RUNGS:
                w = time.perf_counter()
                solved, _bp, _o = verify(s, t, strictness=r)
                cost[r] = time.perf_counter() - w
                verd[r] = bool(solved)
            W.append({"pid": pid, "verdict": verd, "cost": cost})
    return W


def sound_cascade(W):
    """Cheapest-first cascade to gold. Returns (E[cost], correctness vs gold, unsound_cases)."""
    gold = RUNGS[-1]
    total, correct, unsound = 0.0, 0, 0
    always_gold = sum(w["cost"][gold] for w in W) / len(W)
    for w in W:
        c, decided = 0.0, None
        for r in RUNGS:
            c += w["cost"][r]
            if not w["verdict"][r]:            # reject at rung r -> certain gold reject (nesting)
                decided = False
                if w["verdict"][gold]:          # would only happen if nesting violated
                    unsound += 1
                break
            if r == gold:
                decided = w["verdict"][gold]
        total += c
        correct += int(decided == w["verdict"][gold])
    return total / len(W), correct / len(W), unsound, always_gold


def risk_frontier(W, epsilons, delta=0.05, seed=0):
    """Split calib/test; for each epsilon pick cheapest STOP-AND-ACCEPT rung with CP-upper FP<=epsilon on
    calib, measure E[cost] and actual FP on test. FP(rung r) = accepts at r that gold rejects."""
    gold = RUNGS[-1]
    rng = random.Random(seed)
    idx = list(range(len(W)))
    rng.shuffle(idx)
    half = len(idx) // 2
    calib = [W[i] for i in idx[:half]]
    test = [W[i] for i in idx[half:]]

    def fp_rate_calib(r):
        acc = [w for w in calib if w["verdict"][r]]        # candidates this rung ACCEPTS
        if not acc:
            return 0, 0
        fp = sum(1 for w in acc if not w["verdict"][gold])  # ... that gold REJECTS
        return fp, len(acc)

    out = []
    for eps in epsilons:
        # cheapest rung (excluding gold) whose stop-and-accept FP upper-bound <= eps; else gold (sound)
        chosen = gold
        for r in RUNGS[:-1]:
            fp, nacc = fp_rate_calib(r)
            if _clopper_pearson_upper(fp, nacc, delta) <= eps:
                chosen = r
                break
        # apply on test: cheapest-first cascade, but STOP-AND-ACCEPT at `chosen`
        tot, err = 0.0, 0
        for w in test:
            c, decided = 0.0, None
            for r in RUNGS:
                c += w["cost"][r]
                if not w["verdict"][r]:
                    decided = False
                    break
                if r == chosen:                 # stop and accept here (no escalation past `chosen`)
                    decided = True
                    break
            tot += c
            err += int(decided != w["verdict"][gold])
        out.append({"epsilon": eps, "stop_rung": chosen, "E_cost_ms": round(1e3 * tot / len(test), 4),
                    "actual_fp": round(err / len(test), 4)})
    return out


def run(ids, outdir, seed=0, rand_per=8, sols_per=1):
    os.makedirs(outdir, exist_ok=True)
    rng = random.Random(seed)
    W = build_workload(ids, rng, rand_per=rand_per, sols_per=sols_per, seed=seed)
    gold = RUNGS[-1]
    acc_rate = sum(w["verdict"]["match"] for w in W) / len(W)
    gold_rate = sum(w["verdict"][gold] for w in W) / len(W)
    print(f"E5' routing: |workload|={len(W)} (match-accept {100*acc_rate:.0f}%, gold-accept {100*gold_rate:.0f}%)", flush=True)

    ec, corr, unsound, ag = sound_cascade(W)
    print(f"\n[Sound cascade, epsilon=0]  E[cost]={1e3*ec:.4f} ms  vs always-gold {1e3*ag:.4f} ms  "
          f"(saving {100*(1-ec/ag):.1f}%)  correctness={100*corr:.1f}%  unsound_cases={unsound} (must be 0)", flush=True)

    # per-rung stop-and-accept min-epsilon = the CP-upper FP bound = smallest epsilon that would let you
    # stop (accept) at that rung. Quantifies "how gameable" each cheap rung is.
    min_eps = {}
    print("\n[Per-rung stop-and-accept min-epsilon (Clopper-Pearson upper FP bound, delta=0.05)]", flush=True)
    for r in RUNGS[:-1]:
        acc = [w for w in W if w["verdict"][r]]
        fp = sum(1 for w in acc if not w["verdict"][gold])
        ub = _clopper_pearson_upper(fp, len(acc), 0.05)
        min_eps[r] = ub
        print(f"  stop@{r:5}: accepts={len(acc):>3}  FP={fp}/{len(acc)}={fp/max(1,len(acc)):.2f}  "
              f"min-epsilon(CP upper)={ub:.3f}", flush=True)

    print("\n[Risk-cost frontier]  epsilon -> stop-rung, E[cost], actual test FP:", flush=True)
    fr = risk_frontier(W, [0.0, 0.05, 0.1, 0.25, 0.5, 0.9, 0.95, 0.98, 0.99], delta=0.05, seed=seed)
    for r in fr:
        print(f"  eps<= {r['epsilon']:.2f} | stop@{r['stop_rung']:5} | E[cost]={r['E_cost_ms']:.4f} ms | "
              f"actual FP={r['actual_fp']:.3f}", flush=True)
    json.dump({"n": len(W), "match_accept": acc_rate, "gold_accept": gold_rate,
               "sound": {"E_cost_ms": 1e3 * ec, "always_gold_ms": 1e3 * ag, "correctness": corr,
                         "unsound": unsound}, "min_epsilon": min_eps, "frontier": fr},
              open(os.path.join(outdir, "routing.json"), "w"), indent=1)
    print("\nTakeaway: nesting gives a SOUND (eps=0) cost saving from cheap REJECTS; small-eps accept-"
          "stopping is blocked when cheap rungs are gameable -> robustness must pay the strong rung.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, nargs="+", default=None)
    ap.add_argument("--n", type=int, default=12, help="use the N SHORTEST Eterna targets (<= --max-n)")
    ap.add_argument("--max-n", type=int, default=60, help="cap target length (uMFE subopt is unbounded on "
                                                          "random seqs at large n)")
    ap.add_argument("--rand-per", type=int, default=8, help="random (reject-side) seqs per target; 0 = "
                                                            "accept-heavy workload for the risk frontier")
    ap.add_argument("--sols-per", type=int, default=1, help="solver solutions per target (raise for the "
                                                            "accept-heavy risk-frontier workload)")
    ap.add_argument("--outdir", default=OUT_DEFAULT)
    a = ap.parse_args()
    puz = load_eterna100(version="V2")
    if a.ids:
        ids = a.ids
    else:
        ids = [p["id"] for p in sorted(puz, key=lambda p: len(p["target"]))
               if len(p["target"]) <= a.max_n][: a.n]
    run(ids, a.outdir, rand_per=a.rand_per, sols_per=a.sols_per)
