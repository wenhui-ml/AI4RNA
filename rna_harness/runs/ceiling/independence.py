#!/usr/bin/env python3
"""PRECONDITION for the S(B) metric: are trajectories on the same target independent?

The proposed unified metric is

    S(B) = max over {b_i} of  sum_i [ 1 - (1 - p_i)^{b_i} ]   s.t.  sum_i b_i c_i <= B

and its functional form is DERIVED, not fitted: independent seeds plus the benchmark's rule that a target
counts once per execution give exactly 1 - (1-p)^b. If seeds are not independent, that form over-predicts
and every number built on it is wrong.

There is already a warning sign. In the step-2 allocation run, wins per seed were UNDER-dispersed relative
to Poisson (var 0.38 against mean 0.53 at 20% budget; 0.60 against 1.73 at full budget). Under-dispersion
across seeds is what you see when the seeds share structure -- for instance if the skeleton initialisation
is not really re-randomised, so different seeds explore the same basin.

This tests it three ways on data already on disk:

  1. Per-target dispersion. If successes across the K seeds are Binomial(K, p_i), then across targets the
     total variance of the success counts equals sum_i K p_i (1-p_i). Compare observed to expected; the
     ratio is a dispersion index with an exact reference distribution obtained by simulation.
  2. Seed main effect. If seeds are exchangeable and independent given the target, no seed should be
     systematically luckier. A permutation test on the seed-level totals answers this directly.
  3. Nesting check on the trajectory experiment. delta_S2's seeds are a subset of delta_S4's, which are a
     subset of delta_S8's. Under independence the marginal gain from doubling trajectories must follow
     1 - (1-p)^{2b} against 1 - (1-p)^b. That is a direct prediction of the metric's functional form on
     data collected before the metric was written.

Test 3 is the strongest, because it checks the exact quantity the metric depends on rather than a proxy.
"""
import json, os
from collections import defaultdict
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)
B = 20000


# ---------------------------------------------------------------- data
gate = [json.load(open(os.path.join(H, "gate", f))) for f in os.listdir(os.path.join(H, "gate"))
        if f.endswith(".json")]
gate = [r for r in gate if "error" not in r]
win = defaultdict(dict)
for r in gate:
    win[r["arm"]].setdefault(r["id"], {})[r["seed"]] = int(bool((r.get("panel") or {}).get("unanimous")))

print("=== 1. per-target dispersion: is the success count Binomial(K, p_i)? ===")
print(f"  {'arm':8} {'targets':>8} {'K':>3} {'total wins':>11} {'obs var':>9} {'exp var':>9} "
      f"{'index':>7} {'95% null':>16} {'verdict':>10}")
for arm in ("off_150", "on_400"):
    W = win[arm]
    tids = sorted(W)
    K = len(next(iter(W.values())))
    cnt = np.array([sum(W[t].values()) for t in tids], float)
    p = cnt / K
    # dispersion index: observed variance of counts vs the Binomial expectation, conditional on p-hat.
    # p-hat is estimated from the same counts, so the null is obtained by simulation rather than chi2.
    obs = float(np.var(cnt, ddof=0))
    exp = float(np.mean(K * p * (1 - p)) + np.var(K * p, ddof=0))
    null = []
    for _ in range(B):
        sim = rng.binomial(K, p)
        null.append(float(np.var(sim, ddof=0)))
    lo, hi = np.percentile(null, [2.5, 97.5])
    verdict = "OK" if lo <= obs <= hi else ("UNDER" if obs < lo else "OVER")
    print(f"  {arm:8} {len(tids):8d} {K:3d} {int(cnt.sum()):11d} {obs:9.3f} {np.mean(null):9.3f} "
          f"{obs/max(np.mean(null),1e-9):7.2f} [{lo:6.3f},{hi:6.3f}] {verdict:>10}")
print("  (the count variance across targets is dominated by target heterogeneity, which the simulated")
print("   null reproduces by construction, so this test isolates the WITHIN-target part)")

print("\n=== 2. seed main effect: is any seed systematically luckier? ===")
for arm in ("off_150", "on_400"):
    W = win[arm]
    tids = sorted(W)
    seeds = sorted(next(iter(W.values())))
    M = np.array([[W[t][s] for s in seeds] for t in tids], float)     # targets x seeds
    obs = float(np.var(M.sum(axis=0), ddof=0))
    null = []
    for _ in range(B):
        P = np.array([rng.permutation(row) for row in M])
        null.append(float(np.var(P.sum(axis=0), ddof=0)))
    lo, hi = np.percentile(null, [2.5, 97.5])
    pv = float(np.mean(np.array(null) >= obs))
    print(f"  {arm:8} seed totals {M.sum(axis=0).astype(int).tolist()}  var {obs:.3f}  "
          f"null [{lo:.3f},{hi:.3f}]  p = {pv:.3f}"
          + ("   <- a seed effect would break exchangeability" if pv < 0.05 else ""))

print("\n=== 3. THE DIRECT TEST: does doubling trajectories follow 1-(1-p)^b? ===")
S = json.load(open(os.path.join(H, "seeds_shards.json")))["rows"]
S = [r for r in S if "error" not in r]
by = defaultdict(lambda: defaultdict(list))      # arm -> (rep,target) -> [win,...]
for r in S:
    if r["arm"].startswith("delta_S"):
        by[r["arm"]][(r["rep"], r["id"])].append(int(bool((r.get("panel") or {}).get("unanimous"))))

# p-hat per target from the S8 arm's 8 trajectories x 15 replicates = 120 independent seeds
p_hat = {}
for (rep, t), v in by["delta_S8"].items():
    p_hat.setdefault(t, []).extend(v)
p_hat = {t: float(np.mean(v)) for t, v in p_hat.items()}
print(f"  per-target p-hat from {sum(len(v) for v in by['delta_S8'].values())} S8 trajectories:")
for t in sorted(p_hat, key=lambda x: -p_hat[x])[:8]:
    print(f"    {t.split(':')[-1]:>4}  p-hat {p_hat[t]:.3f}")

print(f"\n  {'arm':9} {'b':>2} {'observed solves/rep':>20} {'predicted 1-(1-p)^b':>21} "
      f"{'obs/pred':>9} {'95% CI on obs':>16}")
for arm, b in (("delta_S2", 2), ("delta_S4", 4), ("delta_S8", 8)):
    reps = sorted({k[0] for k in by[arm]})
    obs = []
    for rp in reps:
        obs.append(sum(1 for (r2, t), v in by[arm].items() if r2 == rp and any(v)))
    obs = np.array(obs, float)
    tset = sorted({t for (_, t) in by[arm]})
    pred = sum(1.0 - (1.0 - p_hat.get(t, 0.0)) ** b for t in tset)
    bs = [obs[rng.integers(0, len(obs), len(obs))].mean() for _ in range(B)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    flag = "" if lo <= pred <= hi else "   <== MODEL OUTSIDE THE CI"
    print(f"  {arm:9} {b:2d} {obs.mean():20.2f} {pred:21.2f} {obs.mean()/max(pred,1e-9):9.2f} "
          f"[{lo:5.2f},{hi:5.2f}]{flag}")
print("  p-hat comes from the S8 arm, so S8's own row is near-tautological; S2 and S4 are the real test")
print("  because the model must EXTRAPOLATE DOWN from 8 trajectories to 2 and 4.")

print("\n=== verdict ===")
ok = True
for arm, b in (("delta_S2", 2), ("delta_S4", 4)):
    reps = sorted({k[0] for k in by[arm]})
    obs = np.array([sum(1 for (r2, t), v in by[arm].items() if r2 == rp and any(v)) for rp in reps], float)
    tset = sorted({t for (_, t) in by[arm]})
    pred = sum(1.0 - (1.0 - p_hat.get(t, 0.0)) ** b for t in tset)
    bs = [obs[rng.integers(0, len(obs), len(obs))].mean() for _ in range(B)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    if not (lo <= pred <= hi):
        ok = False
if ok:
    print("  The independence assumption survives the direct test: extrapolating down from 8 trajectories")
    print("  to 2 and 4 lands inside the observed intervals. S(B) = sum[1-(1-p)^b] can be built.")
else:
    print("  The independence assumption FAILS the direct test. 1-(1-p)^b does not extrapolate across")
    print("  trajectory counts on this data, so the metric needs a dispersion-corrected form (e.g. a")
    print("  beta-binomial mixture on p_i) before any curve is drawn.")
