#!/usr/bin/env python3
"""What in the compatible set actually predicts failure -- and what does not.

ev_margin.py produced three results that need following up, two of which refute earlier work on this
branch:

  * The length--sample-size confound is real but NEGLIGIBLE: 0.867 (length-dependent N) vs 0.870 (fixed
    N = 100) on identical draws. I predicted the corrected number might be lower; it is not.
  * Delta is N-cheap: AUC 0.862 at N = 5 vs 0.875 at N = 200. That is what makes an efficiency result
    possible at all, since the predictor's own oracle calls are charged to the budget.
  * Extrapolating to the true infimum Delta* makes the predictor WORSE (0.818). So what predicts our
    optimizer's failure is not the undesignability infimum but the BULK of the compatible set.

That last point says the estimator should be a quantile, not a minimum. This script tests it, checks
whether Delta is merely length in disguise, and reports whether the slack / information-budget framing
in info_budget.py survives (it does not).

Reads ev_margin.json only. No new oracle calls.
"""
import json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RT = 0.0019872 * 310.15
BIT = RT * math.log(2)


def auc(pos, neg):
    pos, neg = list(pos), list(neg)
    if not pos or not neg:
        return float('nan')
    w = sum(1 if a > b else 0.5 if a == b else 0 for a in pos for b in neg)
    return w / (len(pos) * len(neg))


def auc_ci(pos, neg, B=2000, seed=0):
    """Bootstrap CI, resampling targets within each outcome class."""
    rng = np.random.default_rng(seed)
    p, n = np.asarray(pos, float), np.asarray(neg, float)
    vals = [auc(rng.choice(p, len(p)), rng.choice(n, len(n))) for _ in range(B)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def logit_fit(X, y, iters=200):
    """Newton-Raphson logistic regression with an intercept. Returns (beta, loglik)."""
    X = np.column_stack([np.ones(len(y)), np.asarray(X, float).reshape(len(y), -1)])
    y = np.asarray(y, float)
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-X @ b))
        p = np.clip(p, 1e-10, 1 - 1e-10)
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + 1e-8 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        b = b + step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = np.clip(1.0 / (1.0 + np.exp(-X @ b)), 1e-10, 1 - 1e-10)
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return b, ll


def main():
    rows = json.load(open(os.path.join(HERE, "ev_margin.json")))
    for r in rows:
        r['g'] = np.asarray(r['gaps'], float)
    U = [r for r in rows if not r['solved']]
    S = [r for r in rows if r['solved']]
    print(f"{len(rows)} Eterna100 targets with outcomes: {len(U)} unsolved, {len(S)} solved\n")

    print("=== 1. minimum or quantile? which functional of gap over S(T) predicts failure ===")
    print("    (all from the same 200 stored draws; 'cost' is oracle calls needed to estimate it)")
    print(f"    {'statistic':>18}  {'AUC':>6}  {'95% CI':>16}  {'min cost':>9}")
    cands = [("min (Delta-hat_200)", lambda g: g.min(), 200),
             ("q01", lambda g: np.quantile(g, .01), 100),
             ("q05", lambda g: np.quantile(g, .05), 20),
             ("q10", lambda g: np.quantile(g, .10), 10),
             ("q25", lambda g: np.quantile(g, .25), 4),
             ("median", lambda g: np.quantile(g, .50), 2),
             ("q75", lambda g: np.quantile(g, .75), 4),
             ("mean", lambda g: g.mean(), 2),
             ("sd", lambda g: g.std(), 3),
             ("mean - min", lambda g: g.mean() - g.min(), 200)]
    best = None
    for lab, f, cost in cands:
        pu, ps = [f(r['g']) for r in U], [f(r['g']) for r in S]
        a = auc(pu, ps); lo, hi = auc_ci(pu, ps)
        print(f"    {lab:>18}  {a:6.3f}  [{lo:.3f}, {hi:.3f}]  {cost:9d}")
        if best is None or a > best[1]:
            best = (lab, a, f)
    print(f"    -> best single statistic: {best[0]} at AUC {best[1]:.3f}")

    print("\n=== 2. is Delta just length in disguise? ===")
    q = lambda r: float(np.quantile(r['g'], .10))
    ln = [r['n'] for r in rows]
    ter = [np.quantile(ln, 1 / 3), np.quantile(ln, 2 / 3)]
    print(f"    AUC(length -> failure)          = {auc([r['n'] for r in U], [r['n'] for r in S]):.3f}")
    print(f"    AUC(q10   -> failure)           = {auc([q(r) for r in U], [q(r) for r in S]):.3f}")
    print(f"    AUC(q10/n -> failure)           = {auc([q(r)/r['n'] for r in U], [q(r)/r['n'] for r in S]):.3f}")
    print("    within length terciles (Delta cannot borrow length there):")
    for lab, sel in ((f"n <= {ter[0]:.0f}", lambda r: r['n'] <= ter[0]),
                     (f"{ter[0]:.0f} < n <= {ter[1]:.0f}", lambda r: ter[0] < r['n'] <= ter[1]),
                     (f"n > {ter[1]:.0f}", lambda r: r['n'] > ter[1])):
        g = [r for r in rows if sel(r)]
        u = [r for r in g if not r['solved']]; s = [r for r in g if r['solved']]
        if not u or not s:
            print(f"      {lab:18} n={len(g):3d}  (degenerate: {len(u)} unsolved)")
            continue
        print(f"      {lab:18} n={len(g):3d} unsolved={len(u):2d}  "
              f"AUC(q10) = {auc([q(r) for r in u], [q(r) for r in s]):.3f}   "
              f"AUC(len) = {auc([r['n'] for r in u], [r['n'] for r in s]):.3f}")

    y = [0 if r['solved'] else 1 for r in rows]
    zn = lambda v: (np.asarray(v, float) - np.mean(v)) / (np.std(v) + 1e-12)
    _, ll0 = logit_fit(zn(ln), y)
    b1, ll1 = logit_fit(np.column_stack([zn(ln), zn([q(r) for r in rows])]), y)
    lr = 2 * (ll1 - ll0)
    from scipy.stats import chi2
    print(f"    likelihood-ratio test, adding q10 to a length-only model:")
    print(f"      LR = {lr:.2f} on 1 df, p = {chi2.sf(lr, 1):.2e}   "
          f"beta(length) {b1[1]:+.2f}  beta(q10) {b1[2]:+.2f}")

    print("\n=== 3. does the slack / information-budget framing survive? ===")
    print("    slack(T) = H_avail - Delta/(RT ln2). info_budget.py frames this as the point.")
    for lab, sel in (("n < 150", lambda r: r['n'] < 150), ("n >= 150", lambda r: r['n'] >= 150),
                     ("pooled", lambda r: True)):
        g = [r for r in rows if sel(r)]
        u = [r for r in g if not r['solved']]; s = [r for r in g if r['solved']]
        sl = lambda r: r['h_avail_bits'] - float(np.quantile(r['g'], .10)) / BIT
        a = auc([-sl(r) for r in u], [-sl(r) for r in s])
        print(f"    {lab:9} AUC(low slack -> failure) = {a:.3f}")
    neg = sum(1 for r in rows if r['h_avail_bits'] - float(np.quantile(r['g'], .10)) / BIT < 0)
    print(f"    targets with NEGATIVE slack, the whole point of the framing: {neg}/{len(rows)}")
    print("    VERDICT: slack pools to 0.355 and stratifies to ~0.53-0.59, and no target ever violates")
    print("    the budget. H_avail scales with length and swamps Delta. The bits framing is dropped;")
    print("    raw Delta is the coordinate that works.")

    print("\n=== 4. the estimator to carry forward ===")
    for N in (3, 5, 8, 10, 20, 40):
        # q10 of the FIRST N draws, i.e. what N oracle calls actually buy
        pu = [float(np.quantile(r['g'][:N], .10)) for r in U]
        ps = [float(np.quantile(r['g'][:N], .10)) for r in S]
        mu = [float(r['g'][:N].min()) for r in U]
        ms = [float(r['g'][:N].min()) for r in S]
        du = [float(np.median(r['g'][:N])) for r in U]
        ds = [float(np.median(r['g'][:N])) for r in S]
        print(f"    N={N:3d}  AUC(min)={auc(mu, ms):.3f}  AUC(q10)={auc(pu, ps):.3f}  "
              f"AUC(median)={auc(du, ds):.3f}")


if __name__ == "__main__":
    main()
