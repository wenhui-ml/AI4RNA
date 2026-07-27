#!/usr/bin/env python3
"""Step 1, corrected: a CURE model, because 39 of 60 targets are not solvable by this operator at all.

hazard.py fitted a plain Weibull hazard and FAILED out-of-sample calibration: on ids 30-59, at the very
budgets already spent, it predicted 2.27 solves/cell against 1.40 observed -- a 62% over-prediction. The
diagnosis is structural, not numerical. A single smooth lambda(Delta) has to spread hazard over all 30
targets, but 20 of them are never solved in 15 executions. Summing p_i over targets that will never
succeed inflates the total no matter how lambda is fitted.

The standard survival tool for that is a cure (mixture) model: a target is either reachable by this
operator or it is not, and only reachable ones have a hazard.

    p_i(b) = pi_i * [ 1 - exp( -(lambda_i b)^k ) ],   logit pi_i = g0 + g1 z,   log lambda_i = b0 + b1 z

pi_i is exactly the quantity Delta is supposed to be a search-free proxy for -- "can this target be hit
at all" -- and lambda_i is "how fast once it can". Separating them is the whole point.

k = 1 is tested against k free. It matters for more than parsimony: at k = 1 the objective becomes
CONCAVE in b, so the Lagrangian allocation is exact rather than a relaxation,

    dp_i/db = pi_i lambda_i exp(-lambda_i b) = mu   =>   b_i = ln(pi_i lambda_i / mu) / lambda_i,
    funded iff  pi_i lambda_i > mu

and the abstention threshold is a threshold on the product pi_i lambda_i, fixed by the budget. Nothing
is swept. hazard.py's printed nu_crit = 9.6 was a boundary artifact of minimising on (1e-9, 50) when
k < 1 makes the true supremum infinite; the concave k = 1 case has no such pathology.

Fitted on ids 0-29. Everything reported for 30-59 is out-of-sample, including the calibration check,
which uses the sweep already run and costs no oracle calls.
"""
import json, math, os, sys
import numpy as np
from scipy.optimize import minimize, brentq
from scipy.stats import chi2

HERE = os.path.dirname(os.path.abspath(__file__))
ARM = os.environ.get("ARM", "panel_auto")
NDRAW = int(os.environ.get("NDRAW", "8"))     # declared on the fit split (pk_ev.log, ids 0-29 peak)
SPLIT = 30


def unpack(th, fix_k):
    if fix_k:
        g0, g1, b0, b1 = th
        return 1.0, g0, g1, b0, b1
    logk, g0, g1, b0, b1 = th
    return math.exp(logk), g0, g1, b0, b1


def nll(th, z, b, ev, fix_k):
    k, g0, g1, b0, b1 = unpack(th, fix_k)
    pi = 1.0 / (1.0 + np.exp(-(g0 + g1 * z)))
    pi = np.clip(pi, 1e-9, 1 - 1e-9)
    lam = np.exp(np.clip(b0 + b1 * z, -50, 20))
    u = np.clip(lam * b, 1e-12, None)
    uk = u ** k
    e = np.exp(-uk)
    # event: pi * f(b);  censored: (1-pi) + pi*exp(-(lam b)^k)
    lf = np.log(pi) + math.log(k) + np.log(lam) + (k - 1) * np.log(u) - uk
    ls = np.log(np.clip((1 - pi) + pi * e, 1e-300, None))
    return -float(np.sum(np.where(ev > 0, lf, ls)))


def fit(z, b, ev, fix_k=True, g1_free=True, b1_free=True, seed=0):
    rng = np.random.default_rng(seed)
    best = (None, -1e18)
    base = [0.0, -1.0, -2.0, math.log(1.0 / max(1.0, float(np.mean(b)))), 0.0]
    if fix_k:
        base = base[1:]
    for t in range(30):
        x0 = np.array(base) + (0 if t == 0 else rng.normal(0, 0.6, len(base)))
        def fn(x):
            v = np.array(x, float)
            if fix_k:
                if not g1_free: v[1] = 0.0
                if not b1_free: v[3] = 0.0
            else:
                if not g1_free: v[2] = 0.0
                if not b1_free: v[4] = 0.0
            return nll(v, z, b, ev, fix_k)
        r = minimize(fn, x0, method="Nelder-Mead",
                     options={"maxiter": 40000, "maxfev": 40000, "xatol": 1e-11, "fatol": 1e-11})
        if -r.fun > best[1]:
            v = np.array(r.x, float)
            if fix_k:
                if not g1_free: v[1] = 0.0
                if not b1_free: v[3] = 0.0
            else:
                if not g1_free: v[2] = 0.0
                if not b1_free: v[4] = 0.0
            best = (v, -r.fun)
    return best


def pi_lam(th, z, fix_k):
    k, g0, g1, b0, b1 = unpack(th, fix_k)
    return (1.0 / (1.0 + np.exp(-(g0 + g1 * z))), np.exp(np.clip(b0 + b1 * z, -50, 20)), k)


def psolve(pi, lam, k, b):
    return pi * (1.0 - np.exp(-((np.clip(lam * b, 0, None)) ** k)))


def nu_crit(k):
    """sup_{u>0} (1 - exp(-u^k)) / u.  Finite iff k > 1; +inf for k <= 1 (so nothing is ever cut)."""
    if k <= 1.0 + 1e-9:
        return float('inf')
    from scipy.optimize import minimize_scalar
    r = minimize_scalar(lambda u: -(1.0 - math.exp(-(u ** k))) / u,
                        bounds=(1e-6, 40.0), method="bounded", options={"xatol": 1e-12})
    return float(-r.fun)


def u_star(nu, k):
    """argmax_{u>=0} (1 - exp(-u^k)) - nu u.  Zero when no positive u beats u = 0."""
    if nu <= 0:
        return float('inf')
    from scipy.optimize import minimize_scalar
    r = minimize_scalar(lambda u: -((1.0 - math.exp(-(u ** k))) - nu * u),
                        bounds=(1e-9, max(60.0, 30.0 / nu)), method="bounded",
                        options={"xatol": 1e-11})
    return float(r.x) if -r.fun > 1e-14 else 0.0


def _g(u, k):
    """Marginal return in the u = lam*b variable: d/du [1 - exp(-u^k)] = k u^(k-1) exp(-u^k)."""
    return k * (u ** (k - 1)) * math.exp(-(u ** k))


def _u_peak(k):
    """argmax_u g(u).  For k > 1, g(0) = 0 and g rises then falls, so this is where g is maximal."""
    return 0.0 if k <= 1.0 else ((k - 1.0) / k) ** (1.0 / k)


def _u_upper(nu, k, up, gmax):
    """Largest root of g(u) = nu -- the CONCAVE branch, which is where a maximum can sit."""
    if nu >= gmax:
        return up
    hi = max(up * 2.0, 1.0)
    for _ in range(200):
        if _g(hi, k) < nu:
            break
        hi *= 1.6
    lo = up
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _g(mid, k) > nu:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def allocate(pi, lam, k, B):
    """Budget-FEASIBLE allocation for the cure model.

    The Lagrangian relaxation alone is not enough here. For k > 1, dp/db|_0 = 0, so u*(nu) jumps
    discontinuously from 0 as nu crosses nu_crit; spend(mu) is then a 30-step staircase and root-finding
    on it returns allocations that do not sum to B (an earlier version of this function reported a flat
    E[solves] = 2.65 across a 10x budget range, which is what exposed the bug). The duality gap is real,
    not numerical.

    Exact-given-the-nesting fix: the funded set is nested in the ranking by pi_i lam_i, so enumerate its
    size m = 1..n. For a FIXED funded set the interior KKT is solved on the concave branch of g,

        pi_i lam_i g(u_i) = mu,   u_i = largest root,   b_i = u_i / lam_i

    which is continuous and monotone in mu, so bisection hits sum(b) = B exactly. Any m whose MINIMUM
    feasible spend (all u_i at their peak) already exceeds B is infeasible and skipped -- pushing a
    target onto the convex branch is never better than funding one fewer target. Take the best m.
    """
    n = len(pi)
    r = pi * lam
    order = np.argsort(-r)
    up, gmax = _u_peak(k), (_g(_u_peak(k), k) if k > 1 else float('inf'))
    best = (np.zeros(n), 0.0, float('nan'), 0)
    for m in range(1, n + 1):
        S = order[:m]
        rs, ls = r[S], lam[S]
        # smallest spend this funded set can have: every u at its peak
        if k > 1 and float(np.sum(up / ls)) > B:
            continue
        def spend(mu):
            return float(np.sum([_u_upper(mu / ri, k, up, gmax) / li for ri, li in zip(rs, ls)]))
        hi = float(np.min(rs)) * gmax if k > 1 else float(np.min(rs))
        lo = hi
        for _ in range(500):
            if spend(lo) >= B:
                break
            lo *= 0.5
        mu = lo if spend(lo) < B else (hi if spend(hi) >= B else
                                      brentq(lambda x: spend(x) - B, lo, hi, xtol=1e-300,
                                             rtol=1e-13, maxiter=500))
        b = np.zeros(n)
        b[S] = np.array([_u_upper(mu / ri, k, up, gmax) / li for ri, li in zip(rs, ls)])
        tot = float(b.sum())
        if tot > B * (1 + 1e-6):                       # still infeasible: this m cannot be afforded
            continue
        es = float(np.sum(psolve(pi, lam, k, b)))
        if es > best[1]:
            best = (b, es, mu, m)
    return best[0], best[1], best[2]


def main():
    d = json.load(open(os.path.join(HERE, "pk_ev.json")))
    rows = {r["id"]: r for r in d["rows"]}
    delta = {t: float(min(r["draws_norm"][:NDRAW])) for t, r in rows.items()}
    idx = lambda t: int(t.split(":")[-1])
    att = [a for a in d["attempts"] if a["arm"] == ARM and a["n_pkiss"] > 0]
    ncells = d["n_cells"]
    dv = np.array(list(delta.values()))
    zm, zs = float(dv.mean()), float(dv.std() + 1e-12)
    z = lambda t: (delta[t] - zm) / zs

    def pack(sel):
        A = [a for a in att if sel(idx(a["id"]))]
        return (np.array([z(a["id"]) for a in A]), np.array([float(a["n_pkiss"]) for a in A]),
                np.array([a["unanimous"] for a in A]), A)

    zf, bf, ef, Af = pack(lambda i: i < SPLIT)
    zh, bh, eh, Ah = pack(lambda i: i >= SPLIT)
    print(f"arm={ARM}  Delta from N={NDRAW} draws (declared on ids 0-{SPLIT-1})")
    print(f"  fit   ids 0-{SPLIT-1}: {len(Af)} attempts, {int(ef.sum())} events, {ncells} cells")
    print(f"  held  ids {SPLIT}-59 : {len(Ah)} attempts, {int(eh.sum())} events")

    print("\n=== model selection on the fit split (nested LR tests) ===")
    m_full1, ll1 = fit(zf, bf, ef, fix_k=True)
    m_fullk, llk = fit(zf, bf, ef, fix_k=False)
    m_nog1, ll_g = fit(zf, bf, ef, fix_k=True, g1_free=False)
    m_nob1, ll_b = fit(zf, bf, ef, fix_k=True, b1_free=False)
    # no-cure comparison: pi == 1, i.e. plain Weibull (hazard.py's model), k = 1
    def nll_nocure(x):
        b0, b1 = x
        lam = np.exp(np.clip(b0 + b1 * zf, -50, 20))
        u = np.clip(lam * bf, 1e-12, None)
        return -float(np.sum(np.where(ef > 0, np.log(lam) - u, -u)))
    r_nc = minimize(nll_nocure, [math.log(1 / max(1., bf.mean())), 0.0], method="Nelder-Mead",
                    options={"maxiter": 40000, "xatol": 1e-11, "fatol": 1e-11})
    ll_nc = -r_nc.fun
    tests = [("cure vs no-cure (k=1)", 2 * (ll1 - ll_nc), 2),
             ("k free vs k=1", 2 * (llk - ll1), 1),
             ("Delta in pi (g1)", 2 * (ll1 - ll_g), 1),
             ("Delta in lambda (b1)", 2 * (ll1 - ll_b), 1)]
    for lab, lr, df in tests:
        print(f"  {lab:26} LR = {max(lr,0):8.2f} on {df} df   p = {chi2.sf(max(lr,0), df):.3g}")
    k_free = math.exp(m_fullk[0])
    print(f"  k when free = {k_free:.3f}")
    FIXK = 2 * (llk - ll1) < chi2.ppf(0.95, 1)
    th = m_full1 if FIXK else m_fullk
    print(f"  -> using {'k = 1 (concave, exact allocation)' if FIXK else 'k free'}")
    g0, g1, b0, b1 = (th if FIXK else th[1:])
    print(f"  logit pi = {g0:+.3f} {g1:+.3f} z      log lambda = {b0:+.3f} {b1:+.3f} z")

    # ---- out-of-sample calibration, the check hazard.py failed
    pih, lamh, kk = pi_lam(th, zh, FIXK)
    pred = float(np.sum(psolve(pih, lamh, kk, bh)))
    pif, lamf, _ = pi_lam(th, zf, FIXK)
    predf = float(np.sum(psolve(pif, lamf, kk, bf)))
    print(f"\n=== calibration on the sweep ALREADY RUN (no new oracle calls) ===")
    print(f"  in-sample  ids 0-{SPLIT-1} : predicted {predf:6.1f}  observed {int(ef.sum()):3d}  "
          f"per cell {predf/ncells:.2f} vs {ef.sum()/ncells:.2f}")
    print(f"  OUT-OF-SAMPLE ids {SPLIT}-59: predicted {pred:6.1f}  observed {int(eh.sum()):3d}  "
          f"per cell {pred/ncells:.2f} vs {eh.sum()/ncells:.2f}   "
          f"ratio {pred/max(1,eh.sum()):.2f}")
    print(f"  (hazard.py's no-cure model on the same data: 34.0 predicted vs 21 observed, ratio 1.62)")

    # per-target calibration, held out
    held = sorted([t for t in rows if idx(t) >= SPLIT], key=idx)
    print(f"\n  held-out per-target, Delta terciles:")
    zt = np.array([z(t) for t in held])
    pit, lamt, _ = pi_lam(th, zt, FIXK)
    bbar = np.array([np.mean([a["n_pkiss"] for a in Ah if a["id"] == t]) for t in held])
    obs = np.array([rows[t]["panel_win"] for t in held], float)
    order = np.argsort([delta[t] for t in held])
    for lab, sl in (("low Delta", order[:10]), ("mid", order[10:20]), ("high", order[20:])):
        p = psolve(pit[sl], lamt[sl], kk, bbar[sl]) * ncells
        print(f"    {lab:10} n={len(sl):2d}  predicted wins {p.sum():6.1f}  observed {obs[sl].sum():4.0f}"
              f"   median pi {np.median(pit[sl]):.3f}")

    # ---- the frontier, predicted on held-out targets only
    B0 = float(np.mean([sum(a["n_pkiss"] for a in Ah if a["cell"] == c)
                        for c in sorted({a["cell"] for a in Ah})]))
    obs_pc = eh.sum() / ncells
    ever = [t for t in held if rows[t]["panel_win"] > 0]
    cal = float(eh.sum()) / max(pred, 1e-9)          # out-of-sample calibration factor, < 1 = over-predicts
    print(f"\n=== PREDICTED FRONTIER, ids {SPLIT}-59 (n={len(held)}), B0 = {B0:.0f} calls/cell ===")
    print(f"  observed baseline {obs_pc:.2f} solves/cell; {len(ever)}/{len(held)} ever solved")
    print(f"  nu_crit(k={kk:.3f}) = {nu_crit(kk):.4f}  ->  fund iff pi_i lambda_i > mu / nu_crit")
    print(f"  LEVELS ARE INFLATED: multiply any E[solves] below by the measured OOS factor {cal:.2f}.")
    print(f"  The RATIO column is the pre-registered quantity; the levels are not trustworthy.")
    print(f"  {'B/B0':>6} {'B':>7} | {'funded':>6} {'E[slv]':>7} {'cal.':>6} | "
          f"{'unif E[slv]':>11} {'cal.':>6} | {'rank-m E[slv]':>13} | {'ratio':>6}")
    fr = []
    for frac in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00, 1.50, 2.00):
        B = B0 * frac
        b, es, mu = allocate(pit, lamt, kk, B)
        eu = float(np.sum(psolve(pit, lamt, kk, np.full(len(held), B / len(held)))))
        # model-light comparator: same number funded, but budget split EQUALLY among the funded
        m = max(1, int((b > 1e-12).sum()))
        keep = np.argsort(-(pit * lamt))[:m]
        br = np.zeros(len(held)); br[keep] = B / m
        er = float(np.sum(psolve(pit, lamt, kk, br)))
        fr.append((frac, B, es, eu, er, int(m)))
        print(f"  {frac:6.2f} {B:7.0f} | {m:6d} {es:7.2f} {es*cal:6.2f} | {eu:11.2f} {eu*cal:6.2f} | "
              f"{er:13.2f} | {es/max(eu,1e-9):6.2f}")

    tgt = float(np.sum(psolve(pit, lamt, kk, np.full(len(held), B0 / len(held)))))
    try:
        Bp = brentq(lambda B: allocate(pit, lamt, kk, B)[1] - tgt, B0 * 1e-4, B0, xtol=1e-6)
        dcost = NDRAW * len(held)
        print(f"\n  *** PRE-REGISTERED PREDICTION (ids {SPLIT}-59, panel-unanimous, N={NDRAW}) ***")
        print(f"  uniform allocation at B0 = {B0:.0f} yields {tgt:.2f} solves/cell (model).")
        print(f"  the derived policy reaches the same at {Bp:.0f} calls = {B0/Bp:.2f}x cheaper.")
        print(f"  charging Delta's own {dcost} calls: {B0/(Bp+dcost):.2f}x single-run, "
              f"{B0/(Bp+dcost/ncells):.2f}x amortised over {ncells} cells.")
    except Exception as e:
        Bp = float('nan')
        print(f"\n  equal-solves solve failed: {e}")

    # empirical skyline on the held-out set: spend only on targets that ever solved
    sky = float(np.mean([sum(a["n_pkiss"] for a in Ah if a["cell"] == c and rows[a["id"]]["panel_win"] > 0)
                         for c in sorted({a["cell"] for a in Ah})]))
    print(f"\n  bracket: uniform = 1.00x (null);  oracle skyline = {B0/sky:.2f}x "
          f"({sky:.0f} of {B0:.0f} calls);  policy predicted {B0/Bp:.2f}x")

    json.dump({"arm": ARM, "ndraw": NDRAW, "fix_k": bool(FIXK), "theta": list(map(float, th)),
               "g": [float(g0), float(g1)], "b": [float(b0), float(b1)], "k": float(kk),
               "B0": B0, "skyline": sky, "obs_per_cell": float(obs_pc),
               "pred_oos": pred, "obs_oos": int(eh.sum()),
               "equal_solves_target": tgt, "equal_solves_budget": float(Bp),
               "held": held, "pi": pit.tolist(), "lambda": lamt.tolist(),
               "frontier": fr},
              open(os.path.join(HERE, "cure.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
