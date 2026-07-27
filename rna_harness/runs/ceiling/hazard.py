#!/usr/bin/env python3
"""Step 1: derive the allocation policy instead of sweeping a threshold, and PREDICT the frontier.

A budget B of oracle calls is split across targets. Spending b calls on target T succeeds with
probability p(T,b). Model the search as a Weibull process in CALLS:

    p(T, b) = 1 - exp( -(lambda_T b)^k ),      log lambda_T = beta0 + beta1 * z(Delta_T)

k = 1 is a memoryless search (constant per-call hazard); k > 1 means the search needs a warm-up before
it can succeed; k < 1 means it exhausts. The 15-cell sweep gives 900 attempts, each an EXACT event time
(the loop stops on success, so a win's n_pkiss IS the time-to-event) or a right-censored observation
(a loss burned its whole wall-clock allowance). That is textbook censored survival data.

THE ALLOCATION IS THEN SOLVED, NOT SWEPT. Lagrangian relaxation of

    max_b  sum_i p_i(b_i)   s.t.  sum_i b_i <= B

is separable: b_i(mu) = argmax_{b>=0} [ p_i(b) - mu b ]. Substituting u = lambda_i b makes the inner
problem depend on lambda_i only through nu_i = mu / lambda_i:

    h(u; nu) = 1 - exp(-u^k) - nu u,      b_i = u*(nu_i) / lambda_i

For k > 1 the objective is NOT concave (h'(0) = 0), so u = 0 is always a stationary point and funding a
target is worthwhile only when max_u h > 0. That happens iff

    nu_i < nu_crit(k) = max_{u>0} (1 - exp(-u^k)) / u        i.e.   lambda_i > mu / nu_crit(k)

**The abstention threshold is a threshold on lambda -- hence on Delta -- and mu is fixed by the budget.**
It is derived from the fitted hazard, not tuned on the outcome. Funded targets also do NOT get equal
budget: b_i = u*(nu_i)/lambda_i gives easier targets fewer calls, which is a second, sharper prediction
than "abstain and split the rest evenly".

Declared split (METHOD.md 3): the model is fitted on ids 0-29 and everything reported for 30-59 is
out-of-sample. Free validation: the stored sweep IS the uniform-allocation arm, so predicted held-out
solve counts can be checked against observed ones with zero new oracle calls.
"""
import json, math, os, sys
import numpy as np
from scipy.optimize import minimize, minimize_scalar, brentq
from scipy.stats import chi2

HERE = os.path.dirname(os.path.abspath(__file__))
ARM = os.environ.get("ARM", "panel_auto")
NDRAW = int(os.environ.get("NDRAW", "5"))          # Delta from N draws; set from the pk_ev AUC curve
SPLIT = 30                                          # ids 0-29 fit, 30-59 held out


# ---------------------------------------------------------------- Weibull censored MLE
def nll(theta, z, b, ev):
    """Negative log-likelihood. Wins contribute log f(b), losses log S(b)."""
    logk, b0, b1 = theta
    k = math.exp(logk)
    lam = np.exp(b0 + b1 * z)
    u = np.clip(lam * b, 1e-12, None)
    uk = u ** k
    ll = np.where(ev > 0, np.log(k) + np.log(lam) + (k - 1) * np.log(u) - uk, -uk)
    return -float(np.sum(ll))


def fit(z, b, ev, free_b1=True):
    x0 = np.array([0.0, math.log(1.0 / max(1.0, float(np.mean(b)))), 0.0])
    fn = (lambda t: nll(t, z, b, ev)) if free_b1 else \
         (lambda t: nll(np.array([t[0], t[1], 0.0]), z, b, ev))
    x0 = x0 if free_b1 else x0[:2]
    r = minimize(fn, x0, method="Nelder-Mead",
                 options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-10})
    r2 = minimize(fn, r.x, method="Nelder-Mead",
                  options={"maxiter": 20000, "xatol": 1e-12, "fatol": 1e-12})
    th = r2.x if r2.fun <= r.fun else r.x
    if not free_b1:
        th = np.array([th[0], th[1], 0.0])
    return th, -min(r.fun, r2.fun)


# ---------------------------------------------------------------- the derived policy
def u_star(nu, k):
    """argmax_{u>=0} 1 - exp(-u^k) - nu u.  Returns 0 when funding is not worthwhile."""
    if nu <= 0:
        return float('inf')
    f = lambda u: -((1.0 - math.exp(-(u ** k))) - nu * u)
    r = minimize_scalar(f, bounds=(1e-9, max(50.0, 20.0 / nu)), method="bounded",
                        options={"xatol": 1e-10})
    return float(r.x) if -r.fun > 0 else 0.0


def nu_crit(k):
    """max_{u>0} (1 - exp(-u^k)) / u -- above this nu, a target is not worth any budget."""
    g = lambda u: -(1.0 - math.exp(-(u ** k))) / u
    r = minimize_scalar(g, bounds=(1e-9, 50.0), method="bounded", options={"xatol": 1e-12})
    return float(-r.fun)


def allocate(lam, k, B):
    """Budget-feasible Lagrangian allocation. Returns (b, expected_solves, mu)."""
    nc = nu_crit(k)
    def spend(mu):
        return float(np.sum([u_star(mu / L, k) / L for L in lam]))
    lo, hi = 1e-12, max(lam) * nc * 0.999999
    if spend(hi) > B:                 # even the stingiest mu overspends: cannot happen, but guard
        mu = hi
    elif spend(lo) <= B:
        mu = lo
    else:
        mu = brentq(lambda m: spend(m) - B, lo, hi, xtol=1e-14, rtol=1e-12, maxiter=500)
    b = np.array([u_star(mu / L, k) / L for L in lam])
    p = 1.0 - np.exp(-((lam * b) ** k))
    return b, float(np.sum(p)), mu


def main():
    d = json.load(open(os.path.join(HERE, "pk_ev.json")))
    delta = {r["id"]: float(min(r["draws_norm"][:NDRAW])) for r in d["rows"]}
    nlen = {r["id"]: r["n"] for r in d["rows"]}
    ids = [r["id"] for r in d["rows"]]
    idx = {t: int(t.split(":")[-1]) for t in ids}
    att = [a for a in d["attempts"] if a["arm"] == ARM and a["n_pkiss"] > 0]
    ncells = d["n_cells"]
    print(f"arm={ARM}  N_draws for Delta={NDRAW}  {len(att)} attempts over {ncells} cells")

    zall = np.array([delta[t] for t in ids])
    zm, zs = float(zall.mean()), float(zall.std() + 1e-12)
    z = lambda t: (delta[t] - zm) / zs

    def pack(sel):
        A = [a for a in att if sel(idx[a["id"]])]
        return (np.array([z(a["id"]) for a in A]), np.array([float(a["n_pkiss"]) for a in A], float),
                np.array([a["unanimous"] for a in A]), A)

    zf, bf, ef, Af = pack(lambda i: i < SPLIT)
    zh, bh, eh, Ah = pack(lambda i: i >= SPLIT)
    print(f"  fit split   ids 0-{SPLIT-1}: {len(Af)} attempts, {int(ef.sum())} events")
    print(f"  held out    ids {SPLIT}-59 : {len(Ah)} attempts, {int(eh.sum())} events")
    if ef.sum() < 5:
        print("  !! too few events in the fit split for a 3-parameter model; reporting anyway with CIs")

    th, ll1 = fit(zf, bf, ef, True)
    th0, ll0 = fit(zf, bf, ef, False)
    k, b0, b1 = math.exp(th[0]), th[1], th[2]
    lr = 2 * (ll1 - ll0)
    print(f"\n=== fitted hazard (ids 0-{SPLIT-1}) ===")
    print(f"  k = {k:.3f}   beta0 = {b0:.4f}   beta1(Delta) = {b1:+.4f}")
    print(f"  LR test, Delta vs no-Delta: LR = {lr:.2f} on 1 df, p = {chi2.sf(max(lr,0),1):.3g}")
    print(f"  k interpretation: {'warm-up needed, abstention is optimal' if k > 1.05 else ('memoryless' if k > 0.95 else 'search exhausts')}")
    nc = nu_crit(k)
    print(f"  nu_crit(k) = {nc:.4f}  ->  fund target i iff lambda_i > mu / {nc:.4f}")

    # bootstrap over CELLS (the unit the project's own execution-variance result says is exchangeable)
    cells = sorted({a["cell"] for a in Af})
    rng = np.random.default_rng(0)
    ks, b1s = [], []
    for _ in range(200):
        pick = rng.choice(cells, len(cells))
        A = [a for c in pick for a in Af if a["cell"] == c]
        if sum(a["unanimous"] for a in A) < 3:
            continue
        t, _ = fit(np.array([z(a["id"]) for a in A]),
                   np.array([float(a["n_pkiss"]) for a in A]),
                   np.array([a["unanimous"] for a in A]), True)
        ks.append(math.exp(t[0])); b1s.append(t[2])
    if ks:
        print(f"  cell-bootstrap 95% CI:  k [{np.percentile(ks,2.5):.2f}, {np.percentile(ks,97.5):.2f}]"
              f"   beta1 [{np.percentile(b1s,2.5):+.2f}, {np.percentile(b1s,97.5):+.2f}]  (B={len(ks)})")

    # ---------------- free out-of-sample check: predict the held-out sweep we already ran
    lam_h = np.exp(b0 + b1 * zh)
    pred = float(np.sum(1.0 - np.exp(-((lam_h * bh) ** k))))
    print(f"\n=== out-of-sample check, ids {SPLIT}-59 under the allocation ALREADY RUN ===")
    print(f"  predicted solves {pred:.1f}   observed {int(eh.sum())}   "
          f"({len(Ah)} attempts, {ncells} cells)")
    print(f"  per cell: predicted {pred/ncells:.2f}/30   observed {eh.sum()/ncells:.2f}/30")

    # ---------------- the frontier, PREDICTED before the experiment
    held = [t for t in ids if idx[t] >= SPLIT]
    lam = np.exp(np.array([b0 + b1 * z(t) for t in held]))
    # baseline: what the stored sweep actually spent on these targets, per cell
    B0 = float(np.mean([sum(a["n_pkiss"] for a in Ah if a["cell"] == c) for c in sorted({a["cell"] for a in Ah})]))
    obs = eh.sum() / ncells
    ever = {t for t in held if [r for r in d["rows"] if r["id"] == t][0]["panel_win"] > 0}
    print(f"\n=== PREDICTED FRONTIER, ids {SPLIT}-59 (n={len(held)}), baseline B0 = {B0:.0f} calls/cell ===")
    print(f"  observed baseline: {obs:.2f} solves/cell;  {len(ever)}/{len(held)} targets ever solved")
    print(f"  {'B/B0':>6}  {'B':>7}  {'funded':>7}  {'E[solves]':>10}  {'vs baseline':>12}")
    for frac in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00, 1.50, 2.00):
        B = B0 * frac
        b, es, mu = allocate(lam, k, B)
        print(f"  {frac:6.2f}  {B:7.0f}  {int((b>1e-9).sum()):7d}  {es:10.2f}  {es-obs:+12.2f}")
    print("  uniform-allocation comparator at the same budgets:")
    print(f"  {'B/B0':>6}  {'b each':>7}  {'E[solves]':>10}")
    for frac in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00, 2.00):
        bb = B0 * frac / len(held)
        es = float(np.sum(1.0 - np.exp(-((lam * bb) ** k))))
        print(f"  {frac:6.2f}  {bb:7.1f}  {es:10.2f}")

    # equal-solves cost, the headline number
    def solves_policy(B):
        return allocate(lam, k, B)[1]
    def solves_uniform(B):
        return float(np.sum(1.0 - np.exp(-((lam * (B / len(held))) ** k))))
    tgt = solves_uniform(B0)
    try:
        Bp = brentq(lambda B: solves_policy(B) - tgt, B0 * 0.005, B0, xtol=1e-6)
        print(f"\n  PREDICTION: policy reaches the uniform baseline's {tgt:.2f} solves at "
              f"{Bp:.0f} calls = {B0/Bp:.2f}x cheaper (before charging Delta's own "
              f"{NDRAW*len(held)} calls -> {B0/(Bp+NDRAW*len(held)):.2f}x all-in)")
    except Exception as e:
        print(f"\n  equal-solves solve failed: {e}")

    json.dump({"arm": ARM, "ndraw": NDRAW, "k": k, "beta0": b0, "beta1": b1,
               "nu_crit": nc, "lr": lr, "B0": B0, "obs_per_cell": obs,
               "held_ids": held, "lambda": lam.tolist(),
               "k_ci": [float(np.percentile(ks,2.5)), float(np.percentile(ks,97.5))] if ks else None,
               "beta1_ci": [float(np.percentile(b1s,2.5)), float(np.percentile(b1s,97.5))] if ks else None},
              open(os.path.join(HERE, "hazard.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
