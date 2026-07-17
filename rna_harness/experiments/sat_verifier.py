"""E6' -- SAT as Paper D's SECOND deterministic domain: the same verifier cost-robustness / Goodhart
structure, in a domain unrelated to RNA, to show the phenomenon is a property of cheap-proxy verification,
not an RNA artifact. See paper/E6_SAT_second_domain.md for design + the anti-collision boundary vs family
B (B pins correctness ungameable to study COMPOUNDING; we DIAL verifier strength to study gaming/cost --
opposite sides of the verifier axis; and we sharpen "exact != robustness-ungameable").

Verifier ladder (mirror of RNA exact-match / uMFE / P(T)>=theta):
  V0(x)      : x satisfies phi                         (cheap, O(m); the proxy)
  rho(x)     : fraction of single-variable flips that KEEP x satisfying  (continuous robustness in [0,1])
  V_rob(x,t) : x satisfies phi AND rho(x) >= t         (dial-able strong rung; t is the knob)
Hardness knob = clause density alpha = m/n (3-SAT threshold ~4.267). Satisfiability is confirmed by a
WITNESS (a found satisfying assignment proves SAT), so no complete-solver dependency. Model-relative,
decidable, priced. No claim to own SAT solution-robustness (established statistical-physics of SAT: clusters,
frozen variables, backbone) -- we cite it; the contribution is the cross-domain verifier-economics framing.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

OUT_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "runs", "sat_verifier")


# ---- instances + evaluation -----------------------------------------------------------------------
def gen_3sat(n, alpha, rng):
    m = int(round(alpha * n))
    clauses = []
    for _ in range(m):
        vs = rng.sample(range(1, n + 1), 3)
        clauses.append(tuple(v if rng.random() < 0.5 else -v for v in vs))
    return clauses


def _lit_true(lit, x):
    return x[lit] == 1 if lit > 0 else x[-lit] == 0


def sat_count(clauses, x):
    return sum(1 for c in clauses if any(_lit_true(l, x) for l in c))


def slack_count(clauses, x):
    """# clauses satisfied by >=2 literals (flip-robust clauses) -- a cheap O(m) robustness proxy."""
    return sum(1 for c in clauses if sum(_lit_true(l, x) for l in c) >= 2)


def satisfies(clauses, x):
    return all(any(_lit_true(l, x) for l in c) for c in clauses)


def rho(clauses, x, n):
    """Single-flip satisfying fraction: of the n single-variable flips, how many keep x satisfying."""
    ok = 0
    for v in range(1, n + 1):
        x[v] ^= 1
        if satisfies(clauses, x):
            ok += 1
        x[v] ^= 1
    return ok / n


# ---- fixed proposer; only the score differs (mirror of RNA E2') -----------------------------------
def optimize(clauses, n, arm, budget, seed, x0=None):
    rng = random.Random(seed)
    x = list(x0) if x0 is not None else [0] + [rng.randint(0, 1) for _ in range(n)]   # x[0] unused
    m = len(clauses)

    def score(xx):
        sc = sat_count(clauses, xx)
        if arm == "cheap":
            return sc * (m + 1)                          # only satisfaction; slack invisible (proxy)
        return sc * (m + 1) + slack_count(clauses, xx)   # satisfaction + margin (robustness-aware)

    cur = score(x)
    best_x, best = list(x), cur
    for step in range(budget):
        T = max(0.05, 1.0 * (1.0 - step / budget))
        v = rng.randint(1, n)
        x[v] ^= 1
        cand = score(x)
        if cand >= cur or rng.random() < math.exp((cand - cur) / (T * (m + 1))):
            cur = cand
            if cand > best:
                best_x, best = list(x), cand
        else:
            x[v] ^= 1                                    # reject: undo
    return best_x


# ---- SAT instance pool (witness-confirmed satisfiable) --------------------------------------------
def _sat_pool(n, alpha, k, rng, witness_budget=60000):
    """k satisfiable 3-SAT instances at density alpha, each with a witness assignment proving SAT.
    Returns list of (clauses, witness). SAT is certified by the found assignment (no complete solver)."""
    pool = []
    tries = 0
    while len(pool) < k and tries < k * 30:
        tries += 1
        cl = gen_3sat(n, alpha, rng)
        w = optimize(cl, n, "cheap", witness_budget, seed=rng.randint(1, 10 ** 9))
        if satisfies(cl, w):
            pool.append((cl, w))
    return pool


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float("nan")


# ---- E6'-frontier: fragility of cheap-search solutions vs hardness alpha (mirror E1') -------------
def run_frontier(n, alphas, k, outdir, theta=0.3, seed=0, budget=20000):
    os.makedirs(outdir, exist_ok=True)
    rng = random.Random(seed)
    print(f"E6' SAT frontier: n={n}, alphas={alphas}, k={k}/alpha, theta={theta}, cheap-budget={budget} "
          f"(median rho conditioned on SAT to avoid solve-failure contamination)", flush=True)
    rows = []
    for a in alphas:
        pool = _sat_pool(n, a, k, rng)
        rhos, v0 = [], 0                                     # rhos: ONLY over satisfying outputs
        for cl, _w in pool:
            x = optimize(cl, n, "cheap", budget, seed=rng.randint(1, 10 ** 9))   # cost-optimal cheap search
            s = satisfies(cl, x)
            v0 += int(s)
            r = rho(cl, x, n) if s else None
            if s:
                rhos.append(r)
            rows.append({"alpha": a, "rho": (round(r, 4) if s else None), "sat": s})
        med = _median(rhos) if rhos else float("nan")
        robust = (sum(r >= theta for r in rhos) / len(rhos)) if rhos else float("nan")
        print(f"  alpha={a:>5} | n_sat={len(pool):>2} | V0-accept={100*v0/max(1,len(pool)):3.0f}% | "
              f"median rho|sat={med:.3f} (nrho={len(rhos)}) | robust(rho>= {theta})={100*robust:4.0f}%  "
              f"fragile={100*(1-robust):4.0f}%", flush=True)
        json.dump({"rows": rows, "theta": theta}, open(os.path.join(outdir, "frontier.json"), "w"), indent=1)
    print("\nMirror of RNA E1': cheap V0 accepts ~all it SOLVES; robustness (median rho | SAT) DROPS as "
          "alpha -> threshold. V0-accept<100% at high alpha = cheap-solver reach limit (raise --budget), "
          "reported separately so it does not contaminate the fragility signal.", flush=True)


# ---- E6'-Goodhart: cheap vs robust reward, same kernel, budget sweep (mirror E2') -----------------
def run_goodhart(n, alpha, budgets, k, outdir, theta=0.3, seed=0):
    """Both arms START from the same witness satisfying assignment (so V0=100% throughout), isolating the
    ROBUSTNESS-optimization contrast: cheap score is flat on the sat manifold (random walk -> typical low
    rho); strong score keeps climbing slack (-> higher rho). Mirror of RNA E2', confound-free."""
    os.makedirs(outdir, exist_ok=True)
    rng = random.Random(seed)
    pool = _sat_pool(n, alpha, k, rng)
    print(f"E6' SAT Goodhart: n={n}, alpha={alpha}, k={len(pool)} instances, budgets={budgets}, "
          f"theta={theta} (witness-start; median rho is primary)", flush=True)
    rows = []
    for B in budgets:
        for arm in ("cheap", "strong"):
            v0 = robust = 0
            rhos = []
            for cl, w in pool:
                x = optimize(cl, n, arm, B, seed=rng.randint(1, 10 ** 9), x0=w)   # start from witness
                s = satisfies(cl, x)
                r = rho(cl, x, n) if s else 0.0
                v0 += int(s)
                robust += int(s and r >= theta)
                rhos.append(r)
                rows.append({"budget": B, "arm": arm, "sat": s, "rho": round(r, 4)})
            m = len(pool)
            print(f"  B={B:>6} arm={arm:6} | V0-accept={100*v0/m:5.1f}%  median rho={_median(rhos):.3f}  "
                  f"robust-accept(rho>= {theta})={100*robust/m:5.1f}%", flush=True)
            json.dump({"rows": rows, "budgets": budgets, "theta": theta, "alpha": alpha},
                      open(os.path.join(outdir, "goodhart.json"), "w"), indent=1)
    print("\nGoodhart signature (mirror RNA E2'): both keep V0=100%; STRONG arm's median rho RISES with "
          "budget while CHEAP arm's stays at the typical (low) value -> optimizing the cheap proxy does "
          "not buy robustness; optimizing the robustness-aware objective does.", flush=True)


# ---- E6'-cost: end-to-end verifier cost per rung vs (n, r) (mirror E4') ----------------------------
def run_cost(ns, alpha, reps, outdir, seed=0):
    os.makedirs(outdir, exist_ok=True)
    rng = random.Random(seed)
    print(f"E6' SAT cost: ns={ns}, alpha={alpha}, reps={reps}", flush=True)
    per = {"V0": [], "rho(r=1)": [], "V(r=2)": []}
    for n in ns:
        pool = _sat_pool(n, alpha, reps, rng)
        t0s = {"V0": [], "rho(r=1)": [], "V(r=2)": []}
        for cl, x in pool:                                  # x = witness satisfying assignment
            w = time.perf_counter(); satisfies(cl, x); t0s["V0"].append(time.perf_counter() - w)
            w = time.perf_counter(); rho(cl, x, n); t0s["rho(r=1)"].append(time.perf_counter() - w)
            w = time.perf_counter()
            # r=2 ball check cost proxy: all pairs single-then-single (O(n^2 m)); cap to keep bounded
            cnt = 0
            for i in range(1, min(n, 60) + 1):
                x[i] ^= 1
                for j in range(i + 1, min(n, 60) + 1):
                    x[j] ^= 1; satisfies(cl, x); x[j] ^= 1; cnt += 1
                x[i] ^= 1
            t0s["V(r=2)"].append(time.perf_counter() - w)
        line = [f"n={n:>4}"]
        for rung in ("V0", "rho(r=1)", "V(r=2)"):
            med = sorted(t0s[rung])[len(t0s[rung]) // 2]
            per[rung].append((n, med))
            line.append(f"{rung}={med*1e3:8.3f}ms")
        print("  " + "  ".join(line), flush=True)
    json.dump({"per": {k: v for k, v in per.items()}}, open(os.path.join(outdir, "cost.json"), "w"), indent=1)
    print("\nWithin-domain cost-robustness ladder: V0 O(m) -> rho O(n m) -> r=2 O(n^2 m) (mirror RNA E4').", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["frontier", "goodhart", "cost"], required=True)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--alpha", type=float, default=4.2)
    ap.add_argument("--alphas", type=float, nargs="+", default=[3.0, 3.8, 4.0, 4.2, 4.267, 4.4])
    ap.add_argument("--budgets", type=int, nargs="+", default=[500, 4000, 20000])
    ap.add_argument("--ns", type=int, nargs="+", default=[40, 60, 80, 100, 120])
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--theta", type=float, default=0.3)
    ap.add_argument("--budget", type=int, default=20000, help="cheap-search budget for frontier")
    ap.add_argument("--outdir", default=OUT_DEFAULT)
    a = ap.parse_args()
    if a.mode == "frontier":
        run_frontier(a.n, a.alphas, a.k, a.outdir, theta=a.theta, budget=a.budget)
    elif a.mode == "goodhart":
        run_goodhart(a.n, a.alpha, a.budgets, a.k, a.outdir, theta=a.theta)
    elif a.mode == "cost":
        run_cost(a.ns, a.alpha, a.reps, a.outdir)
