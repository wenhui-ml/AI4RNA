"""Reviewer-hardening SAT redesign (kills solver-selection bias). Instead of computing rho on a WalkSAT-
found assignment (which samples solutions with a solver-dependent bias), we ENUMERATE all 2^n assignments
for small n: this gives (a) exact SAT/UNSAT per formula --- not 'WalkSAT failed => UNSAT' --- and (b) the
single-flip robustness rho averaged over the ENTIRE solution set, i.e. the true typical robustness of the
solution space, uniform over solutions, with no solver in the loop. We report both formula-level (fraction
SAT) and assignment-level (median rho over all solutions) quantities separately."""
from __future__ import annotations
import argparse, json, os, random
import numpy as np

OUT_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "runs", "sat_exhaustive")


def gen_3sat(n, m, rng):
    cl = []
    for _ in range(m):
        vs = rng.sample(range(n), 3)
        cl.append([(v, rng.randint(0, 1)) for v in vs])   # (var, sign) sign=1 positive, 0 negated
    return cl


def sat_vector(cl, n):
    """Boolean array over all 2^n assignments: True where the formula is satisfied. Vectorized."""
    N = 1 << n
    a = np.arange(N, dtype=np.int64)
    bit = [((a >> j) & 1).astype(bool) for j in range(n)]   # bit[j][assignment] = value of var j
    sat = np.ones(N, dtype=bool)
    for clause in cl:
        c = np.zeros(N, dtype=bool)
        for (v, s) in clause:
            c |= (bit[v] if s == 1 else ~bit[v])
        sat &= c
    return sat


def rho_over_solutions(sat, n):
    """median single-flip rho over ALL satisfying assignments (uniform over the solution set)."""
    N = 1 << n
    idx = np.arange(N, dtype=np.int64)
    acc = np.zeros(N, dtype=np.float64)
    for j in range(n):
        acc += sat[idx ^ (1 << j)]        # neighbor satisfaction via bit-flip
    rho = acc / n
    S = np.nonzero(sat)[0]
    if len(S) == 0:
        return None, 0
    return float(np.median(rho[S])), int(len(S))


def run(n, alphas, k, outdir, seed=0):
    os.makedirs(outdir, exist_ok=True)
    print(f"SAT exhaustive (n={n}, 2^n={1<<n} enumerated; k={k}/density; rho over ALL solutions)", flush=True)
    rows = []
    rng = random.Random(seed)
    for al in alphas:
        m = max(1, round(al * n))
        rhos, sat_frac, sizes = [], 0, []
        for _ in range(k):
            cl = gen_3sat(n, m, rng)
            sv = sat_vector(cl, n)
            r, sz = rho_over_solutions(sv, n)
            if r is not None:                       # formula is SAT
                sat_frac += 1; rhos.append(r); sizes.append(sz)
            rows.append({"alpha": al, "m": m, "sat": bool(sv.any()),
                         "median_rho_over_solutions": r, "n_solutions": sz})
        med = float(np.median(rhos)) if rhos else None
        print(f"  alpha={al:>5} m={m:>3} | true-SAT={100*sat_frac/k:5.1f}% | "
              f"median rho over ALL solutions={med if med is None else round(med,3)} | "
              f"mean |solutions|={round(np.mean(sizes),1) if sizes else 0}", flush=True)
        json.dump({"rows": rows, "n": n, "k": k}, open(os.path.join(outdir, "sat_exhaustive.json"), "w"), indent=1)
    print(f"\n  wrote {os.path.join(outdir,'sat_exhaustive.json')}", flush=True)
    print("SATEXH_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=14)
    ap.add_argument("--alphas", type=float, nargs="+", default=[3.0, 3.4, 3.8, 4.0, 4.2, 4.267, 4.4, 4.6])
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--outdir", default=OUT_DEFAULT)
    a = ap.parse_args()
    run(a.n, a.alphas, a.k, a.outdir)
