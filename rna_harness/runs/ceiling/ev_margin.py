#!/usr/bin/env python3
"""Task zero: fix the length--sample-size confound in Delta, and make Delta cheap.

info_budget.py drew N = 100/50/30 compatible sequences PER TARGET LENGTH. Delta-hat is a sample
MINIMUM, so it is non-increasing in N; giving long targets fewer draws inflated their Delta-hat, and
long targets are also likelier to be unsolved. The reported AUC 0.867 is therefore confounded, and this
script is the correction. It does three things:

  (1) Draws a fixed N_MAX = 200 for EVERY target and stores the raw gaps, so any N can be evaluated
      after the fact from the same draws.

  (2) Computes E[Delta-hat_N] EXACTLY from the order statistics rather than by resampling. For M stored
      draws with sorted values g_(1) <= ... <= g_(M), a subsample of size N without replacement has

          Pr[min = g_(k)] = C(M-k, N-1) / C(M, N)

      so E[Delta-hat_N] = sum_k g_(k) * C(M-k, N-1) / C(M, N). No Monte Carlo, no seed.

  (3) Fits the extreme-value form  Delta-hat_N ~ Delta* + c * N^(-1/alpha)  by grid-searching Delta*
      for maximum log-log linearity. This estimates the true margin Delta*(T), removes the N dependence
      that caused the defect, and tells us how small N can be before the predictor degrades -- which
      matters because Delta is not free: at N = 60 it costs more oracle calls than the search it is
      supposed to be triaging.

The headline test: does Delta still predict failure at FIXED N, and at what N does it stop?
"""
import json, math, os, random, sys
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
import RNA                                                              # noqa: E402
from rna_harness.benchmarks.eterna100 import load_eterna100 as L        # noqa: E402

RT = 0.0019872 * 310.15
BIT = RT * math.log(2)
WC = [('A', 'U'), ('U', 'A'), ('G', 'C'), ('C', 'G'), ('G', 'U'), ('U', 'G')]

N_MAX = 200
GRID = [5, 10, 20, 30, 40, 60, 80, 100, 140, 200]
# the length-dependent schedule that produced the confounded number, reproduced for comparison
OLD_N = lambda n: 100 if n <= 150 else (50 if n <= 300 else 30)


def pairmap(db):
    st, p = [], {}
    for i, c in enumerate(db):
        if c == '(':
            st.append(i)
        elif c == ')':
            j = st.pop(); p[j] = i; p[i] = j
    return p


def probe(args):
    """All N_MAX gaps for one target. Per-target RNG so the draw set is reproducible."""
    task, db, seed = args
    rng = random.Random(seed)
    p = pairmap(db); n = len(db)
    npair = sum(1 for i, j in p.items() if i < j)
    h_avail = 2 * (n - 2 * npair) + math.log2(6) * npair
    gaps = []
    for _ in range(N_MAX):
        s = ['A'] * n
        for i in range(n):
            if i not in p:
                s[i] = rng.choice('ACGU')
        for i, j in p.items():
            if i < j:
                a, b = rng.choice(WC); s[i] = a; s[j] = b
        x = ''.join(s)
        gaps.append(RNA.energy_of_struct(x, db) - RNA.fold(x)[1])
    return task, gaps, npair, h_avail


def exp_min(sorted_gaps, N):
    """E[min of N drawn without replacement from the M stored draws], exactly."""
    M = len(sorted_gaps)
    if N >= M:
        return sorted_gaps[0]
    denom = math.comb(M, N)
    return sum(g * math.comb(M - k - 1, N - 1) for k, g in enumerate(sorted_gaps)
               if M - k - 1 >= N - 1) / denom


def ev_fit(curve):
    """Fit Delta-hat_N = D + c N^(-1/alpha). Grid D, maximise R^2 of log(y-D) vs log(N).

    Returns (D, alpha, r2). Degenerate cases (a flat or already-zero curve) return D = min with
    alpha = nan, which is the honest answer: there is nothing to extrapolate.
    """
    Ns = [n for n, _ in curve]
    ys = [y for _, y in curve]
    lo, hi = 0.0, min(ys)
    if hi - max(ys) == 0 or hi <= 1e-9:
        return min(ys), float('nan'), float('nan')
    best = (None, -2.0, None)
    span = hi - lo
    for t in range(1, 400):
        D = lo + span * t / 400.0
        r = [y - D for y in ys]
        if min(r) <= 1e-12:
            continue
        X = [math.log(n) for n in Ns]
        Y = [math.log(v) for v in r]
        mx, my = sum(X) / len(X), sum(Y) / len(Y)
        sxx = sum((x - mx) ** 2 for x in X)
        sxy = sum((x - mx) * (y - my) for x, y in zip(X, Y))
        syy = sum((y - my) ** 2 for y in Y)
        if sxx <= 0 or syy <= 0:
            continue
        r2 = sxy * sxy / (sxx * syy)
        if r2 > best[1]:
            best = (D, r2, -sxy / sxx)      # slope = -1/alpha
    if best[0] is None:
        return min(ys), float('nan'), float('nan')
    D, r2, slope = best
    alpha = (1.0 / slope) if slope > 1e-9 else float('nan')
    return D, alpha, r2


def auc(pos, neg):
    if not pos or not neg:
        return float('nan')
    w = sum(1 if a > b else 0.5 if a == b else 0 for a in pos for b in neg)
    return w / (len(pos) * len(neg))


def main():
    T = {f"eterna100:{t['id']}:{t['name']}": t['target'] for t in L()}
    rows = json.load(open(os.path.join(HERE, "..", "..", "_archive_runs/eterna_paired/memon/rows_partial.json")))
    by = {}
    for r in rows:
        by.setdefault(r['task_id'], []).append(r)
    tasks = [t for t in sorted(by, key=lambda x: by[x][0]['n']) if t in T]
    args = [(t, T[t], 7000 + k) for k, t in enumerate(tasks)]

    nw = int(os.environ.get("NW", "64"))
    with mp.get_context("fork").Pool(nw) as pool:
        raw = {t: (g, npair, h) for t, g, npair, h in pool.map(probe, args)}

    out = []
    for t in tasks:
        gaps, npair, h_avail = raw[t]
        g = sorted(gaps)
        curve = [(N, exp_min(g, N)) for N in GRID]
        D, alpha, r2 = ev_fit(curve)
        n = by[t][0]['n']
        out.append({
            "task": t, "n": n, "n_pairs": npair, "solved": any(x['solved'] for x in by[t]),
            "h_avail_bits": round(h_avail, 1),
            "curve": {str(N): round(v, 4) for N, v in curve},
            "d_old": round(exp_min(g, OLD_N(n)), 4),        # the confounded schedule, same draws
            "d_star": round(D, 4), "alpha": None if alpha != alpha else round(alpha, 3),
            "r2": None if r2 != r2 else round(r2, 4),
            "gaps": [round(v, 3) for v in gaps],
        })
        print(f"  {n:4d}nt {'S' if out[-1]['solved'] else 'F'}  "
              f"D_5={curve[0][1]:7.1f} D_60={dict(curve)[60]:7.1f} D_200={curve[-1][1]:7.1f} "
              f"D*={D:7.1f} alpha={alpha if alpha == alpha else float('nan'):5.2f} r2={r2 if r2 == r2 else float('nan'):.3f}  "
              f"{t.split(':', 2)[-1][:30]}", flush=True)

    json.dump(out, open(os.path.join(HERE, "ev_margin.json"), "w"), indent=1)

    U = [r for r in out if not r['solved']]
    S = [r for r in out if r['solved']]
    print(f"\n{len(out)} targets with outcomes, {len(U)} unsolved, {len(S)} solved")

    print("\n=== the defect, demonstrated on identical draws ===")
    a_old = auc([r['d_old'] for r in U], [r['d_old'] for r in S])
    a_fix = auc([dict((int(k), v) for k, v in r['curve'].items())[100] for r in U],
                [dict((int(k), v) for k, v in r['curve'].items())[100] for r in S])
    print(f"  AUC, length-dependent N (100/50/30)  = {a_old:.3f}   <- the published 0.867's schedule")
    print(f"  AUC, fixed N = 100                    = {a_fix:.3f}")
    print(f"  AUC(length -> failure)                = {auc([r['n'] for r in U], [r['n'] for r in S]):.3f}")

    print("\n=== how the predictor degrades with N (this sets what Delta costs) ===")
    print(f"  {'N':>5}  {'AUC(E[D_N])':>12}  {'AUC(1 realisation)':>19}  {'oracle calls/target':>20}")
    for N in GRID:
        e_u = [r['curve'][str(N)] for r in U]
        e_s = [r['curve'][str(N)] for r in S]
        p_u = [min(r['gaps'][:N]) for r in U]
        p_s = [min(r['gaps'][:N]) for r in S]
        print(f"  {N:5d}  {auc(e_u, e_s):12.3f}  {auc(p_u, p_s):19.3f}  {N:20d}")
    a_star = auc([r['d_star'] for r in U], [r['d_star'] for r in S])
    print(f"  {'D* (N->inf)':>5}  {a_star:12.3f}")

    ok = [r for r in out if r['r2'] is not None]
    print(f"\n=== extreme-value fit ===")
    print(f"  targets with a non-degenerate fit: {len(ok)}/{len(out)}")
    if ok:
        al = sorted(r['alpha'] for r in ok if r['alpha'] is not None)
        rr = sorted(r['r2'] for r in ok)
        print(f"  median alpha {al[len(al)//2]:.2f}   median R^2 {rr[len(rr)//2]:.3f}")
        print(f"  targets already at D_200 = 0 (nothing to extrapolate): "
              f"{sum(1 for r in out if r['curve']['200'] <= 1e-9)}")

    print("\n=== bits form, at fixed N = 100 ===")
    for lab, sel in (("<150nt", lambda r: r['n'] < 150), (">=150nt", lambda r: r['n'] >= 150),
                     ("ALL", lambda r: True)):
        g = [r for r in out if sel(r)]
        u = [r for r in g if not r['solved']]; s = [r for r in g if r['solved']]
        if not u or not s:
            continue
        sl = lambda r: r['h_avail_bits'] - r['curve']['100'] / BIT
        print(f"  {lab:8} n={len(g):3d} unsolved={len(u):3d}  "
              f"AUC(slack -> failure) = {auc([-sl(r) for r in u], [-sl(r) for r in s]):.3f}   "
              f"median slack  unsolved {sorted(sl(r) for r in u)[len(u)//2]:8.1f}b  "
              f"solved {sorted(sl(r) for r in s)[len(s)//2]:8.1f}b")


if __name__ == "__main__":
    main()
