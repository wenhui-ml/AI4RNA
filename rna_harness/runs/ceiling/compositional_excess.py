#!/usr/bin/env python3
"""Compositional excess: the cumulant-expansion analogue for RNA inverse design.

Deng-Hani-Ma derive an irreversible kinetic equation from reversible dynamics by carrying the FULL
collision history in a cumulant expansion and then proving that the correlation-creating terms
(recollisions) are negligible. The simple effective theory is derived, not assumed.

Decomposition-based RNA design assumes the analogous molecular-chaos hypothesis without checking it:
that units can be designed independently. Teslagon says otherwise -- 42 units, each of a size we solve
90% of the time, assembled solve rate 0.

Make the assumption a measurable quantity. With Delta(.) the designability margin (info_budget.py),

    X(T) = Delta(T) - sum_i Delta(u_i)

is the part of the difficulty that lives in NO unit -- a total correlation in kcal/mol. Molecular chaos
would give X = 0. X >> 0 is the recollision term, and it is exactly what a per-unit designer cannot see.

Splitting X from the local terms separates our two residual regimes quantitatively rather than by name:
a short target with X ~ 0 is a local designability wall; a long target with X ~ Delta(T) is a pure
composition failure and no amount of better unit design touches it.

Search-free: only ViennaRNA energy evaluations on random compatible sequences.
"""
import json, math, os, random, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import RNA                                                              # noqa: E402
from rna_harness.benchmarks.eterna100 import load_eterna100 as L        # noqa: E402

WC = [('A','U'),('U','A'),('G','C'),('C','G'),('G','U'),('U','G')]


def pairmap(db):
    st, p = [], {}
    for i, c in enumerate(db):
        if c == '(':
            st.append(i)
        elif c == ')':
            j = st.pop(); p[j] = i; p[i] = j
    return p


def margin(db, nsamp):
    """min over random compatible sequences of E(x,db) - E(x,MFE(x)); 0.0 for structures with no pair."""
    p = pairmap(db); N = len(db)
    if not p or N < 5:
        return 0.0
    best = float('inf')
    for _ in range(nsamp):
        s = ['A'] * N
        for i in range(N):
            if i not in p:
                s[i] = random.choice('ACGU')
        for i, j in p.items():
            if i < j:
                a, b = random.choice(WC); s[i] = a; s[j] = b
        x = ''.join(s)
        best = min(best, RNA.energy_of_struct(x, db) - RNA.fold(x)[1])
    return best


def units(db):
    """Every helix-closed substructure, as an independently foldable sub-target."""
    p = pairmap(db); out = []
    for i, c in enumerate(db):
        if c != '(':
            continue
        j = p[i]
        sub = db[i:j + 1]
        # keep only units that are self-contained (balanced) and non-trivial
        if sub.count('(') == sub.count(')') and len(sub) >= 5:
            out.append((i, j, sub))
    # keep maximal disjoint units so the sum is not double-counting nested helices
    out.sort(key=lambda r: (r[0], -(r[1])))
    kept, last_end = [], -1
    for i, j, sub in out:
        if i > last_end:
            kept.append(sub); last_end = j
    return kept


def main():
    random.seed(1)
    T = {f"eterna100:{t['id']}:{t['name']}": t['target'] for t in L()}
    rows = json.load(open(os.path.join(os.path.dirname(__file__), "..", "..",
                                       "_archive_runs/eterna_paired/memon/rows_partial.json")))
    by = {}
    for r in rows:
        by.setdefault(r['task_id'], []).append(r)
    print(f"{'len':>5} {'st':>2} {'Delta(T)':>9} {'sum units':>10} {'excess X':>9} {'X/Delta':>8}  target")
    out = []
    for t in sorted(by, key=lambda x: by[x][0]['n']):
        if t not in T:
            continue
        n = by[t][0]['n']
        ns = 100 if n <= 150 else (50 if n <= 300 else 30)
        dT = margin(T[t], ns)
        us = units(T[t])
        du = sum(margin(u, ns) for u in us)
        X = dT - du
        solved = any(x['solved'] for x in by[t])
        out.append({"task": t, "n": n, "solved": solved, "n_units": len(us),
                    "delta_T": round(dT, 2), "sum_units": round(du, 2), "excess": round(X, 2),
                    "frac_excess": round(X / dT, 3) if dT > 0 else None})
        print(f"{n:5d} {'S' if solved else 'F':>2} {dT:9.1f} {du:10.1f} {X:9.1f} "
              f"{(X/dT if dT>0 else 0):8.2f}  {t.split(':',2)[-1][:32]}", flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "compositional_excess.json"), "w"), indent=1)

    def med(v):
        v = sorted(v); return v[len(v)//2] if v else float('nan')
    for lo, hi, lab in ((0, 150, "<150nt"), (150, 10000, ">=150nt")):
        g = [r for r in out if lo <= r['n'] < hi and r['delta_T'] > 0]
        U = [r for r in g if not r['solved']]; S = [r for r in g if r['solved']]
        if not U or not S:
            continue
        print(f"\n=== {lab} ({len(U)} unsolved, {len(S)} solved) ===")
        print(f"  median Delta(T):   unsolved {med([r['delta_T'] for r in U]):7.1f}  "
              f"solved {med([r['delta_T'] for r in S]):7.1f}")
        print(f"  median excess X:   unsolved {med([r['excess'] for r in U]):7.1f}  "
              f"solved {med([r['excess'] for r in S]):7.1f}")
        print(f"  median X/Delta:    unsolved {med([r['frac_excess'] for r in U]):7.2f}  "
              f"solved {med([r['frac_excess'] for r in S]):7.2f}")


if __name__ == "__main__":
    main()
