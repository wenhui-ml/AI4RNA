#!/usr/bin/env python3
"""Where does the difficulty live? A multi-scale decomposition of the designability margin.

Supersedes compositional_excess.py, whose unit decomposition was void: "maximal disjoint helix-closed
units" returns the whole target (0.86-1.00 of its length for every long target checked), so
sum_i Delta(u_i) ~ Delta(T) was an identity, not a measurement.

The right object is a SCALE CURVE, which is also closer to the layered/multi-scale structure that makes
the Deng-Hani-Ma argument work. Recursively split the target at its multiloops into branches; the
frontier at recursion depth d is a partition of the structure into pieces that get smaller as d grows.
Then

    X(d) = Delta(T) - sum over frontier(d) of Delta(piece).

X(0) = 0 by construction. If difficulty is local, X(d) stays near 0 as pieces shrink: a designer working
piece-by-piece loses nothing. If difficulty is compositional, X(d) grows with d, and the depth at which
it takes off says at what scale the interactions live. The curve, not any single number, is the finding.

Search-free: ViennaRNA energy evaluations on random compatible sequences only.
"""
import json, math, os, random, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import RNA                                                              # noqa: E402
from rna_harness.benchmarks.eterna100 import load_eterna100 as L        # noqa: E402

WC = [('A','U'),('U','A'),('G','C'),('C','G'),('G','U'),('U','G')]
MAXDEPTH = 6


def pairmap(db):
    st, p = [], {}
    for i, c in enumerate(db):
        if c == '(':
            st.append(i)
        elif c == ')':
            j = st.pop(); p[j] = i; p[i] = j
    return p


def margin(db, nsamp):
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


def children(db):
    """Immediate helix-closed substructures at the top level of db (peeling one closing helix)."""
    p = pairmap(db)
    if not p:
        return []
    # peel a leading fully-enclosing helix so recursion makes progress
    if db[0] == '(' and p.get(0) == len(db) - 1:
        inner = db[1:-1]
        return [inner] if inner.strip('.') else []
    out, k = [], 0
    while k < len(db):
        if db[k] == '(':
            out.append(db[k:p[k] + 1]); k = p[k] + 1
        else:
            k += 1
    return out if len(out) > 1 or (out and len(out[0]) < len(db)) else []


def frontier(db, depth):
    cur = [db]
    for _ in range(depth):
        nxt = []
        for piece in cur:
            ch = children(piece)
            nxt.extend(ch if ch else [piece])
        if nxt == cur:
            break
        cur = nxt
    return cur


def main():
    random.seed(2)
    T = {f"eterna100:{t['id']}:{t['name']}": t['target'] for t in L()}
    rows = json.load(open(os.path.join(os.path.dirname(__file__), "..", "..",
                                       "_archive_runs/eterna_paired/memon/rows_partial.json")))
    by = {}
    for r in rows:
        by.setdefault(r['task_id'], []).append(r)
    out = []
    for t in sorted(by, key=lambda x: by[x][0]['n']):
        if t not in T:
            continue
        n = by[t][0]['n']
        ns = 80 if n <= 150 else (40 if n <= 300 else 25)
        db = T[t]
        dT = margin(db, ns)
        if dT <= 0:
            continue
        curve, sizes = [], []
        for d in range(1, MAXDEPTH + 1):
            f = frontier(db, d)
            biggest = max(len(x) for x in f)
            curve.append(round(dT - sum(margin(x, ns) for x in f), 2))
            sizes.append(round(biggest / len(db), 2))
        rec = {"task": t, "n": n, "solved": any(x['solved'] for x in by[t]),
               "delta_T": round(dT, 2), "X_by_depth": curve, "largest_frac_by_depth": sizes}
        out.append(rec)
        print(f"{n:5d} {'S' if rec['solved'] else 'F'} D={dT:7.1f}  X={curve}  frac={sizes}  "
              f"{t.split(':',2)[-1][:26]}", flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "scale_excess.json"), "w"), indent=1)

    import statistics as st
    print("\n=== X(d) as pieces shrink; frac = largest piece / target length ===")
    for lo, hi, lab in ((0, 150, "<150nt"), (150, 10000, ">=150nt")):
        g = [r for r in out if lo <= r['n'] < hi]
        U = [r for r in g if not r['solved']]; S = [r for r in g if r['solved']]
        if not U or not S:
            continue
        print(f"\n{lab}: {len(U)} unsolved, {len(S)} solved")
        for d in range(MAXDEPTH):
            fu = st.median(r['largest_frac_by_depth'][d] for r in g)
            print(f"  depth {d+1} (largest piece {fu:.2f} of target): "
                  f"median X unsolved {st.median(r['X_by_depth'][d] for r in U):7.1f} "
                  f"({st.median(r['X_by_depth'][d]/r['delta_T'] for r in U):5.2f} of Delta)   "
                  f"solved {st.median(r['X_by_depth'][d] for r in S):7.1f} "
                  f"({st.median(r['X_by_depth'][d]/r['delta_T'] for r in S):5.2f})")


if __name__ == "__main__":
    main()
