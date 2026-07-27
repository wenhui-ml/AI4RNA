#!/usr/bin/env python3
"""The mean-field baseline: does a helix need its context, ignore it, or fight it?

Three earlier attempts to measure "composition" failed, and the third failure said why: a fragment folded
IN ISOLATION is a different folding problem from the same fragment in context (free ends instead of a
closing helix and flanking sequence), so sum_i Delta(fragment_i) is not a decomposition of Delta(T) at all.
The Deng-Hani-Ma cumulant expansion works because it expands around a genuinely free baseline --
non-interacting streaming particles. "The fragment on its own" is not that baseline.

The right baseline keeps the molecule and removes the INFORMATION. For a helix H of target T:

  full ensemble        every position compatible with T (WC/GU at T's pairs, uniform elsewhere)
  mean-field for H     H's positions compatible with H; EVERY other position uniform random,
                       including positions T pairs -- so the rest of the molecule is present, at the
                       same length and the same place, carrying no information about T

    p_full(H) = Pr[ MFE(x) contains all of H | x ~ full ]
    p_mf(H)   = Pr[ MFE(x) contains all of H | x ~ mean-field(H) ]
    C(H)      = log p_full(H) - log p_mf(H)

  C > 0   cooperation: H forms only because the rest of T's pairing is there. This is the term a
          per-unit designer cannot see, and the honest candidate for "units easy, assembly impossible".
  C ~ 0   independence: molecular chaos holds at the helix level; difficulty is local and additive.
  C < 0   frustration: the rest of T's pairing COMPETES with H. Designing one unit hurts another.

The three regimes are physically distinct, which is what the earlier one-sided quantities could not
express. Laplace-smoothed at (k+0.5)/(N+1) so a zero count does not produce an infinite C; raw counts are
reported alongside so a reader can see when smoothing is carrying the estimate.

Search-free: random sequences, one ViennaRNA fold each. No optimizer.
"""
import json, math, os, random, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import RNA                                                              # noqa: E402

WC = [('A','U'),('U','A'),('G','C'),('C','G'),('G','U'),('U','G')]


def pairmap(db):
    st, p = [], {}
    for i, c in enumerate(db):
        if c == '(':
            st.append(i)
        elif c == ')':
            j = st.pop(); p[j] = i; p[i] = j
    return p


def helices(db):
    """Maximal stacks of contiguous base pairs: (i,j),(i+1,j-1),... Each is one unit."""
    p = pairmap(db)
    pairs = sorted((i, j) for i, j in p.items() if i < j)
    seen, out = set(), []
    for i, j in pairs:
        if (i, j) in seen:
            continue
        h = []
        a, b = i, j
        while p.get(a) == b:
            h.append((a, b)); seen.add((a, b)); a, b = a + 1, b - 1
            if p.get(a) != b:
                break
        out.append(h)
    return [h for h in out if h]


def sample_full(db, rng):
    p = pairmap(db); N = len(db); s = ['A'] * N
    for i in range(N):
        if i not in p:
            s[i] = rng.choice('ACGU')
    for i, j in p.items():
        if i < j:
            a, b = rng.choice(WC); s[i] = a; s[j] = b
    return ''.join(s)


def sample_mf(db, helix, rng):
    """H compatible; everything else uniform -- context present, information removed."""
    N = len(db); s = [rng.choice('ACGU') for _ in range(N)]
    for i, j in helix:
        a, b = rng.choice(WC); s[i] = a; s[j] = b
    return ''.join(s)


def contains(fold_pairs, helix):
    return all((i, j) in fold_pairs for i, j in helix)


def fold_pairs(x):
    db = RNA.fold(x)[0]
    p = pairmap(db)
    return {(i, j) for i, j in p.items() if i < j}


def analyse(name, db, nsamp, seed=0):
    rng = random.Random(seed)
    H = helices(db)
    full = [fold_pairs(sample_full(db, rng)) for _ in range(nsamp)]
    rows = []
    for h in H:
        kf = sum(contains(f, h) for f in full)
        km = sum(contains(fold_pairs(sample_mf(db, h, rng)), h) for _ in range(nsamp))
        pf = (kf + 0.5) / (nsamp + 1); pm = (km + 0.5) / (nsamp + 1)
        rows.append({"span": [h[0][0], h[0][1]], "len": len(h),
                     "k_full": kf, "k_mf": km, "C": round(math.log(pf) - math.log(pm), 3)})
    return {"target": name, "n": len(db), "n_helices": len(H), "nsamp": nsamp, "helices": rows}


def summarise(r):
    C = [h["C"] for h in r["helices"]]
    coop = sum(1 for c in C if c > 0.35); frus = sum(1 for c in C if c < -0.35)
    ind = len(C) - coop - frus
    med = sorted(C)[len(C) // 2] if C else float('nan')
    print(f"  {r['target'][:34]:34} n={r['n']:4d} helices={r['n_helices']:3d}  "
          f"median C={med:+6.2f}  cooperative={coop:3d} independent={ind:3d} frustrated={frus:3d}")
    return med, coop, ind, frus


if __name__ == "__main__":
    from rna_harness.benchmarks.eterna100 import load_eterna100 as L
    T = {f"{t['id']}:{t['name']}": t['target'] for t in L()}
    WANT = ["62:Bug 18", "57:multilooping fun", "100:Teslagon", "97:Hoglafractal",
            "35:Iron Cross", "83:Fractal 3", "16:Fractile", "1:Simple Hairpin"]
    out = []
    print("mean-field composition index by target (C>0 cooperative, ~0 independent, <0 frustrated)\n")
    for k in WANT:
        if k not in T:
            print(f"  (missing {k})"); continue
        db = T[k]
        ns = 80 if len(db) <= 130 else (40 if len(db) <= 250 else 25)
        r = analyse(k, db, ns)
        summarise(r); out.append(r)
        json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "meanfield.json"), "w"), indent=1)
