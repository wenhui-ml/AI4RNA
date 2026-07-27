#!/usr/bin/env python3
"""An information budget for RNA inverse design -- a search-free, algorithm-independent coordinate.

Frame. Design is entropy reduction. A target T admits a set of sequences merely COMPATIBLE with its
pairing; that set has entropy

    H_avail(T) = 2 * n_unpaired + log2(6) * n_pairs      bits

(4 letters at unpaired positions, 6 admissible WC/GU pairs at paired ones). But compatibility is not
enough: T must also beat every rival fold. Writing gap(x) = E(x,T) - E(x, MFE(x)) >= 0, the Boltzmann
weight of T under x obeys p(T|x) <= exp(-gap/RT), so the bits design must additionally buy are

    H_needed(T) = Delta(T) / (RT * ln 2),   Delta(T) = min over compatible x of gap(x).

Delta is estimated by sampling compatible sequences, which makes H_needed an UPPER bound estimate (a real
designer does better than random). The coordinate we care about is the slack

    slack(T) = H_avail(T) - H_needed(T).

slack < 0 means the target asks for more bits than the sequence can carry: undesignable for reasons no
optimizer can touch. This is the rate-distortion-style statement the solve/fail metric cannot express.

Nothing here runs a search. No optimizer, no budget, no seed.
"""
import json, math, os, random, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import RNA                                                              # noqa: E402
from rna_harness.benchmarks.eterna100 import load_eterna100 as L        # noqa: E402

RT = 0.0019872 * 310.15          # kcal/mol at 37 C
BIT = RT * math.log(2)           # kcal/mol per bit
WC = [('A','U'),('U','A'),('G','C'),('C','G'),('G','U'),('U','G')]


def pairmap(db):
    st, p = [], {}
    for i, c in enumerate(db):
        if c == '(':
            st.append(i)
        elif c == ')':
            j = st.pop(); p[j] = i; p[i] = j
    return p


def probe(db, nsamp):
    """min_x gap(x) over random compatible x, plus the compatible-set entropy."""
    p = pairmap(db); N = len(db)
    npair = sum(1 for i, j in p.items() if i < j)
    h_avail = 2 * (N - 2 * npair) + math.log2(6) * npair
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
    return best, h_avail, npair


def auc(pos, neg):
    if not pos or not neg:
        return float('nan')
    w = sum(1 if a > b else 0.5 if a == b else 0 for a in pos for b in neg)
    return w / (len(pos) * len(neg))


def main():
    random.seed(0)
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
        nsamp = 100 if n <= 150 else (50 if n <= 300 else 30)
        d, h, npair = probe(T[t], nsamp)
        out.append({"task": t, "n": n, "n_pairs": npair, "solved": any(x['solved'] for x in by[t]),
                    "delta_kcal": round(d, 2), "h_avail_bits": round(h, 1),
                    "h_needed_bits": round(d / BIT, 1), "slack_bits": round(h - d / BIT, 1)})
        print(f"  {n:4d}nt {'S' if out[-1]['solved'] else 'F'}  Delta={d:7.1f}  "
              f"avail={h:7.1f}b  needed={d/BIT:7.1f}b  slack={h-d/BIT:8.1f}b  {t.split(':',2)[-1][:34]}",
              flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "info_budget.json"), "w"), indent=1)

    for lo, hi, lab in ((0, 150, "<150nt"), (150, 10000, ">=150nt"), (0, 10000, "ALL")):
        g = [r for r in out if lo <= r['n'] < hi]
        U = [r for r in g if not r['solved']]; S = [r for r in g if r['solved']]
        if not U or not S:
            continue
        print(f"\n=== {lab}: {len(g)} targets, {len(U)} unsolved ===")
        for k, sign in (("delta_kcal", 1), ("h_needed_bits", 1), ("slack_bits", -1), ("n", 1)):
            a = auc([sign * r[k] for r in U], [sign * r[k] for r in S])
            print(f"  AUC({k:14} -> failure) = {a:.3f}")
        print(f"  median slack: unsolved {sorted(r['slack_bits'] for r in U)[len(U)//2]:8.1f}b   "
              f"solved {sorted(r['slack_bits'] for r in S)[len(S)//2]:8.1f}b")
        neg = [r for r in g if r['slack_bits'] < 0]
        print(f"  targets with NEGATIVE slack (budget violated): {len(neg)}"
              + (f" -- {sum(1 for r in neg if not r['solved'])} of them unsolved" if neg else ""))


if __name__ == "__main__":
    main()
