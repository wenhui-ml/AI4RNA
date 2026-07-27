#!/usr/bin/env python3
"""Re-read the chomsky_climb results in the Delta coordinate.

Delta(T) (info_budget.py) predicts our Eterna failures at AUC 0.867 without running any search. The
crossing study never had such a coordinate, so its central numbers -- 43/60 under pKiss, 1/60 panel-
unanimous, +0.048 for the panel operator over the gamer -- have never been checked against target-side
difficulty. This asks the question the paper cannot: is any of that a property of the METHOD, or of
WHICH TARGETS ARE EASY?

pKiss exposes pair sets, not energies, so the crossing analogue of Delta is structural:

    Delta_bp(T) = min over random compatible x of  |pKiss(x) symdiff P(T)| / |P(T)|

-- how close the best of N merely-compatible sequences gets, with no optimizer in the loop. Compatible
means WC/GU at every paired position of T (crossing pairs included) and uniform elsewhere.

Reads the committed sweeps for outcomes; the only new computation is the search-free probe.
"""
import json, os, random, sys
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
from rna_harness.benchmarks.rnainvbench import load_pseudobase          # noqa: E402
from rna_harness.oracles_pk import pkiss_pairs                          # noqa: E402

WC = [('A','U'),('U','A'),('G','C'),('C','G'),('G','U'),('U','G')]
NSAMP = int(os.environ.get("NSAMP", "60"))


def pk_pairmap(db):
    """Pair map for a dot-bracket with multiple bracket classes (crossing pairs included)."""
    OP = {'(': ')', '[': ']', '{': '}', '<': '>'}
    CL = {v: k for k, v in OP.items()}
    st = {k: [] for k in OP}
    p = {}
    for i, c in enumerate(db):
        if c in OP:
            st[c].append(i)
        elif c in CL:
            j = st[CL[c]].pop(); p[j] = i; p[i] = j
    return p


def probe_one(args):
    tid, db, seed = args
    rng = random.Random(seed)
    p = pk_pairmap(db); N = len(db)
    tgt = {(i, j) for i, j in p.items() if i < j}
    best = len(tgt) * 2
    for _ in range(NSAMP):
        s = ['A'] * N
        for i in range(N):
            if i not in p:
                s[i] = rng.choice('ACGU')
        for i, j in p.items():
            if i < j:
                a, b = rng.choice(WC); s[i] = a; s[j] = b
        got = pkiss_pairs(''.join(s))
        if got is None:
            continue
        got = {(min(a, b), max(a, b)) for a, b in got}
        best = min(best, len(got ^ tgt))
    return tid, best, len(tgt)


def main():
    tasks = load_pseudobase(limit=60)
    args = [(t.task_id, t.target, 1000 + k) for k, t in enumerate(tasks)]
    with mp.get_context("fork").Pool(24) as pool:
        res = {tid: (b, n) for tid, b, n in pool.map(probe_one, args)}

    # outcomes from the committed sweeps
    cc = HERE
    gamer_pk = {r["id"] for r in json.load(open(os.path.join(cc, "generous/panel_autonomous.json")))["rows"]
                if r.get("pkiss")}
    panel_mix, gamer_mix = {}, {}
    cells = 0
    for e in range(3):
        for s in range(5):
            f = os.path.join(cc, f"gvp_e{e}_s{s}", "sweep.json")
            if not os.path.exists(f):
                continue
            cells += 1
            for r in json.load(open(f))["rows"]:
                for arm, acc in (("panel_auto", panel_mix), ("autonomous", gamer_mix)):
                    v = r["arms"].get(arm) or {}
                    acc[r["id"]] = acc.get(r["id"], 0) + int(bool((v.get("panel") or {}).get("unanimous")))

    rows = []
    for t in tasks:
        b, npair = res[t.task_id]
        rows.append({"id": t.task_id, "n": len(t.target), "n_pairs": npair,
                     "n_cross": t.meta.get("n_pk_pairs"),
                     "delta_bp": b, "delta_norm": round(b / max(1, npair), 3),
                     "gamer_pkiss": t.task_id in gamer_pk,
                     "panel_mix": panel_mix.get(t.task_id, 0), "gamer_mix": gamer_mix.get(t.task_id, 0)})
    json.dump({"n_cells": cells, "nsamp": NSAMP, "rows": rows},
              open(os.path.join(HERE, "pk_margin.json"), "w"), indent=1)

    import statistics as st
    def med(v):
        v = sorted(v); return v[len(v)//2] if v else float('nan')
    def auc(p, q):
        return sum(1 if a > b else .5 if a == b else 0 for a in p for b in q)/(len(p)*len(q)) if p and q else float('nan')

    A = [r for r in rows if r["gamer_pkiss"]]; B = [r for r in rows if not r["gamer_pkiss"]]
    print(f"probe: {NSAMP} random compatible sequences per target, {cells} sweep cells for outcomes\n")
    print("=== Q1. Is the 43/60 pKiss rate a property of the targets? ===")
    print(f"  median Delta_norm  pKiss-SOLVED (n={len(A)}) {med([r['delta_norm'] for r in A]):.3f}   "
          f"unsolved (n={len(B)}) {med([r['delta_norm'] for r in B]):.3f}")
    print(f"  AUC(Delta_norm -> pKiss failure) = {auc([r['delta_norm'] for r in B],[r['delta_norm'] for r in A]):.3f}")

    print("\n=== Q2. Where do the panel operator's r_mix wins sit on Delta? ===")
    W = [r for r in rows if r["panel_mix"] > 0]; L = [r for r in rows if r["panel_mix"] == 0]
    print(f"  targets the panel operator EVER wins: {len(W)}/60  median Delta_norm {med([r['delta_norm'] for r in W]):.3f}")
    print(f"  never wins:                           {len(L)}/60  median Delta_norm {med([r['delta_norm'] for r in L]):.3f}")
    print(f"  AUC(Delta_norm -> panel never wins) = {auc([r['delta_norm'] for r in L],[r['delta_norm'] for r in W]):.3f}")

    print("\n=== Q3. Does the panel-vs-gamer gap come from closing margin, or from easy targets? ===")
    for lab, sel in (("low  Delta (easiest third)", lambda r: r["delta_norm"] <= sorted(x["delta_norm"] for x in rows)[19]),
                     ("mid  Delta", lambda r: sorted(x["delta_norm"] for x in rows)[19] < r["delta_norm"] <= sorted(x["delta_norm"] for x in rows)[39]),
                     ("high Delta (hardest third)", lambda r: r["delta_norm"] > sorted(x["delta_norm"] for x in rows)[39])):
        g = [r for r in rows if sel(r)]
        pm = sum(r["panel_mix"] for r in g); gm = sum(r["gamer_mix"] for r in g)
        print(f"  {lab:28} n={len(g):2d}  panel wins {pm:3d}  gamer wins {gm:3d}  "
              f"paired diff {(pm-gm)/(len(g)*max(1,cells)):+.4f}")


if __name__ == "__main__":
    main()
