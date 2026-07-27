#!/usr/bin/env python3
"""Step 0b: the same N-dependence question on the crossing benchmark, where the allocation runs.

On Eterna, Delta is nearly N-free: AUC 0.859 at N = 5 vs 0.875 at N = 200 (ev_margin.py). If that holds
here, the predictor costs 300 pKiss calls per 60-target pass instead of 3600, which is the difference
between a 1.6x and a 5x efficiency result.

But Delta_bp is DISCRETE and bounded below by 0, and pk_margin.py already found 30 of 43 pKiss-solves
sitting at Delta_bp = 0 with N = 60. Raising N can only push more targets to 0, creating ties and
DESTROYING discrimination. So the prediction here is the opposite of Eterna's: small N should be at
least as good, possibly better. That is a real fork, and it is what this measures.

Stores every draw so any N, and any functional of the draws, can be scored afterwards without new
oracle calls. Also stores the per-attempt cost/outcome table the hazard model needs.
"""
import json, os, random, sys
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
CC = os.path.abspath(os.path.join(HERE, "..", "cc_ascent"))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
from rna_harness.benchmarks.rnainvbench import load_pseudobase          # noqa: E402
from rna_harness.oracles_pk import pkiss_pairs                          # noqa: E402

WC = [('A', 'U'), ('U', 'A'), ('G', 'C'), ('C', 'G'), ('G', 'U'), ('U', 'G')]
N_MAX = int(os.environ.get("NMAX", "200"))
GRID = [1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 60, 100, 150, 200]


def pk_pairmap(db):
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


def probe(args):
    tid, db, seed = args
    rng = random.Random(seed)
    p = pk_pairmap(db); n = len(db)
    tgt = {(i, j) for i, j in p.items() if i < j}
    draws = []
    for _ in range(N_MAX):
        s = ['A'] * n
        for i in range(n):
            if i not in p:
                s[i] = rng.choice('ACGU')
        for i, j in p.items():
            if i < j:
                a, b = rng.choice(WC); s[i] = a; s[j] = b
        got = pkiss_pairs(''.join(s))
        if got is None:
            draws.append(None); continue
        got = {(min(a, b), max(a, b)) for a, b in got}
        draws.append(len(got ^ tgt))
    return tid, draws, len(tgt)


def auc(pos, neg):
    if not pos or not neg:
        return float('nan')
    return sum(1 if a > b else .5 if a == b else 0 for a in pos for b in neg) / (len(pos) * len(neg))


def main():
    tasks = load_pseudobase(limit=60)
    args = [(t.task_id, t.target, 9000 + k) for k, t in enumerate(tasks)]
    nw = int(os.environ.get("NW", "48"))
    with mp.get_context("fork").Pool(nw) as pool:
        raw = {tid: (d, npair) for tid, d, npair in pool.map(probe, args)}

    # outcomes AND per-attempt realised cost, from the committed 15-cell sweep
    panel_win, gamer_win, attempts = {}, {}, []
    cells = 0
    for e in range(3):
        for s in range(5):
            f = os.path.join(CC, f"gvp_e{e}_s{s}", "sweep.json")
            if not os.path.exists(f):
                continue
            cells += 1
            for r in json.load(open(f))["rows"]:
                for arm, acc in (("panel_auto", panel_win), ("autonomous", gamer_win)):
                    v = r["arms"].get(arm) or {}
                    u = int(bool((v.get("panel") or {}).get("unanimous")))
                    acc[r["id"]] = acc.get(r["id"], 0) + u
                    st = v.get("stats") or {}
                    attempts.append({"cell": f"e{e}s{s}", "arm": arm, "id": r["id"],
                                     "n_pkiss": st.get("n_pkiss", 0), "unanimous": u,
                                     "in_loop_solved": int(bool(v.get("solved")))})
    pk_solved = {r["id"] for r in json.load(open(os.path.join(CC, "generous/panel_autonomous.json")))["rows"]
                 if r.get("pkiss")}

    rows = []
    for t in tasks:
        draws, npair = raw[t.task_id]
        ok = [d for d in draws if d is not None]
        rows.append({"id": t.task_id, "n": len(t.target), "n_pairs": npair,
                     "n_cross": t.meta.get("n_pk_pairs"), "n_ok": len(ok),
                     "draws": ok, "draws_norm": [round(d / max(1, npair), 4) for d in ok],
                     "pk_solved": t.task_id in pk_solved,
                     "panel_win": panel_win.get(t.task_id, 0),
                     "gamer_win": gamer_win.get(t.task_id, 0)})
    json.dump({"n_cells": cells, "n_max": N_MAX, "rows": rows, "attempts": attempts},
              open(os.path.join(HERE, "pk_ev.json"), "w"))
    print(f"{len(rows)} crossing targets, {N_MAX} draws each, {cells} sweep cells, "
          f"{len(attempts)} stored attempts")
    bad = [r['id'] for r in rows if r['n_ok'] < N_MAX]
    if bad:
        print(f"  targets with pKiss failures among draws: {len(bad)}")

    def dmin(r, N):
        d = r['draws_norm'][:N]
        return min(d) if d else float('nan')

    for lab, key in (("pKiss ever solves (43/60)", 'pk_solved'),
                     ("panel operator ever wins (21/60)", 'panel_win')):
        pos = [r for r in rows if not (r[key] if isinstance(r[key], bool) else r[key] > 0)]
        neg = [r for r in rows if (r[key] if isinstance(r[key], bool) else r[key] > 0)]
        print(f"\n=== AUC(Delta_bp -> failure), outcome = {lab};  {len(pos)} fail / {len(neg)} succeed ===")
        print(f"    {'N':>5}  {'AUC(min)':>9}  {'AUC(median)':>12}  {'AUC(mean)':>10}  "
              f"{'frac at 0':>10}  {'pKiss calls/60 targets':>23}")
        import statistics as stt
        for N in GRID:
            mn_p = [dmin(r, N) for r in pos]; mn_n = [dmin(r, N) for r in neg]
            md_p = [stt.median(r['draws_norm'][:N]) for r in pos]
            md_n = [stt.median(r['draws_norm'][:N]) for r in neg]
            mu_p = [stt.mean(r['draws_norm'][:N]) for r in pos]
            mu_n = [stt.mean(r['draws_norm'][:N]) for r in neg]
            f0 = sum(1 for r in rows if dmin(r, N) <= 0) / len(rows)
            print(f"    {N:5d}  {auc(mn_p, mn_n):9.3f}  {auc(md_p, md_n):12.3f}  {auc(mu_p, mu_n):10.3f}  "
                  f"{f0:10.2f}  {N*60:23d}")

    print("\n=== the ledger this fixes (panel-unanimous metric, 7682 pKiss calls/cell baseline) ===")
    print(f"    {'N':>5}  {'Delta cost':>10}  {'+ oracle-skyline 1088':>21}  {'x vs 7682':>10}  {'AUC(min)':>9}")
    pos = [r for r in rows if r['panel_win'] == 0]; neg = [r for r in rows if r['panel_win'] > 0]
    for N in (1, 2, 3, 5, 10, 20, 60):
        c = N * 60
        print(f"    {N:5d}  {c:10d}  {c+1088:21d}  {7682/(c+1088):10.2f}  "
              f"{auc([dmin(r, N) for r in pos], [dmin(r, N) for r in neg]):9.3f}")


if __name__ == "__main__":
    main()
