#!/usr/bin/env python3
"""M17d: the ladder's delta=0 rung over-rejected. Rule 76 says that measures the MAIN arm's
type-I error, so measure it properly instead of widening the probe.

Two nulls, 40 draws each, because 8 draws cannot estimate a rate against a 0.15 threshold.

  A -- the ladder's own construction: permute y within cell, test by permuting gc_paired.
  B -- exchangeability of the test itself: take ONE within-cell permutation of gc_paired as the
       observed treatment, test it against fresh permutations. Real y untouched.
  C -- same as B but the null permutes at the TARGET level (whole cells' treatment vectors
       swapped between cells of the same target), the candidate repair.

P1 already returned 0/72 with a Gaussian noise column; gc_paired is not Gaussian (ties, clumping,
cell-specific spread), so B is the check that column shape does not break the test.

COST: $0, offline.
"""
from __future__ import annotations

import argparse, collections, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", ".."))); sys.path.insert(0, HERE)
from m13b_gc_placement import assemble, match, point, perm_cols, pval, POPS, DELTA   # noqa: E402

NPERM, DRAWS, P_GATE = 400, 40, 0.005


def _D(fn, seed):
    C = json.load(open(os.path.join(HERE, "m13_features.json")))
    E = json.load(open(os.path.join(HERE, "m13b_extra.json")))
    return assemble(C[fn]["rows"], E[fn], seed)


def cells_permuted_y(d, rng):
    out = []
    for c in d["cells"]:
        y = c["y"][rng.permutation(len(c["y"]))]
        out.append({**c, "y": y - y.mean()})
    return {**d, "cells": out}


def one(d, treat, rng2, nperm=NPERM):
    M = match(d, treat, DELTA, panel=False)
    if M is None or M["npair"] < 30:
        return None
    obs = point(M)
    nl = np.sort(np.array([point(match(d, perm_cols(d, rng2), DELTA, panel=False)
                                 or {"sums": {0: 0.0}, "cnts": {0: 1}}) for _ in range(nperm)]))
    return obs, pval(obs, nl)


def job(spec):
    fn, seed, test, k0, k1 = spec
    d = _D(fn, seed)
    hits, ps, effs = 0, [], []
    for k in range(k0, k1):
        rng = np.random.default_rng(500000 + 7919 * seed + 131 * k + {"A": 1, "B": 2}[test])
        rng2 = np.random.default_rng(900000 + 7919 * seed + 131 * k + {"A": 1, "B": 2}[test])
        if test == "A":
            r = one(cells_permuted_y(d, rng), [c["gcp"] for c in d["cells"]], rng2)
        else:
            r = one(d, perm_cols(d, rng), rng2)
        if r is None:
            continue
        obs, p = r; ps.append(round(p, 5)); effs.append(round(obs, 4)); hits += int(p <= P_GATE)
    return fn, test, {"n": len(ps), "hits": hits, "p": ps, "eff": effs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default=os.path.join(HERE, "m17d_typeI_diag.json"))
    a = ap.parse_args()
    print("=== M17d:第76条诊断 —— 主臂在 panel=False 下的一类错误率 ===")
    print(f"  两种零假设 × {DRAWS} 次抽样 × {NPERM} 次置换。名义 {P_GATE}。离线 $0。\n")
    CH = [(0, 10), (10, 20), (20, 30), (30, 40)]
    specs = [(fn, a.seed, t, lo, hi) for _n, fn in POPS for t in ("A", "B") for lo, hi in CH]
    from concurrent.futures import ProcessPoolExecutor
    ACC = collections.defaultdict(lambda: collections.defaultdict(lambda: {"n": 0, "hits": 0, "p": [], "eff": []}))
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for fn, t, r in ex.map(job, specs):
            s = ACC[fn][t]
            s["n"] += r["n"]; s["hits"] += r["hits"]; s["p"] += r["p"]; s["eff"] += r["eff"]
    print(f"  {'人群':12s} {'A 打乱y':>16s} {'A 效应SD':>10s} | {'B 置换处理列':>16s} {'B 效应SD':>10s}")
    tot = {"A": [0, 0], "B": [0, 0]}
    for nm, fn in POPS:
        A, B = ACC[fn]["A"], ACC[fn]["B"]
        for t, s in (("A", A), ("B", B)):
            tot[t][0] += s["hits"]; tot[t][1] += s["n"]
        print(f"  {nm:12s} {A['hits']:6d}/{A['n']:<4d}={A['hits']/max(A['n'],1):5.3f} "
              f"{np.std(A['eff']):10.4f} | {B['hits']:6d}/{B['n']:<4d}={B['hits']/max(B['n'],1):5.3f} "
              f"{np.std(B['eff']):10.4f}")
    print(f"\n  合计  A {tot['A'][0]}/{tot['A'][1]} = {tot['A'][0]/tot['A'][1]:.4f}"
          f"   B {tot['B'][0]}/{tot['B'][1]} = {tot['B'][0]/tot['B'][1]:.4f}   名义 {P_GATE}")
    lo = 0.005 * tot['A'][1]
    print(f"  名义下期望命中数 {lo:.2f};A 实测 {tot['A'][0]},B 实测 {tot['B'][0]}")
    verdict = ("test_is_exact__ladder_selfcheck_was_small_sample_noise"
               if tot['A'][0] <= 3 and tot['B'][0] <= 3 else
               "main_arm_type_I_inflated__repair_the_main_arm")
    print(f"  **诊断:{verdict}**")
    json.dump({"A": {fn: ACC[fn]["A"] for _n, fn in POPS},
               "B": {fn: ACC[fn]["B"] for _n, fn in POPS},
               "total": tot, "nominal": P_GATE, "verdict": verdict},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
