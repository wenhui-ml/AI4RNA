#!/usr/bin/env python3
"""G4: does the panel's SIGNED disagreement structure add over its UNSIGNED spread?

Pre-registered in g4_signed_vs_spread_declaration.json. Kill conditions there are frozen.

THE OBJECTION FIRST. F2 already holds all fifteen fine columns from the three judges, so a linear model can
already form any pairwise difference -- signed differences add exactly zero as LINEAR features. Any increment
must come from the non-linear part: which judge is the outlier, and the sign pattern. Both new blocks are
therefore non-linear functions of the raw columns.

WHAT SURVIVED REVIEW. The "shared error cancels in the signed difference but only partly in the spread" claim
was retracted on 8-21: the shared term is the same number for all three judges, so it cancels exactly from
the spread too. What survived is informational: the signed pairwise differences DETERMINE the spread, not the
other way round. That makes a PAIR of nested predictions, and the pair is the whole test:

    (a) adding D on top of B+S   should gain            <- only says D is useful
    (b) adding S on top of B+D   should NOT gain        <- this is the claim itself

Testing only (a) is what g1_audit.md caught me doing with gamma*: nesting in one direction proves each term
is useful and cannot show the construction beats its own strongest part.

Estimator is R1's, unchanged: within-cell demeaning, within-cell permutation null to subtract what feature
COUNT alone buys (S is 8 columns, D is 9, so raw in-sample R^2 is not comparable), cluster bootstrap over
cells. Nested sets => in-sample R^2 monotone by exact algebra, and a violation aborts.

Sample 3 is primary, not the fourth batch: rule 34 puts the primary where the clue was found, and D3's
control arm (+0.0040 [+0.0006, +0.0103] over the F2 u f1c = 0.196 bar) was measured on sample 3.

COST: $0, offline, minutes. Columns already on disk.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import statistics as st
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from arbitration_confirm import load_fresh                                      # noqa: E402
from r1_info_ladder import r2_within, demean_by_cell                            # noqa: E402

ORACLES = ("pkiss", "probknot", "shapeknots")
FINE = ("recall", "precision", "f1", "n_pred_cross", "n_pred_total")
SPREAD_ON = ("recall", "precision", "f1", "n_pred_cross")
FLOOR_CELLS = 20


def blocks(r):
    """(B, S, D) feature blocks for one molecule, or None if any column is missing.

    B  F2 (k dummies + per-oracle recall_product + the fifteen fine columns) + f1c   -- the 0.196 bar
    S  unsigned dispersion across the three judges: std and range, per scalar        -- magnitude only
    D  argmax judge, argmin judge, and the three pairwise-difference signs           -- identity/sign only
    """
    po = r.get("per_oracle") or {}
    if sorted(po) != sorted(ORACLES):
        return None
    rp = [po[o].get("recall_product") for o in ORACLES]
    f1c = r.get("f1c")
    if any(x is None for x in rp) or f1c is None:
        return None
    k = sum(1 for x in rp if x >= 1.0)
    B = [1.0 if k == j else 0.0 for j in (1, 2, 3)] + [float(x) for x in rp]
    vals = {}
    for c in FINE:
        v = [po[o].get(c) for o in ORACLES]
        if any(x is None for x in v):
            return None
        vals[c] = [float(x) for x in v]
        B += vals[c]
    B.append(float(f1c))

    S = []
    for c in SPREAD_ON:
        v = vals[c]
        S += [st.pstdev(v), max(v) - min(v)]

    # signed structure is read off the f1 column -- the observable, F1-consistent one
    v = vals["f1"]
    hi, lo = max(range(3), key=lambda i: v[i]), min(range(3), key=lambda i: v[i])
    D = [1.0 if i == hi else 0.0 for i in range(3)] + [1.0 if i == lo else 0.0 for i in range(3)]
    for (i, j) in ((0, 1), (0, 2), (1, 2)):
        D.append(0.0 if v[i] == v[j] else (1.0 if v[i] > v[j] else -1.0))
    return B, S, D


def run_pop(name, path, draws, nperm, seed):
    rec, _sp, _meta = load_fresh(path, "panel_recall_product")
    cells = collections.defaultdict(list)
    cols, dropped = {}, 0
    nB = nS = nD = 0
    for r in rec:
        b = blocks(r)
        if b is None:
            dropped += 1
            continue
        nB, nS, nD = len(b[0]), len(b[1]), len(b[2])
        cols[id(r)] = b[0] + b[1] + b[2]
        cells[(r["puzzle"], r["method"])].append(r)
    rows = [(c, g) for c, g in cells.items() if len(g) >= 3]
    print(f"\n=== {name} ===", flush=True)
    print(f"  分子 {len(rec)},列不全被丢 {dropped};格 {len(rows)}(每格 >=3);列数 B={nB} S={nS} D={nD}")
    if len(rows) < FLOOR_CELLS:
        print(f"  低于下限 {FLOOR_CELLS} 格 -> 无数据")
        return {"no_data": True, "n_cells": len(rows)}
    y, X, ci = demean_by_cell(rows, cols)
    NT = nB + nS + nD
    dof = len(y) - NT - len(rows)
    print(f"  分子 {len(y)};格内去均值后自由度 {len(y)} − {NT} − {len(rows)} = {dof}")
    if dof <= 0:
        print("  自由度不足 -> 无数据")
        return {"no_data": True, "dof": dof}

    iB = list(range(nB))
    iS = list(range(nB, nB + nS))
    iD = list(range(nB + nS, NT))
    SETS = {"B": iB, "BS": iB + iS, "BD": iB + iD, "BSD": iB + iS + iD}

    rng = np.random.default_rng(seed)
    obs, nul = {}, {}
    for nm, idx in SETS.items():
        Xa = X[:, idx]
        obs[nm] = r2_within(y, Xa)
        vals = []
        for _ in range(nperm):
            yp = y.copy()
            for c in range(ci.max() + 1):
                m = ci == c
                yp[m] = rng.permutation(yp[m])
            vals.append(r2_within(yp, Xa))
        nul[nm] = (float(np.mean(vals)), float(np.quantile(vals, 0.975)))

    print(f"  {'集合':5s} {'列数':>4s} {'样本内R²':>10s} {'置换零':>9s} {'零97.5%':>9s} {'净R²':>9s}")
    for nm in ("B", "BS", "BD", "BSD"):
        print(f"  {nm:5s} {len(SETS[nm]):4d} {obs[nm]:10.4f} {nul[nm][0]:9.4f} {nul[nm][1]:9.4f} "
              f"{obs[nm]-nul[nm][0]:9.4f}" + ("   > 零97.5%" if obs[nm] > nul[nm][1] else "   未超过零"))

    viol = [(a, b) for a, b in (("B", "BS"), ("B", "BD"), ("BS", "BSD"), ("BD", "BSD"))
            if obs[b] < obs[a] - 1e-12]
    print(f"  自检(嵌套 ⇒ 样本内 R² 单调不减,精确代数):违反 {len(viol)} 处"
          + ("  OK" if not viol else f"  BROKEN {viol} —— 不报结果"))
    if viol:
        return {"estimator_broken": True, "violations": viol, "r2": obs}

    # net-R^2 increments, cluster bootstrap over cells. Net, not raw: the blocks differ in width.
    net = {nm: obs[nm] - nul[nm][0] for nm in SETS}

    def raw_inc(sel_rows, a, b):
        """RAW in-sample increment on a resample. The permutation offset is a constant (same feature
        counts, same cell structure), so it is estimated ONCE on the full sample below and subtracted
        from both the point estimate and the interval -- shifting a CI by a constant keeps its width."""
        yy, XX, _cc = demean_by_cell(sel_rows, cols)
        ra, rb = r2_within(yy, XX[:, SETS[a]]), r2_within(yy, XX[:, SETS[b]])
        return None if (ra is None or rb is None) else rb - ra

    out = {"n_cells": len(rows), "n_mol": int(len(y)), "dof": int(dof), "n_cols": {k: len(v) for k, v in SETS.items()},
           "r2": obs, "null": {k: v[0] for k, v in nul.items()},
           "null_p975": {k: v[1] for k, v in nul.items()}, "net": net}
    prng = random.Random(seed)
    TESTS = (("(a) 净ΔR²(BSD−BS) 带符号加在无符号之上", "BS", "BSD"),
             ("(b) 净ΔR²(BSD−BD) 无符号加在带符号之上", "BD", "BSD"),
             ("    净ΔR²(BS−B)  无符号相对基线", "B", "BS"),
             ("    净ΔR²(BD−B)  带符号相对基线", "B", "BD"))
    for lab, a, b in TESTS:
        off = nul[b][0] - nul[a][0]                    # what feature count alone buys, estimated once
        pt = raw_inc(rows, a, b) - off
        bs = []
        for _ in range(draws):
            sel = [rows[prng.randrange(len(rows))] for _ in range(len(rows))]
            v = raw_inc(sel, a, b)
            if v is not None:
                bs.append(v - off)
        bs.sort()
        lo, hi = bs[int(0.025 * len(bs))], bs[min(len(bs) - 1, int(0.975 * len(bs)))]
        excl = lo > 0 or hi < 0
        print(f"  {lab:42s} {pt:+.4f}  聚类CI [{lo:+.4f}, {hi:+.4f}]"
              f"{'  排除 0' if excl else '  含 0'}   (置换零偏移 {off:+.4f})", flush=True)
        out[a + "->" + b] = {"delta": pt, "ci": [lo, hi], "excludes_zero": bool(excl),
                             "positive": bool(lo > 0), "null_offset": off, "n_boot": len(bs)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=600)
    ap.add_argument("--nperm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "g4_signed_vs_spread.json"))
    a = ap.parse_args()
    D = json.load(open(os.path.join(HERE, "g4_signed_vs_spread_declaration.json")))
    POPS = (("样本3(主)", "panel_f1_third.json"), ("第四批(复现)", "panel_f1_fourth.json"),
            ("样本2(复现)", "panel_f1_fresh.json"))
    res = {"declaration_id": D["id"], "populations": {}}
    for nm, fn in POPS:
        res["populations"][fn] = run_pop(nm, os.path.join(HERE, fn), a.draws, a.nperm, a.seed)

    print("\n=== 判定(按冻结的打死条件)===")
    P = res["populations"]
    prim = P["panel_f1_third.json"]
    if prim.get("estimator_broken"):
        v = "estimator_broken"
    elif prim.get("no_data"):
        v = "no_data"
    else:
        a_pos = prim["BS->BSD"]["positive"]
        b_excl = prim["BD->BSD"]["excludes_zero"] and prim["BD->BSD"]["positive"]
        anyadd = prim["B->BS"]["positive"] or prim["B->BD"]["positive"]
        if not anyadd:
            v = "K3_whole_line_empty"
        elif b_excl:
            v = "K2_premise_contradicted"
        elif not a_pos:
            v = "K1_signed_adds_nothing_over_spread"
        else:
            v = "signed_beats_spread"
    reps = []
    for nm, fn in POPS[1:]:
        p = P[fn]
        reps.append((nm, "无数据" if p.get("no_data") else
                     "估计量坏了" if p.get("estimator_broken") else
                     ("复现" if p["BS->BSD"]["positive"] else "未复现")))
    print(f"  主人群(样本3)裁定:{v}")
    for nm, s in reps:
        print(f"  {nm}:{s}")
    res["verdict"] = v
    res["replication"] = dict(reps)
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
