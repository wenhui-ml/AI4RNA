#!/usr/bin/env python3
"""R1: where on the coarse-graining chain does the information about the TUBE die?

Declared in `r1_info_ladder_declaration.json`, pushed before anything was computed.

WHERE THE IDEA COMES FROM, AND WHY IT IS NOT DECORATION. Deng, Hani and Ma derived the Boltzmann equation
from Newtonian particle dynamics on arbitrarily long times (2026 Fields Medal). Reversible microscopic
dynamics, irreversible macroscopic equation; the bridge works by tracking only the one-particle distribution
and DISCARDING correlations, and the technical obstruction is the correlation that recollisions carry. The
accurate statement is not that information vanishes but that it MIGRATES from the marginal we track into the
high-order correlations we do not -- and the equation we can write down is exactly the one that has forgotten
it. ViennaRNA's partition function IS a Boltzmann distribution over structures, so this is the same object
rather than a metaphor, and our certificate is a chain of coarse-grainings each of which strictly loses
information:

    sequence -> ensemble P(structure|s) -> predicted pair set -> per-oracle crossing recall -> k -> tier

This week measured that the k>=1 boundary carries information about the tube while the k=max rung barely
does, and three separate hunts for a signal AT THE TOP returned zero. In this language that is one sentence:
the information survives down to "confirmed or not" and dies at the last rounding. This run tests that
sentence and locates the arrow.

THE SELF-CHECK I FIRST PROMISED WAS WRONG, AND THAT IS WHY THE ESTIMATOR LOOKS LIKE THIS. I said the chain
came with a built-in data-processing check. It does not, for Spearman: DPI governs mutual information, and a
coarser variable's rank correlation can easily EXCEED a finer one's, because a monotone collapse can remove
noise. Estimating mutual information directly is worse -- plug-in bias grows with the number of bins, and the
number of bins differs by rung, so the bias would masquerade as the ladder. So the rungs are built as
LITERALLY NESTED FEATURE SETS, F0 subset F1 subset F2, where in-sample R^2 is monotone non-decreasing by
exact algebra. THAT is the self-check: any rung scoring below a coarser one means the estimator is broken, and
the run aborts instead of reporting.

    F0  dummies for k = 1,2,3                                     the shipped confirm count
    F1  F0 + per-oracle recall_product                            the vector k is a function of
    F2  F1 + per-oracle recall, precision, f1, n_pred_cross, n_pred_total

F0's features are KEPT in F1 and F1's in F2 on purpose. If F1 held only the recall vector the inclusion would
fail -- k is a threshold function of it, not a linear one -- and with it the self-check.

Cell fixed effects (within-cell demeaning) absorb target difficulty and method composition by construction;
this project has been burned once by each (+4.75 -> +1.90). A within-cell permutation null subtracts the R^2
that feature count alone buys. The unit for every interval is the CELL.

COST: $0, offline, minutes. Columns already on disk; no oracle, no solver, no API.
"""
from __future__ import annotations

import argparse, collections, json, os, random, statistics as st, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)

from arbitration_confirm import load_fresh                                       # noqa: E402

ORACLES = ("pkiss", "probknot", "shapeknots")
FINE = ("recall", "precision", "f1", "n_pred_cross", "n_pred_total")
FLOOR_CELLS = 20


def feats(r):
    """Returns (F0, F1_extra, F2_extra) blocks for one molecule, or None if a column is missing."""
    po = r.get("per_oracle") or {}
    if sorted(po) != sorted(ORACLES):
        return None
    rp = [po[o].get("recall_product") for o in ORACLES]
    if any(x is None for x in rp):
        return None
    k = sum(1 for x in rp if x >= 1.0)
    f0 = [1.0 if k == j else 0.0 for j in (1, 2, 3)]          # k=0 is the reference level
    fine = []
    for o in ORACLES:
        for c in FINE:
            v = po[o].get(c)
            if v is None:
                return None
            fine.append(float(v))
    return f0, [float(x) for x in rp], fine


def r2_within(y, X):
    """In-sample R^2 of already-demeaned y on already-demeaned X, via least squares. No intercept."""
    sst = float(y @ y)
    if sst <= 0:
        return None
    if X.shape[1] == 0:
        return 0.0
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    return 1.0 - float(res @ res) / sst


def demean_by_cell(rows, cols):
    """Within-cell demeaning. Returns (y, X, cell_index) with cell means removed from every column."""
    ys, Xs, ci = [], [], []
    for ci_i, (_cell, grp) in enumerate(rows):
        y = np.array([r["wet"] for r in grp], dtype=float)
        X = np.array([cols[id(r)] for r in grp], dtype=float)
        ys.append(y - y.mean()); Xs.append(X - X.mean(axis=0)); ci += [ci_i] * len(grp)
    return np.concatenate(ys), np.vstack(Xs), np.array(ci)


def run_pop(name, path, draws, nperm, seed):
    rec, _sp, _meta = load_fresh(path, "panel_recall_product")
    cells = collections.defaultdict(list)
    dropped = 0
    cols = {}
    for r in rec:
        f = feats(r)
        if f is None:
            dropped += 1; continue
        cols[id(r)] = f[0] + f[1] + f[2]
        cells[(r["puzzle"], r["method"])].append(r)
    # a cell needs enough molecules that within-cell demeaning leaves degrees of freedom for F2
    NF = 3 + 3 + len(ORACLES) * len(FINE)
    rows = [(c, g) for c, g in cells.items() if len(g) >= 3]
    print(f"\n=== {name} ===")
    print(f"  分子 {len(rec)},列不全被丢的 {dropped};格 {len(rows)}(要求每格 >=3 条);特征总数 F2 = {NF}")
    if len(rows) < FLOOR_CELLS:
        print(f"  低于声明下限 {FLOOR_CELLS} 格 → 无数据"); return {"no_data": True, "n_cells": len(rows)}
    y, X, ci = demean_by_cell(rows, cols)
    dof = len(y) - NF - len(rows)
    print(f"  参与拟合的分子 {len(y)};格内去均值后自由度 {len(y)} − {NF} − {len(rows)} = {dof}")
    if dof <= 0:
        print("  自由度不足 → 无数据"); return {"no_data": True, "dof": dof}

    BLOCKS = {"F0": slice(0, 3), "F1": slice(0, 6), "F2": slice(0, NF)}
    obs, nul = {}, {}
    rng = np.random.default_rng(seed)
    for nm, sl in BLOCKS.items():
        obs[nm] = r2_within(y, X[:, sl])
        vals = []
        for _ in range(nperm):
            yp = y.copy()
            for c in range(ci.max() + 1):              # permute the tube score WITHIN each cell
                m = ci == c
                yp[m] = rng.permutation(yp[m])
            vals.append(r2_within(yp, X[:, sl]))
        nul[nm] = (float(np.mean(vals)), float(np.quantile(vals, 0.975)))

    print(f"  {'级':4s} {'样本内 R^2':>11s} {'置换零均值':>11s} {'零 97.5%':>10s} {'净 R^2':>9s}")
    for nm in ("F0", "F1", "F2"):
        print(f"  {nm:4s} {obs[nm]:11.4f} {nul[nm][0]:11.4f} {nul[nm][1]:10.4f} {obs[nm]-nul[nm][0]:9.4f}"
              f"{'   > 零 97.5%' if obs[nm] > nul[nm][1] else '   未超过零'}")

    # THE DECLARED SELF-CHECK: nested feature sets => in-sample R^2 monotone non-decreasing. Exact algebra.
    viol = [(a, b) for a, b in (("F0", "F1"), ("F1", "F2")) if obs[b] < obs[a] - 1e-12]
    print(f"  自检(嵌套 ⇒ 样本内 R^2 单调不减,精确代数):违反 {len(viol)} 处"
          + ("  ✓" if not viol else f"  ✗ {viol} —— 估计量坏了,不报结果"))
    if viol:
        return {"estimator_broken": True, "violations": viol, "r2": obs}

    # cluster bootstrap over cells for the two increments
    def inc(sel_rows, a, b):
        yy, XX, _ = demean_by_cell(sel_rows, cols)
        ra, rb = r2_within(yy, XX[:, BLOCKS[a]]), r2_within(yy, XX[:, BLOCKS[b]])
        return None if (ra is None or rb is None) else rb - ra
    out = {"n_cells": len(rows), "n_mol": int(len(y)), "dof": int(dof),
           "r2": obs, "null": {k: v[0] for k, v in nul.items()},
           "null_p975": {k: v[1] for k, v in nul.items()}}
    prng = random.Random(seed)
    for lab, a, b in (("D1  ΔR²(F1−F0) 取整成 k 销毁的", "F0", "F1"),
                      ("D2  ΔR²(F2−F1) 比召回向量更细的", "F1", "F2")):
        pt = inc(rows, a, b)
        bs = []
        for _ in range(draws):
            sel = [rows[prng.randrange(len(rows))] for _ in range(len(rows))]
            v = inc(sel, a, b)
            if v is not None:
                bs.append(v)
        bs.sort()
        lo, hi = bs[int(0.025 * len(bs))], bs[min(len(bs) - 1, int(0.975 * len(bs)))]
        print(f"  {lab:34s} {pt:+.4f}  聚类 CI [{lo:+.4f}, {hi:+.4f}]"
              f"{'  排除 0' if lo > 0 else '  跨过 0'}")
        out[a + "->" + b] = {"delta": pt, "ci": [lo, hi], "excludes_zero_positive": bool(lo > 0)}

    # siblings, NOT rungs: how much of F2's gain does each single shipped/observable column capture alone?
    print(f"  --- 兄弟列(不是阶梯上的一级):各自单独的净 R^2 ---")
    sib = {}
    for nm, key in (("panel_recall_product(出厂)", "panel_recall_product"), ("panel_f1(观测)", "panel_f1")):
        c2 = {id(r): [float(r[key])] for _c, g in rows for r in g if r.get(key) is not None}
        if len(c2) < len(y):
            print(f"  {nm:26s} 列不全,跳过"); continue
        yy, XX, cc = demean_by_cell(rows, c2)
        o = r2_within(yy, XX)
        vals = []
        for _ in range(nperm):
            yp = yy.copy()
            for c in range(cc.max() + 1):
                m = cc == c
                yp[m] = rng.permutation(yp[m])
            vals.append(r2_within(yp, XX))
        print(f"  {nm:26s} R² {o:.4f}  零均值 {np.mean(vals):.4f}  净 {o-np.mean(vals):+.4f}")
        sib[key] = {"r2": o, "null": float(np.mean(vals)), "net": o - float(np.mean(vals))}
    out["siblings"] = sib

    # reference ceiling from replicate noise -- explicitly not a primary
    within_var = float(np.var(y))
    sd_rep = 4.097 / (2 ** 0.5)
    ceil = 1.0 - (sd_rep ** 2) / within_var if within_var > 0 else None
    print(f"  --- 参考上限(不是主检验):复现噪声 spread 中位 4.097 → SD≈{sd_rep:.2f};"
          f"格内方差 {within_var:.2f} → 可解释上限 R² ≈ {ceil:.3f}(31 对重复太薄,只作参考)")
    out["reference_ceiling_r2"] = ceil
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--nperm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "r1_info_ladder.json"))
    a = ap.parse_args()
    D = json.load(open(os.path.join(HERE, "r1_info_ladder_declaration.json")))
    res = {"declaration": D, "populations": {}}
    POPS = (("第四批(主)", "panel_f1_fourth.json"), ("样本2(复现)", "panel_f1_fresh.json"),
            ("样本3(复现)", "panel_f1_third.json"))
    for nm, fn in POPS:
        res["populations"][fn] = run_pop(nm, os.path.join(HERE, fn), a.draws, a.nperm, a.seed)

    print(f"\n=== 判定(按声明)===")
    P = [res["populations"][fn] for _n, fn in POPS]
    if any(p.get("estimator_broken") for p in P):
        v = "estimator_broken"
    elif any(p.get("no_data") for p in P):
        v = "no_data"
    elif not all(p["r2"]["F2"] > p["null_p975"]["F2"] for p in P):
        v = "whole_chain_reads_nothing"
    else:
        d1 = all(p["F0->F1"]["excludes_zero_positive"] for p in P)
        d2 = all(p["F1->F2"]["excludes_zero_positive"] for p in P)
        v = ("information_dies_at_the_rounding" if d1 else
             "finer_is_better_but_not_by_rounding" if d2 else "information_dies_upstream")
    for (nm, fn) in POPS:
        p = res["populations"][fn]
        if p.get("no_data") or p.get("estimator_broken"):
            print(f"  {nm}: {'无数据' if p.get('no_data') else '估计量坏了'}"); continue
        print(f"  {nm}: D1 {p['F0->F1']['delta']:+.4f} [{p['F0->F1']['ci'][0]:+.4f},{p['F0->F1']['ci'][1]:+.4f}]"
              f"   D2 {p['F1->F2']['delta']:+.4f} [{p['F1->F2']['ci'][0]:+.4f},{p['F1->F2']['ci'][1]:+.4f}]")
    print(f"  【结论】{v} —— {D['outcomes_fixed_in_advance'][v]}")
    print(f"  声明的限制必须随结论走:{D['known_limitations_declared_upfront'][1]}")
    res["verdict"] = v
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n写出 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
