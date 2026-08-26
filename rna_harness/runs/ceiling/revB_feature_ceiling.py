#!/usr/bin/env python3
"""REVIEW B: the boosting proposal, and the claim it should have been.

WHAT I SAID, AND WHY IT IS WRONG. I claimed M4c/M4d is a "boostability diagnostic": GC fails on the
same targets every batch (6/6) so it cannot be boosted, the panel fails on different targets (2/6) so
it can. That is wrong twice over.

  (i) AdaBoost's weak-learning condition is about whether SOME learner in the class beats chance on the
      REWEIGHTED distribution. One learner failing consistently is not an obstruction -- it is exactly
      the situation the ensemble exists for, provided another column succeeds there.
  (ii) More seriously, boosting over a FIXED feature set cannot manufacture information the features do
      not carry. If the 22 columns hold 23.7% of the explainable variance, no reweighting reaches 76.3%.

So the question that should have been asked is the one measured here:

  **How much of the ceiling is reachable by ANY function of the existing 22 columns, rather than by the
  linear combination we happen to fit?**

and a second one the occupancy table quietly assumes:

  **0.1651 is an IN-SAMPLE R^2 minus a within-cell permutation null (r1_info_ladder.py). That is a
  first-order overfit correction, not a held-out estimate. What is the held-out value?**

Arms: OLS in-sample(net), OLS target-grouped CV, gradient-boosted stumps CV, k-NN CV, and a
per-column reweighted check that is the ACTUAL weak-learning condition.

Offline, $0.
"""
from __future__ import annotations
import argparse, collections, json, math, os, random, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from e1_ensemble_marginals import base_block, load_pop                        # noqa: E402

POPS = (("样本2(主)", "panel_f1_fresh.json"), ("基准4560", "panel_f1_designs.json"),
        ("样本3", "panel_f1_third.json"), ("第四批", "panel_f1_fourth.json"))
CEILING = 0.6956


def r2_within(y, X):
    if X.shape[1] == 0 or len(y) <= X.shape[1]:
        return 0.0
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    ss = float(np.sum(y * y))
    return 0.0 if ss <= 0 else 1.0 - float(np.sum((y - X @ beta) ** 2)) / ss


def r2_of_pred(y, p):
    ss = float(np.sum(y * y))
    return 0.0 if ss <= 0 else 1.0 - float(np.sum((y - p) ** 2)) / ss


def load(fn):
    rec, _s, _m = load_pop(fn)
    cells, cols = collections.defaultdict(list), {}
    for r in rec:
        b = base_block(r)
        if b is None:
            continue
        cols[id(r)] = b
        cells[(r["puzzle"], r["method"])].append(r)
    rws = [(c, g) for c, g in cells.items() if len(g) >= 3]
    ys, Xs, ci, tg = [], [], [], []
    for k, (cell, grp) in enumerate(rws):
        y = np.array([r["wet"] for r in grp], float)
        X = np.array([cols[id(r)] for r in grp], float)
        ys.append(y - y.mean()); Xs.append(X - X.mean(axis=0))
        ci += [k] * len(grp); tg += [cell[0]] * len(grp)
    if not ys:
        return None
    return np.concatenate(ys), np.vstack(Xs), np.array(ci), np.array(tg), len(rws)


# ---- gradient-boosted regression stumps, written out so nothing is hidden in a library default ----
def stump(X, g, idx):
    best = (float("inf"), 0, 0.0, 0.0, 0.0)
    for j in range(X.shape[1]):
        v = X[idx, j]
        qs = np.unique(np.quantile(v, [.1, .25, .5, .75, .9]))
        for t in qs:
            m = v <= t
            if m.sum() < 5 or (~m).sum() < 5:
                continue
            lo, hi = g[idx][m].mean(), g[idx][~m].mean()
            loss = float(((g[idx][m] - lo) ** 2).sum() + ((g[idx][~m] - hi) ** 2).sum())
            if loss < best[0]:
                best = (loss, j, float(t), float(lo), float(hi))
    return best


def gb_fit(X, y, rounds, lr, seed):
    rng = np.random.default_rng(seed)
    trees, pred = [], np.zeros(len(y))
    n = len(y)
    for _ in range(rounds):
        g = y - pred
        idx = rng.choice(n, size=max(50, int(.7 * n)), replace=False)
        _l, j, t, lo, hi = stump(X, g, idx)
        upd = np.where(X[:, j] <= t, lo, hi)
        pred = pred + lr * upd
        trees.append((j, t, lo, hi))
    return trees


def gb_pred(trees, X, lr):
    p = np.zeros(len(X))
    for j, t, lo, hi in trees:
        p = p + lr * np.where(X[:, j] <= t, lo, hi)
    return p


def knn_pred(Xtr, ytr, Xte, k):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    out = np.empty(len(B))
    for i in range(len(B)):
        d = ((A - B[i]) ** 2).sum(1)
        out[i] = ytr[np.argpartition(d, min(k, len(d) - 1))[:k]].mean()
    return out


def group_folds(tg, nfold, seed):
    u = sorted(set(tg.tolist())); random.Random(seed).shuffle(u)
    a = {t: i % nfold for i, t in enumerate(u)}
    return np.array([a[t] for t in tg])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nperm", type=int, default=120)
    ap.add_argument("--nfold", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "revB_feature_ceiling.json"))
    a = ap.parse_args()
    print("=== REVIEW B:22 列到底还剩多少没被榨出来,以及 0.1651 是不是留出值 ===")
    print("  声明:本文件是【事后复核】,不是预注册实验。它检验的是我自己上一条消息里的两个断言。\n")
    RES = {}
    for nm, fn in POPS:
        d = load(fn)
        if d is None:
            print(f"  {nm}: 无数据"); continue
        y, X, ci, tg, ncell = d
        rng = np.random.default_rng(a.seed)
        print(f"=== {nm} ===")
        print(f"  分子 {len(y)}  格 {ncell}  靶标 {len(set(tg.tolist()))}  列 {X.shape[1]}")

        ins = r2_within(y, X)
        nul = []
        for _ in range(a.nperm):
            yp = y.copy()
            for c in range(ci.max() + 1):
                m = ci == c
                yp[m] = rng.permutation(yp[m])
            nul.append(r2_within(yp, X))
        net_ins = ins - float(np.mean(nul))

        fold = group_folds(tg, a.nfold, a.seed)
        po, pg, pk = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y))
        for f in range(a.nfold):
            te, tr = fold == f, fold != f
            if tr.sum() < X.shape[1] + 10 or te.sum() == 0:
                po[te] = pg[te] = pk[te] = 0.0; continue
            beta, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
            po[te] = X[te] @ beta
            trees = gb_fit(X[tr], y[tr], a.rounds, a.lr, a.seed + f)
            pg[te] = gb_pred(trees, X[te], a.lr)
            pk[te] = knn_pred(X[tr], y[tr], X[te], 25)
        cv_ols, cv_gb, cv_knn = r2_of_pred(y, po), r2_of_pred(y, pg), r2_of_pred(y, pk)

        print(f"  {'臂':<34s} {'R^2':>9s}   注")
        print(f"  {'样本内 OLS(减置换零)= 现行口径':<34s} {net_ins:9.4f}   <- 0.1651 是这一行的量")
        print(f"  {'留出 OLS(按靶标分组 CV)':<34s} {cv_ols:9.4f}   <- 同一模型,同一列,只是留出")
        print(f"  {'留出 梯度提升树桩':<34s} {cv_gb:9.4f}   <- 非线性能不能再榨出东西")
        print(f"  {'留出 k-NN(k=25)':<34s} {cv_knn:9.4f}   <- 特征空间里的无参数上界")
        print(f"  可解释上限(r1b)                     {CEILING:9.4f}")
        print(f"  ⇒ 现行口径占上限 {net_ins/CEILING*100:5.1f}%;**留出口径占 {max(cv_ols,cv_gb,cv_knn)/CEILING*100:5.1f}%**")
        print(f"  ⇒ 非线性相对线性的增量(留出):{cv_gb-cv_ols:+.4f}")

        # the ACTUAL weak-learning condition, on the reweighted distribution
        res = y - po
        w = res ** 2
        hard = np.argsort(-w)[:max(30, len(w) // 4)]
        beat = []
        for j in range(X.shape[1]):
            xv = X[hard, j]
            if np.std(xv) < 1e-12:
                continue
            rr = float(np.corrcoef(xv, y[hard])[0, 1])
            if not math.isnan(rr):
                beat.append((abs(rr), j, rr))
        beat.sort(reverse=True)
        nb = sum(1 for b in beat if b[0] > 2 / math.sqrt(len(hard)))
        print(f"  弱学习条件(真正的那条):在 OLS 残差最大的 {len(hard)} 个分子上,"
              f"**{nb}/{X.shape[1]}** 根列的 |r| 超过 2/sqrt(n)"
              f";最强 |r|={beat[0][0]:.3f}(列 {beat[0][1]})" if beat else "  弱学习条件:无可用列")

        RES[fn] = {"pop": nm, "n": int(len(y)), "cells": int(ncell),
                   "targets": int(len(set(tg.tolist()))), "cols": int(X.shape[1]),
                   "net_insample": round(net_ins, 4), "cv_ols": round(cv_ols, 4),
                   "cv_gb": round(cv_gb, 4), "cv_knn": round(cv_knn, 4),
                   "nonlinear_gain": round(cv_gb - cv_ols, 4),
                   "occ_insample": round(net_ins / CEILING, 4),
                   "occ_heldout": round(max(cv_ols, cv_gb, cv_knn) / CEILING, 4),
                   "weak_cols_on_hard": int(nb) if beat else None}
        print()

    if RES:
        ins = [v["net_insample"] for v in RES.values()]
        ho = [max(v["cv_ols"], v["cv_gb"], v["cv_knn"]) for v in RES.values()]
        gain = [v["nonlinear_gain"] for v in RES.values()]
        print("=== 合并 ===")
        print(f"  样本内净 R^2 均值 {np.mean(ins):.4f}  ⇒ 占上限 {np.mean(ins)/CEILING*100:.1f}%")
        print(f"  留出最好      均值 {np.mean(ho):.4f}  ⇒ 占上限 {np.mean(ho)/CEILING*100:.1f}%")
        print(f"  非线性增量    均值 {np.mean(gain):+.4f},人群里为正的 {sum(1 for g in gain if g>0)}/{len(gain)}")
        v = ("features_exhausted__new_columns_required" if np.mean(gain) <= 0.01
             else "nonlinearity_left_on_the_table")
        print(f"\n=== 裁定 ===\n  **{v}**")
        RES["_summary"] = {"verdict": v, "mean_insample": round(float(np.mean(ins)), 4),
                           "mean_heldout": round(float(np.mean(ho)), 4),
                           "mean_nonlinear_gain": round(float(np.mean(gain)), 4)}
    json.dump(RES, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
