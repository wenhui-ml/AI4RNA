#!/usr/bin/env python3
"""REVIEW of M2/M2b before believing any of it. Five checks, each aimed at a defect this pipeline has
actually produced before.

R1  Are the statistics themselves right? spearman / permutation p / bootstrap vs scipy on the real vectors.
R2  THE C1 DEFECT. h2's `pair_accuracy_rows` DROPS prediction ties, so BC and BC_ed are scored on their own
    pair sets with different denominators. That is exactly what manufactured "the cheap tier beats the panel"
    in C1. Measure the per-block tie rate, then rescore every pair on the COMMON set with a prediction tie
    worth 0.5 (C1's fixed rule) and recompute arm1 and arm2 from scratch.
R3  DATA-QUANTITY CONFOUND. Cells per target run 5..10 and pairs per target vary with it. If the selected
    half is just "targets with more data", the split is about sample size, not signal. n_cells / n_designs /
    n_pairs were NOT among M2's ten candidates -- a gap in the screen.
R4  METHOD-COMPOSITION CONFOUND. Cells are (puzzle, method) over 13 methods with a very uneven mix. If the
    gain differs by method and the method mix at a target is stable across populations, cross-population rho
    is high without the TARGET being the causal unit. Partial out the method-share vector and recompute.
R5  RULE 48. Any new column must first show it is not matched by a GC count. Applied to the empirical
    invariant: does splitting on gc_cross, or on n_pairs, reproduce arm2's separation?

COST: $0, offline.
"""
from __future__ import annotations

import argparse, collections, json, math, os, random, statistics as st, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from e1_ensemble_marginals import base_block                                      # noqa: E402
from h1_heldout_ensemble_defect import fit_predict                                # noqa: E402
import h2_heldout_cluster_ci as H                                                 # noqa: E402
from m2_invariant_split import spearman, rho_ci, half_stats                       # noqa: E402
from m2b_gt_target_property import fold                                           # noqa: E402

POPS = (("样本2(主)", "panel_f1_fresh.json"), ("基准4560", "panel_f1_designs.json"),
        ("样本3", "panel_f1_third.json"), ("第四批", "panel_f1_fourth.json"))
PRIMARY = "panel_f1_fresh.json"


def acc_pair_of_blocks(rows, pA, pB, rule):
    """Score BOTH blocks. rule='drop' = h2's (each block its own set); rule='common05' = C1's fix."""
    okA = okB = 0.0
    n = tt = tA = tB = 0
    nA = nB = 0
    oA = oB = 0.0
    for _c, g in rows:
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                dw = g[a]["wet"] - g[b]["wet"]
                if dw == 0:
                    tt += 1; continue
                dA = pA[id(g[a])] - pA[id(g[b])]
                dB = pB[id(g[a])] - pB[id(g[b])]
                if dA == 0: tA += 1
                if dB == 0: tB += 1
                if rule == "drop":                        # h2 / M2 / M2b as run
                    if dA != 0:
                        nA += 1; oA += 1.0 if ((dw > 0) == (dA > 0)) else 0.0
                    if dB != 0:
                        nB += 1; oB += 1.0 if ((dw > 0) == (dB > 0)) else 0.0
                else:                                     # C1's fixed rule: common set, tie = 0.5
                    n += 1
                    okA += 0.5 if dA == 0 else (1.0 if ((dw > 0) == (dA > 0)) else 0.0)
                    okB += 0.5 if dB == 0 else (1.0 if ((dw > 0) == (dB > 0)) else 0.0)
    if rule == "drop":
        return ((oA / nA if nA else None), (oB / nB if nB else None), nA, nB, tt, tA, tB)
    return ((okA / n if n else None), (okB / n if n else None), n, n, tt, tA, tB)


def g_from(keep, M, splits, seed, rule):
    cells, cols = collections.defaultdict(list), {}
    for r, m in zip(keep, M):
        if "error" in m:
            continue
        rr = dict(r)
        cols[id(rr)] = base_block(r) + m["C"] + [m["ed_skel"]] + [m["ed_mfe"]]
        cells[(r["puzzle"], r["method"])].append(rr)
    rws = [(c, g) for c, g in cells.items() if len(g) >= 2]
    tg = sorted({c[0] for c, _g in rws})
    nB = len(base_block(keep[0]))
    iBC = list(range(nB + 2)); iED = iBC + [nB + 2]
    by_t = collections.defaultdict(list)
    for c, gp in rws:
        by_t[c[0]].append((c, gp))
    per_t = {t: [] for t in tg}
    rng = random.Random(seed); order = list(tg)
    S = dict(tie_tube=0, tie_A=0, tie_B=0, nA=0, nB=0)
    for _s in range(splits):
        rng.shuffle(order)
        cut = max(1, int(0.7 * len(order)))
        tr, te = set(order[:cut]), set(order[cut:])
        TR = [x for t in tr for x in by_t[t]]; TE = [x for t in te for x in by_t[t]]
        if not TR or not TE:
            continue
        PA = fit_predict(TR, TE, cols, iBC); PB = fit_predict(TR, TE, cols, iED)
        for t in te:
            aA, aB, nA, nB2, tt, tA, tB = acc_pair_of_blocks(by_t[t], PA, PB, rule)
            S["tie_tube"] += tt; S["tie_A"] += tA; S["tie_B"] += tB; S["nA"] += nA; S["nB"] += nB2
            if aA is None or aB is None or min(nA, nB2) < H.MIN_PAIRS_PER_TARGET:
                continue
            per_t[t].append(aB - aA)
    g = {t: st.fmean(v) for t, v in per_t.items() if len(v) >= H.MIN_SPLITS_PER_TARGET}
    return g, S, by_t, rws


def counts_per_target(by_t):
    out = {}
    for t, rows in by_t.items():
        nd = sum(len(g) for _c, g in rows)
        npair = sum(len(g) * (len(g) - 1) // 2 for _c, g in rows)
        out[t] = {"n_cells": len(rows), "n_designs": nd, "n_pairs": npair}
    return out


def method_shares(by_t, methods):
    out = {}
    for t, rows in by_t.items():
        c = collections.Counter()
        for (p, m), g in rows:
            c[m] += len(g)
        n = sum(c.values())
        out[t] = [c[m] / n for m in methods]
    return out


def residualise(g, X):
    """OLS residuals of g on the design matrix X (with intercept), over the common targets."""
    ts = sorted(set(g) & set(X))
    A = np.array([[1.0] + list(X[t]) for t in ts], dtype=float)
    y = np.array([g[t] for t in ts], dtype=float)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ beta
    return {t: float(v) for t, v in zip(ts, r)}


def arm2(G, sel_pops, primary):
    sel = {}
    for t in set().union(*[set(G[f]) for f in sel_pops]):
        v = [G[f][t] for f in sel_pops if t in G[f]]
        if len(v) >= 2:
            sel[t] = st.fmean(v)
    thr = st.median(list(sel.values()))
    gp = G[primary]
    hi = [gp[t] for t in gp if t in sel and sel[t] > thr]
    lo = [gp[t] for t in gp if t in sel and sel[t] <= thr]
    return sel, thr, half_stats(hi), half_stats(lo), hi, lo


def perm_diff_p(hi, lo, seed=20260825, N=20000):
    v = hi + lo; d = st.fmean(hi) - st.fmean(lo)
    rng = random.Random(seed); ge = 0
    for _ in range(N):
        rng.shuffle(v)
        if abs(st.fmean(v[:len(hi)]) - st.fmean(v[len(hi):])) >= abs(d):
            ge += 1
    return d, (ge + 1) / (N + 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "m2b_review_audit.json"))
    a = ap.parse_args()
    R = {}
    KE, ME = {}, {}
    print("=== 折叠 ===", flush=True)
    for nm, fn in POPS:
        KE[fn], ME[fn] = fold(fn, a.workers)

    # ---------------- R2: the C1 tie defect ----------------------------------------------------------
    print("\n=== R2 C1 的并列缺陷:两个块是不是各按各的对集打分 ===", flush=True)
    GD, GC, BYT = {}, {}, {}
    for nm, fn in POPS:
        gd, sd_, by_t, rws = g_from(KE[fn], ME[fn], a.splits, a.seed, "drop")
        gc, sc, _b, _r = g_from(KE[fn], ME[fn], a.splits, a.seed, "common05")
        GD[fn], GC[fn], BYT[fn] = gd, gc, by_t
        print(f"  {nm:10s} 试管并列 {sd_['tie_tube']}  预测并列 BC {sd_['tie_A']} BC_ed {sd_['tie_B']}"
              f"  分母 BC {sd_['nA']} vs BC_ed {sd_['nB']}"
              f"  差 {abs(sd_['nA']-sd_['nB'])} = {abs(sd_['nA']-sd_['nB'])/max(1,sd_['nA'])*100:.4f}%")
        print(f"  {'':10s} g_t 均值 丢弃口径 {st.fmean(list(gd.values())):+.5f}"
              f"  |  共同对集+0.5 口径 {st.fmean(list(gc.values())):+.5f}"
              f"  两口径 rho {spearman([gd[t] for t in sorted(set(gd)&set(gc))],[gc[t] for t in sorted(set(gd)&set(gc))]):+.5f}", flush=True)
        R.setdefault("R2", {})[fn] = {"ties": sd_, "mean_drop": round(st.fmean(list(gd.values())), 5),
                                      "mean_common": round(st.fmean(list(gc.values())), 5)}

    for tag, G in (("丢弃口径(实际跑的)", GD), ("共同对集+0.5 口径(C1 修正)", GC)):
        print(f"\n  --- arm1 六对 rho,{tag} ---")
        rs = []
        for i, (n1, f1) in enumerate(POPS):
            for n2, f2 in POPS[i + 1:]:
                ts = sorted(set(G[f1]) & set(G[f2]))
                r = spearman([G[f1][t] for t in ts], [G[f2][t] for t in ts])
                ci = rho_ci([G[f1][t] for t in ts], [G[f2][t] for t in ts], 11, 0.95)
                rs.append(r)
                print(f"    {n1:10s} vs {n2:10s} n {len(ts):3d} rho {r:+.4f} 95% {ci}"
                      f" {'排除0' if ci[0]>0 or ci[1]<0 else '含0'}")
        print(f"    六对均值 {st.fmean(rs):+.4f}")
        sel, thr, sh, sl, hi, lo = arm2(G, [f for _n, f in POPS if f != PRIMARY], PRIMARY)
        d, p = perm_diff_p(list(hi), list(lo))
        print(f"    arm2 阈值 {thr:+.5f} 高半 n {sh['n']} {sh['mean']:+.4f} (效应/MDE {sh['eff_over_mde']:.2f})"
              f"  低半 n {sl['n']} {sl['mean']:+.4f}  半间差 {d:+.4f} 置换 p {p:.5f}")
        R.setdefault("arm", {})[tag] = {"mean_rho": round(st.fmean(rs), 4), "thr": thr,
                                        "high": sh, "low": sl, "diff": round(d, 4), "perm_p": p}
        json.dump(R, open(a.out, "w"), indent=1, ensure_ascii=False)

    G = GC          # everything below uses the C1-corrected scoring
    print("\n  **以下全部改用共同对集 +0.5 口径。**")

    # ---------------- R3: data-quantity confound -----------------------------------------------------
    print("\n=== R3 数据量混杂:被选中的一半是不是只是【数据更多的靶标】 ===")
    CN = {fn: counts_per_target(BYT[fn]) for _n, fn in POPS}
    sel, thr, sh, sl, hi, lo = arm2(G, [f for _n, f in POPS if f != PRIMARY], PRIMARY)
    for key in ("n_cells", "n_designs", "n_pairs"):
        ts = sorted(set(sel) & set(CN[PRIMARY]))
        r = spearman([sel[t] for t in ts], [CN[PRIMARY][t][key] for t in ts])
        ci = rho_ci([sel[t] for t in ts], [CN[PRIMARY][t][key] for t in ts], 13, 0.95)
        print(f"  经验不变量 vs {key:10s} rho {r:+.4f} 95% {ci} {'排除0' if ci[0]>0 or ci[1]<0 else '含0'}")
        R.setdefault("R3", {})[key] = {"rho": round(r, 4), "ci": ci}
    # does splitting on n_pairs reproduce the separation?
    for key in ("n_pairs", "n_cells"):
        col = {t: CN[PRIMARY][t][key] for t in CN[PRIMARY]}
        th2 = st.median([col[t] for t in G[PRIMARY] if t in col])
        h2 = [G[PRIMARY][t] for t in G[PRIMARY] if t in col and col[t] > th2]
        l2 = [G[PRIMARY][t] for t in G[PRIMARY] if t in col and col[t] <= th2]
        d2, p2 = perm_diff_p(list(h2), list(l2))
        print(f"  按 {key} 中位切分:高 {st.fmean(h2):+.4f}(n {len(h2)}) 低 {st.fmean(l2):+.4f}(n {len(l2)})"
              f"  半间差 {d2:+.4f} 置换 p {p2:.4f}  {'**也切得开 -> 混杂**' if p2 < 0.05 else '切不开 ✓'}")
        R.setdefault("R3_split", {})[key] = {"diff": round(d2, 4), "p": p2}

    # ---------------- R5: rule 48, matched by a GC count? --------------------------------------------
    print("\n=== R5 第48条:这个切分会不会被一个 GC 计数配上 ===")
    gcm = {}
    for _n, fn in POPS:
        acc = collections.defaultdict(list)
        for r, m in zip(KE[fn], ME[fn]):
            if "error" not in m:
                acc[r["puzzle"]].append(m["C"][0])
        gcm[fn] = {t: st.fmean(v) for t, v in acc.items() if v}
    ts = sorted(set(sel) & set(gcm[PRIMARY]))
    r = spearman([sel[t] for t in ts], [gcm[PRIMARY][t] for t in ts])
    ci = rho_ci([sel[t] for t in ts], [gcm[PRIMARY][t] for t in ts], 15, 0.95)
    print(f"  经验不变量 vs 交叉茎 GC 计数 rho {r:+.4f} 95% {ci} {'排除0' if ci[0]>0 or ci[1]<0 else '含0'}")
    th3 = st.median([gcm[PRIMARY][t] for t in G[PRIMARY] if t in gcm[PRIMARY]])
    h3 = [G[PRIMARY][t] for t in G[PRIMARY] if t in gcm[PRIMARY] and gcm[PRIMARY][t] > th3]
    l3 = [G[PRIMARY][t] for t in G[PRIMARY] if t in gcm[PRIMARY] and gcm[PRIMARY][t] <= th3]
    d3, p3 = perm_diff_p(list(h3), list(l3))
    print(f"  按 GC 计数中位切分:高 {st.fmean(h3):+.4f} 低 {st.fmean(l3):+.4f} 半间差 {d3:+.4f}"
          f" 置换 p {p3:.4f}  {'**也切得开 -> 第48条未过**' if p3 < 0.05 else '切不开 ⇒ 第48条通过'}")
    R["R5"] = {"rho_gc": round(r, 4), "ci": ci, "split_diff": round(d3, 4), "p": p3}

    # ---------------- R4: method-composition confound -------------------------------------------------
    print("\n=== R4 方法构成混杂:格是 (靶标, 方法),13 个方法且分布极不均 ===")
    meth = sorted({c[1] for _n, fn in POPS for rows in BYT[fn].values() for (c, _g) in rows})
    print(f"  方法 {len(meth)} 个")
    MS = {fn: method_shares(BYT[fn], meth) for _n, fn in POPS}
    # is the method mix at a target stable across populations?
    stab = []
    for i, (n1, f1) in enumerate(POPS):
        for n2, f2 in POPS[i + 1:]:
            tt = sorted(set(MS[f1]) & set(MS[f2]))
            v = [float(np.corrcoef(MS[f1][t], MS[f2][t])[0, 1]) for t in tt
                 if np.std(MS[f1][t]) > 0 and np.std(MS[f2][t]) > 0]
            stab.append(st.fmean(v))
    print(f"  同一靶标在两人群间的方法构成相关(均值) {st.fmean(stab):+.4f}"
          f"  ⇒ 方法构成{'高度稳定,是个真混杂候选' if st.fmean(stab) > 0.5 else '不稳定'}")
    RES = {fn: residualise(G[fn], MS[fn]) for _n, fn in POPS}
    rs2 = []
    for i, (n1, f1) in enumerate(POPS):
        for n2, f2 in POPS[i + 1:]:
            tt = sorted(set(RES[f1]) & set(RES[f2]))
            rr = spearman([RES[f1][t] for t in tt], [RES[f2][t] for t in tt])
            ci = rho_ci([RES[f1][t] for t in tt], [RES[f2][t] for t in tt], 17, 0.95)
            rs2.append(rr)
            print(f"    {n1:10s} vs {n2:10s} 扣掉方法构成后 rho {rr:+.4f} 95% {ci}"
                  f" {'排除0' if ci[0]>0 or ci[1]<0 else '含0'}")
    print(f"  扣掉方法构成后六对均值 {st.fmean(rs2):+.4f}(原 {R['arm']['共同对集+0.5 口径(C1 修正)']['mean_rho']:+.4f})")
    R["R4"] = {"method_mix_stability": round(st.fmean(stab), 4), "mean_rho_resid": round(st.fmean(rs2), 4),
               "n_methods": len(meth)}
    json.dump(R, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
