#!/usr/bin/env python3
"""M31: can a predictor tell you which compensatory flips will rescue, inside one design?

Declaration m31_which_flips_rescue_declaration.json, frozen and pushed BEFORE this file existed.
Offline, $0 -- the same interventions as M26, the neural scores cached by M26c, the selection
labels of M30 and the leakage cut of M26d. No API, no GPU, no new data.

The causal contrast in this paper conditions on design cell, helix depth and crossing, because
depth confounds pair type. The question an experimenter asks is a different one and needs a
different conditioning set: given ONE design and a budget, which flips should be built? Depth is
information they already hold, so removing it would be removing part of the answer. Same corpus,
two questions, two conditioning sets -- stated, not blurred.

K1 refuses the reframe if no predictor ranks within a design. K2 refuses it if none beats a
depth-only baseline, which costs nothing.
"""
from __future__ import annotations

import collections, csv, json, math, os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m26_model_scored as M26

NEURAL_CACHE = os.path.join(HERE, "m26c_pneural_cache.npz")
NMIN, THRESH_MAIN, TOPK_FRAC = 8, 0.5, 0.25
THRESH_SWEEP = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
NBOOT, NPERM = 2000, 2000
PREDS = (("P_vienna", "pmut"), ("dP_vienna", "dp"),
         ("P_neural", "nmut"), ("dP_neural", "ndp"),
         ("depth_baseline", "depth"))
LOG = []


def log(s=""):
    print(s, flush=True); LOG.append(s)


def cz(v):
    """centre and normalise; returns None when the column is constant."""
    v = np.asarray(v, float) - np.mean(v)
    n = float(np.sqrt((v * v).sum()))
    return v / n if n > 0 else None


def boot_ci(vals, lo=0.025, hi=0.975):
    v = np.sort(np.asarray([x for x in vals if x is not None and np.isfinite(x)], float))
    if len(v) < 20:
        return [None, None]
    return [round(float(v[int(lo * len(v))]), 5), round(float(v[int(hi * len(v))]), 5)]


# ------------------------------------------------------------------ cells
def make_cells(recs, nmin=NMIN):
    by = collections.defaultdict(list)
    for r in recs:
        by[r["design"]].append(r)
    return [(k, v) for k, v in by.items() if len(v) >= nmin]


def ranked(cells, key):
    """per cell: the centred, normalised rank vector of `key`. None when constant."""
    out = []
    for k, rs in cells:
        out.append((k, rs, cz(rankdata([r[key] for r in rs]))))
    return out


def weighted_rho(cells, key, ry_of):
    """weighted mean within-cell Spearman, weight = cell size. Also the per-cell values."""
    num = den = 0.0; per = []
    for k, rs in cells:
        rx = cz(rankdata([r[key] for r in rs]))
        ry = ry_of[k]
        if rx is None or ry is None:
            continue
        rho = float(rx @ ry)
        per.append((k, len(rs), rho))
        num += len(rs) * rho; den += len(rs)
    return (num / den if den else None), per


def ry_map(cells, ykey="rf"):
    return {k: cz(rankdata([r[ykey] for r in rs])) for k, rs in cells}


def arm_A(cells, label, seed, do_perm=True):
    """within-design ranking, every predictor, with a clustered CI and a rule-80 permutation."""
    ry = ry_map(cells)
    pz = sorted({rs[0]["puzzle"] for _k, rs in cells})
    bypz = collections.defaultdict(list)
    for k, rs in cells:
        bypz[rs[0]["puzzle"]].append((k, rs))
    out = {}
    for nm, key in PREDS:
        obs, per = weighted_rho(cells, key, ry)
        if obs is None:
            out[nm] = None; continue
        rng = np.random.default_rng(seed)
        bs = []
        for _ in range(NBOOT):
            ix = rng.integers(0, len(pz), len(pz))
            sub = [c for q in ix for c in bypz[pz[q]]]
            v, _p = weighted_rho(sub, key, ry)
            if v is not None:
                bs.append(v)
        ci = boot_ci(bs)
        pval = None
        if do_perm:
            rp = np.random.default_rng(seed + 1); ge = 0
            rx_pre = {k: cz(rankdata([r[key] for r in rs])) for k, rs in cells}
            for _ in range(NPERM):
                num = den = 0.0
                for k, rs in cells:
                    rx = rx_pre[k]; y = ry[k]
                    if rx is None or y is None:
                        continue
                    yp = y[rp.permutation(len(y))]
                    num += len(rs) * float(rx @ yp); den += len(rs)
                if den and abs(num / den) >= abs(obs):
                    ge += 1
            pval = (ge + 1) / (NPERM + 1)
        out[nm] = {"rho": round(obs, 5), "ci": ci, "p_perm": pval,
                   "n_cells": len(per), "n_records": sum(x[1] for x in per),
                   "median_cell_rho": round(float(np.median([x[2] for x in per])), 5),
                   "frac_cells_positive": round(float(np.mean([x[2] > 0 for x in per])), 4)}
        pv = "" if pval is None else f"  置换 p {pval:.5f}"
        log(f"  [{label}] {nm:16s} rho {obs:+.5f}  CI {ci}{pv}  "
            f"格 {len(per)}  格内为正 {out[nm]['frac_cells_positive']:.2f}")
    return out


def main():
    t0 = time.time()
    log("=== M31:预测哪些补偿翻转会 rescue ===")
    log("  声明已冻结并推送于本文件存在之前。离线 $0,不用新数据。")
    log("  因果问题条件在(设计格, 深度, 交叉)上;预测问题只条件在设计格上 ——")
    log("  深度是实验者本来就有的信息,把它条件掉等于把答案的一部分扔掉。\n")
    OUT = {"NMIN": NMIN, "THRESH_MAIN": THRESH_MAIN, "TOPK_FRAC": TOPK_FRAC}

    recs, _dr = M26.build()
    z = np.load(NEURAL_CACHE, allow_pickle=True)
    pn = {k: v for k, v in zip(z["keys"], z["mats"])}
    miss = [r for r in recs if r["mut_seq"] not in pn or r["wt_seq"] not in pn]
    if miss:
        log(f"  **神经缓存缺 {len(miss)} 条 ⇒ 作废**"); return 1
    for r in recs:
        i, j = min(r["a"], r["b"]), max(r["a"], r["b"])
        r["nmut"] = float(pn[r["mut_seq"]][i, j])
        r["nwt"] = float(pn[r["wt_seq"]][i, j])
        r["ndp"] = r["nmut"] - r["nwt"]
    del pn, z
    log(f"  纳入 {len(recs)} 次干预,{len({r['design'] for r in recs})} 个设计格")

    cells = make_cells(recs)
    ncell_rec = sum(len(v) for _k, v in cells)
    log(f"  至少 {NMIN} 个翻转的格 {len(cells)},覆盖 {ncell_rec} 次干预 "
        f"({100*ncell_rec/len(recs):.1f}%)")
    OUT["n_records"] = len(recs)
    OUT["n_cells_used"] = len(cells)
    OUT["n_records_in_cells"] = ncell_rec

    # ---------------- A: within-design ranking
    log("\n=== A 格内排名:给定这条设计,预测器能不能把会 rescue 的翻转排到前面 ===")
    OUT["A_within_design"] = arm_A(cells, "格内", seed=310)

    # A': the same statistic under the CAUSAL conditioning set, to show they differ
    log("\n=== A' 同一个统计量,换成因果那套条件集(设计格 x 深度 x 交叉)===")
    log("  这不是同一个问题。放在这里是为了让两者的差别有数,而不是靠说。")
    byfull = collections.defaultdict(list)
    for r in recs:
        byfull[(r["design"], min(r["depth"], 3), r["crossed"])].append(r)
    fcells = [(k, v) for k, v in byfull.items() if len(v) >= NMIN]
    log(f"  这样切出来的层里满足 >= {NMIN} 的只有 {len(fcells)} 个,"
        f"覆盖 {sum(len(v) for _k,v in fcells)} 次干预")
    OUT["Aprime_causal_conditioning"] = arm_A(fcells, "因果层内", seed=311) if fcells else None

    # ---------------- B: detection, threshold-swept
    log("\n=== B 二分类检出(连续统计量为主,这一节是次要且要扫阈值)===")
    rfall = np.array([r["rf"] for r in recs], float)
    log(f"  rescue factor 分位 min {rfall.min():.3f} p10 {np.quantile(rfall,.1):.3f} "
        f"中位 {np.median(rfall):.3f} p90 {np.quantile(rfall,.9):.3f} max {rfall.max():.3f}")
    sweep = []
    for th in THRESH_SWEEP:
        base = float((rfall >= th).mean())
        row = {"threshold": th, "base_rate": round(base, 4)}
        for nm, key in PREDS:
            num = den = 0.0
            for _k, rs in cells:
                y = np.array([r["rf"] >= th for r in rs])
                if y.all() or not y.any():
                    continue
                x = np.array([r[key] for r in rs], float)
                rk = rankdata(x)
                npos, nneg = int(y.sum()), int((~y).sum())
                auc = (rk[y].sum() - npos * (npos + 1) / 2) / (npos * nneg)
                w = npos * nneg
                num += w * auc; den += w
            row[nm] = round(num / den, 5) if den else None
        sweep.append(row)
        log(f"  阈值 {th:.1f}  正例率 {base:.4f}   " +
            "  ".join(f"{nm.split('_')[0]}:{row[nm]}" for nm, _k in PREDS if row[nm] is not None))
    OUT["B_auroc_sweep"] = sweep
    main_row = next(r for r in sweep if abs(r["threshold"] - THRESH_MAIN) < 1e-9)
    K0 = bool(not (0.05 <= main_row["base_rate"] <= 0.95))
    log(f"  **K0(主阈值下正例率越界 ⇒ 二分类作废)= {K0}**")
    # does the ordering of the two model classes survive the sweep
    order = [(r["P_neural"] or 0) >= (r["P_vienna"] or 0) for r in sweep]
    K3 = bool(len(set(order)) > 1)
    log(f"  两个模型类的先后在扫描中{'翻过' if K3 else '没翻'} ⇒ **K3 = {K3}**")

    # ---------------- C: the decision claim
    log("\n=== C 决定性主张:只造排名前 k 个,能捞回多少会 rescue 的翻转 ===")
    # the depth baseline is given its best global direction -- the conservative choice for us
    dsum = 0.0
    for _k, rs in cells:
        y = np.array([r["rf"] >= THRESH_MAIN for r in rs])
        if y.all() or not y.any():
            continue
        rk = rankdata([r["depth"] for r in rs])
        npos, nneg = int(y.sum()), int((~y).sum())
        dsum += (rk[y].sum() - npos * (npos + 1) / 2) / (npos * nneg) - 0.5
    depth_sign = 1.0 if dsum >= 0 else -1.0
    log(f"  深度基线取全局最有利方向 sign={depth_sign:+.0f}(这是对我们不利的一侧,故意的)")

    def topk_recovery(cells, key, sign=1.0, thresh=THRESH_MAIN, frac=TOPK_FRAC):
        num = den = 0.0; rnd_num = 0.0; rows = []
        for k, rs in cells:
            y = np.array([r["rf"] >= thresh for r in rs])
            npos, n = int(y.sum()), len(rs)
            if npos == 0 or npos == n:
                continue
            kk = max(2, math.ceil(frac * n))
            x = sign * np.array([r[key] for r in rs], float)
            top = np.argsort(-x, kind="stable")[:kk]
            rec = float(y[top].sum()) / npos
            num += npos * rec; den += npos; rnd_num += npos * (kk / n)
            rows.append({"cell": list(k), "n": n, "k": kk, "n_pos": npos, "rec": round(rec, 4)})
        return ((num / den if den else None), (rnd_num / den if den else None), rows)

    def k_to_half(cells, key, sign=1.0, thresh=THRESH_MAIN):
        """constructs you must build to capture half a cell's rescuing flips."""
        ks, kr, ns = [], [], []
        for _k, rs in cells:
            y = np.array([r["rf"] >= thresh for r in rs])
            npos, n = int(y.sum()), len(rs)
            if npos == 0 or npos == n:
                continue
            x = sign * np.array([r[key] for r in rs], float)
            order_ = np.argsort(-x, kind="stable")
            hit = np.cumsum(y[order_])
            need = int(np.argmax(hit >= math.ceil(npos / 2))) + 1
            ks.append(need); kr.append(n * 0.5); ns.append(n)
        return (float(np.mean(ks)) if ks else None,
                float(np.mean(kr)) if kr else None, len(ks), float(np.mean(ns)) if ns else None)

    cres = {}
    for nm, key in PREDS:
        sgn = depth_sign if nm == "depth_baseline" else 1.0
        rec, rnd, _rows = topk_recovery(cells, key, sgn)
        kh, khr, ncell, mn = k_to_half(cells, key, sgn)
        cres[nm] = {"recovery_topk": round(rec, 5) if rec is not None else None,
                    "recovery_random": round(rnd, 5) if rnd is not None else None,
                    "gain_points": round(100 * (rec - rnd), 2) if rec is not None else None,
                    "k_to_half": round(kh, 3) if kh is not None else None,
                    "k_to_half_random": round(khr, 3) if khr is not None else None,
                    "constructs_saved_ratio": round(khr / kh, 3) if kh else None,
                    "n_cells_scored": ncell, "mean_cell_size": round(mn, 2) if mn else None}
        log(f"  {nm:16s} 前 {int(100*TOPK_FRAC)}% 捞回 {cres[nm]['recovery_topk']}  "
            f"随机 {cres[nm]['recovery_random']}  净增 {cres[nm]['gain_points']} 点   "
            f"捞到一半要造 {cres[nm]['k_to_half']} 条 vs 随机 {cres[nm]['k_to_half_random']} 条"
            f"({cres[nm]['constructs_saved_ratio']}x)")
    OUT["C_decision"] = cres
    best_model = max(("P_vienna", "dP_vienna", "P_neural", "dP_neural"),
                     key=lambda n: cres[n]["gain_points"] or -99)
    K2 = bool((cres[best_model]["gain_points"] or -99)
              <= (cres["depth_baseline"]["gain_points"] or -99))
    log(f"  最好的模型列 {best_model};**K2(打不过免费的深度基线)= {K2}**")

    # ---------------- D: model class, paired by cell
    log("\n=== D 两个模型类,按格配对比较 ===")
    ry = ry_map(cells)
    pz = sorted({rs[0]["puzzle"] for _k, rs in cells})
    bypz = collections.defaultdict(list)
    for k, rs in cells:
        bypz[rs[0]["puzzle"]].append((k, rs))
    rng = np.random.default_rng(312); dif = []
    for _ in range(NBOOT):
        ix = rng.integers(0, len(pz), len(pz))
        sub = [c for q in ix for c in bypz[pz[q]]]
        a_, _p = weighted_rho(sub, "nmut", ry)
        b_, _p = weighted_rho(sub, "pmut", ry)
        if a_ is not None and b_ is not None:
            dif.append(a_ - b_)
    an = OUT["A_within_design"]["P_neural"]["rho"]
    av = OUT["A_within_design"]["P_vienna"]["rho"]
    ci_d = boot_ci(dif)
    log(f"  神经 {an:+.5f} - 热力学 {av:+.5f} = {an-av:+.5f}   CI {ci_d}")
    OUT["D_model_class"] = {"neural": an, "vienna": av, "diff": round(an - av, 5),
                            "ci_diff": ci_d,
                            "ci_contains_zero": bool(ci_d[0] is not None
                                                     and ci_d[0] <= 0 <= ci_d[1])}

    # ---------------- E: the controls already built
    log("\n=== E 用已经建好的两套对照再问一次 ===")
    E = {}
    try:
        import m30_selection_audit as M30S
        pool, tested = M30S.read_pool(), M30S.read_tested()
        bs = collections.defaultdict(list)
        for p in pool:
            bs[p["design_sequence"]].append(p)
        okc = {k: bool(max(r["RNet_F1"] for r in bs[v]) >= M30S.CUT)
               for k, v in tested.items() if v in bs}
        fb = [(k, rs) for k, rs in cells if okc.get(k) is False]
        log(f"  E1 M30 里不过已发表筛选阈值的格 {len(fb)} 个")
        E["E1_selection_fallback"] = arm_A(fb, "fallback", seed=313, do_perm=False) if fb else None
    except Exception as e:                                                      # noqa: BLE001
        log(f"  E1 取用失败 {repr(e)[:80]} ⇒ 记为未做,不静默略过")
        E["E1_selection_fallback"] = {"void": repr(e)[:200]}
    try:
        import m26c_leakage as LEAK
        train = set()
        for rid in LEAK.TRAIN_DEPOSITS:
            u = LEAK.rdat_url(rid)
            if u:
                train.update(LEAK.stream_sequences(u, lambda *a: None))
        log(f"  E2 训练源唯一序列 {len(train)}")
        csv.field_size_limit(10 ** 9)
        win = {}
        for d in csv.DictReader(open(M26.SRC)):
            try:
                win[(d["puzzle"], d["method"])] = (int(float(d["sub_start"])) - 1,
                                                   int(float(d["sub_end"])))
            except (ValueError, KeyError):
                continue
        import m26d_clean_subset as M26D
        q = {}
        for k, rs in cells:
            if k in win:
                lo, hi = win[k]; q[k] = rs[0]["wt_seq"][lo:hi]
        best, _p1 = M26D.max_identity(q, train, lambda *a: None)
        leaked = {k for k, v in best.items() if v >= M26D.THR}
        clean = [(k, rs) for k, rs in cells if k not in leaked]
        log(f"  E2 泄漏格 {len(leaked)}/{len(q)};干净格 {len(clean)}")
        E["E2_leakage_clean"] = arm_A(clean, "去泄漏", seed=314, do_perm=False)
        E["E2_n_leaked_cells"] = len(leaked)
    except Exception as e:                                                      # noqa: BLE001
        log(f"  E2 取用失败 {repr(e)[:80]} ⇒ 记为未做,不静默略过")
        E["E2_leakage_clean"] = {"void": repr(e)[:200]}
    OUT["E_controls"] = E

    # ---------------- verdict
    A = OUT["A_within_design"]
    modelnames = ("P_vienna", "dP_vienna", "P_neural", "dP_neural")
    excl = [n for n in modelnames
            if A[n] and A[n]["ci"][0] is not None and not (A[n]["ci"][0] <= 0 <= A[n]["ci"][1])]
    K1 = bool(not excl)
    ap = OUT["Aprime_causal_conditioning"]
    st = {
        "1_some_predictor_ranks": bool(excl),
        "2_within_design_exceeds_causal_conditioning":
            bool(ap and ap.get("P_neural") and A["P_neural"]["rho"] > ap["P_neural"]["rho"]),
        "3_neural_at_least_vienna": bool(A["P_neural"]["rho"] >= A["P_vienna"]["rho"]),
        "4_topk_beats_random_by_5pts":
            bool((cres[best_model]["gain_points"] or -99) >= 5.0),
        "5_beats_depth_baseline": bool(not K2),
        "6_base_rate_in_band": bool(0.2 <= main_row["base_rate"] <= 0.8),
        "7_survives_controls": None,
    }
    for kk in ("E1_selection_fallback", "E2_leakage_clean"):
        v = E.get(kk)
        if isinstance(v, dict) and "void" not in v and v.get("P_neural"):
            st["7_survives_controls"] = bool(st["7_survives_controls"] is not False
                                             and v["P_neural"]["rho"] > 0)
    log("\n=== 对照声明里预先押下的七条 ===")
    for k, v in st.items():
        log(f"  {k:44s} {v}")
    OUT["staked"] = st
    OUT["kill"] = {"K0_degenerate_base_rate": K0, "K1_no_ranking": K1,
                   "K2_no_gain_over_depth": K2, "K3_ordering_flips": K3}
    OUT["ci_excluding_zero"] = excl
    OUT["verdict"] = ("reframe_refused_no_ranking" if K1 else
                      "reframe_refused_depth_baseline_explains_it" if K2 else
                      "reframe_supported")
    log(f"\n  裁定 {OUT['verdict']}   ({time.time()-t0:.0f}s)")

    json.dump(OUT, open(os.path.join(HERE, "m31_which_flips_rescue.json"), "w"),
              ensure_ascii=False, indent=1)
    open(os.path.join(HERE, "m31_which_flips_rescue.log"), "w").write("\n".join(LOG) + "\n")
    log("写出 m31_which_flips_rescue.json / .log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
