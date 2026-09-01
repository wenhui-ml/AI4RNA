#!/usr/bin/env python3
"""M60b: the corollary bound, with the four implementation defects of M60 repaired.

Governed by m60_carrying_bound_declaration.json, which is NOT edited.  The defects, the diagnosis
and the fixes are recorded in m60_declaration_erratum.json; M60's own output is kept in the
repository as the thing being corrected.

The one substantive repair is F1.  M60 compared a CDE weighted over (cell x bin) against a total
weighted over (cell), with h and d_TV pooled across cells on a third weighting.  Those are not the
two sides of the proposition.  Here every quantity is carried on ONE set of cell weights
w_c = min(n_Ac, n_Bc): the total contrast, the bin-standardised contrast, the conditional mean h,
and the two arms' laws on the mediator.

Offline, $0, reads the committed caches; no sequence is folded and no predictor is called.
"""
from __future__ import annotations

import collections, json, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/data/wenhui/openclaude_rna/rna_harness/tools")

SEED = 20260901
BINGRID = (3, 5, 8, 10, 15, 20, 30, 40)
SEPGRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
FGRID = (0.0, 0.25, 0.5, 0.75, 1.0)
NMIN = 30
LOG = []


def log(s=""):
    print(s, flush=True)
    LOG.append(s)


# --------------------------------------------------------------------------- primitives
def qbins(x, nbins):
    q = np.linspace(0, 1, nbins + 1)[1:-1]
    edges = np.unique([float(np.quantile(x, v)) for v in q])
    return np.searchsorted(edges, x, side="right"), len(edges) + 1


def levelbins(x):
    """F3: an integer-valued mediator is binned on its own levels, never on quantiles."""
    lv = np.unique(x)
    return np.searchsorted(lv, x), len(lv)


def cell_weight(ix, t):
    return min(int((t[ix] == 1).sum()), int((t[ix] == 0).sum()))


def total_and_cde(cells, t, y, b, nb):
    """F1: both contrasts on ONE set of cell weights.

    total(c)  = mean_A(y) - mean_B(y)                       inside cell c
    cde(c)    = sum_k p_ck * (mean_{A,c,k} - mean_{B,c,k})  standardised on cell c's own M law,
                renormalised over the bins where both arms are present
    Both are aggregated with w_c = min(n_Ac, n_Bc), which is the weight the paper's estimator uses.
    """
    num_t = num_c = w_t = w_c = 0.0
    for _c, ix in cells:
        a, bx = ix[t[ix] == 1], ix[t[ix] == 0]
        if len(a) == 0 or len(bx) == 0:
            continue
        ww = min(len(a), len(bx))
        num_t += ww * (y[a].mean() - y[bx].mean())
        w_t += ww
        acc = p = 0.0
        for k in range(nb):
            ak, bk = a[b[a] == k], bx[b[bx] == k]
            if len(ak) and len(bk):
                pk = (len(ak) + len(bk)) / len(ix)
                acc += pk * (y[ak].mean() - y[bk].mean())
                p += pk
        if p > 0:
            num_c += ww * (acc / p)
            w_c += ww
    if w_t == 0 or w_c == 0:
        return None, None, 0.0
    return num_t / w_t, num_c / w_c, w_t


def h_and_tv(cells, t, y, b, nb):
    """F1: h and the two arms' laws, carried on the same cell weights as the contrasts."""
    hn = np.zeros(nb)
    hd = np.zeros(nb)
    pa = np.zeros(nb)
    pb = np.zeros(nb)
    for _c, ix in cells:
        a, bx = ix[t[ix] == 1], ix[t[ix] == 0]
        if len(a) == 0 or len(bx) == 0:
            continue
        ww = min(len(a), len(bx))
        cen = y[ix] - y[ix].mean()
        for k in range(nb):
            sel = b[ix] == k
            if sel.any():
                hn[k] += ww * cen[sel].sum() / len(ix)
                hd[k] += ww * sel.sum() / len(ix)
        for k in range(nb):
            pa[k] += ww * (b[a] == k).sum() / len(a)
            pb[k] += ww * (b[bx] == k).sum() / len(bx)
    live = hd > 0
    h = np.where(live, hn / np.where(live, hd, 1.0), np.nan)
    pa = pa / max(pa.sum(), 1e-12)
    pb = pb / max(pb.sum(), 1e-12)
    dtv = 0.5 * float(np.abs(pa - pb).sum())
    return h, live, dtv


def bound_row(cells, t, y, m, nbins, integer_levels=False):
    b, nb = levelbins(m) if integer_levels else qbins(m, nbins)
    d_tot, d_cde, w = total_and_cde(cells, t, y, b, nb)
    if d_tot is None or abs(d_tot) < 1e-12:
        return None
    h, live, dtv = h_and_tv(cells, t, y, b, nb)
    if int(live.sum()) < 3:
        return None
    hs = h[live]
    ms = np.array([m[b == k].mean() for k in range(nb)])[live]
    beta = None
    sxy = sxx = 0.0
    for _c, ix in cells:
        if len(ix) < 2:
            continue
        xa, ya = m[ix] - m[ix].mean(), y[ix] - y[ix].mean()
        sxy += float((xa * ya).sum())
        sxx += float((xa * xa).sum())
    beta = sxy / sxx if sxx else 0.0
    g_num = g_w = 0.0
    for _c, ix in cells:
        a, bx = ix[t[ix] == 1], ix[t[ix] == 0]
        if len(a) and len(bx):
            ww = min(len(a), len(bx))
            g_num += ww * (m[a].mean() - m[bx].mean())
            g_w += ww
    g = g_num / g_w if g_w else 0.0

    osc_h = float(hs.max() - hs.min())
    osc_r = float((hs - beta * ms).max() - (hs - beta * ms).min())
    strat, prod = d_tot - d_cde, beta * g
    return {
        "nbins": int(nb), "n_used_bins": int(live.sum()),
        "d_tot": round(d_tot, 6), "d_cde": round(d_cde, 6),
        "beta": round(beta, 8), "g": round(g, 5),
        "route_stratified": round(strat, 6), "route_product": round(prod, 6),
        "gap_points": round(100 * (strat - prod), 3),
        "osc_h": round(osc_h, 6), "osc_r": round(osc_r, 6), "d_TV": round(dtv, 5),
        "route_lhs": round(abs(strat - prod), 7), "route_rhs": round(osc_r * dtv, 7),
        "route_bound_holds": bool(abs(strat - prod) <= osc_r * dtv + 1e-12),
        "carry_lhs": round(abs(strat), 7), "carry_rhs": round(osc_h * dtv, 7),
        "carry_bound_holds": bool(abs(strat) <= osc_h * dtv + 1e-12),
        "corollary_bound_frac": round(osc_h * dtv / abs(d_tot), 5),
        "mediated_frac_stratified": round(strat / d_tot, 5),
        "product_reading_frac": round(abs(prod / d_tot), 5),
        "weight": w,
    }


# --------------------------------------------------------------------------- corpora
def rna_records():
    import m26_model_scored as M26
    import m21_thermo_mediator as M21
    recs = M21.build_with_dg()
    pm = {}
    for r in M26.build()[0]:
        pm.setdefault((r["design"], r["pt"], round(r["rf"], 6)), r["pmut"])
    for r in recs:
        r["pmut"] = pm.get((r["design"], r["pt"], round(r["rf"], 6)))
    return recs


def rna_arm(recs, mediator):
    keep = [r for r in recs if r["pt"] in ("CG", "AU")]
    if mediator == "pmut":
        keep = [r for r in keep if r.get("pmut") is not None]
    y = np.array([r["rf"] for r in keep], float)
    t = np.array([1 if r["pt"] == "CG" else 0 for r in keep], int)
    m = np.array([(r["dG"] if mediator == "dG"
                   else r["pmut"] if mediator == "pmut"
                   else min(r["depth"], 3)) for r in keep], float)
    by = collections.defaultdict(list)
    key = ((lambda r: (r["design"], r["crossed"])) if mediator == "depth"
           else (lambda r: (r["design"], min(r["depth"], 3), r["crossed"])))
    for i, r in enumerate(keep):
        by[key(r)].append(i)
    cells = [(k, np.asarray(v)) for k, v in by.items()
             if len(set(t[v])) > 1 and len(v) >= 2]
    return cells, t, y, m, len(keep)


def binder_arm():
    import m29_binder_crossdomain as M29
    import pandas as pd
    d = pd.read_csv(M29.CSV, low_memory=False).dropna(subset=[M29.TAU, M29.ZCOL, M29.YCOL])
    d["method"] = np.where(d["source"].str.startswith("Cao"), "Cao et al",
                  np.where(d["source"].str.contains("Watson"), "Watson et al", "other"))
    d = d[(d["target_id"].isin(M29.TARGETS)) & (d["method"].isin(M29.METHODS))].reset_index(drop=True)
    y = d[M29.YCOL].astype(float).to_numpy()
    m = d[M29.ZCOL].astype(float).to_numpy()
    tau = d[M29.TAU].astype(float).to_numpy()
    ic = M29.cells(d)
    t = np.zeros(len(d), int)
    for _c, ix in ic:
        t[ix] = (tau[ix] > np.median(tau[ix])).astype(int)
    cells = [(c, ix) for c, ix in ic
             if len(np.unique(y[ix])) > 1 and len(np.unique(t[ix])) > 1]
    return cells, t, y, m, len(d)


# --------------------------------------------------------------------------- A3 first (F4)
def interaction(cells, t, y, m, nbins=5, integer_levels=False):
    b, nb = levelbins(m) if integer_levels else qbins(m, nbins)
    bins = []
    for k in range(nb):
        cs = [(c, ix[b[ix] == k]) for c, ix in cells]
        cs = [(c, ix) for c, ix in cs if len(ix) >= 2 and len(set(t[ix])) > 1]
        nk = sum(len(ix) for _c, ix in cs)
        if nk < NMIN:
            continue
        num = w = 0.0
        for _c, ix in cs:
            a, bx = ix[t[ix] == 1], ix[t[ix] == 0]
            ww = min(len(a), len(bx))
            num += ww * (y[a].mean() - y[bx].mean())
            w += ww
        if w:
            bins.append({"bin": int(k), "n": int(nk), "contrast": round(num / w, 5)})
    if len(bins) < 2:
        return None
    cs_ = [x["contrast"] for x in bins]
    mean, spread = float(np.mean(cs_)), float(max(cs_) - min(cs_))
    return {"bins": bins, "mean": round(mean, 5), "spread": round(spread, 5),
            "same_sign_all_bins": bool(len({np.sign(c) for c in cs_}) == 1),
            "spread_over_mean": round(spread / abs(mean), 3) if abs(mean) > 1e-9 else None,
            "assumption7_supported": bool(len({np.sign(c) for c in cs_}) == 1
                                          and abs(mean) > 1e-9 and spread / abs(mean) < 2.0)}


# --------------------------------------------------------------------------- A2 (F2)
def resample_global(cells, t, m, s, b, nb, rng):
    """F2: reweight toward or away from the GLOBAL pooled law, which is what d_TV is measured on."""
    inall = np.concatenate([ix for _c, ix in cells])
    pooled = np.array([(b[inall] == k).sum() for k in range(nb)], float)
    pooled /= max(pooled.sum(), 1)
    parm = {}
    for arm in (0, 1):
        sel = inall[t[inall] == arm]
        c = np.array([(b[sel] == k).sum() for k in range(nb)], float)
        parm[arm] = c / max(c.sum(), 1)
    out = []
    for c_, ix in cells:
        picked = []
        for arm in (0, 1):
            sub = ix[t[ix] == arm]
            if len(sub) < 2:
                picked.append(sub)
                continue
            q = np.clip(pooled + s * (parm[arm] - pooled), 0.0, None)
            if q.sum() <= 0:
                picked.append(sub)
                continue
            q /= q.sum()
            w = np.where(parm[arm] > 0, q / np.where(parm[arm] > 0, parm[arm], 1.0), 0.0)[b[sub]]
            if w.sum() <= 0:
                picked.append(sub)
                continue
            picked.append(rng.choice(sub, size=len(sub), replace=True, p=w / w.sum()))
        out.append((c_, np.concatenate(picked)))
    return out


# --------------------------------------------------------------------------- main
def main():
    log("=== M60b:修好四处实现缺陷后的推论界 ===")
    log("  声明未改;缺陷、诊断与修复记录在 m60_declaration_erratum.json。全离线,$0。\n")
    out = {"seed": SEED, "governed_by": "m60_carrying_bound_declaration.json",
           "erratum": "m60_declaration_erratum.json",
           "bin_grid": list(BINGRID), "sep_grid": list(SEPGRID), "f_grid": list(FGRID)}
    recs = rna_records()
    log(f"RNA 记录 {len(recs)};P_model 连上 {sum(r.get('pmut') is not None for r in recs)}\n")

    # ---- V
    log("【V】先复算已发表的数")
    pub = json.load(open(os.path.join(HERE, "m21b_direct_contrast.json")))
    cells, t, y, m, n = rna_arm(recs, "dG")
    b5, nb5 = qbins(m, 5)
    d_rf, _, _ = total_and_cde(cells, t, y, b5, nb5)
    d_dg, _, _ = total_and_cde(cells, t, m, b5, nb5)
    vok = True
    vch = []
    for nm, mine, theirs, tol in (("rescue 对比", d_rf, pub["rf"]["effect"], 5e-4),
                                  ("ΔG_loc 对比", d_dg, pub["dg"]["effect"], 5e-3)):
        good = abs(mine - theirs) <= tol
        vok &= good
        vch.append({"name": nm, "mine": round(mine, 5), "published": theirs,
                    "delta": round(abs(mine - theirs), 6), "ok": bool(good)})
        log(f"  {nm:<14} 我 {mine:+.5f}  已发表 {theirs:+.5f}  |Δ| {abs(mine-theirs):.2e}  "
            f"-> {'通过' if good else '**不符**'}")
    out["V"] = {"checks": vch, "all_ok": bool(vok)}
    if not vok:
        log("\n**K0 触发。停。**")
        out["verdict"] = "K0_v_arm_failed"
        json.dump(out, open(os.path.join(HERE, "m60b_carrying_bound.json"), "w"),
                  ensure_ascii=False, indent=1)
        open(os.path.join(HERE, "m60b_run.out"), "w").write("\n".join(LOG) + "\n")
        return 1

    # ---- A3 first (F4): the corollary's premise before the corollary
    log("\n【A3】先报假设 (7):没有 T×M 交互。推论只在它成立处被许可。")
    a3 = {}
    for corpus, med, intlev in (("rna", "dG", False), ("rna", "pmut", False),
                                ("rna", "depth", True), ("binder", "A_length", False)):
        cs, tt, yy, mm, nn = rna_arm(recs, med) if corpus == "rna" else binder_arm()
        r = interaction(cs, tt, yy, mm, 5, intlev)
        if not r:
            log(f"  {corpus}/{med}: 可用箱 < 2,剔除")
            continue
        a3[f"{corpus}:{med}"] = r
        log(f"  {corpus}/{med}: " + "  ".join(f"bin{x['bin']}(n={x['n']}) {x['contrast']:+.5f}"
                                              for x in r["bins"]))
        log(f"     mean {r['mean']:+.5f}  spread {r['spread']:.5f}  同号 "
            f"{'是' if r['same_sign_all_bins'] else '**否**'}  "
            f"spread/|mean| {r['spread_over_mean']}  -> 假设(7) "
            f"{'成立' if r['assumption7_supported'] else '**不成立,推论不受许可**'}")
    out["A3"] = a3

    # ---- A1
    log("\n【A1】推论界 osc(h)·d_TV / |Δ_tot|,与论文印着的乘积读数并排")
    published = {"dG": 0.586, "pmut": 0.068}
    res = {}
    for corpus, med, intlev in (("rna", "dG", False), ("rna", "pmut", False),
                                ("rna", "depth", True), ("binder", "A_length", False)):
        cs, tt, yy, mm, nn = rna_arm(recs, med) if corpus == "rna" else binder_arm()
        grid = (0,) if intlev else BINGRID
        rows = [r for r in (bound_row(cs, tt, yy, mm, nb, intlev) for nb in grid) if r]
        if not rows:
            log(f"  {corpus}/{med}: 无可用箱数,按 K5 剔除")
            continue
        key = f"{corpus}:{med}"
        cb = [r["corollary_bound_frac"] for r in rows]
        lic = a3.get(key, {}).get("assumption7_supported", False)
        res[key] = {"n": nn, "n_cells": len(cs), "rows": rows,
                    "assumption7_supported": bool(lic),
                    "corollary_bound_min": min(cb), "corollary_bound_max": max(cb),
                    "corollary_bound_at_5bins": next(
                        (r["corollary_bound_frac"] for r in rows if r["nbins"] in (5, len(rows) and rows[0]["nbins"])), rows[0]["corollary_bound_frac"]),
                    "product_reading_frac": rows[0]["product_reading_frac"],
                    "published_bound": published.get(med),
                    "carry_bound_holds_all": all(r["carry_bound_holds"] for r in rows),
                    "route_bound_holds_all": all(r["route_bound_holds"] for r in rows)}
        log(f"\n  {key}  n={nn} 格={len(cs)}  假设(7) {'成立' if lic else '**不成立**'}")
        log(f"    {'箱':>4}{'用箱':>5}{'Δ_tot':>10}{'Δ_cde':>10}{'中介部分':>10}"
            f"{'osc(h)':>9}{'d_TV':>8}{'推论界':>9}{'界':>6}{'乘积读数':>9}")
        for r in rows:
            log(f"    {r['nbins']:>4}{r['n_used_bins']:>5}{r['d_tot']:>+10.5f}{r['d_cde']:>+10.5f}"
                f"{r['route_stratified']:>+10.5f}{r['osc_h']:>9.5f}{r['d_TV']:>8.4f}"
                f"{100*r['corollary_bound_frac']:>8.1f}%{'  是' if r['carry_bound_holds'] else ' **否**':>6}"
                f"{100*r['product_reading_frac']:>8.1f}%")
        if med in published:
            log(f"    -> 推论界区间 [{100*min(cb):.1f}%, {100*max(cb):.1f}%]   "
                f"论文印的乘积读数 {100*published[med]:.1f}%")
    out["A1"] = res

    # ---- A2
    log("\n【A2】分离度扫描(F2:旋钮对准全局分箱律)")
    cs, tt, yy, mm, _ = rna_arm(recs, "dG")
    rng = np.random.default_rng(SEED)
    b5, nb5 = qbins(mm, 5)
    rows = []
    for s in SEPGRID:
        cc = resample_global(cs, tt, mm, s, b5, nb5, rng)
        r = bound_row(cc, tt, yy, mm, 5)
        if r:
            r["separation"] = s
            rows.append(r)
            log(f"    s={s:<5} d_TV {r['d_TV']:.4f}  |条件−乘积| {r['route_lhs']:.6f}  "
                f"osc(r)·d_TV {r['route_rhs']:.6f}  界 "
                f"{'成立' if r['route_bound_holds'] else '**不成立**'}")
    from scipy.stats import spearmanr
    rho = float(spearmanr([r["route_lhs"] for r in rows],
                          [r["route_rhs"] for r in rows]).statistic) if len(rows) > 2 else float("nan")
    dtvs = [r["d_TV"] for r in rows]
    log(f"    -> d_TV 实际扫过 [{min(dtvs):.4f}, {max(dtvs):.4f}];"
        f"|条件−乘积| 与 osc(r)·d_TV 的 Spearman {rho:+.4f}")

    cen = np.concatenate([yy[ix] - yy[ix].mean() for _c, ix in cs])
    sigma = float(cen.std(ddof=1))
    d_tot_real, _, _ = total_and_cde(cs, tt, yy, b5, nb5)
    hshape = np.zeros(len(mm))
    hh, live, _ = h_and_tv(cs, tt, yy, b5, nb5)
    for k in range(nb5):
        if live[k]:
            hshape[b5 == k] = hh[k]
    log(f"    层内残差 SD {sigma:.4f}(实测,非选定);注入总对比 {d_tot_real:+.5f}")
    sweep, spans = [], {}
    for s in SEPGRID:
        cc = resample_global(cs, tt, mm, s, b5, nb5, rng)
        gh_num = gh_w = 0.0
        for _c, ix in cc:
            a, bx = ix[tt[ix] == 1], ix[tt[ix] == 0]
            if len(a) and len(bx):
                w_ = min(len(a), len(bx))
                gh_num += w_ * (hshape[a].mean() - hshape[bx].mean())
                gh_w += w_
        ghs = gh_num / gh_w if gh_w else 0.0
        if abs(ghs) < 1e-9:
            continue
        pts = []
        for f in FGRID:
            ysyn = ((1.0 - f) * d_tot_real * tt + f * d_tot_real * hshape / ghs
                    + rng.normal(0.0, sigma, size=len(yy)))
            r = bound_row(cc, tt, ysyn, mm, 5)
            if r and abs(r["d_tot"]) > 1e-9:
                pts.append({"separation": s, "d_TV": r["d_TV"], "f_true": f,
                            "f_strat": r["mediated_frac_stratified"],
                            "carry_bound_holds": r["carry_bound_holds"]})
        sweep += pts
        if len(pts) >= 2:
            sp = max(p["f_strat"] for p in pts) - min(p["f_strat"] for p in pts)
            tr = max(p["f_true"] for p in pts) - min(p["f_true"] for p in pts)
            spans[s] = {"d_TV": pts[0]["d_TV"], "span_frac": round(sp / tr, 4)}
            log(f"    s={s:<5} d_TV {pts[0]['d_TV']:.4f}  读回跨度/真值跨度 {100*sp/tr:7.1f}%")
    out["A2"] = {"a2a_rows": rows, "spearman_lhs_rhs": round(rho, 5), "n_rungs": len(rows),
                 "d_TV_swept": [round(min(dtvs), 4), round(max(dtvs), 4)],
                 "sigma_measured": round(sigma, 5), "a2b_sweep": sweep,
                 "a2b_spans": {str(k): v for k, v in spans.items()},
                 "route_bound_holds_all": all(r["route_bound_holds"] for r in rows)}

    # ---- scoring, on the declaration's own seven, unedited
    log("\n=== 预注册记分(按冻结声明原文,未改写)===")
    S, txt = {}, {}
    dg, pm_ = res.get("rna:dG"), res.get("rna:pmut")
    lic_rows = [(k, v) for k, v in res.items() if v["assumption7_supported"]]
    S["S1"] = bool(dg and dg["corollary_bound_at_5bins"] < 0.586)
    txt["S1"] = (f"ΔG 推论界 {100*dg['corollary_bound_at_5bins']:.1f}% vs 印着的 58.6%"
                 if dg else "n/a")
    S["S2"] = bool(pm_ and pm_["corollary_bound_at_5bins"] < 0.068)
    txt["S2"] = (f"P_model 推论界 {100*pm_['corollary_bound_at_5bins']:.1f}% vs 印着的 6.8%"
                 if pm_ else "n/a")
    n_ok = sum(1 for _k, v in lic_rows for r in v["rows"] if r["carry_bound_holds"])
    n_all = sum(len(v["rows"]) for _k, v in lic_rows)
    S["S3"] = bool(n_all > 0 and n_ok == n_all)
    txt["S3"] = f"在假设(7)成立的 {len(lic_rows)} 个臂上,界成立 {n_ok}/{n_all}"
    S["S4"] = bool(rho >= 0.8)
    txt["S4"] = f"Spearman {rho:+.4f},d_TV 扫过 [{min(dtvs):.3f}, {max(dtvs):.3f}]"
    sp = sorted((float(k), v["span_frac"]) for k, v in out["A2"]["a2b_spans"].items())
    S["S5"] = bool(len(sp) >= 2 and sp[0][1] <= sp[-1][1])
    txt["S5"] = f"跨度比 从 {100*sp[0][1]:.1f}% 到 {100*sp[-1][1]:.1f}%" if sp else "n/a"
    r3 = a3.get("rna:dG")
    S["S6"] = bool(r3 and r3["same_sign_all_bins"])
    txt["S6"] = f"逐箱同号 {r3['same_sign_all_bins']}" if r3 else "n/a"
    S["S7"] = bool(r3 and r3["spread_over_mean"] is not None and r3["spread_over_mean"] < 2.0)
    txt["S7"] = f"spread/|mean| {r3['spread_over_mean']}" if r3 else "n/a"
    for k in ("S1", "S2", "S3", "S4", "S5", "S6", "S7"):
        log(f"  {k}  {'成立  ' if S[k] else '**失败**'}  {txt[k]}")
    out["stakes"], out["stake_text"] = S, txt

    if not S["S3"]:
        v = "K1_corollary_violated"
    elif not (S["S1"] and S["S2"]):
        v = "K2_product_reading_was_not_a_bound"
    elif not S["S4"]:
        v = "K3_tracking_unresolved"
    elif not S["S6"]:
        v = "K4_assumption_7_fails"
    else:
        v = "corollary_bound_replaces_the_product_reading"
    out["verdict"] = v
    log(f"\n  裁定:{v}   {sum(S.values())}/7 成立")

    json.dump(out, open(os.path.join(HERE, "m60b_carrying_bound.json"), "w"),
              ensure_ascii=False, indent=1)
    open(os.path.join(HERE, "m60b_run.out"), "w").write("\n".join(LOG) + "\n")
    log("\n写出 m60b_carrying_bound.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
