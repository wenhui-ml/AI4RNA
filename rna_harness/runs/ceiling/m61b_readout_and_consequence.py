#!/usr/bin/env python3
"""M61b: M61 with its six recorded defects repaired, and the cut that actually separates chemistry.

Declaration: m61_readout_and_consequence_declaration.json, unedited. Defects: m61_declaration_erratum.json.
The stakes were graded once, on M61, in the declaration's wording; they are NOT reissued here.

The cut. A composition artifact in R lives at the two mutated bases: those are the positions whose
identity changed, and 2A3 acylates a flexible 2'-OH whose exposure they set. A structural effect
does not: a pair that forms holds a helix together and its loss moves the profile away from those
two positions as well. So recompute all three RMSDs with the mutated positions and a neighbourhood
EXCLUDED, and sweep the radius. Conditioning on the denominator, which is what M61 did, cannot do
this: the denominator is how much breaking the pair disrupts the molecule, which is downstream of
the pair forming at all.

COST: $0, offline, CPU only.
"""
from __future__ import annotations
import collections, json, math, os, random as _r, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, "/data/wenhui/openclaude_rna/rna_harness/tools")
import m61_readout_and_consequence as M61
import m21_thermo_mediator as M21
import m60b_carrying_bound as B60
import m60c_tight_bound as C60
from designaudit import core

SEED = 20260901
RADII = (None, 0, 1, 2, 3)     # None = no exclusion, the published readout
LOG = []


def log(s=""):
    print(s, flush=True); LOG.append(s)


def rmsd_mask(a, b, drop):
    m = np.isfinite(a) & np.isfinite(b)
    if drop is not None:
        m = m & drop
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.sum() >= 10 else np.nan


def main():
    rng = np.random.default_rng(SEED)
    OUT = {"seed": SEED, "repairs": "m61_declaration_erratum.json"}
    log("=== M61b:修好 M61 记录在案的六处缺陷,并补上真正能分开化学的那一刀 ===")
    log("  声明未改。全离线,$0。\n")

    recs, et = M61.build()
    log(f"记录 {len(recs)}  设计格 {len({r['design'] for r in recs})}  靶标 {len({r['puzzle'] for r in recs})}")
    rows = M61.flat(recs)
    ev = M61.ctr(rows, "rf")
    log(f"  基线 rescue 对比 {ev.effect:+.5f}  [{ev.ci[0]:+.4f}, {ev.ci[1]:+.4f}]  (已发表 +0.05393)")
    OUT["baseline"] = {"effect": round(ev.effect, 5), "ci": [round(x, 4) for x in ev.ci]}

    # ---------------------------------------------------------------- A6
    log("\n【A6】把突变位点本身挖掉再算一遍(F5 的修复)")
    log("  组成赝像坐在被突变的两个碱基上;结构效应不坐在那里。半径逐档扫,不是挑一档。")
    need = ("a_idx", "b_idx")
    A6 = {}
    ref = [None]   # the no-exclusion recomputation; the retention denominator
    log("  半径 >= 1 时,进入任何一个 RMSD 的位点全都没有改变过身份:突变只在 i 和 j 两处。")
    log(f"\n  {'挖掉的位置':<26}{'用的位点':>9}{'G-C 减 A-U':>12}{'95% CI':>22}{'p':>9}{'留存':>9}")
    for rad in RADII:
        newrows = []
        nsite = []
        for r, f in zip(recs, rows):
            P = r["prof"]
            L = len(P[0])
            keep = np.ones(L, bool)
            if rad is not None:
                lo1, hi1 = max(0, r["a_idx"] - rad), min(L, r["a_idx"] + rad + 1)
                lo2, hi2 = max(0, r["b_idx"] - rad), min(L, r["b_idx"] + rad + 1)
                keep[lo1:hi1] = False; keep[lo2:hi2] = False
            ri = rmsd_mask(P[0], P[1], keep); rj = rmsd_mask(P[0], P[2], keep)
            rij = rmsd_mask(P[0], P[3], keep)
            if not (np.isfinite(ri) and np.isfinite(rj) and np.isfinite(rij)):
                continue
            D = math.sqrt(ri ** 2 + rj ** 2)
            if D <= 0:
                continue
            g = dict(f); g["rex"] = 1 - rij / D
            g["rho_ij_ex"] = rij; g["D_ex"] = D
            newrows.append(g)
            nsite.append(int((keep & np.isfinite(P[0]) & np.isfinite(P[3])).sum()))
        e = M61.ctr(newrows, "rex")
        if ref[0] is None:
            ref[0] = e.effect
        lab = "无(本文重算的基线)" if rad is None else f"两个突变位点 ±{rad}"
        A6[str(rad)] = {"radius": rad, "n": len(newrows),
                        "median_sites_used": int(np.median(nsite)),
                        "effect": round(e.effect, 5), "ci": [round(x, 4) for x in e.ci],
                        "p": round(e.p, 5), "weight": e.weight,
                        "retention_vs_recomputed": round(float(e.effect / ref[0]), 4),
                        "retention_vs_released": round(float(e.effect / ev.effect), 4),
                        "ci_excludes_zero": bool(e.ci[0] * e.ci[1] > 0)}
        log(f"  {lab:<26}{int(np.median(nsite)):>9}{e.effect:>+12.5f}"
            f"   [{e.ci[0]:+.4f}, {e.ci[1]:+.4f}]{e.p:>9.4f}{100*e.effect/ref[0]:>8.1f}%")
    # the complement: only the mutated neighbourhood, where a composition artifact must live
    log("\n  反向:只用被挖掉的那一小块位点算(赝像若存在,必须集中在这里)")
    for rad in (0, 1, 2):
        newrows = []
        for r, f in zip(recs, rows):
            P = r["prof"]; L = len(P[0])
            sel = np.zeros(L, bool)
            lo1, hi1 = max(0, r["a_idx"] - rad), min(L, r["a_idx"] + rad + 1)
            lo2, hi2 = max(0, r["b_idx"] - rad), min(L, r["b_idx"] + rad + 1)
            sel[lo1:hi1] = True; sel[lo2:hi2] = True
            ri = rmsd_mask(P[0], P[1], sel); rj = rmsd_mask(P[0], P[2], sel)
            rij = rmsd_mask(P[0], P[3], sel)
            if not (np.isfinite(ri) and np.isfinite(rj) and np.isfinite(rij)):
                continue
            D = math.sqrt(ri ** 2 + rj ** 2)
            if D <= 0:
                continue
            g = dict(f); g["rin"] = 1 - rij / D
            newrows.append(g)
        if len(newrows) < 300:
            log(f"  只用 ±{rad}:有效记录 {len(newrows)},太少,不读"); continue
        e = M61.ctr(newrows, "rin")
        A6[f"only_{rad}"] = {"radius": rad, "n": len(newrows), "effect": round(e.effect, 5),
                             "ci": [round(x, 4) for x in e.ci], "p": round(e.p, 5),
                             "share_of_baseline": round(float(e.effect / ref[0]), 4),
                             "caveat": "a few positions only, and the structural disruption is "
                                       "largest here too; this is not a chemistry-only readout"}
        log(f"  只用突变位点 ±{rad}(n={len(newrows)}){e.effect:>+12.5f}"
            f"   [{e.ci[0]:+.4f}, {e.ci[1]:+.4f}]{e.p:>9.4f}{100*e.effect/ref[0]:>8.1f}%")
    A6["recomputed_baseline"] = round(float(ref[0]), 5)
    A6["released_column_baseline"] = round(float(ev.effect), 5)
    OUT["A6_off_mutation_cut"] = A6

    # ---------------------------------------------------------------- B3 repaired
    log("\n【B3'】对半可靠性,换成有内容的量(F3 的修复)")
    pz = sorted({r["puzzle"] for r in recs})
    H = []
    for _ in range(200):
        pm = rng.permutation(len(pz)); h1 = {pz[i] for i in pm[:len(pz) // 2]}
        vals = []
        for sel in (h1, set(pz) - h1):
            sub = [f for f in rows if f["puzzle"] in sel]
            try:
                vals.append(core.contrast(core.Corpus.from_records(
                    sub, design="design", treatment="pt", outcome="rf",
                    covariates=("depth3", "crossed"), cluster="puzzle"),
                    ("CG", "AU"), strata=("depth3", "crossed"), nperm=1, nboot=1, seed=1).effect)
            except Exception:
                vals = None; break
        if vals:
            H.append(vals)
    H = np.array(H)
    OUT["B3_repaired"] = {
        "n_splits": len(H), "half_sd": round(float(H.std()), 5),
        "half_mean": round(float(H.mean()), 5),
        "both_halves_positive": round(float(np.mean((H[:, 0] > 0) & (H[:, 1] > 0))), 4),
        "either_half_negative": round(float(np.mean((H[:, 0] <= 0) | (H[:, 1] <= 0))), 4),
        "note": "complementary halves are anti-correlated by arithmetic; the correlation is not reported"}
    log(f"  200 次按靶标对半:半样本对比 均值 {H.mean():+.5f}  SD {H.std():.5f}")
    log(f"  两半同时为正的比例 {100*OUT['B3_repaired']['both_halves_positive']:.1f}%"
        f"(任一半为负 {100*OUT['B3_repaired']['either_half_negative']:.1f}%)")

    # ---------------------------------------------------------------- D2 repaired
    log("\n【D2'】主通道紧界的靶标自助区间,用 M60 自己的函数(F1 的修复)")
    r60 = B60.rna_records()
    keep = [r for r in r60 if r["pt"] in ("CG", "AU")]
    y = np.array([r["rf"] for r in keep], float)
    t = np.array([1 if r["pt"] == "CG" else 0 for r in keep], int)
    m = np.array([r["dG"] for r in keep], float)
    puz = np.array([r["puzzle"] for r in keep])
    ckey = lambda r: (r["design"], min(r["depth"], 3), r["crossed"])

    def cells_from(idx):
        """M60b's own cell filter: a cell counts only if it holds both arms and at least 2 records."""
        by = collections.defaultdict(list)
        for i in idx:
            by[ckey(keep[i])].append(i)
        return [(k, np.asarray(v)) for k, v in by.items()
                if len(set(t[np.asarray(v)])) > 1 and len(v) >= 2]

    def tightpct(idx, nb):
        idx = np.asarray(idx)
        cl = cells_from(idx)
        rr = C60.row(cl, t, y, m, nb)
        return rr

    # M60c's grid REQUESTS 15 and qbins ACHIEVES 13 after duplicate quantile edges collapse;
    # its table prints the achieved count. Requesting 13 achieves 12 and is a different row.
    REQ = 15
    base13 = tightpct(np.arange(len(keep)), REQ)
    log(f"  点估计(请求 {REQ} 箱,实得 {base13['nbins']},用上 {base13['used']}):"
        f"紧界 {100*base13['tight_frac']:.1f}%  粗界 {100*base13['loose_frac']:.1f}%"
        f"   (M60c 发表值 53.1% / 101.9%)")
    bycl = {p: np.where(puz == p)[0] for p in sorted(set(puz))}
    pzl = sorted(bycl)
    bs = []
    for _ in range(400):
        pick = [pzl[i] for i in rng.integers(0, len(pzl), len(pzl))]
        idx = np.concatenate([bycl[p] for p in pick])
        try:
            rr = tightpct(idx, REQ)
        except Exception:
            continue
        if rr:
            bs.append(100 * rr["tight_frac"])
    bs = np.array(bs)
    OUT["D2_repaired"] = {"bins_requested": REQ, "bins_achieved": int(base13["nbins"]),
                          "bins_used": int(base13["used"]), "point_tight_pct": round(100 * base13["tight_frac"], 1),
                          "point_crude_pct": round(100 * base13["loose_frac"], 1),
                          "tight_holds": bool(base13["tight_holds"]),
                          "nboot": int(len(bs)),
                          "ci": [round(float(np.quantile(bs, 0.025)), 1),
                                 round(float(np.quantile(bs, 0.975)), 1)],
                          "median": round(float(np.median(bs)), 1),
                          "frac_ge_100": round(float(np.mean(bs >= 100)), 4)}
    log(f"  400 次靶标自助:紧界 95% 区间 {OUT['D2_repaired']['ci']},中位 {OUT['D2_repaired']['median']}%,"
        f"重采样里落到 100% 以上的比例 {100*OUT['D2_repaired']['frac_ge_100']:.1f}%")

    # ---------------------------------------------------------------- D3 repaired
    log("\n【D3'】ΔG_loc 的零结果按方法子集重跑,用 M21 自己的留存估计量(F2 的修复)")
    r21 = M21.build_with_dg()
    LO, HI, PUB = 93.2, 107.0, 101.9
    d3 = {}
    for nm, ms in (("all", None), ("thermodynamic", M61.THERMO), ("neural", M61.NEURAL)):
        sub = [r for r in r21 if ms is None or r["method"] in ms]
        if len(sub) < 300:
            d3[nm] = {"skipped": len(sub)}; continue
        edges = M21.bin_edges(sub, M21.NBINS)
        for r in sub:
            r["bin"] = M21.assign_bin(r["dG"], edges)
        d1, w1, s1 = M21.matched_diff(sub, "CG", "AU", M21.strat_dg)
        kept = M21.retained(sub, "CG", "AU", M21.strat_dg)
        d_same, w_s, s_s = M21.matched_diff(kept, "CG", "AU", M21.strat_base)
        ret = 100 * d1 / d_same if d_same else float("nan")
        inb = LO <= ret <= HI
        d3[nm] = {"n": len(sub), "cde": round(d1, 5), "base_same_population": round(d_same, 5),
                  "retention_pct": round(ret, 1), "weight": w1, "n_strata": s1,
                  "in_placebo_band": bool(inb)}
        log(f"  {nm:<16}n={len(sub):5d}  同群体基线 {d_same:+.5f}  CDE(固定 ΔG) {d1:+.5f}"
            f"  留存 {ret:6.1f}%  {'在安慰剂带内' if inb else '**出带**'}")
    d3["published_all"] = PUB
    d3["band"] = [LO, HI]
    d3["both_hold"] = bool(d3.get("thermodynamic", {}).get("in_placebo_band")
                           and d3.get("neural", {}).get("in_placebo_band"))
    OUT["D3_repaired"] = d3

    # ---------------------------------------------------------------- A7
    log("\n【A7】depth 通道上 Prop.3 的闭式残差,直接算一遍而不是相减")
    dep = np.array([min(r["depth"], 3) for r in keep], int)
    ckey2 = lambda r: (r["design"], r["crossed"])
    by = collections.defaultdict(list)
    for i, r in enumerate(keep):
        by[ckey2(r)].append(i)
    cells2 = [(k, np.array(v)) for k, v in by.items()]
    b, nb = B60.levelbins(dep.astype(float))
    dtot, dcde, _w = B60.total_and_cde(cells2, t, y, b, nb)
    h, live, dtv = B60.h_and_tv(cells2, t, y, b, nb)
    pa = np.zeros(nb); pb = np.zeros(nb); gA = gB = wA = 0.0
    for _c, ix in cells2:
        a, bx = ix[t[ix] == 1], ix[t[ix] == 0]
        if len(a) == 0 or len(bx) == 0:
            continue
        ww = min(len(a), len(bx))
        for k in range(nb):
            pa[k] += ww * (b[a] == k).sum() / len(a)
            pb[k] += ww * (b[bx] == k).sum() / len(bx)
        gA += ww * dep[a].mean(); gB += ww * dep[bx].mean(); wA += ww
    pa /= pa.sum(); pb /= pb.sum(); g = (gA - gB) / wA
    lv = np.unique(dep.astype(float))
    hs, pas, pbs, ms = h[live], pa[live], pb[live], lv[live]
    wgt = (pas + pbs) / 2
    mbar = float(np.average(ms, weights=wgt)); hbar = float(np.average(hs, weights=wgt))
    beta = float((wgt * (ms - mbar) * (hs - hbar)).sum() / (wgt * (ms - mbar) ** 2).sum())
    rres = hs - beta * ms
    direct = float((rres * (pas - pbs)).sum())
    bysub = float((dtot - dcde) - beta * g)
    OUT["A7_prop3_identity"] = {
        "delta_tot": round(dtot, 5), "delta_cde": round(dcde, 5),
        "attenuation": round(dtot - dcde, 5), "beta": round(beta, 5), "g": round(g, 4),
        "beta_g": round(beta * g, 5),
        "residual_computed_directly": round(direct, 5),
        "residual_by_subtraction": round(bysub, 5),
        "agreement": round(abs(direct - bysub), 6),
        "linear_share_pct": round(100 * beta * g / (dtot - dcde), 1) if dtot != dcde else None,
        "h": [round(float(x), 5) for x in hs], "levels": [float(x) for x in ms],
        "pA": [round(float(x), 4) for x in pas], "pB": [round(float(x), 4) for x in pbs]}
    log(f"  Δ_tot {dtot:+.5f}  Δ_cde {dcde:+.5f}  衰减 {dtot-dcde:+.5f}")
    log(f"  β {beta:+.5f}  g {g:+.4f}  βg {beta*g:+.5f}  线性项占衰减 "
        f"{100*beta*g/(dtot-dcde):.1f}%")
    log(f"  E[r|A]−E[r|B] 直接算 {direct:+.5f} vs 相减 {bysub:+.5f}  差 {abs(direct-bysub):.2e}"
        f"  -> {'恒等式成立' if abs(direct-bysub) < 5e-4 else '**不成立,(7) 被违反的量'}")

    json.dump(OUT, open(os.path.join(HERE, "m61b_readout_and_consequence.json"), "w"),
              ensure_ascii=False, indent=1, default=float)
    open(os.path.join(HERE, "m61b_readout_and_consequence.log"), "w").write("\n".join(LOG) + "\n")
    log("\n写出 m61b_readout_and_consequence.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
