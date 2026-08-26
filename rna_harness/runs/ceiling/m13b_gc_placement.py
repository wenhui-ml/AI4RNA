#!/usr/bin/env python3
"""M13b: is it the GC, or is it WHERE the GC sits?

Declaration: m13b_gc_placement_declaration.json, frozen and pushed at 4c37b11e BEFORE this file
existed (the eleventh done that way). It states at the top that the main arm is NOT blind to me:
M13 already measured the point estimates. What is new here is (1) permutation inference, because
M13's probe proved the cluster bootstrap's interval too narrow (rule 76), and (2) P4, the probe
M13 never had.

The wet readout is the OpenKnot Score, a chemical-mapping measurement. "More GC at paired
positions scores better" therefore has a second explanation: the probing chemistry responding to
base composition. P4 circularly shifts the target's paired-position set -- identical size,
identical contiguous-run structure, wrong alignment -- and the placement claim is gated on the
real set beating the 95th percentile of twenty shifts.

COST: $0, offline.
"""
from __future__ import annotations

import argparse, collections, json, math, os, random, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from m13_cache import POPS                                                            # noqa: E402

CAL_GC, CAL_P, CAL_DG, DELTA = 0.02, 0.20, 0.20, 0.05
NPERM, NFOLD, R_NOISE, DRAWS, NSHIFT = 2000, 5, 12, 20, 20
RHOS = (0.02, 0.05, 0.10, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.80)
DOSE = ((0.05, 0.10), (0.10, 0.15), (0.15, 9.0))
MIN_PAIRS, MIN_TGT, MAX_SMD, PASS = 100, 15, 0.10, 0.8
P_GATE = 0.005


# ------------------------------------------------------------------ assembly
def assemble(rows, extra, seed):
    cells = collections.defaultdict(list)
    for r, e in zip(rows, extra):
        cells[(r["puzzle"], r["method"])].append((r, e))
    rws = [(c, g) for c, g in sorted(cells.items()) if len(g) >= 2]
    y, B, tg = [], [], []
    for cell, grp in rws:
        w = np.array([r["wet"] for r, _e in grp], float)
        X = np.array([r["B"] for r, _e in grp], float)
        y.append(w - w.mean()); B.append(X - X.mean(axis=0)); tg += [cell[0]] * len(grp)
    y = np.concatenate(y); B = np.vstack(B); tg = np.array(tg)
    u = sorted(set(tg.tolist())); random.Random(seed).shuffle(u)
    fa = {t: i % NFOLD for i, t in enumerate(u)}
    fold = np.array([fa[t] for t in tg])
    pB = np.zeros(len(y))
    for f in range(NFOLD):
        te, tr = fold == f, fold != f
        if tr.sum() <= B.shape[1] + 10 or te.sum() == 0:
            continue
        b, *_ = np.linalg.lstsq(B[tr], y[tr], rcond=None)
        pB[te] = B[te] @ b
    sp = float(np.std(pB)) or 1.0
    dgall = np.array([e["dG_target"] for _c, g in rws for _r, e in g], float)
    sdg = float(np.std(dgall)) or 1.0
    CELLS, o = [], 0
    for cell, grp in rws:
        k = len(grp); sl = slice(o, o + k); o += k
        e0 = grp[0][1]; n = e0["n"]
        ISGC = np.array([[1.0 if ch in "GC" else 0.0 for ch in e["seq"]] for _r, e in grp])
        CELLS.append({"tgt": cell[0], "method": cell[1], "n": n,
                      "paired": np.array(e0["paired"], int), "helix_mean": e0["helix_mean"],
                      "y": y[sl], "gcf": np.array([e["gc_frac"] for _r, e in grp]) if False else
                      np.array([r["F"]["gc_frac"] for r, _e in grp], float),
                      "gcp": np.array([r["F"]["gc_paired"] for r, _e in grp], float),
                      "pB": pB[sl], "dg": np.array([e["dG_target"] for _r, e in grp], float),
                      "ISGC": ISGC})
    return {"cells": CELLS, "sp": sp, "sdg": sdg, "n": len(y),
            "sd_gcp": float(np.std(np.concatenate([c["gcp"] for c in CELLS]))),
            "sd_y": float(np.std(y))}


# ------------------------------------------------------------------ matching
def match(D, treat, delta, panel=True, dg=False):
    """treat: list of per-cell arrays. -> per-target (sum, count) plus diagnostics."""
    sums, cnts = collections.defaultdict(float), collections.defaultdict(int)
    dose, dgap, ppgap, diff, tgs, gcpgap = [], [], [], [], [], []
    for c, t in zip(D["cells"], treat):
        if len(t) < 2:
            continue
        dt = t[:, None] - t[None, :]
        ok = dt >= delta
        if not ok.any():
            continue
        g = c["gcf"]; ok &= np.abs(g[:, None] - g[None, :]) <= CAL_GC
        if panel:
            p = c["pB"]; ok &= np.abs(p[:, None] - p[None, :]) <= CAL_P * D["sp"]
        if dg:
            q = c["dg"]; ok &= np.abs(q[:, None] - q[None, :]) <= CAL_DG * D["sdg"]
        if not ok.any():
            continue
        a, b = np.nonzero(ok)
        d = c["y"][a] - c["y"][b]
        sums[c["tgt"]] += float(d.sum()); cnts[c["tgt"]] += len(d)
        dose.append(dt[a, b]); diff.append(d); tgs += [c["tgt"]] * len(d)
        dgap.append(g[a] - g[b]); ppgap.append(c["pB"][a] - c["pB"][b])
        gcpgap.append(c["gcp"][a] - c["gcp"][b])
    if not diff:
        return None
    return {"sums": sums, "cnts": cnts, "dose": np.concatenate(dose),
            "diff": np.concatenate(diff), "tg": np.array(tgs),
            "dgcf": np.concatenate(dgap), "dpB": np.concatenate(ppgap),
            "dgcp": np.concatenate(gcpgap),
            "npair": int(sum(cnts.values())), "ntgt": len(cnts)}


def point(M):
    return float(sum(M["sums"].values()) / sum(M["cnts"].values()))


def cluster_ci(M, seed, nb=2000):
    T = sorted(M["cnts"]); s = np.array([M["sums"][t] for t in T])
    c = np.array([M["cnts"][t] for t in T], float)
    rng = np.random.default_rng(seed); ix = rng.integers(0, len(T), size=(nb, len(T)))
    num, den = s[ix].sum(1), c[ix].sum(1)
    B = np.sort(num[den > 0] / den[den > 0])
    return [round(float(B[int(.025 * len(B))]), 4), round(float(B[int(.975 * len(B))]), 4)]


def perm_cols(D, rng):
    return [c["gcp"][rng.permutation(len(c["gcp"]))] for c in D["cells"]]


def null_dist(D, seed, nperm=NPERM, delta=DELTA, panel=True, dg=False):
    rng = np.random.default_rng(seed); out = []
    for _ in range(nperm):
        M = match(D, perm_cols(D, rng), delta, panel, dg)
        out.append(point(M) if M else 0.0)
    return np.sort(np.array(out))


def pval(obs, null):
    return float((null >= obs).sum() + 1) / (len(null) + 1)


def qdelta(D, treat, ref_delta=DELTA):
    """rule 71 width matching: same tail fraction of the within-cell ordered-pair gaps."""
    def gaps(cols):
        o = []
        for t in cols:
            if len(t) < 2:
                continue
            g = t[:, None] - t[None, :]; o.append(g[g > 0])
        return np.concatenate(o) if o else np.array([0.0])
    ref = gaps([c["gcp"] for c in D["cells"]])
    q = float((ref >= ref_delta).mean()); gc = gaps(treat)
    return float(np.quantile(gc, 1.0 - q)) if q > 0 and len(gc) else float(gc.max()) + 1.0


def shifted_gc(D, s_frac):
    """P4: GC on the target's paired-position set, rotated by s_frac of the length."""
    out, shifts, corr = [], [], []
    for c in D["cells"]:
        n = c["n"]; s = int(round(s_frac * n)) % n
        S = (c["paired"] + s) % n
        v = c["ISGC"][:, S].mean(1)
        out.append(v); shifts.append(s)
        if len(v) > 2 and np.std(v) > 0 and np.std(c["gcp"]) > 0:
            corr.append(float(np.corrcoef(v, c["gcp"])[0, 1]))
    return out, float(np.mean(corr)) if corr else float("nan")


def synth(D, rho, rng):
    out = []
    sd = D["sd_y"] or 1.0
    for c in D["cells"]:
        v = rho * (c["y"] / sd) + math.sqrt(max(0.0, 1 - rho * rho)) * rng.standard_normal(len(c["y"]))
        out.append(v - v.mean())
    return out


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nperm", type=int, default=NPERM)
    ap.add_argument("--out", default=os.path.join(HERE, "m13b_gc_placement.json"))
    a = ap.parse_args()
    print("=== M13b:是 GC,还是 GC 放在哪里 ===")
    print("  声明 4c37b11e(代码存在之前已推送)。离线 $0。")
    print("  **主臂对我不是盲的** —— M13 已量过点估计。本轮新的是置换推断与 P4。\n")
    C = json.load(open(os.path.join(HERE, "m13_features.json")))
    E = json.load(open(os.path.join(HERE, "m13b_extra.json")))
    D = {fn: assemble(C[fn]["rows"], E[fn], a.seed) for _nm, fn in POPS}
    RES, NULLS = {}, {}

    print("--- 主臂:配平对照 + 置换检验 ---")
    for nm, fn in POPS:
        d = D[fn]; treat = [c["gcp"] for c in d["cells"]]
        M = match(d, treat, DELTA)
        if M is None:
            RES[fn] = {"pop": nm, "A": None}; print(f"  {nm:12s} 零配对"); continue
        obs = point(M)
        null = null_dist(d, a.seed, a.nperm); NULLS[fn] = null
        smd_g = abs(float(M["dgcf"].mean())) / (d["sd_gcp"] if False else
                    float(np.std(np.concatenate([c["gcf"] for c in d["cells"]]))) or 1.0)
        smd_p = abs(float(M["dpB"].mean())) / (d["sp"] or 1.0)
        usable = (M["npair"] >= MIN_PAIRS and M["ntgt"] >= MIN_TGT
                  and smd_g <= MAX_SMD and smd_p <= MAX_SMD)
        RES[fn] = {"pop": nm, "n": d["n"], "cells": len(d["cells"]),
                   "A": {"effect": round(obs, 4), "p_perm": round(pval(obs, null), 5),
                         "null_mean": round(float(null.mean()), 4),
                         "null_sd": round(float(null.std()), 4),
                         "null_p995": round(float(np.quantile(null, .995)), 4),
                         "z_vs_null": round((obs - float(null.mean())) / (float(null.std()) or 1), 2),
                         "boot_ci_secondary": cluster_ci(M, a.seed),
                         "npair": M["npair"], "ntgt": M["ntgt"],
                         "smd_gcf": round(smd_g, 4), "smd_pB": round(smd_p, 4)},
                   "usable": bool(usable)}
        r = RES[fn]["A"]
        print(f"  {nm:12s} 效应 {r['effect']:+7.3f}  置换 p {r['p_perm']:.5f}  "
              f"零 {r['null_mean']:+.3f}±{r['null_sd']:.3f}  z={r['z_vs_null']:5.1f}  "
              f"配对 {r['npair']:5d}/{r['ntgt']:3d}靶  SMD {r['smd_gcf']:.3f}/{r['smd_pB']:.3f}"
              f"  {'可用' if usable else '不可用'}")
        print(f"               次要:bootstrap 区间 {r['boot_ci_secondary']}  "
              f"(半宽 {(r['boot_ci_secondary'][1]-r['boot_ci_secondary'][0])/2:.3f} "
              f"vs 置换SD {r['null_sd']:.3f} —— 第76条要看的就是这两个数)")

    # -------------------------------------------------- P4 structural placebo (GATED)
    print(f"\n--- P4 结构性安慰剂:配对位集合整体环形平移 × {NSHIFT}(本轮的核心闸门) ---")
    rng = np.random.default_rng(4000 + a.seed)
    p4 = {}
    for nm, fn in POPS:
        d = D[fn]
        if not RES[fn].get("A"):
            continue
        eff, cors, leak = [], [], []
        for k in range(NSHIFT):
            sf = float(rng.uniform(0.10, 0.90))
            cols, cr = shifted_gc(d, sf)
            M = match(d, cols, qdelta(d, cols))
            if M is None:
                continue
            eff.append(point(M)); cors.append(cr); leak.append(float(M["dgcp"].mean()))
        eff = np.array(eff)
        obs = RES[fn]["A"]["effect"]
        q95 = float(np.quantile(eff, .95)) if len(eff) else float("nan")
        ok = bool(obs > q95)
        p4[fn] = {"pop": nm, "n_shift": len(eff), "shift_mean": round(float(eff.mean()), 4),
                  "shift_max": round(float(eff.max()), 4), "shift_q95": round(q95, 4),
                  "real": obs, "pass": ok, "ratio": round(obs / (float(eff.mean()) or 1e-9), 2),
                  "mean_corr_with_gcp": round(float(np.nanmean(cors)), 3),
                  "leaked_gcp_gap": round(float(np.mean(leak)), 5),
                  "real_gcp_gap": round(float(match(d, [c["gcp"] for c in d["cells"]],
                                                    DELTA)["dgcp"].mean()), 5)}
        print(f"  {nm:12s} 真实 {obs:+7.3f}  平移均值 {eff.mean():+7.3f}  平移最大 {eff.max():+7.3f}"
              f"  95分位 {q95:+7.3f}  比值 {p4[fn]['ratio']:5.2f}x  "
              f"重叠相关 {p4[fn]['mean_corr_with_gcp']:+.3f}  "
              f"漏进的真实Δgc_paired {p4[fn]['leaked_gcp_gap']:+.4f}(真臂 {p4[fn]['real_gcp_gap']:+.4f})"
              f"  {'通过' if ok else '**不通过**'}")
    npass = sum(1 for v in p4.values() if v["pass"])
    print(f"  P4 通过 {npass}/6")

    # -------------------------------------------------- mechanism M1: match dG_target too
    print("\n--- 机制臂 M1:把目标结构自由能 dG_target 也配平 ---")
    m1 = {}
    for nm, fn in POPS:
        d = D[fn]; M = match(d, [c["gcp"] for c in d["cells"]], DELTA, dg=True)
        if M is None or M["npair"] < 30:
            print(f"  {nm:12s} 配对不足({0 if M is None else M['npair']})"); continue
        obs = point(M)
        null = null_dist(d, a.seed, max(400, a.nperm // 5), dg=True)
        base = RES[fn]["A"]["effect"]
        m1[fn] = {"pop": nm, "effect": round(obs, 4), "p_perm": round(pval(obs, null), 5),
                  "npair": M["npair"], "vs_main": round(obs / (base or 1e-9), 2)}
        print(f"  {nm:12s} 效应 {obs:+7.3f}(主臂 {base:+.3f},保留 {m1[fn]['vs_main']:.0%})  "
              f"p {m1[fn]['p_perm']:.4f}  配对 {M['npair']}")

    # -------------------------------------------------- dose response + moderation
    print("\n--- M2 剂量-反应(预先声明:单调递增) ---")
    m2 = {}
    for nm, fn in POPS:
        d = D[fn]; M = match(d, [c["gcp"] for c in d["cells"]], DELTA)
        if M is None:
            continue
        row = []
        for lo, hi in DOSE:
            m = (M["dose"] >= lo) & (M["dose"] < hi)
            if m.sum() < 30:
                row.append(None); continue
            s, c2 = collections.defaultdict(float), collections.defaultdict(int)
            for t, dd in zip(M["tg"][m], M["diff"][m]):
                s[t] += float(dd); c2[t] += 1
            row.append({"lo": lo, "n": int(m.sum()),
                        "effect": round(sum(s.values()) / sum(c2.values()), 3)})
        m2[fn] = row
        print(f"  {nm:12s} " + "  ".join(
            f"[{DOSE[i][0]:.2f}) n={r['n']:5d} {r['effect']:+7.3f}" if r else f"[{DOSE[i][0]:.2f}) —"
            for i, r in enumerate(row)))

    print("\n--- M3 调节:按靶标平均螺旋长度中位数分半(预先声明:短螺旋那半更大) ---")
    m3 = {}
    for nm, fn in POPS:
        d = D[fn]
        hm = np.array([c["helix_mean"] for c in d["cells"]])
        med = float(np.median(hm)); half = {}
        for lab, sel in (("短", hm <= med), ("长", hm > med)):
            sub = {"cells": [c for c, s in zip(d["cells"], sel) if s], "sp": d["sp"],
                   "sdg": d["sdg"], "n": d["n"], "sd_gcp": d["sd_gcp"], "sd_y": d["sd_y"]}
            M = match(sub, [c["gcp"] for c in sub["cells"]], DELTA)
            half[lab] = round(point(M), 3) if M and M["npair"] >= 30 else None
        m3[fn] = {"median_helix": round(med, 2), **half}
        print(f"  {nm:12s} 中位螺旋 {med:.2f}  短半 {half['短']}  长半 {half['长']}")

    # -------------------------------------------------- P0 / P1 / P2
    print("\n--- P0 结构性恒等式(处理列换成已被配平的 gc_frac) ---")
    p0 = {}
    for nm, fn in POPS:
        d = D[fn]; M = match(d, [c["gcf"] for c in d["cells"]], DELTA)
        p0[fn] = 0 if M is None else M["npair"]
        print(f"  {nm:12s} 存活配对 {p0[fn]}")
    p0_ok = all(v == 0 for v in p0.values())
    print(f"  P0 {'通过' if p0_ok else '不通过'}")

    print(f"\n--- P1 等宽纯噪声 {R_NOISE} 根(以置换 p < {P_GATE} 计数) ---")
    p1 = {}
    for nm, fn in POPS:
        d = D[fn]
        if fn not in NULLS:
            continue
        rng2 = np.random.default_rng(1000 + a.seed); pos = 0
        for k in range(R_NOISE):
            cols = []
            for c in d["cells"]:
                v = rng2.standard_normal(len(c["y"])) * d["sd_gcp"]
                cols.append(v - v.mean())
            M = match(d, cols, qdelta(d, cols))
            if M and pval(point(M), NULLS[fn]) < P_GATE:
                pos += 1
        p1[fn] = pos
        print(f"  {nm:12s} 正向显著 {pos}/{R_NOISE}")
    p1_ok = all(v <= 1 for v in p1.values())
    print(f"  P1 {'通过' if p1_ok else '不通过'}")

    print("\n--- P2 梯级(第74条:量程按主臂实测上界设,不按预言设) ---")
    lad = {}
    for rho in RHOS:
        eff, hit, tot = [], 0, 0
        for nm, fn in POPS:
            d = D[fn]
            if fn not in NULLS:
                continue
            rng2 = np.random.default_rng(int(rho * 1e4) + a.seed)
            thr = float(np.quantile(NULLS[fn], .995))
            for k in range(DRAWS):
                cols = synth(d, rho, rng2)
                M = match(d, cols, qdelta(d, cols))
                if M is None:
                    continue
                e = point(M); eff.append(e); hit += (e > thr); tot += 1
        lad[rho] = {"E": round(float(np.mean(eff)), 4) if eff else None,
                    "detect": round(hit / tot, 3) if tot else None, "n": tot}
        print(f"  rho={rho:.2f}  实测 E[配对差]={lad[rho]['E']:+.4f} wet  达标 {lad[rho]['detect']:.2f}")
    top = max(v["E"] for v in lad.values() if v["E"] is not None)
    mx = max(r["A"]["effect"] for r in RES.values() if r.get("A"))
    thr = next((lad[r]["E"] for r in RHOS if lad[r]["detect"] and lad[r]["detect"] >= PASS), None)
    p2_ok = top >= mx
    print(f"  顶档 {top:+.4f}  主臂最大 {mx:+.4f}  检出阈值 {thr}")
    print(f"  P2(b) {'通过' if p2_ok else '不通过 => K0'};K4 "
          f"{'触发' if (thr is None or thr > mx) else f'不触发(余量 {mx/thr:.1f}x)'}")

    # -------------------------------------------------- verdict
    ok = [fn for fn, r in RES.items() if r.get("A") and r.get("usable")]
    nsig = sum(1 for fn in ok if RES[fn]["A"]["p_perm"] < P_GATE)
    k0 = not (p0_ok and p1_ok and p2_ok)
    print(f"\n可用人群 {len(ok)}/6;置换 p<{P_GATE} 的 {nsig};P4 通过 {npass}/6")
    if k0:
        verdict = "void_probe_failed"
    elif len(ok) < 3:
        verdict = "no_data"
    elif nsig >= 4 and npass >= 4:
        verdict = "gc_placement_rule_holds"
    elif nsig >= 4:
        verdict = "gc_helps_but_placement_unproven"      # K5
    elif all(RES[fn]["A"]["effect"] <= 0 or RES[fn]["A"]["p_perm"] >= 0.05 for fn in ok):
        verdict = "no_placement_effect"
    else:
        verdict = "under_replicated"
    print("裁定:", verdict)
    json.dump({"pops": RES, "P4": p4, "M1_dg_matched": m1, "M2_dose": m2, "M3_moderation": m3,
               "P0": p0, "P1": p1, "P2_ladder": {str(k): v for k, v in lad.items()},
               "top_rung": top, "main_max": mx, "threshold": thr,
               "gates": {"P0": p0_ok, "P1": p1_ok, "P2": p2_ok, "P4_pass": npass},
               "verdict": verdict}, open(a.out, "w"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
