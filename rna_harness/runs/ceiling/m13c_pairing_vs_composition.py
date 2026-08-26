#!/usr/bin/env python3
"""M13c: is it the base composition at paired positions, or is it the base PAIRS?

Declaration: m13c_pairing_vs_composition_declaration.json, frozen and pushed at af3236ca BEFORE
this file existed (the twelfth). Unlike M13b, this one IS blind to me -- I have looked at exactly
one number, and it contains no wet readout: after conditioning on gc_paired, the GC-pair fraction
keeps 40.8% of its marginal SD.

The OpenKnot Score rewards low 2A3 SHAPE reactivity at target-paired positions. So M13b's result
has two readings: the GC really makes the target base pairs stronger so the molecule folds (甲), or
SHAPE chemistry is base-dependent and G/C residues simply read as unreactive (乙). A circular shift
cannot separate them because it moves composition and pairing together.

This does separate them. A target base pair can be G-C, A-U, G-U wobble, or non-canonical. Holding
2*n_GC + n_GU fixed -- the composition at paired positions -- n_GC can still move.

  arm P : match composition, contrast pairing   -> 甲 predicts positive, 乙 predicts zero
  arm C : match pairing, contrast composition   -> 乙 predicts positive, 甲 predicts zero

Compared as SLOPES (wet units per unit of treatment), never as raw effects: the two arms cannot
reach the same treatment gap, so raw effects are not comparable.

COST: $0, offline.
"""
from __future__ import annotations

import argparse, collections, json, math, os, random, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from m12_sequence_intrinsic import parse_pairs                                        # noqa: E402
from m13_cache import POPS                                                            # noqa: E402

CAL_GC, CAL_P, CAL_T, DELTA = 0.02, 0.20, 0.20, 0.05
NPERM, NFOLD, R_NOISE, DRAWS = 2000, 5, 12, 20
RHOS = (0.02, 0.05, 0.10, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.80)
MIN_PAIRS, MIN_TGT, MAX_SMD, PASS, P_GATE = 100, 15, 0.10, 0.8, 0.005


def classify(seq, P):
    c = collections.Counter()
    for i, j in P:
        b = {seq[i], seq[j]}
        c["GC" if b == {"G", "C"} else "AU" if b == {"A", "U"} else
          "GU" if b == {"G", "U"} else "NC"] += 1
    n = max(1, sum(c.values()))
    return c["GC"] / n, c["AU"] / n, c["GU"] / n, c["NC"] / n


def assemble(rows, extra, PM, seed):
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
    fold = np.array([fa[t] for t in tg]); pB = np.zeros(len(y))
    for f in range(NFOLD):
        te, tr = fold == f, fold != f
        if tr.sum() <= B.shape[1] + 10 or te.sum() == 0:
            continue
        b, *_ = np.linalg.lstsq(B[tr], y[tr], rcond=None); pB[te] = B[te] @ b
    CELLS, o = [], 0
    for cell, grp in rws:
        k = len(grp); sl = slice(o, o + k); o += k
        P = PM[cell[0]]
        cls = np.array([classify(e["seq"], P) for _r, e in grp])
        CELLS.append({"tgt": cell[0], "y": y[sl], "pB": pB[sl],
                      "gcf": np.array([r["F"]["gc_frac"] for r, _e in grp], float),
                      "gcp": np.array([r["F"]["gc_paired"] for r, _e in grp], float),
                      "pair": cls[:, 0], "au": cls[:, 1], "gu": cls[:, 2], "nc": cls[:, 3]})
    cat = lambda k: np.concatenate([c[k] for c in CELLS])
    return {"cells": CELLS, "n": len(y),
            "sp": float(np.std(pB)) or 1.0,
            "sd_gcp": float(np.std(cat("gcp"))) or 1.0,
            "sd_pair": float(np.std(cat("pair"))) or 1.0,
            "sd_gcf": float(np.std(cat("gcf"))) or 1.0, "sd_y": float(np.std(y)) or 1.0}


def match(D, treat, delta, hold):
    """hold: name of the third caliper column ('gcp' or 'pair' or None)."""
    sd = {"gcp": D["sd_gcp"], "pair": D["sd_pair"]}.get(hold, 1.0)
    sums, cnts = collections.defaultdict(float), collections.defaultdict(int)
    gap, hgap, dgcf, dpB, diff, tgs, ncg = [], [], [], [], [], [], []
    for c, t in zip(D["cells"], treat):
        if len(t) < 2:
            continue
        dt = t[:, None] - t[None, :]; ok = dt >= delta
        if not ok.any():
            continue
        g = c["gcf"]; ok &= np.abs(g[:, None] - g[None, :]) <= CAL_GC
        p = c["pB"]; ok &= np.abs(p[:, None] - p[None, :]) <= CAL_P * D["sp"]
        if hold:
            h = c[hold]; ok &= np.abs(h[:, None] - h[None, :]) <= CAL_T * sd
        if not ok.any():
            continue
        a, b = np.nonzero(ok); d = c["y"][a] - c["y"][b]
        sums[c["tgt"]] += float(d.sum()); cnts[c["tgt"]] += len(d)
        gap.append(dt[a, b]); diff.append(d); tgs += [c["tgt"]] * len(d)
        dgcf.append(g[a] - g[b]); dpB.append(p[a] - p[b])
        ncg.append(c["nc"][a] - c["nc"][b])
        if hold:
            hgap.append(c[hold][a] - c[hold][b])
    if not diff:
        return None
    return {"sums": sums, "cnts": cnts, "gap": np.concatenate(gap), "diff": np.concatenate(diff),
            "tg": np.array(tgs), "dgcf": np.concatenate(dgcf), "dpB": np.concatenate(dpB),
            "dnc": np.concatenate(ncg),
            "hgap": np.concatenate(hgap) if hgap else np.zeros(1),
            "npair": int(sum(cnts.values())), "ntgt": len(cnts)}


def point(M):
    return float(sum(M["sums"].values()) / sum(M["cnts"].values()))


def cluster_ci(M, seed, nb=2000):
    T = sorted(M["cnts"]); s = np.array([M["sums"][t] for t in T])
    c = np.array([M["cnts"][t] for t in T], float)
    rng = np.random.default_rng(seed); ix = rng.integers(0, len(T), size=(nb, len(T)))
    num, den = s[ix].sum(1), c[ix].sum(1)
    B = np.sort(num[den > 0] / den[den > 0])
    return [round(float(B[int(.025 * len(B))]), 3), round(float(B[int(.975 * len(B))]), 3)]


def gaps_of(cols):
    o = []
    for t in cols:
        if len(t) < 2:
            continue
        g = t[:, None] - t[None, :]; o.append(g[g > 0])
    return np.concatenate(o) if o else np.array([0.0])


def qdelta(D, treat):
    ref = gaps_of([c["gcp"] for c in D["cells"]])
    q = float((ref >= DELTA).mean()); gc = gaps_of(treat)
    return float(np.quantile(gc, 1.0 - q)) if q > 0 and len(gc) else float(gc.max()) + 1.0


def null_dist(D, key, delta, hold, seed, nperm):
    rng = np.random.default_rng(seed); out = []
    for _ in range(nperm):
        cols = [c[key][rng.permutation(len(c[key]))] for c in D["cells"]]
        M = match(D, cols, delta, hold)
        out.append(point(M) if M else 0.0)
    return np.sort(np.array(out))


def pval(obs, null):
    return (float((null >= obs).sum()) + 1) / (len(null) + 1)


def run_arm(D, key, hold, seed, nperm, label):
    treat = [c[key] for c in D["cells"]]
    delta = DELTA if key == "gcp" else qdelta(D, treat)
    M = match(D, treat, delta, hold)
    if M is None:
        return None
    obs = point(M); g = float(M["gap"].mean())
    null = null_dist(D, key, delta, hold, seed, nperm)
    smd = {"gcf": abs(float(M["dgcf"].mean())) / D["sd_gcf"],
           "pB": abs(float(M["dpB"].mean())) / D["sp"],
           hold: abs(float(M["hgap"].mean())) / ({"gcp": D["sd_gcp"], "pair": D["sd_pair"]}[hold])}
    return {"arm": label, "treat": key, "held": hold, "delta": round(delta, 4),
            "effect": round(obs, 4), "gap": round(g, 5), "slope": round(obs / g, 2),
            "p_perm": round(pval(obs, null), 5), "null_sd": round(float(null.std()), 4),
            "z": round((obs - float(null.mean())) / (float(null.std()) or 1), 2),
            "boot_ci": cluster_ci(M, seed), "npair": M["npair"], "ntgt": M["ntgt"],
            "smd": {k: round(v, 4) for k, v in smd.items()},
            "usable": bool(M["npair"] >= MIN_PAIRS and M["ntgt"] >= MIN_TGT
                           and max(smd.values()) <= MAX_SMD),
            "d_noncanonical": round(float(M["dnc"].mean()), 5), "_null": null, "_M": M}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nperm", type=int, default=NPERM)
    ap.add_argument("--out", default=os.path.join(HERE, "m13c_pairing_vs_composition.json"))
    a = ap.parse_args()
    print("=== M13c:是配对位上的【碱基组成】,还是那些【碱基对】本身 ===")
    print("  声明 af3236ca(代码存在之前已推送)。**本实验对我是盲的。** 离线 $0。")
    print("  P 臂:配平组成,对照配对  ->  (甲)折叠 预测正,(乙)化学 预测零")
    print("  C 臂:配平配对,对照组成  ->  (乙)化学 预测正,(甲)折叠 预测零")
    print("  两臂只能按【斜率】比(wet/单位处理),因为可达 gap 天生不同。\n")
    from leaderboard_design import targets as load_targets
    T = load_targets(); PM = {p: sorted(parse_pairs(d["target"])) for p, d in T.items()}
    C = json.load(open(os.path.join(HERE, "m13_features.json")))
    E = json.load(open(os.path.join(HERE, "m13b_extra.json")))
    D = {fn: assemble(C[fn]["rows"], E[fn], PM, a.seed) for _nm, fn in POPS}

    RES = {}
    for nm, fn in POPS:
        d = D[fn]
        P = run_arm(d, "pair", "gcp", a.seed, a.nperm, "P_pairing")
        Q = run_arm(d, "gcp", "pair", a.seed, a.nperm, "C_composition")
        RES[fn] = {"pop": nm, "P": P, "C": Q}
        print(f"=== {nm} ===")
        for r in (P, Q):
            if r is None:
                print("   (零配对)"); continue
            print(f"  {r['arm']:14s} delta {r['delta']:.4f}  效应 {r['effect']:+7.3f}  "
                  f"实测gap {r['gap']:.4f}  **斜率 {r['slope']:+8.2f}**  p {r['p_perm']:.5f}  "
                  f"z {r['z']:5.1f}  配对 {r['npair']:5d}/{r['ntgt']:3d}靶  "
                  f"SMD最大 {max(r['smd'].values()):.3f}  {'可用' if r['usable'] else '不可用'}")
            print(f"                 配平残余 {r['smd']}  Δ非规范对 {r['d_noncanonical']:+.5f}"
                  f"  bootstrap {r['boot_ci']}")
        if P and Q:
            print(f"  斜率比 P/C = {P['slope']/(Q['slope'] or 1e-9):.2f};配对数比 P/C = "
                  f"{P['npair']/max(1,Q['npair']):.2f}")
        print()

    # ---- shape_3: is the pairing arm driven by designs that simply broke the target pairing?
    print("--- 形状三自检:P 臂是不是由『非规范/未成对』那一档驱动的 ---")
    sh3 = {}
    for nm, fn in POPS:
        r = RES[fn]["P"]
        if not r:
            continue
        M = r["_M"]; nc = M["dnc"]
        lo = np.abs(nc) < 1e-9
        if lo.sum() >= 30:
            s, c2 = collections.defaultdict(float), collections.defaultdict(int)
            for t, dd in zip(M["tg"][lo], M["diff"][lo]):
                s[t] += float(dd); c2[t] += 1
            e = sum(s.values()) / sum(c2.values())
        else:
            e = None
        sh3[fn] = {"n_nc_equal": int(lo.sum()), "effect_when_nc_equal": round(e, 3) if e else None}
        print(f"  {nm:12s} 非规范对数完全相同的配对 {int(lo.sum()):5d}/{M['npair']:5d}  "
              f"该子集效应 {sh3[fn]['effect_when_nc_equal']}  (全臂 {r['effect']:+.3f})")

    # ---- probes
    print("\n--- P0 结构性恒等式(处理列换成该臂已配平的那一根) ---")
    p0 = {}
    for nm, fn in POPS:
        d = D[fn]
        a1 = match(d, [c["gcp"] for c in d["cells"]], DELTA, "gcp")
        a2 = match(d, [c["pair"] for c in d["cells"]], qdelta(d, [c["pair"] for c in d["cells"]]), "pair")
        p0[fn] = [0 if a1 is None else a1["npair"], 0 if a2 is None else a2["npair"]]
        print(f"  {nm:12s} P臂形式 {p0[fn][0]}  C臂形式 {p0[fn][1]}")
    p0_ok = all(v == [0, 0] for v in p0.values())
    print(f"  P0 {'通过' if p0_ok else '不通过'}")

    print(f"\n--- P1 等宽噪声 {R_NOISE} 根(以 p<{P_GATE} 计数,两臂各做) ---")
    p1 = {}
    for nm, fn in POPS:
        d = D[fn]; out = {}
        for lab, hold in (("P", "gcp"), ("C", "pair")):
            r = RES[fn][{"P": "P", "C": "C"}[lab]]
            if not r:
                out[lab] = None; continue
            rng = np.random.default_rng(1000 + a.seed); pos = 0
            for k in range(R_NOISE):
                cols = []
                for c in d["cells"]:
                    v = rng.standard_normal(len(c["y"])) * d["sd_pair"]
                    cols.append(v - v.mean())
                M = match(d, cols, qdelta(d, cols), hold)
                if M and pval(point(M), r["_null"]) < P_GATE:
                    pos += 1
            out[lab] = pos
        p1[fn] = out
        print(f"  {nm:12s} P臂 {out['P']}/{R_NOISE}   C臂 {out['C']}/{R_NOISE}")
    p1_ok = all(v is None or v <= 1 for o in p1.values() for v in o.values())
    print(f"  P1 {'通过' if p1_ok else '不通过'}")

    print("\n--- P2 梯级(第74条:顶档下限 3.8) ---")
    lad = {}
    for rho in RHOS:
        eff, hit, tot = [], 0, 0
        for nm, fn in POPS:
            d = D[fn]; r = RES[fn]["P"]
            if not r:
                continue
            rng = np.random.default_rng(int(rho * 1e4) + a.seed)
            thr = float(np.quantile(r["_null"], .995))
            for k in range(DRAWS):
                cols = []
                for c in d["cells"]:
                    v = (rho * (c["y"] / d["sd_y"])
                         + math.sqrt(max(0.0, 1 - rho * rho)) * rng.standard_normal(len(c["y"])))
                    cols.append(v - v.mean())
                M = match(d, cols, qdelta(d, cols), "gcp")
                if M is None:
                    continue
                e = point(M); eff.append(e); hit += (e > thr); tot += 1
        lad[rho] = {"E": round(float(np.mean(eff)), 4) if eff else None,
                    "detect": round(hit / tot, 3) if tot else None}
        print(f"  rho={rho:.2f}  E={lad[rho]['E']:+.4f}  达标 {lad[rho]['detect']:.2f}")
    top = max(v["E"] for v in lad.values() if v["E"] is not None)
    mx = max([r[k]["effect"] for r in RES.values() for k in ("P", "C") if r[k]] or [0])
    p2_ok = top >= max(3.8, mx)
    print(f"  顶档 {top:+.4f}  两臂最大 {mx:+.4f}  P2(b) {'通过' if p2_ok else '不通过 => K0'}")

    # ---- verdict
    usP = [fn for fn in RES if RES[fn]["P"] and RES[fn]["P"]["usable"]]
    usC = [fn for fn in RES if RES[fn]["C"] and RES[fn]["C"]["usable"]]
    nP = sum(1 for fn in usP if RES[fn]["P"]["p_perm"] < P_GATE)
    nC = sum(1 for fn in usC if RES[fn]["C"]["p_perm"] < P_GATE)
    sP = [RES[fn]["P"]["slope"] for fn in usP]; sC = [RES[fn]["C"]["slope"] for fn in usC]
    mP = float(np.median(sP)) if sP else 0.0; mC = float(np.median(sC)) if sC else 0.0
    ratio = mP / mC if mC else float("inf")
    print(f"\n可用人群 P {len(usP)}/6  C {len(usC)}/6;p<{P_GATE}: P {nP}  C {nC}")
    print(f"中位斜率 P {mP:+.2f}  C {mC:+.2f}  比值 {ratio:.2f}")
    npair_ratio = (np.median([RES[fn]["P"]["npair"] for fn in usP]) /
                   max(1, np.median([RES[fn]["C"]["npair"] for fn in usC]))) if usP and usC else 0
    if not (p0_ok and p1_ok and p2_ok):
        v = "void_probe_failed"
    elif len(usP) < 3 and len(usC) < 3:
        v = "no_data"
    elif nP >= 4 and nC >= 4:
        v = "both_contribute"
    elif nP >= 4 and 0.8 <= ratio <= 1.25:
        v = "both_contribute"
    elif nP >= 4 and mP > mC:
        v = "pairing_not_composition"
    elif nC >= 4 and mC > mP:
        v = "composition_not_pairing"
    else:
        v = "underpowered_or_inseparable"
    print("裁定:", v)
    for r in RES.values():
        for k in ("P", "C"):
            if r[k]:
                r[k].pop("_null", None); r[k].pop("_M", None)
    json.dump({"pops": RES, "shape3": sh3, "P0": p0, "P1": p1,
               "P2_ladder": {str(k): v2 for k, v2 in lad.items()}, "top_rung": top,
               "median_slope_P": round(mP, 3), "median_slope_C": round(mC, 3),
               "slope_ratio_P_over_C": round(ratio, 3), "npair_ratio_P_over_C": round(float(npair_ratio), 2),
               "gates": {"P0": p0_ok, "P1": p1_ok, "P2": p2_ok}, "verdict": v},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
