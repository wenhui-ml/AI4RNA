#!/usr/bin/env python3
"""M13: two real molecules, matched, differing only in where the GC sits.

Declaration: m13_gc_placement_declaration.json, frozen and pushed at c080083e BEFORE this file
existed (the tenth declaration done that way).

M12 said a sequence-intrinsic block adds. gc_paired's share of that was a drop-one coefficient in
a linear model. This file fits no model to produce the estimate: it compares pairs of real
molecules from the same puzzle and the same method, with global GC matched to two percentage
points and the shipped panel's prognostic score matched to 0.2 SD, differing by at least five
percentage points in the fraction of GC sitting at target-paired positions.

The output is a design rule stated in wet units. It mentions no verifier.

COST: $0, offline.
"""
from __future__ import annotations

import argparse, collections, json, math, os, random, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)

POPS = (("样本2(主)", "panel_f1_fresh.json"), ("基准4560", "panel_f1_designs.json"),
        ("样本3", "panel_f1_third.json"), ("第四批", "panel_f1_fourth.json"),
        ("第五批(盲)", "panel_f1_fifth.json"), ("第六批(盲)", "panel_f1_sixth.json"))

CAL_GC, CAL_P, DELTA, NBOOT, NFOLD = 0.02, 0.20, 0.05, 2000, 5
R_NOISE, DRAWS = 12, 20
RHOS = (0.02, 0.05, 0.10, 0.15, 0.25, 0.35, 0.45)
DOSE = ((0.05, 0.10), (0.10, 0.15), (0.15, 9.0))
MIN_PAIRS, MIN_TGT = 100, 15


# ---------------------------------------------------------------- population assembly
def assemble(rows, seed):
    """-> dict of aligned arrays; pB is the cross-fitted panel-only prediction."""
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["puzzle"], r["method"])].append(r)
    rws = [(c, g) for c, g in sorted(cells.items()) if len(g) >= 2]
    if not rws:
        return None
    y, B, ci, tg, gcf, gcp = [], [], [], [], [], []
    for k, (cell, grp) in enumerate(rws):
        w = np.array([r["wet"] for r in grp], float)
        X = np.array([r["B"] for r in grp], float)
        y.append(w - w.mean()); B.append(X - X.mean(axis=0))
        ci += [k] * len(grp); tg += [cell[0]] * len(grp)
        gcf += [r["F"]["gc_frac"] for r in grp]; gcp += [r["F"]["gc_paired"] for r in grp]
    y = np.concatenate(y); B = np.vstack(B)
    ci = np.array(ci); tg = np.array(tg)
    gcf = np.array(gcf, float); gcp = np.array(gcp, float)

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
    cellslice = [np.where(ci == k)[0] for k in range(len(rws))]
    return {"y": y, "pB": pB, "ci": ci, "tg": tg, "gcf": gcf, "gcp": gcp,
            "cells": cellslice, "ncell": len(rws), "n": len(y)}


# ---------------------------------------------------------------- matching
def match(D, treat, delta, use_panel=True, cal_gc=CAL_GC, cal_p=CAL_P):
    """all ordered pairs (i,j) inside a cell with treat_i - treat_j >= delta and calipers met.

    Returns per-target (sum of wet differences, count) plus the raw dose/diff lists.
    """
    y, gcf, pB = D["y"], D["gcf"], D["pB"]
    sp = float(np.std(pB)) or 1.0
    sums, cnts = collections.defaultdict(float), collections.defaultdict(int)
    doses, diffs, tgs, ij = [], [], [], []
    for idx in D["cells"]:
        if len(idx) < 2:
            continue
        t = treat[idx]; dt = t[:, None] - t[None, :]
        ok = dt >= delta
        if not ok.any():
            continue
        g = gcf[idx]
        ok &= np.abs(g[:, None] - g[None, :]) <= cal_gc
        if use_panel:
            p = pB[idx]
            ok &= np.abs(p[:, None] - p[None, :]) <= cal_p * sp
        if not ok.any():
            continue
        a, b = np.nonzero(ok)
        w = y[idx]
        d = w[a] - w[b]
        tt = D["tg"][idx[0]]
        sums[tt] += float(d.sum()); cnts[tt] += len(d)
        doses.append(dt[a, b]); diffs.append(d); tgs += [tt] * len(d)
        ij.append((idx[a], idx[b]))
    if not diffs:
        return None
    return {"sums": sums, "cnts": cnts, "dose": np.concatenate(doses),
            "diff": np.concatenate(diffs), "tg": np.array(tgs),
            "npair": int(sum(cnts.values())), "ntgt": len(cnts), "ij": ij}


def boot(M, seed, nb=NBOOT):
    """target cluster bootstrap of the pair-weighted mean difference."""
    T = sorted(M["cnts"])
    s = np.array([M["sums"][t] for t in T]); c = np.array([M["cnts"][t] for t in T], float)
    obs = float(s.sum() / c.sum())
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(T), size=(nb, len(T)))
    num = s[ix].sum(1); den = c[ix].sum(1)
    good = den > 0
    B = np.sort(num[good] / den[good])
    lo, hi = float(B[int(.025 * len(B))]), float(B[int(.975 * len(B))])
    tmeans = s / c
    return {"mean": round(obs, 4), "ci": [round(lo, 4), round(hi, 4)],
            "excl_pos": bool(lo > 0), "excl": bool(lo > 0 or hi < 0),
            "npair": M["npair"], "ntgt": M["ntgt"],
            "target_mean_of_means": round(float(tmeans.mean()), 4)}


def quantile_matched_delta(D, col, ref_col=None, ref_delta=DELTA):
    """rule 71 width-matching: the delta putting `col` at the same tail fraction as ref_delta puts
    the real column, over the within-cell ordered-pair gap distribution."""
    def gaps(v):
        out = []
        for idx in D["cells"]:
            if len(idx) < 2:
                continue
            t = v[idx]; g = t[:, None] - t[None, :]
            out.append(g[g > 0])
        return np.concatenate(out) if out else np.array([0.0])
    ref = gaps(ref_col if ref_col is not None else D["gcp"])
    q = float((ref >= ref_delta).mean())
    gc = gaps(col)
    if q <= 0 or len(gc) == 0:
        return float(np.max(gc)) + 1.0
    return float(np.quantile(gc, 1.0 - q))


def synth(D, rho, rng):
    y = D["y"]; sd = float(np.std(y)) or 1.0
    c = rho * (y / sd) + math.sqrt(max(0.0, 1 - rho * rho)) * rng.standard_normal(len(y))
    for idx in D["cells"]:
        c[idx] -= c[idx].mean()
    return c


def perm_within_cell(D, rng):
    c = D["gcp"].copy()
    for idx in D["cells"]:
        c[idx] = c[idx][rng.permutation(len(idx))]
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default=os.path.join(HERE, "m13_features.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "m13_gc_placement.json"))
    a = ap.parse_args()
    print("=== M13:配平之后的成对对照 —— 把 GC 放在配对位上,试管里测得更好吗 ===")
    print("  声明 c080083e(代码存在之前已推送)。离线 $0。不拟合模型产生估计量。\n")
    CACHE = json.load(open(a.cache))

    RES, DATA = {}, {}
    for nm, fn in POPS:
        rows = CACHE[fn]["rows"]
        D = assemble(rows, a.seed)
        if D is None:
            print(f"=== {nm} === 无可用格\n"); continue
        DATA[fn] = (nm, D)
        MA = match(D, D["gcp"], DELTA, use_panel=True)
        MB = match(D, D["gcp"], DELTA, use_panel=False)
        rec = {"pop": nm, "n": D["n"], "cells": D["ncell"]}
        if MA is None:
            rec["A"] = None; print(f"=== {nm} === A 臂零配对\n"); RES[fn] = rec; continue
        rec["A"] = boot(MA, a.seed)
        rec["B"] = boot(MB, a.seed) if MB else None
        rec["usable"] = bool(rec["A"]["npair"] >= MIN_PAIRS and rec["A"]["ntgt"] >= MIN_TGT)
        print(f"=== {nm} ===  分子 {D['n']}  格 {D['ncell']}")
        for k in ("A", "B"):
            r = rec.get(k)
            if r:
                print(f"  {k} 臂  配对 {r['npair']:6d}  靶标 {r['ntgt']:3d}  "
                      f"配对差 {r['mean']:+.4f} [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}]"
                      f"  {'排除0' if r['excl_pos'] else ('负向排除0' if r['excl'] else '含0')}"
                      f"   逐靶标均值 {r['target_mean_of_means']:+.4f}")
        # ---- D arm: dose response
        dose = []
        for lo, hi in DOSE:
            m = (MA["dose"] >= lo) & (MA["dose"] < hi)
            if m.sum() >= 30:
                s, c = collections.defaultdict(float), collections.defaultdict(int)
                for t, d in zip(MA["tg"][m], MA["diff"][m]):
                    s[t] += float(d); c[t] += 1
                dose.append({"lo": lo, "hi": hi,
                             **boot({"sums": s, "cnts": c, "npair": int(m.sum()),
                                     "ntgt": len(c)}, a.seed)})
            else:
                dose.append({"lo": lo, "hi": hi, "npair": int(m.sum()), "mean": None})
        rec["D_dose"] = dose
        print("  D 剂量-反应:", "  ".join(
            f"[{d['lo']:.2f},{d['hi']:.2f}) n={d['npair']}"
            + (f" {d['mean']:+.4f}{'*' if d.get('excl_pos') else ''}" if d["mean"] is not None else " —")
            for d in dose))
        # ---- C arm: greedy 1-to-1
        used = set(); s, c = collections.defaultdict(float), collections.defaultdict(int)
        order = np.argsort(-MA["dose"])
        A_i = np.concatenate([x[0] for x in MA["ij"]]); A_j = np.concatenate([x[1] for x in MA["ij"]])
        for k in order:
            i, j = int(A_i[k]), int(A_j[k])
            if i in used or j in used:
                continue
            used.add(i); used.add(j)
            t = D["tg"][i]; s[t] += float(D["y"][i] - D["y"][j]); c[t] += 1
        if sum(c.values()) >= 30:
            rec["C_1to1"] = boot({"sums": s, "cnts": c, "npair": int(sum(c.values())),
                                  "ntgt": len(c)}, a.seed)
            r = rec["C_1to1"]
            print(f"  C 臂 1对1  配对 {r['npair']:5d}  {r['mean']:+.4f} "
                  f"[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] {'排除0' if r['excl_pos'] else '含0'}")
        # ---- shape 2: leave-one-target-out / leave-one-method-out
        T = sorted(MA["cnts"]); ss = np.array([MA["sums"][t] for t in T])
        cc = np.array([MA["cnts"][t] for t in T], float)
        loo = [(ss.sum() - ss[i]) / (cc.sum() - cc[i]) for i in range(len(T)) if cc.sum() > cc[i]]
        rec["loo_target_range"] = [round(min(loo), 4), round(max(loo), 4)] if loo else None
        print(f"  留一靶标极差 {rec['loo_target_range']}")
        RES[fn] = rec
        print()

    json.dump({"pops": RES}, open(a.out, "w"), ensure_ascii=False, indent=1)
    # ---------------- gates on the main arm only; probes run in m13_probes.py
    ok = [r for r in RES.values() if r.get("A") and r.get("usable")]
    npos = sum(1 for r in ok if r["A"]["excl_pos"])
    print(f"可用人群 {len(ok)}/6;主臂正向排除 0 的 {npos}")
    verdict = ("no_data" if len(ok) < 3 else
               "gc_placement_rule_holds" if npos >= 4 else
               "no_placement_effect" if all(r["A"]["mean"] <= 0 or not r["A"]["excl_pos"] for r in ok)
               else "under_replicated")
    print("裁定(未含探针,探针在 m13_probes.py):", verdict)
    j = json.load(open(a.out)); j["verdict_main_arm_only"] = verdict
    json.dump(j, open(a.out, "w"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
