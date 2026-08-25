#!/usr/bin/env python3
"""M5: the fourth family. Not "what is the target" but "what state are the tube and the panel in there".

Pre-registered in m5_measurement_geometry_declaration.json.

WHERE THIS COMES FROM. Three representation families are now measured empty -- target structure (13
candidates with a proven-power screen), data quantity, and the predicted pair set (classified bookkeeping
never beat its own collapse on either index set). And yet the invariant exists: M2b arm2 selected targets by
g_t measured on three populations, evaluated on a fourth the selection never touched, and separated the
halves by +0.0344 at permutation p=0.0002; M2d then showed the cross-population correlation sits at 84% of
its attenuation ceiling. Something real is there and I cannot name it.

g_t is not a function of the target. It is a function of target x tube x panel: how much does adding
ensemble_defect improve the ranking HERE. The first three families all asked about the target alone.

THE LEAKAGE TRAP AND THE FIX. `base_acc` is one of the two terms of g_t = A_ed - A, so correlating them on
the same data is regression to the mean, not a finding; the same objection hits every wet-derived column.
So every candidate is computed by LEAVE-ONE-POPULATION-OUT cross-fitting: the evaluation population's g_t is
correlated only against candidate values built from the other three populations, whose designs share zero
molecules with it.

THE BENCHMARK IS THE PROBE. The must-fire column is the empirical invariant itself -- the other populations'
g_t, the thing M2b arm2 used. It has to clear Bonferroni or the screen has no power and the run is void; and
it is also the yardstick, because it IS the quantity we are trying to explain.

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
from m2_invariant_split import spearman, rho_ci, rho_perm_p, half_stats, R_SEED    # noqa: E402
from m2b_gt_target_property import fold                                           # noqa: E402
from m2b_review_audit import acc_pair_of_blocks, perm_diff_p                      # noqa: E402

POPS = (("样本2", "panel_f1_fresh.json"), ("基准4560", "panel_f1_designs.json"),
        ("样本3", "panel_f1_third.json"), ("第四批", "panel_f1_fourth.json"))
CAND = ["wet_sd", "tie_rate", "hard_frac", "wet_mean", "base_acc", "base_room",
        "pred_sd", "ed_sd", "ed_range", "oracle_room"]
ALPHA_B = 0.05 / len(CAND)
BENCH = "__BENCHMARK_gt_other_pops"


def g_and_geometry(keep, M, splits, seed, global_med_dw=None):
    """Per-target g_t (C1 scoring) plus the measurement-geometry columns for the SAME population."""
    cells, cols = collections.defaultdict(list), {}
    for r, m in zip(keep, M):
        if "error" in m:
            continue
        rr = dict(r)
        cols[id(rr)] = base_block(r) + m["C"] + [m["ed_skel"], m["ed_mfe"]]
        cells[(r["puzzle"], r["method"])].append(rr)
    rws = [(c, g) for c, g in cells.items() if len(g) >= 2]
    tg = sorted({c[0] for c, _g in rws})
    nB = len(base_block(keep[0]))
    iBC = list(range(nB + 2)); iED = iBC + [nB + 2]
    by_t = collections.defaultdict(list)
    for c, gp in rws:
        by_t[c[0]].append((c, gp))

    # ---- geometry that needs no model -------------------------------------------------------------
    all_dw = []
    for _c, g in rws:
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                all_dw.append(abs(g[a]["wet"] - g[b]["wet"]))
    med = global_med_dw if global_med_dw is not None else (st.median(all_dw) if all_dw else 0.0)
    geo = {}
    for t in tg:
        sds, ties, hard, npair, orooms, edsd, edrg, wets = [], 0, 0, 0, [], [], [], []
        for _c, g in by_t[t]:
            w = [r["wet"] for r in g]
            e = [cols[id(r)][nB + 2] for r in g]
            wets += w
            if len(w) >= 2:
                sds.append(st.stdev(w)); edsd.append(st.stdev(e))
            edrg.append(max(e) - min(e))
            orooms.append(max(w) - st.fmean(w))
            for a in range(len(g)):
                for b in range(a + 1, len(g)):
                    d = abs(w[a] - w[b]); npair += 1
                    if d == 0:
                        ties += 1
                    if d < med:
                        hard += 1
        geo[t] = {"wet_sd": st.fmean(sds) if sds else 0.0,
                  "tie_rate": ties / max(1, npair),
                  "hard_frac": hard / max(1, npair),
                  "wet_mean": st.fmean(wets) if wets else 0.0,
                  "ed_sd": st.fmean(edsd) if edsd else 0.0,
                  "ed_range": st.fmean(edrg) if edrg else 0.0,
                  "oracle_room": st.fmean(orooms) if orooms else 0.0}

    # ---- g_t and the model-dependent geometry ------------------------------------------------------
    per_t = {t: [] for t in tg}; acc_t = {t: [] for t in tg}; psd_t = {t: [] for t in tg}
    rng = random.Random(seed); order = list(tg); leak = 0
    for _s in range(splits):
        rng.shuffle(order)
        cut = max(1, int(0.7 * len(order)))
        tr, te = set(order[:cut]), set(order[cut:])
        TR = [x for t in tr for x in by_t[t]]; TE = [x for t in te for x in by_t[t]]
        if not TR or not TE:
            continue
        if {id(r) for _c, g in TR for r in g} & {id(r) for _c, g in TE for r in g}:
            leak += 1
        PA = fit_predict(TR, TE, cols, iBC); PB = fit_predict(TR, TE, cols, iED)
        for t in te:
            aA, aB, nA, _nb, _tt, _ta, _tb = acc_pair_of_blocks(by_t[t], PA, PB, "common05")
            if aA is None or aB is None or nA < H.MIN_PAIRS_PER_TARGET:
                continue
            per_t[t].append(aB - aA); acc_t[t].append(aA)
            sds = [st.stdev([PA[id(r)] for r in g]) for _c, g in by_t[t] if len(g) >= 2]
            psd_t[t].append(st.fmean(sds) if sds else 0.0)
    g = {t: st.fmean(v) for t, v in per_t.items() if len(v) >= H.MIN_SPLITS_PER_TARGET}
    for t in g:
        geo[t]["base_acc"] = st.fmean(acc_t[t])
        geo[t]["base_room"] = 1.0 - st.fmean(acc_t[t])
        geo[t]["pred_sd"] = st.fmean(psd_t[t])
    return g, {t: geo[t] for t in g}, leak, med


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "m5_measurement_geometry.json"))
    a = ap.parse_args()
    R = {}; G = {}; GEO = {}; SETS = {}
    print("=== M5 第四族:测量几何。留一人群交叉拟合,候选与被解释量来自不相交的设计 ===", flush=True)
    keeps = {}
    for nm, fn in POPS:
        keeps[fn] = fold(fn, a.workers)
    meds = []
    for _n, fn in POPS:
        keep, M = keeps[fn]
        g, geo, lk, md = g_and_geometry(keep, M, a.splits, a.seed)
        G[fn], GEO[fn] = g, geo
        SETS[fn] = {(r["puzzle"], r["method"], r["seq"]) for r in keep}
        meds.append(md)
        print(f"  {[n for n, f in POPS if f == fn][0]:10s} 靶标 {len(g)}  g 均值 {st.fmean(list(g.values())):+.4f}"
              f"  泄漏 {lk}/{a.splits} {'✓' if not lk else '✗'}", flush=True)

    print("\n=== 闸门 V1:四人群分子级零重叠(候选与被解释量不共享任何设计)===")
    worst = 0
    for i, (n1, f1) in enumerate(POPS):
        for n2, f2 in POPS[i + 1:]:
            k = len(SETS[f1] & SETS[f2]); worst = max(worst, k)
            print(f"  {n1} ∩ {n2} = {k}")
    print(f"  {'✓ 零重叠' if worst == 0 else '✗ 有重叠 -> 作废'}")
    R["overlap_max"] = worst
    if worst:
        R["verdict"] = "void_overlap"; json.dump(R, open(a.out, "w"), indent=1, ensure_ascii=False); return 0

    print(f"\n=== 留一轮换筛选:{len(CAND)} 个候选,Bonferroni {ALPHA_B:.4f},区间取 {100*(1-ALPHA_B):.1f}% ===")
    ROT = {}
    for c in CAND + [BENCH]:
        rhos, ns = [], []
        for nm, fn in POPS:
            others = [f for _n2, f in POPS if f != fn]
            col = {}
            for t in G[fn]:
                if c == BENCH:
                    v = [G[o][t] for o in others if t in G[o]]
                else:
                    v = [GEO[o][t][c] for o in others if t in GEO[o] and c in GEO[o][t]]
                if len(v) >= 2:
                    col[t] = st.fmean(v)
            ts = sorted(col)
            if len(ts) < 30:
                continue
            rhos.append(spearman([col[t] for t in ts], [G[fn][t] for t in ts])); ns.append(len(ts))
        ROT[c] = rhos
    # permutation null on the MEAN of the four rotation rhos
    scr = {}
    for c in CAND + [BENCH]:
        if not ROT[c]:
            scr[c] = {"skipped": True}; continue
        mr = st.fmean(ROT[c])
        # rebuild columns once per rotation for the permutation
        pv_hits = 0; NP = 2000
        rngp = random.Random(a.seed + 55)
        cols_rot = []
        for nm, fn in POPS:
            others = [f for _n2, f in POPS if f != fn]
            col = {}
            for t in G[fn]:
                v = ([G[o][t] for o in others if t in G[o]] if c == BENCH
                     else [GEO[o][t][c] for o in others if t in GEO[o] and c in GEO[o][t]])
                if len(v) >= 2:
                    col[t] = st.fmean(v)
            ts = sorted(col)
            if len(ts) >= 30:
                cols_rot.append(([col[t] for t in ts], [G[fn][t] for t in ts]))
        for _ in range(NP):
            s = 0.0
            for x, y in cols_rot:
                yy = list(y); rngp.shuffle(yy)
                s += spearman(x, yy)
            if abs(s / len(cols_rot)) >= abs(mr):
                pv_hits += 1
        pv = (pv_hits + 1) / (NP + 1)
        # interval on the mean rotation rho: cluster bootstrap inside each rotation, averaged
        cis = []
        for x, y in cols_rot:
            cis.append(rho_ci(x, y, a.seed + 31, 1 - ALPHA_B))
        lo = st.fmean([c2[0] for c2 in cis]); hi = st.fmean([c2[1] for c2 in cis])
        ok = bool((lo > 0 or hi < 0) and pv < ALPHA_B)
        scr[c] = {"rho_mean": round(mr, 4), "rho_by_rotation": [round(x, 3) for x in ROT[c]],
                  "ci_bonf_mean": [round(lo, 4), round(hi, 4)], "perm_p": round(pv, 5), "passes": ok}
        tag = "**基准 = 要被解释的那个量(必响)**" if c == BENCH else ""
        print(f"  {c:24s} rho均 {mr:+.4f}  四轮 {scr[c]['rho_by_rotation']}"
              f"  {100*(1-ALPHA_B):.1f}% [{lo:+.3f},{hi:+.3f}]  置换 p {pv:.5f}"
              f"  {'**通过**' if ok else '不通过'} {tag}")
    R["screen"] = scr; R["alpha_bonf"] = ALPHA_B

    K0 = not scr[BENCH].get("passes")
    b = scr[BENCH]
    print(f"\n  闸门 V5 分辨力:基准 rho {b.get('rho_mean')},Bonferroni 区间半宽 "
          f"{(b['ci_bonf_mean'][1]-b['ci_bonf_mean'][0])/2:.3f} ⇒ **本次筛选分辨得了 |rho| ≳ "
          f"{(b['ci_bonf_mean'][1]-b['ci_bonf_mean'][0])/2:.2f}**")
    print(f"  K0 基准列没响:{'**FIRES -> 整个实验作废(第 54 条)**' if K0 else '不触发 ✓ 筛选有检出力'}")
    if K0:
        R["verdict"] = "void_benchmark_did_not_fire"
        json.dump(R, open(a.out, "w"), indent=1, ensure_ascii=False)
        print("  **不许事后换探针给这一跑背书。**"); return 0

    passing = [c for c in CAND if scr.get(c, {}).get("passes")]
    K1 = not passing
    print(f"\n  K1 十个候选无一通过:{'FIRES' if K1 else f'不触发(通过:{passing})'}")
    if K1:
        rank = sorted([c for c in CAND if "rho_mean" in scr[c]], key=lambda c: -abs(scr[c]["rho_mean"]))[:3]
        print("  **强制标签:`no_invariant_found_in_measurement_geometry` —— 十个不是全空间。**")
        print("  下一份声明的线索(不是发现):" + ", ".join(
            f"{c} rho {scr[c]['rho_mean']:+.3f} p {scr[c]['perm_p']:.3f}" for c in rank))
        R["leads"] = rank; R["verdict"] = "no_invariant_found_in_measurement_geometry"
        json.dump(R, open(a.out, "w"), indent=1, ensure_ascii=False)
        print(f"  -> {a.out}"); return 0

    best = max(passing, key=lambda c: abs(scr[c]["rho_mean"]))
    print(f"\n=== K2 最优候选 `{best}` 切不切得开库(基准列给的是 +0.0344,p=0.00020)===")
    diffs = []
    for nm, fn in POPS:
        others = [f for _n2, f in POPS if f != fn]
        col = {t: st.fmean([GEO[o][t][best] for o in others if t in GEO[o]])
               for t in G[fn] if sum(1 for o in others if t in GEO[o]) >= 2}
        if len(col) < 30:
            continue
        thr = st.median(list(col.values()))
        hi = [G[fn][t] for t in col if col[t] > thr]; lo = [G[fn][t] for t in col if col[t] <= thr]
        d, p = perm_diff_p(list(hi), list(lo))
        rb = random.Random(77); bs = []
        for _ in range(5000):
            A = [hi[rb.randrange(len(hi))] for _ in range(len(hi))]
            B = [lo[rb.randrange(len(lo))] for _ in range(len(lo))]
            bs.append(st.fmean(A) - st.fmean(B))
        bs.sort(); ci = [round(bs[125], 4), round(bs[4874], 4)]
        diffs.append({"pop": nm, "diff": round(d, 4), "p": p, "ci": ci,
                      "excl": bool(ci[0] > 0 or ci[1] < 0), "n_hi": len(hi), "n_lo": len(lo)})
        print(f"  {nm:10s} 半间差 {d:+.4f}  95% {ci}  置换 p {p:.5f}"
              f"  {'排除 0' if diffs[-1]['excl'] else '含 0'}")
    R["K2_splits"] = diffs
    K2 = not any(d["excl"] for d in diffs)
    K3 = abs(scr[best]["rho_mean"]) < 0.5 * abs(scr[BENCH]["rho_mean"])
    print(f"\n=== 判定 ===")
    print(f"  K0 筛选无检出力:不触发")
    print(f"  K1 十个候选无一通过:不触发(通过 {passing})")
    print(f"  K2 最优候选切不开:{'FIRES' if K2 else '不触发 ⇒ 它切得开'}")
    print(f"  K3 解释不到基准的一半({abs(scr[best]['rho_mean']):.3f} vs 基准 "
          f"{abs(scr[BENCH]['rho_mean']):.3f} 的一半 {0.5*abs(scr[BENCH]['rho_mean']):.3f}):"
          f"{'FIRES' if K3 else '不触发'}")
    v = ("measurement_geometry_column_found" if not (K2 or K3) else
         "found_but_does_not_split" if K2 else "found_but_explains_less_than_half")
    print(f"  裁定:{v}")
    print("  **口径:区间含 0 一律写 `unresolved`,绝不写 `refuted`。**")
    print("  边界:**没有盲人群**;候选按留一人群交叉拟合,与被解释量零分子重叠;不判上线。")
    R["verdict"] = v; R["best"] = best
    json.dump(R, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
