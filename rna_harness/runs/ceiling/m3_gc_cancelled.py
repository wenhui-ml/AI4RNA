#!/usr/bin/env python3
"""M3: Deng move three -- when term-by-term bounding fails, look for CANCELLATION.

Pre-registered in m3_gc_cancelled_declaration.json.

THE CORRECTION THIS CARRIES. Today I said "matched by a GC count" a dozen times. The actual number for
gamma* is -0.0020 with a 95% interval of [-0.0791, +0.0728] -- wide enough to hold twice the effect either
way. That is not "matched", it is "not measured". Nine of today's twelve results were of that kind and I
wrote all of them up as closed.

WHY EXACT CANCELLATION BEATS CONTROLLING FOR IT. Adding GC to a regression subtracts it linearly; residual
correlation can still carry it. But if two designs have IDENTICAL gc_cross and gc_frac, then in any linear
model the two GC columns contribute exactly zero to the DIFFERENCE of predictions, whatever their fitted
coefficients. So restricting the pairwise ranking metric to GC-matched pairs cancels GC by construction. The
question becomes the positive one: with GC gone, what still ranks?

A_G minus A on matched pairs is the built-in must-be-zero control, and it is not vacuous: get the matching
condition wrong and it fires. The probe at the end deliberately matches on gc_frac only and must trip it.

COST: $0, offline.
"""
from __future__ import annotations

import argparse, collections, json, math, os, random, statistics as st, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from e1_ensemble_marginals import base_block, load_pop, marginals, E_FREE      # noqa: E402
from h1_heldout_ensemble_defect import fit_predict                            # noqa: E402
import h2_heldout_cluster_ci as H                                             # noqa: E402

SPLITS, R_NULL, MIN_PAIRS = 200, 12, 30


def acc_matched(rows, pred, key, mode="exact"):
    """mode: 'exact' = gc_cross AND gc_frac equal; 'gfrac' = gc_frac only; 'none' = no matching at all."""
    """Pairwise ranking accuracy counting ONLY GC-matched pairs. Prediction ties score 0.5."""
    ok = 0.0
    n = tt = tp = skipped = 0
    for _c, g in rows:
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                ka, kb = key[id(g[a])], key[id(g[b])]
                same = True if mode == "none" else (ka[1] == kb[1]) if mode == "gfrac" else (ka == kb)
                if not same:
                    skipped += 1
                    continue
                dw = g[a]["wet"] - g[b]["wet"]
                if dw == 0:
                    tt += 1
                    continue
                n += 1
                dp = pred[id(g[a])] - pred[id(g[b])]
                if dp == 0:
                    tp += 1; ok += 0.5
                else:
                    ok += 1.0 if ((dw > 0) == (dp > 0)) else 0.0
    return (ok / n if n else None), n, tt, tp, skipped


def heldout(rws, cols, idx_of, key, seed, mode="exact"):
    by_t = collections.defaultdict(list)
    for c, gp in rws:
        by_t[c[0]].append((c, gp))
    tg = sorted(by_t); order = list(tg)
    REF = next(iter(idx_of))
    per = {k: {t: [] for t in tg} for k in idx_of}
    rng = random.Random(seed)
    st_ = dict(pairs=0, tie_tube=0, leak_mol=0, leak_tg=0, predtie={})
    for _s in range(SPLITS):
        rng.shuffle(order)
        cut = max(1, int(0.7 * len(order)))
        tr, te = set(order[:cut]), set(order[cut:])
        TR = [x for t in tr for x in by_t[t]]; TE = [x for t in te for x in by_t[t]]
        if not TR or not TE:
            continue
        if {id(r) for _c, g in TR for r in g} & {id(r) for _c, g in TE for r in g}:
            st_["leak_mol"] += 1
        if {c[0] for c, _g in TR} & {c[0] for c, _g in TE}:
            st_["leak_tg"] += 1
        P = {k: fit_predict(TR, TE, cols, ix) for k, ix in idx_of.items()}
        for t in te:
            vals, nt = {}, None
            for k in idx_of:
                v, n, tt, tp, _sk = acc_matched(by_t[t], P[k], key, mode)
                vals[k] = v
                st_["predtie"][k] = st_["predtie"].get(k, 0) + tp
                if k == REF:
                    nt = n; st_["pairs"] += n; st_["tie_tube"] += tt
            if nt is None or nt < MIN_PAIRS or any(v is None for v in vals.values()):
                continue
            for k in idx_of:
                per[k][t].append(vals[k])
    used = [t for t in tg if len(per[REF][t]) >= H.MIN_SPLITS_PER_TARGET]
    return per, used, st_, REF


def contrast(per, used, a, b, seed, n_cells, n_mol):
    """MDE is the half-width of THIS contrast's own bootstrap interval (amendment A).

    The first version took sd of block A's per-target ACCURACY, which is between-target difficulty and has
    nothing to do with the interval the verdict reads. That produced the impossible label "interval excludes
    zero AND below MDE" in the smoke. Defined this way, "excludes 0" and "above MDE" are equivalent by
    construction, which is what a resolution floor is supposed to mean.
    """
    g = [st.fmean(per[a][t]) - st.fmean(per[b][t]) for t in used]
    ci, n = H.boot_ci(g, seed + 17)
    assert n == len(used) and n != n_cells and n != n_mol, "闸门6:重抽单位不是靶标"
    mde_c = 1.96 * st.stdev(g) / math.sqrt(len(g))
    return {"mean": round(st.fmean(g), 4), "ci": ci, "n_clusters": n, "MDE": round(mde_c, 4),
            "pos": sum(1 for x in g if x > 0), "excl": bool(ci[0] > 0 or ci[1] < 0),
            "over_mde": abs(st.fmean(g)) > mde_c}


def build(fn, workers):
    from leaderboard_design import targets
    T = targets()
    rec, _s, _m = load_pop(fn)
    keep = [r for r in rec if base_block(r) is not None]
    jobs = [(r["puzzle"], r["seq"], T[r["puzzle"]]["target"]) for r in keep]
    from concurrent.futures import ProcessPoolExecutor
    M = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for m in ex.map(marginals, jobs, chunksize=8):
            M.append(m)
    return keep, M


def run_arm(nm, keep, M, formable_only, seed, mode="exact"):
    """Arm 1 (formable subset) uses ensemble_defect; arm 2 (full) uses the five free ensemble columns."""
    cells, cols, key = collections.defaultdict(list), {}, {}
    for r, m in zip(keep, M):
        if "error" in m:
            continue
        has_ed = "ens_defect_skel" in m
        if formable_only and not has_ed:
            continue
        rr = dict(r)
        B = base_block(r)
        ext = [m["ens_defect_skel"]] if formable_only else [m[k] for k in E_FREE]
        cols[id(rr)] = B + [m["gc_cross"], m["gc_frac"]] + ext
        key[id(rr)] = (m["gc_cross"], round(m["gc_frac"], 3))
        cells[(r["puzzle"], r["method"])].append(rr)
    rws = [(c, g) for c, g in cells.items() if len(g) >= 2]
    mols = [r for _c, g in rws for r in g]
    nB = len(base_block(keep[0])); nE = 1 if formable_only else len(E_FREE)
    iA = list(range(nB)); iG = iA + [nB, nB + 1]
    iX = iA + list(range(nB + 2, nB + 2 + nE))
    xname = "A_ed" if formable_only else "A_E5"
    idx = {"A": iA, xname: iX, "A_G": iG}
    print(f"\n--- {nm} | 分子 {len(mols)} 格 {len(rws)} 靶标 {len({c[0] for c,_g in rws})}"
          f" | 块 A={nB} {xname}={nB+nE} A_G={nB+2} ---", flush=True)
    per, used, s, REF = heldout(rws, cols, idx, key, seed, mode)
    print(f"  泄漏 分子级 {s['leak_mol']}/{SPLITS} 靶标级 {s['leak_tg']}/{SPLITS}"
          f"  {'✓' if not s['leak_mol'] and not s['leak_tg'] else '✗ 作废'}")
    if len(used) < H.MIN_TARGETS:
        print(f"  产生 g_t 的靶标 {len(used)} < {H.MIN_TARGETS} -> 无数据")
        return {"no_data": True, "n_used": len(used)}
    tot = s["pairs"] + s["tie_tube"]
    print(f"  GC 配平且试管可判的对 {s['pairs']},试管并列剔除 {s['tie_tube']}"
          f" = {s['tie_tube']/max(1,tot)*100:.2f}%;各块预测并列率 "
          + " ".join(f"{k} {v/max(1,s['pairs'])*100:.2f}%" for k, v in s["predtie"].items()))
    gA = [st.fmean(per["A"][t]) for t in used]
    print(f"  cluster {len(used)};A 在配平对上的准确率 {st.fmean(gA):.4f}"
          f"(靶标间 sd {st.stdev(gA):.4f} —— 这是难度差异,**不是** MDE)")
    out = {"n_used": len(used), "pairs": s["pairs"],
           "acc": {k: round(st.fmean([st.fmean(per[k][t]) for t in used]), 4) for k in idx},
           "contrasts": {}}
    for a in (xname, "A_G"):
        c = contrast(per, used, a, "A", seed, len(rws), len(mols))
        out["contrasts"][f"{a}_minus_A"] = c
        tag = "**内建必须为零对照**" if a == "A_G" else ""
        print(f"    {a}−A  {c['mean']:+.4f}  95% {c['ci']}  (本对比 MDE +{c['MDE']:.4f})"
              f"  {'排除 0' if c['excl'] else '含 0 -> unresolved,不是 refuted'}"
              f"  正号 {c['pos']}/{c['n_clusters']}  {tag}")
    # width-matched null: add nE useless columns to A (rule 45 revised)
    rngn = random.Random(4242); ms = []
    base = len(cols[id(mols[0])])
    for j in range(R_NULL):
        nz = {k: [rngn.gauss(0, 1.0) for _ in range(nE)] for k in cols}
        aug = {k: v + nz[k] for k, v in cols.items()}
        p2, u2, _s2, _r2 = heldout(rws, aug, {"A": iA, "An": iA + list(range(base, base + nE))},
                                   key, seed, mode)
        uu = [t for t in u2 if len(p2["A"][t]) >= H.MIN_SPLITS_PER_TARGET]
        if len(uu) >= H.MIN_TARGETS:
            ms.append(st.fmean([st.fmean(p2["An"][t]) - st.fmean(p2["A"][t]) for t in uu]))
    if ms:
        out["null"] = {"center": round(st.fmean(ms), 5), "n": len(ms), "width": nE}
        print(f"  无信息列零点(宽度 {nE},{len(ms)} 次):{st.fmean(ms):+.5f}"
              f"  ⇒ {xname} 校正后 {out['contrasts'][xname+'_minus_A']['mean'] - st.fmean(ms):+.4f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "m3_gc_cancelled.json"))
    a = ap.parse_args()
    PRIMARY = "panel_f1_fresh.json"
    POPS = (("样本2(主)", PRIMARY), ("基准4560", "panel_f1_designs.json"),
            ("样本3", "panel_f1_third.json"), ("第四批", "panel_f1_fourth.json"))
    res = {}
    for nm, fn in POPS:
        print(f"\n=== {nm} ===", flush=True)
        keep, M = build(fn, a.workers)
        res[fn] = {"arm1_formable": run_arm(f"{nm} 臂1 骨架可形成 + ensemble_defect", keep, M, True, a.seed),
                   "arm2_full": run_arm(f"{nm} 臂2 全人群 + 五根自由集合边际", keep, M, False, a.seed)}
        if fn == PRIMARY:                      # rule 47 dual obligation, on the primary only
            print(f"\n  [第47条 对偶义务] 探针:只配平 gc_frac(故意写错)—— K0 必须响")
            for pname, pmode, must in (("完全不配平(必须响)", "none", True),
                                       ("只配 gc_frac(冒烟里已知不响)", "gfrac", False)):
                pr = run_arm(f"{nm} 探针 {pname}", keep, M, False, a.seed, mode=pmode)
                res[fn][f"probe_{pmode}"] = pr
                if not pr.get("no_data"):
                    fired = pr["contrasts"]["A_G_minus_A"]["excl"]
                    ok = (fired == must)
                    print(f"  探针[{pname}] K0 {'响' if fired else '不响'} —— "
                          f"{'✓ 符合预期' if ok else '✗ 闸门空转,K0 测不出相消是否失效'}")
        json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)

    print("\n=== 判定(冻结的打死条件)===")
    P1 = res[PRIMARY]["arm1_formable"]; P2 = res[PRIMARY]["arm2_full"]
    def ex(d, k): 
        c = d["contrasts"][k]["ci"]; return c[0] > 0 or c[1] < 0
    K0 = (not P1.get("no_data") and ex(P1, "A_G_minus_A")) or (not P2.get("no_data") and ex(P2, "A_G_minus_A"))
    print(f"  K0 内建必须为零对照(相消是否失效):{'FIRES -> 主人群作废' if K0 else '不触发 ✓ 相消生效'}")
    if K0:
        v = "void_cancellation_failed"
    else:
        K1 = not ex(P1, "A_ed_minus_A"); K2 = not ex(P2, "A_E5_minus_A")
        c1 = P1["contrasts"]["A_ed_minus_A"]; c2 = P2["contrasts"]["A_E5_minus_A"]
        K4 = (not c1["over_mde"]) and (not c2["over_mde"])
        reps = sum(1 for fn in ("panel_f1_designs.json", "panel_f1_third.json", "panel_f1_fourth.json")
                   if not res[fn]["arm1_formable"].get("no_data") and ex(res[fn]["arm1_formable"], "A_ed_minus_A"))
        K3 = (not K1) and reps < 2
        print(f"  K1 臂1 ensemble_defect 在 GC 相消后({c1['mean']:+.4f},本对比 MDE +{c1['MDE']:.4f},"
              f"效应/MDE {abs(c1['mean'])/c1['MDE']:.2f}):"
              f"{'FIRES -> unresolved,不是 refuted' if K1 else '不触发 ⇒ 排除 0'}")
        print(f"  K2 臂2 五根自由集合边际({c2['mean']:+.4f},本对比 MDE +{c2['MDE']:.4f},"
              f"效应/MDE {abs(c2['mean'])/c2['MDE']:.2f}):"
              f"{'FIRES -> unresolved,不是 refuted' if K2 else '不触发 ⇒ 排除 0'}")
        print(f"  K3 复现不足(复现 {reps}/3):{'FIRES' if K3 else '不触发'}")
        print(f"  K4 两臂都低于 MDE:{'FIRES -> 报【测不出】' if K4 else '不触发'}")
        v = ("unresolved_after_gc_cancellation" if (K1 and K2) else
             "gc_free_signal_in_ensemble_defect" if not K1 else
             "gc_free_signal_in_free_marginals")
        if K3: v += "__underreplicated"
        if K4: v += "__not_measurable"
    print(f"  裁定:{v}")
    print("  **口径(声明冻结):区间含 0 一律写 `unresolved`,绝不写 `refuted`。**")
    print("  边界:没有盲人群(H2/C1 读过全部六个);仍在 Turner 族内,不修判官独立性;只量信息量,不判上线。")
    res["verdict"] = v
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
