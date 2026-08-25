#!/usr/bin/env python3
"""M1d: the placebo arm M1c's new column never had. Pre-registered in m1d_meanp_placebo_declaration.json.

M1c handed over a simpler column -- the mean pair probability over the skeleton pairs -- at +0.0148
excluding zero, level with seven classified columns while ensemble_defect at the same padded width sat at
+0.0059 containing zero. Two things were never measured: the single-column-versus-single-column contrast
(M1c padded both to width 7, and its direct ed-vs-mean contrast was unresolved at -0.0089 containing zero),
and a placebo. Rule 52 makes the placebo mandatory, and a "mean pair probability" is itself a measure of
ensemble concentration, so it needs that control MORE than ensemble_defect does, not less.

Everything is scored only on pairs with identical gc_cross and gc_frac, inheriting M3's construction and its
built-in A_G control. Every arm is a SINGLE added column, so M1c's width penalty cannot recur.

COST: $0, offline.
"""
from __future__ import annotations

import argparse, collections, json, math, os, random, statistics as st, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from e1_ensemble_marginals import base_block, load_pop                            # noqa: E402
from h1_heldout_ensemble_defect import fit_predict                                # noqa: E402
import h2_heldout_cluster_ci as H                                                 # noqa: E402
from m1_indexset_bookkeeping import acc_multi, stats_of                            # noqa: E402

COLS = ["p_all_skel", "ed_skel", "p_all_mfe", "p_all_cross"]
R_NOISE = 12
MIN_PAIRS = 30
CANON = {frozenset({"A", "U"}), frozenset({"G", "C"}), frozenset({"G", "U"})}
POPS = (("样本2(主)", "panel_f1_fresh.json"), ("基准4560", "panel_f1_designs.json"),
        ("样本3", "panel_f1_third.json"), ("第四批", "panel_f1_fourth.json"))
PRIMARY = "panel_f1_fresh.json"


def work(item):
    puz, seq, target = item
    import RNA
    from rna_harness.benchmarks.rnainvbench import _round_only, pk_pairs
    from gamma_star import crossing_set
    try:
        cross = sorted(crossing_set(target))
        if not cross:
            return {"puzzle": puz, "error": "no crossing pairs"}
        skel = _round_only(target)
        sp = sorted(pk_pairs(skel))
        if not sp:
            return {"puzzle": puz, "error": "no skeleton pairs"}
        if [1 for (i, j) in sp if frozenset({seq[i], seq[j]}) not in CANON]:
            return {"puzzle": puz, "error": "undefined: skeleton not formable"}
        n = len(seq)
        fc = RNA.fold_compound(seq)
        mfe_s, _e = fc.mfe()
        fc.pf()
        bpp = fc.bpp()
        mp = sorted(pk_pairs(mfe_s))

        def meanp(pairs):
            if not pairs:
                return None
            v = []
            for (i, j) in pairs:
                a, b = (i + 1, j + 1) if i < j else (j + 1, i + 1)
                v.append(float(bpp[a][b]))
            return st.fmean(v)
        ps, pm, pc = meanp(sp), meanp(mp), meanp(cross)
        if ps is None or pc is None:
            return {"puzzle": puz, "error": "empty index set"}
        if pm is None:                      # an unpaired MFE: the placebo is undefined, use 0.0 and count
            pm = 0.0
        cs = set(cross); gc = 0
        for (i, j) in sorted(cs):
            if (i + 1, j - 1) in cs:
                for (x, y) in ((i, j), (i + 1, j - 1)):
                    if {seq[x], seq[y]} == {"G", "C"}:
                        gc += 1
        return {"puzzle": puz, "V": [ps, fc.ensemble_defect(skel), pm, pc],
                "gc": [float(gc), (seq.count("G") + seq.count("C")) / n],
                "mfe_unpaired": not mp}
    except Exception as exc:                                                      # noqa: BLE001
        return {"puzzle": puz, "error": repr(exc)[:140]}


def build(fn, workers):
    from leaderboard_design import targets
    T = targets()
    rec, _s, _m = load_pop(fn)
    keep = [r for r in rec if base_block(r) is not None]
    jobs = [(r["puzzle"], r["seq"], T[r["puzzle"]]["target"]) for r in keep]
    from concurrent.futures import ProcessPoolExecutor
    M = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for m in ex.map(work, jobs, chunksize=8):
            M.append(m)
    return keep, M


def run_pop(nm, keep, M, splits, seed, noise_seed, variant="real", mode="exact", verbose=False):
    """variant: 'real' | 'noisePS' (K1 probe) | 'psIsEd' (K2 probe: p_all_skel := ed_skel)."""
    cells, cols, key = collections.defaultdict(list), {}, {}
    rngn = random.Random(noise_seed)
    for r, m in zip(keep, M):
        if "error" in m:
            continue
        rr = dict(r)
        V = list(m["V"])
        if variant == "noisePS":
            V[0] = rngn.gauss(0, 1.0)
        elif variant == "psIsEd":
            V[0] = V[1]
        cols[id(rr)] = base_block(r) + m["gc"] + V + [rngn.gauss(0, 1.0) for _ in range(R_NOISE)]
        key[id(rr)] = (m["gc"][0], round(m["gc"][1], 3))
        cells[(r["puzzle"], r["method"])].append(rr)
    rws = [(c, g) for c, g in cells.items() if len(g) >= 2]
    if not rws:
        return {"no_data": True}
    nB = len(base_block(keep[0])); o_gc = nB; o_v = nB + 2; o_n = o_v + len(COLS)
    iA = list(range(nB))
    idx = {"A": iA, "A_G": iA + [o_gc, o_gc + 1],
           "A_ps": iA + [o_v], "A_ed": iA + [o_v + 1],
           "A_pm": iA + [o_v + 2], "A_pc": iA + [o_v + 3]}
    for j in range(R_NOISE):
        idx[f"A_n{j}"] = iA + [o_n + j]
    keys = list(idx)
    by_t = collections.defaultdict(list)
    for c, gp in rws:
        by_t[c[0]].append((c, gp))
    tg = sorted(by_t)
    per = {k: {t: [] for t in tg} for k in keys if k != "A"}
    per["ps_minus_ed"] = {t: [] for t in tg}
    rng = random.Random(seed); order = list(tg)
    S = dict(leak=0, pairs=0)
    for _s in range(splits):
        rng.shuffle(order)
        cut = max(1, int(0.7 * len(order)))
        tr, te = set(order[:cut]), set(order[cut:])
        TR = [x for t in tr for x in by_t[t]]; TE = [x for t in te for x in by_t[t]]
        if not TR or not TE:
            continue
        if {id(r) for _c, g in TR for r in g} & {id(r) for _c, g in TE for r in g}:
            S["leak"] += 1
        P = {k: fit_predict(TR, TE, cols, ix) for k, ix in idx.items()}
        for t in te:
            acc, n, _tt, _sk = acc_multi(by_t[t], P, keys, key, mode)
            S["pairs"] += n
            if n < MIN_PAIRS or any(v is None for v in acc.values()):
                continue
            for k in keys:
                if k != "A":
                    per[k][t].append(acc[k] - acc["A"])
            per["ps_minus_ed"][t].append(acc["A_ps"] - acc["A_ed"])
    used = [t for t in tg if len(per["A_ps"][t]) >= H.MIN_SPLITS_PER_TARGET]
    print(f"  {nm:28s} 格 {len(rws)} 靶标 {len(used)} 配平可判对 {S['pairs']}"
          f" | 泄漏 {S['leak']}/{splits} {'✓' if not S['leak'] else '✗'}", flush=True)
    if verbose:
        V = {COLS[i]: [cols[id(r)][o_v + i] for _c, g in rws for r in g] for i in range(len(COLS))}
        gv = [cols[id(r)][o_gc] for _c, g in rws for r in g]
        print("    V2 方差:" + ", ".join(f"{k} {np.var(v):.4g}" for k, v in V.items()))
        print("    V2 两两相关:" + ", ".join(
            f"{COLS[i]}~{COLS[j]} {np.corrcoef(V[COLS[i]], V[COLS[j]])[0,1]:+.3f}"
            for i in range(len(COLS)) for j in range(i + 1, len(COLS))))
        print("    V2 与 gc_cross:" + ", ".join(f"{k} {np.corrcoef(v, gv)[0,1]:+.3f}" for k, v in V.items()))
        nu = sum(1 for m in M if "error" not in m and m.get("mfe_unpaired"))
        print(f"    安慰剂无定义(MFE 全无配对)的设计:{nu}")
    if len(used) < H.MIN_TARGETS:
        return {"no_data": True, "n_used": len(used)}
    return {"n_used": len(used), **{k: stats_of([st.fmean(per[k][t]) for t in used], seed, k) for k in per}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "m1d_meanp_placebo.json"))
    a = ap.parse_args()
    R = {}; KEEP = {}
    print("=== M1d 单列对单列:骨架平均边际概率 vs ensemble_defect,带安慰剂与指标集对照 ===", flush=True)
    for nm, fn in POPS:
        KEEP[fn] = build(fn, a.workers)
        print(f"  折叠完 {nm}", flush=True)

    print("\n=== 三个必响探针(第 47/54 条)===", flush=True)
    keep, M = KEEP[PRIMARY]
    p0 = run_pop("探针 完全不配平(K0 必响)", keep, M, a.splits, a.seed, 8801, "real", "none")
    p1 = run_pop("探针 主列换噪声(K1 必响)", keep, M, a.splits, a.seed, 8802, "noisePS")
    p2 = run_pop("探针 主列 := ed(K2 必响)", keep, M, a.splits, a.seed, 8803, "psIsEd")
    ok0 = p0["A_G"]["excl"]; ok1 = not p1["A_ps"]["excl"]; ok2 = not p2["ps_minus_ed"]["excl"]
    for tag, o, ok in (("K0 A_G−A", p0["A_G"], ok0), ("K1 A_ps−A", p1["A_ps"], ok1),
                       ("K2 ps−ed", p2["ps_minus_ed"], ok2)):
        print(f"  {tag:14s} {o['mean']:+.4f} {o['ci']}  {'**响 ✓**' if ok else '✗ 没响'}")
    R["probes"] = {"K0": bool(ok0), "K1": bool(ok1), "K2": bool(ok2)}
    if not (ok0 and ok1 and ok2):
        R["verdict"] = "void_probe_did_not_fire"
        json.dump(R, open(a.out, "w"), indent=1, ensure_ascii=False)
        print("  **第 54 条:本次运行作废,不许事后换探针。**"); return 0

    print("\n=== 主运行 ===", flush=True)
    for j, (nm, fn) in enumerate(POPS):
        k2, m2 = KEEP[fn]
        R[fn] = run_pop(nm, k2, m2, a.splits, a.seed, 8900 + 101 * j, verbose=(fn == PRIMARY))
        json.dump(R, open(a.out, "w"), indent=1, ensure_ascii=False)

    P = R[PRIMARY]
    print("\n=== 主人群(全部单列)===")
    LB = {"A_G": "内建必须为零", "A_ps": "**骨架平均边际概率(主臂)**", "A_ed": "ensemble_defect(主比较)",
          "A_pm": "**安慰剂:MFE 配对的平均 p**", "A_pc": "指标集对照:交叉配对的平均 p",
          "ps_minus_ed": "**主判据:主臂 − ensemble_defect**"}
    for k in ("A_G", "A_ps", "A_ed", "A_pm", "A_pc", "ps_minus_ed"):
        o = P[k]
        print(f"  {k:14s} {o['mean']:+.4f}  95% {o['ci']}  MDE +{o['MDE']:.4f}  效应/MDE {o['eff_over_mde']:.2f}"
              f"  正号 {o['pos']}/{o['n']}  {'排除 0' if o['excl'] else '含 0'}"
              f"  [种子 {o['seed_flip']}]  {LB[k]}")

    print(f"\n=== 逐人群与同侧假阳(第 53 条改写版,{R_NOISE} 根单列)===")
    drop = []
    for nm, fn in POPS:
        d = R[fn]
        if d.get("no_data"):
            print(f"  {nm:10s} 无数据"); continue
        pos = sum(1 for j in range(R_NOISE) if d[f"A_n{j}"]["excl"] and d[f"A_n{j}"]["ci"][0] > 0)
        neg = sum(1 for j in range(R_NOISE) if d[f"A_n{j}"]["excl"] and d[f"A_n{j}"]["ci"][1] < 0)
        nz = [d[f"A_n{j}"]["mean"] for j in range(R_NOISE)]
        if fn != PRIMARY and pos >= 2:
            drop.append(nm)
        print(f"  {nm:10s} 零点 {st.fmean(nz):+.5f} | **同侧(正) {pos}/{R_NOISE}**,负 {neg}/{R_NOISE}"
              f" | A_ps−A {d['A_ps']['mean']:+.4f} {'排除0' if d['A_ps']['excl'] else '含0 '}"
              f" | A_ed−A {d['A_ed']['mean']:+.4f} {'排除0' if d['A_ed']['excl'] else '含0 '}"
              f" | 主判据 {d['ps_minus_ed']['mean']:+.4f} {'排除0' if d['ps_minus_ed']['excl'] else '含0'}")

    print("\n=== 判定(冻结的打死条件)===")
    ex = lambda d, k: d[k]["excl"]
    K0 = ex(P, "A_G"); K1 = not ex(P, "A_ps"); K2 = not ex(P, "ps_minus_ed")
    K3 = ex(P, "A_pm") and abs(P["A_pm"]["mean"]) >= 0.5 * abs(P["A_ps"]["mean"])
    K4 = ex(P, "A_pc") and abs(P["A_pc"]["mean"]) >= 0.5 * abs(P["A_ps"]["mean"])
    reps = 0; tot_r = 0
    for nm, fn in POPS:
        if fn == PRIMARY or R[fn].get("no_data") or nm in drop:
            continue
        tot_r += 1; reps += 1 if ex(R[fn], "A_ps") else 0
    K5 = (not K1) and reps < 2
    print(f"  K0 相消失效:{'FIRES -> 作废' if K0 else '不触发 ✓'}")
    print(f"  K1 主臂无增量({P['A_ps']['mean']:+.4f}):{'FIRES -> unresolved' if K1 else '不触发 ⇒ 排除 0'}")
    print(f"  K2 **主臂打不过 ensemble_defect**({P['ps_minus_ed']['mean']:+.4f}):"
          f"{'FIRES' if K2 else '**不触发 ⇒ 排除 0,它确实更好**'}")
    print(f"  K3 安慰剂也解释得了({P['A_pm']['mean']:+.4f} vs 主臂一半 {0.5*abs(P['A_ps']['mean']):+.4f}):"
          f"{'FIRES' if K3 else '不触发'}")
    print(f"  K4 指标集不特异(交叉配对 {P['A_pc']['mean']:+.4f}):{'FIRES' if K4 else '不触发'}")
    print(f"  K5 复现不足({reps}/{tot_r}{';第53条作废:' + ','.join(drop) if drop else ''}):"
          f"{'FIRES' if K5 else '不触发'}")
    if K0:
        v = "void_cancellation_failed"
    elif K1:
        v = "unresolved_mean_p"
    elif K3:
        v = "explained_by_generic_concentration"
    elif K4:
        v = "index_set_not_specific"
    elif K2:
        v = "mean_p_indistinguishable_from_ensemble_defect"
    else:
        v = "mean_p_beats_ensemble_defect"
    if K5:
        v += "__underreplicated"
    print(f"  裁定:{v}")
    print("  **口径:区间含 0 一律写 `unresolved` 或强制标签,绝不写 `refuted`。**")
    print("  边界:**没有盲人群**;第五/六批在 GC 精确配平下只剩 16 和 1 个靶标,本实验不用它们,"
          "而且它们本来也不是盲(H2 读过);只用骨架可形成子集;仍在 Turner 族内;**不判上线**。")
    R["verdict"] = v
    json.dump(R, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
