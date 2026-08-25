#!/usr/bin/env python3
"""M1c: on the SKELETON index set, does keeping the classified bookkeeping beat summing it away?

Pre-registered in m1c_skeleton_bookkeeping_declaration.json.

WHY THIS EXISTS. M1 measured the crossing-pair index set and found it empty (+0.0007, contains 0), which
also explains M2 -- all ten of its candidates were built on that set. The largest number in M1's table came
from the arm I had declared as the index-set PLACEBO: classified bookkeeping over the skeleton pairs,
+0.0159 excluding zero. That label was wrong at design time and provably so without looking at the number,
because the one positive on this line is ensemble_defect(skeleton), a skeleton quantity. M1 never computed
the contrast that matters, and its skeleton arm was 7 columns against ensemble_defect's 1.

So: same index set, two summaries. Kept (7 classified columns) versus collapsed (ensemble_defect, and
separately the plain mean), with BOTH collapsed arms padded with noise to the same width.

Everything is scored only on pairs with identical gc_cross AND gc_frac, so GC contributes exactly zero to
any prediction difference (inherited from M3, with its built-in A_G control and its must-fire probe).

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
from m1_indexset_bookkeeping import classify, CLASSES, acc_multi, stats_of         # noqa: E402

R_NOISE = 12
MIN_PAIRS = 30
CANON = {frozenset({"A", "U"}), frozenset({"G", "C"}), frozenset({"G", "U"})}


def work(item):
    puz, seq, target = item
    import RNA
    from rna_harness.benchmarks.rnainvbench import _round_only, pk_pairs
    from gamma_star import crossing_set
    try:
        cross = sorted(crossing_set(target))
        if not cross:
            return {"puzzle": puz, "error": "target has no crossing pairs"}
        skel = _round_only(target)
        sp = sorted(pk_pairs(skel))
        if not sp:
            return {"puzzle": puz, "error": "target has no skeleton pairs"}
        bad = [(i, j) for (i, j) in sp if frozenset({seq[i], seq[j]}) not in CANON]
        if bad:
            return {"puzzle": puz, "error": "undefined: skeleton not formable"}
        n = len(seq)
        fc = RNA.fold_compound(seq)
        fc.pf()
        bpp = fc.bpp()
        pm = {}
        for (i, j) in set(cross) | set(sp):
            a, b = (i + 1, j + 1) if i < j else (j + 1, i + 1)
            pm[(i, j)] = float(bpp[a][b])
        cc = classify(cross, pm)
        cs_ = classify(sp, pm)
        if cc is None or cs_ is None:
            return {"puzzle": puz, "error": "empty index set"}
        gc = 0
        S = set(cross)
        for (i, j) in sorted(S):
            if (i + 1, j - 1) in S:
                for (x, y) in ((i, j), (i + 1, j - 1)):
                    if {seq[x], seq[y]} == {"G", "C"}:
                        gc += 1
        return {"puzzle": puz, "Ccross": cc[0], "Cskel": cs_[0],
                "imp_cross": sorted(cc[2]), "imp_skel": sorted(cs_[2]),
                "p_all_skel": cs_[1], "ed_skel": fc.ensemble_defect(skel),
                "gc": [float(gc), (seq.count("G") + seq.count("C")) / n]}
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


def run_pop(nm, keep, M, splits, seed, noise_seed, keepc, variant="real", mode="exact", verbose=False):
    """variant: 'real' | 'noiseC' (K1 probe) | 'edlike' (K2 probe: ed + 6 fresh noise, same shape as the
    comparator so K2 must contain 0)."""
    W = len(keepc)
    cells, cols, key = collections.defaultdict(list), {}, {}
    rngn = random.Random(noise_seed)
    for r, m in zip(keep, M):
        if "error" in m:
            continue
        rr = dict(r)
        Cs = [m["Cskel"][i] for i in keepc]
        if variant == "noiseC":
            Cs = [rngn.gauss(0, 1.0) for _ in range(W)]
        elif variant == "edlike":
            Cs = [m["ed_skel"]] + [rngn.gauss(0, 1.0) for _ in range(W - 1)]
        Cc = [m["Ccross"][i] for i in keepc]
        ed_pad = [m["ed_skel"]] + [rngn.gauss(0, 1.0) for _ in range(W - 1)]
        pa_pad = [m["p_all_skel"]] + [rngn.gauss(0, 1.0) for _ in range(W - 1)]
        noise = [rngn.gauss(0, 1.0) for _ in range(W * R_NOISE)]
        cols[id(rr)] = base_block(r) + m["gc"] + Cs + ed_pad + pa_pad + Cc + noise
        key[id(rr)] = (m["gc"][0], round(m["gc"][1], 3))
        cells[(r["puzzle"], r["method"])].append(rr)
    rws = [(c, g) for c, g in cells.items() if len(g) >= 2]
    if not rws:
        return {"no_data": True, "reason": "no cells"}
    nB = len(base_block(keep[0]))
    o_gc = nB; o_cs = nB + 2; o_ed = o_cs + W; o_pa = o_ed + W; o_cc = o_pa + W; o_nz = o_cc + W
    iA = list(range(nB))
    idx = {"A": iA, "A_G": iA + [o_gc, o_gc + 1],
           "A_skelC7": iA + list(range(o_cs, o_cs + W)),
           "A_ed_pad": iA + list(range(o_ed, o_ed + W)),
           "A_pall_pad": iA + list(range(o_pa, o_pa + W)),
           "A_crossC7": iA + list(range(o_cc, o_cc + W))}
    for j in range(R_NOISE):
        idx[f"A_n{j}"] = iA + list(range(o_nz + j * W, o_nz + (j + 1) * W))
    keys = list(idx)
    by_t = collections.defaultdict(list)
    for c, gp in rws:
        by_t[c[0]].append((c, gp))
    tg = sorted(by_t)
    DERIVED = [("skelC7_minus_edpad", "A_skelC7", "A_ed_pad"),
               ("skelC7_minus_pallpad", "A_skelC7", "A_pall_pad"),
               ("edpad_minus_pallpad", "A_ed_pad", "A_pall_pad")]
    per = {k: {t: [] for t in tg} for k in keys if k != "A"}
    for d, _a, _b in DERIVED:
        per[d] = {t: [] for t in tg}
    rng = random.Random(seed); order = list(tg)
    S = dict(pairs=0, tie=0, skip=0, leak=0)
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
            acc, n, tt, sk = acc_multi(by_t[t], P, keys, key, mode)
            S["pairs"] += n; S["tie"] += tt; S["skip"] += sk
            if n < MIN_PAIRS or any(v is None for v in acc.values()):
                continue
            for k in keys:
                if k != "A":
                    per[k][t].append(acc[k] - acc["A"])
            for d, x, y in DERIVED:
                per[d][t].append(acc[x] - acc[y])
    used = [t for t in tg if len(per["A_skelC7"][t]) >= H.MIN_SPLITS_PER_TARGET]
    tot = S["pairs"] + S["tie"] + S["skip"]
    print(f"  {nm:28s} 格 {len(rws)} 靶标 {len(used)} | 配平可判对 {S['pairs']}"
          f"(配平剔除 {S['skip']/max(1,tot)*100:.1f}%) | 泄漏 {S['leak']}/{splits}"
          f" {'✓' if not S['leak'] else '✗ 作废'}", flush=True)
    if verbose and rws:
        vals = {CLASSES[c]: [cols[id(r)][o_cs + i] for _c2, g in rws for r in g] for i, c in enumerate(keepc)}
        pa = [cols[id(r)][o_pa] for _c2, g in rws for r in g]
        gv = [cols[id(r)][o_gc] for _c2, g in rws for r in g]
        print("    V2 骨架分类列与 mean p 的相关:"
              + ", ".join(f"{c} {np.corrcoef(vals[c], pa)[0,1]:+.3f}" for c in vals))
        print("    V7 与 gc_cross 的相关:"
              + ", ".join(f"{c} {np.corrcoef(vals[c], gv)[0,1]:+.3f}" for c in vals))
    if len(used) < H.MIN_TARGETS:
        return {"no_data": True, "n_used": len(used)}
    out = {"n_used": len(used)}
    for k in per:
        out[k] = stats_of([st.fmean(per[k][t]) for t in used], seed, k)
    return out


def sameside(d, claim_positive=True):
    """Rule 53 rewritten: count only noise arms whose interval excludes 0 on the SAME side as the claim."""
    p = n = 0
    for j in range(R_NOISE):
        o = d.get(f"A_n{j}")
        if not o or not o["excl"]:
            continue
        if o["ci"][0] > 0:
            p += 1
        else:
            n += 1
    return (p if claim_positive else n), p, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "m1c_skeleton_bookkeeping.json"))
    a = ap.parse_args()
    PRIMARY = "panel_f1_fresh.json"
    POPS = (("样本2(主)", PRIMARY), ("基准4560", "panel_f1_designs.json"),
            ("样本3", "panel_f1_third.json"), ("第四批", "panel_f1_fourth.json"),
            ("第五批(本线未读)", "panel_f1_fifth.json"), ("第六批(本线未读)", "panel_f1_sixth.json"))
    R = {"note": "第五批/第六批是【本条线未读】,不是【盲】—— H2 读过它们。"}
    KEEP = {}
    print("=== M1c:同一个骨架指标集,保留分类记账 vs 求和掉;全部在 GC 精确配平的对上评估 ===", flush=True)
    print("=== 先折叠六个人群,汇总骨架七类的填补比例,执行闸门 V3(>30% 作废),再跑 ===", flush=True)
    for nm, fn in POPS:
        KEEP[fn] = build(fn, a.workers)
        print(f"  折叠完 {nm}", flush=True)
    impc = collections.Counter(); ntot = 0
    for _n, fn in POPS:
        for m in KEEP[fn][1]:
            if "error" in m:
                continue
            ntot += 1
            for x in m["imp_skel"]:
                impc[x] += 1
    rates = {c: impc[c] / ntot for c in CLASSES}
    keepc = [i for i, c in enumerate(CLASSES) if rates[c] <= 0.30]
    voided = [c for c in CLASSES if rates[c] > 0.30]
    print("  闸门 V3 骨架七类填补比例(六人群 %d 条):" % ntot
          + ", ".join(f"{c} {rates[c]*100:.1f}%" for c in CLASSES))
    print(f"  **V3 作废 {len(voided)} 根:{voided or '无'} ⇒ 用 {len(keepc)} 类;"
          f"两个塌缩臂各补噪声到 {len(keepc)} 列,每根噪声臂 {len(keepc)} 列**")
    R["V3"] = {"rates": {c: round(rates[c], 4) for c in CLASSES}, "voided": voided,
               "classes_used": [CLASSES[i] for i in keepc]}
    if len(keepc) < 3:
        R["verdict"] = "void_too_few_classes"; json.dump(R, open(a.out, "w"), indent=1, ensure_ascii=False)
        print("  **存活的类少于 3 根 ⇒ 作废**"); return 0
    for j, (nm, fn) in enumerate(POPS):
        keep, M = KEEP[fn]
        R[fn] = run_pop(nm, keep, M, a.splits, a.seed, 8000 + 173 * j, keepc, verbose=(fn == PRIMARY))
        json.dump(R, open(a.out, "w"), indent=1, ensure_ascii=False)

    print("\n=== 三个必响探针(第 47/54 条;任一不响 -> 本次作废)===")
    keep, M = KEEP[PRIMARY]
    pr = {}
    pr["K0"] = run_pop("探针 完全不配平(K0 必响)", keep, M, a.splits, a.seed, 8801, keepc, "real", "none")
    pr["K1"] = run_pop("探针 分类列换噪声(K1 必响)", keep, M, a.splits, a.seed, 8802, keepc, "noiseC")
    pr["K2"] = run_pop("探针 分类换成 ed+噪声(K2 必响)", keep, M, a.splits, a.seed, 8803, keepc, "edlike")
    p0 = pr["K0"]["A_G"]["excl"]
    p1 = not pr["K1"]["A_skelC7"]["excl"]
    p2 = not pr["K2"]["skelC7_minus_edpad"]["excl"]
    for tag, o, ok in (("K0 A_G−A", pr["K0"]["A_G"], p0),
                       ("K1 A_skelC7−A", pr["K1"]["A_skelC7"], p1),
                       ("K2 skelC7−ed_pad", pr["K2"]["skelC7_minus_edpad"], p2)):
        print(f"  {tag:22s} {o['mean']:+.4f} {o['ci']}  {'**响 ✓**' if ok else '✗ 没响'}")
    R["probes"] = {"K0": p0, "K1": p1, "K2": p2}
    if not (p0 and p1 and p2):
        R["verdict"] = "void_probe_did_not_fire"
        print("\n  **第 54 条:必响探针没响 ⇒ 本次作废。不许事后换探针给这一跑背书。**")
        json.dump(R, open(a.out, "w"), indent=1, ensure_ascii=False); return 0

    P = R[PRIMARY]
    print("\n=== 主人群 ===")
    LBL = {"A_G": "内建必须为零", "A_skelC7": "**骨架分类记账(主臂)**",
           "A_ed_pad": "ensemble_defect(宽度配平)", "A_pall_pad": "朴素均值 p(宽度配平)",
           "A_crossC7": "交叉配对分类(M1 已测,继续报)",
           "skelC7_minus_edpad": "**主判据:分类 − ensemble_defect**",
           "skelC7_minus_pallpad": "次要 S1:分类 − 朴素均值",
           "edpad_minus_pallpad": "次要 S2:ensemble_defect − 朴素均值"}
    for k in ("A_G", "A_skelC7", "A_ed_pad", "A_pall_pad", "A_crossC7",
              "skelC7_minus_edpad", "skelC7_minus_pallpad", "edpad_minus_pallpad"):
        o = P[k]
        print(f"  {k:22s} {o['mean']:+.4f}  95% {o['ci']}  MDE +{o['MDE']:.4f}  效应/MDE {o['eff_over_mde']:.2f}"
              f"  正号 {o['pos']}/{o['n']}  {'排除 0' if o['excl'] else '含 0'}"
              f"  [种子 {o['seed_flip']}]  {LBL[k]}")

    print(f"\n=== 零点与【同侧】假阳(第 53 条改写版,{R_NOISE} 根 × {len(keepc)} 列)===")
    drop = []
    for nm, fn in POPS:
        d = R[fn]
        if d.get("no_data"):
            print(f"  {nm:16s} 无数据"); continue
        same, pos, neg = sameside(d, claim_positive=True)
        nz = [d[f"A_n{j}"]["mean"] for j in range(R_NOISE)]
        if fn != PRIMARY and same >= 2:
            drop.append(nm)
        print(f"  {nm:16s} 零点中心 {st.fmean(nz):+.5f} | **同侧(正向) {pos}/{R_NOISE}**,负向 {neg}/{R_NOISE}"
              f" | A_skelC7 {d['A_skelC7']['mean']:+.4f} {'排除0' if d['A_skelC7']['excl'] else '含0 '}"
              f" | 主判据 {d['skelC7_minus_edpad']['mean']:+.4f}"
              f" {'排除0' if d['skelC7_minus_edpad']['excl'] else '含0'}")

    print("\n=== 判定(冻结的打死条件)===")
    ex = lambda d, k: d[k]["excl"]
    K0 = ex(P, "A_G")
    nz = [P[f"A_n{j}"]["mean"] for j in range(R_NOISE)]
    K1 = not ex(P, "A_skelC7")
    K2 = not ex(P, "skelC7_minus_edpad")
    K3 = P["A_skelC7"]["mean"] <= st.fmean(nz) + sorted(nz)[int(.95 * R_NOISE)]
    K4 = ex(P, "A_crossC7") and abs(P["A_crossC7"]["mean"]) >= 0.5 * abs(P["A_skelC7"]["mean"])
    reps = 0; tot_r = 0
    for nm, fn in POPS:
        if fn == PRIMARY or R[fn].get("no_data") or nm in drop:
            continue
        tot_r += 1
        reps += 1 if ex(R[fn], "A_skelC7") else 0
    K5 = (not K1) and reps < 3
    print(f"  K0 GC 相消失效:{'FIRES -> 作废' if K0 else '不触发 ✓'}")
    print(f"  K1 骨架分类无增量({P['A_skelC7']['mean']:+.4f}):{'FIRES -> unresolved' if K1 else '不触发 ⇒ 排除 0'}")
    print(f"  K2 **分类打不过 ensemble_defect**({P['skelC7_minus_edpad']['mean']:+.4f}):"
          f"{'FIRES' if K2 else '**不触发 ⇒ 排除 0,分类确实更好**'}")
    print(f"  K3 宽度混杂:{'FIRES' if K3 else '不触发'}")
    print(f"  K4 骨架不特异(交叉臂 {P['A_crossC7']['mean']:+.4f}):{'FIRES' if K4 else '不触发'}")
    print(f"  K5 复现不足(复现 {reps}/{tot_r}"
          f"{';按第 53 条作废:' + ','.join(drop) if drop else ''}):{'FIRES' if K5 else '不触发'}")
    if K0:
        v = "void_cancellation_failed"
    elif K1:
        v = "unresolved_skeleton_bookkeeping"
    elif K2:
        v = "classification_does_not_beat_ensemble_defect"
        if K5: v += "__underreplicated"
    elif K4:
        v = "index_set_not_specific"
    elif K3:
        v = "width_confounded"
    else:
        v = "classification_beats_ensemble_defect" + ("__underreplicated" if K5 else "")
    print(f"  裁定:{v}")
    print("  **口径:区间含 0 一律写 `unresolved` 或强制标签,绝不写 `refuted`。**")
    print("  边界:**没有盲人群**;第五批/第六批是【本条线未读】,H2 读过它们。只用骨架可形成子集;"
          "仍在 Turner 族内;只量信息量,不判上线。")
    R["verdict"] = v
    json.dump(R, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
