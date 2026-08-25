#!/usr/bin/env python3
"""M1: Deng move one -- keep bookkeeping over a classifiable discrete index set instead of collapsing the
distribution into a scalar.

Pre-registered in m1_indexset_bookkeeping_declaration.json.

THE GAP THIS FILLS. Of the four moves, this is the one I have never used. I took the L x L pair-probability
matrix and collapsed it into seven scalars; `ensemble_defect` is itself a SUM, and a sum is a collapse. Two
independent results now point at the representation: the ceiling review (79.9% of within-cell explainable
variance untouched) and M2 (ten structural invariants all unresolved, and that run is void by its own gate).

THE CONSTRUCTION. The index set is the target's crossing pairs. Per design we keep p_ij for each of them and
form eight columns, each a mean over a CLASSIFIABLE subclass (terminal vs interior, short vs long span,
short vs long stem, the weakest pair, the weakest stem). The thing they are compared against is p_all, the
mean over the same index set -- the collapse itself, padded to the same width with noise so the comparison
is not a width comparison (C1's T0pad fix).

Everything is scored ONLY on pairs with identical gc_cross AND gc_frac, so the two GC columns contribute
exactly zero to any prediction difference. Move one composes with move three; the built-in A_G control and
its must-fire probe are inherited from M3.

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

CLASSES = ["p_min", "p_term", "p_int", "p_span_lo", "p_span_hi", "p_stem_sh", "p_stem_lo", "p_worststem"]
NC = len(CLASSES)
R_NOISE = 12
MIN_PAIRS = 30
CANON = {frozenset({"A", "U"}), frozenset({"G", "C"}), frozenset({"G", "U"})}


def _stems(S):
    """Maximal chains under (i,j) -> (i+1, j-1). Returns a list of lists of pairs."""
    heads = [(i, j) for (i, j) in S if (i - 1, j + 1) not in S]
    out = []
    for (i, j) in sorted(heads):
        ch, n = [], 0
        while (i + n, j - n) in S:
            ch.append((i + n, j - n)); n += 1
        out.append(ch)
    return out


def classify(pairs, pmap):
    """The eight class columns plus the collapse p_all, over one index set. Returns (cols, p_all, imputed)."""
    P = [pmap[q] for q in pairs]
    if not P:
        return None
    p_all = st.fmean(P)
    stems = _stems(set(pairs))
    term, inter = [], []
    for ch in stems:
        for k, q in enumerate(ch):
            (term if (k == 0 or k == len(ch) - 1) else inter).append(pmap[q])
    spans = sorted(j - i for (i, j) in pairs)
    med = spans[len(spans) // 2]
    lo = [pmap[(i, j)] for (i, j) in pairs if (j - i) < med]
    hi = [pmap[(i, j)] for (i, j) in pairs if (j - i) >= med]
    sh = [pmap[q] for ch in stems if len(ch) <= 2 for q in ch]
    lg = [pmap[q] for ch in stems if len(ch) >= 3 for q in ch]
    worst = min((st.fmean([pmap[q] for q in ch]) for ch in stems), default=p_all)
    imp = set()
    def m(v, fallback, name):
        if not v:
            imp.add(name); return fallback
        return st.fmean(v)
    cols = [min(P),
            m(term, p_all, "p_term"),
            m(inter, st.fmean(term) if term else p_all, "p_int"),
            m(lo, p_all, "p_span_lo"),
            m(hi, p_all, "p_span_hi"),
            m(sh, p_all, "p_stem_sh"),
            m(lg, p_all, "p_stem_lo"),
            worst]
    return cols, p_all, imp


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
        c1 = classify(cross, pm)
        c2 = classify(sp, pm)
        if c1 is None or c2 is None:
            return {"puzzle": puz, "error": "empty index set"}
        gc = 0
        cs = set(cross)
        for (i, j) in sorted(cs):
            if (i + 1, j - 1) in cs:
                for (x, y) in ((i, j), (i + 1, j - 1)):
                    if {seq[x], seq[y]} == {"G", "C"}:
                        gc += 1
        return {"puzzle": puz, "C8": c1[0], "p_all": c1[1], "imp": sorted(c1[2]),
                "S8": c2[0], "sp_all": c2[1],
                "gc": [float(gc), (seq.count("G") + seq.count("C")) / n],
                "ed_skel": fc.ensemble_defect(skel)}
    except Exception as exc:                                                      # noqa: BLE001
        return {"puzzle": puz, "error": repr(exc)[:140]}


def acc_multi(rows, preds, keys, key, mode="exact"):
    ok = {k: 0.0 for k in keys}
    n = tt = skipped = 0
    for _c, g in rows:
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                if mode != "none" and key[id(g[a])] != key[id(g[b])]:
                    skipped += 1; continue
                dw = g[a]["wet"] - g[b]["wet"]
                if dw == 0:
                    tt += 1; continue
                n += 1
                for k in keys:
                    d = preds[k][id(g[a])] - preds[k][id(g[b])]
                    ok[k] += 0.5 if d == 0 else (1.0 if ((dw > 0) == (d > 0)) else 0.0)
    return ({k: (ok[k] / n if n else None) for k in keys}, n, tt, skipped)


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


def stats_of(g, seed, tag):
    ci, nb = H.boot_ci(g, seed + 17)
    mde = 1.96 * st.stdev(g) / math.sqrt(len(g))
    ex = 0
    for s in range(50):
        c, _n = H.boot_ci(g, 9500 + 13 * s)
        ex += 1 if (c[0] > 0 or c[1] < 0) else 0
    return {"tag": tag, "mean": round(st.fmean(g), 4), "ci": ci, "n": nb, "MDE": round(mde, 4),
            "excl": bool(ci[0] > 0 or ci[1] < 0), "pos": sum(1 for x in g if x > 0),
            "eff_over_mde": round(abs(st.fmean(g)) / mde, 2) if mde > 0 else 0.0,
            "seed_flip": f"{ex}/50"}


def run_pop(nm, keep, M, splits, seed, noise_seed, keepc, variant="real", mode="exact"):
    """variant: 'real' | 'noiseC8' (K1 probe) | 'degenerate' (K2 probe).

    `keepc` is the list of class indices that survived void gate V7 (imputation rate <= 30%). The collapse
    arm is padded to len(keepc) and every noise arm is len(keepc) wide, so K2 is never a width comparison.
    """
    NCe = len(keepc)
    cells, cols, key = collections.defaultdict(list), {}, {}
    rngn = random.Random(noise_seed)
    impc = collections.Counter(); ntot = 0
    for r, m in zip(keep, M):
        if "error" in m:
            continue
        ntot += 1
        for x in m["imp"]:
            impc[x] += 1
        rr = dict(r)
        C8 = [m["C8"][i] for i in keepc]
        if variant == "noiseC8":
            C8 = [rngn.gauss(0, 1.0) for _ in range(NCe)]
        elif variant == "degenerate":
            C8 = [m["p_all"]] * NCe
        pad = [rngn.gauss(0, 1.0) for _ in range(NCe - 1)]
        noise = [rngn.gauss(0, 1.0) for _ in range(NCe * R_NOISE)]
        S8 = [m["S8"][i] for i in keepc]
        cols[id(rr)] = (base_block(r) + m["gc"] + C8 + [m["p_all"]] + pad
                        + S8 + [m["ed_skel"]] + noise)
        key[id(rr)] = (m["gc"][0], round(m["gc"][1], 3))
        cells[(r["puzzle"], r["method"])].append(rr)
    rws = [(c, g) for c, g in cells.items() if len(g) >= 2]
    nB = len(base_block(keep[0]))
    o_gc, o_c8, o_s1 = nB, nB + 2, nB + 2 + NCe
    o_sk = o_s1 + NCe; o_ed = o_sk + NCe; o_nz = o_ed + 1
    iA = list(range(nB))
    idx = {"A": iA, "A_G": iA + [o_gc, o_gc + 1],
           "A_C8": iA + list(range(o_c8, o_c8 + NCe)),
           "A_s1pad": iA + list(range(o_s1, o_s1 + NCe)),
           "A_skelC8": iA + list(range(o_sk, o_sk + NCe)),
           "A_ed": iA + [o_ed]}
    for j in range(R_NOISE):
        idx[f"A_n{j}"] = iA + list(range(o_nz + j * NCe, o_nz + (j + 1) * NCe))
    keys = list(idx)
    by_t = collections.defaultdict(list)
    for c, gp in rws:
        by_t[c[0]].append((c, gp))
    tg = sorted(by_t)
    per = {k: {t: [] for t in tg} for k in keys if k != "A"}
    per["C8_minus_s1pad"] = {t: [] for t in tg}
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
            for k in per:
                if k == "C8_minus_s1pad":
                    per[k][t].append(acc["A_C8"] - acc["A_s1pad"])
                else:
                    per[k][t].append(acc[k] - acc["A"])
    used = [t for t in tg if len(per["A_C8"][t]) >= H.MIN_SPLITS_PER_TARGET]
    tot = S["pairs"] + S["tie"] + S["skip"]
    print(f"  {nm:26s} 格 {len(rws)} 靶标 {len(used)} | 配平可判对 {S['pairs']}"
          f"(配平剔除 {S['skip']/max(1,tot)*100:.1f}%) | 泄漏 {S['leak']}/{splits}"
          f" {'✓' if not S['leak'] else '✗ 作废'}", flush=True)
    if variant == "real" and mode == "exact" and ntot:
        print("    闸门 V7 该人群填补比例:"
              + ", ".join(f"{c} {impc[c]/ntot*100:.1f}%" for c in CLASSES if impc[c]) or "全部 0%")
        vals = {CLASSES[c]: [cols[id(r)][o_c8 + i] for _c2, g in rws for r in g] for i, c in enumerate(keepc)}
        pa = [cols[id(r)][o_s1] for _c2, g in rws for r in g]
        print("    闸门 V2 每根分类列与塌缩 p_all 的相关:"
              + ", ".join(f"{c} {np.corrcoef(vals[c], pa)[0,1]:+.3f}" for c in vals))
        gcv = [cols[id(r)][o_gc] for _c2, g in rws for r in g]
        print("    闸门 V3 每根分类列与 gc_cross 的相关:"
              + ", ".join(f"{c} {np.corrcoef(vals[c], gcv)[0,1]:+.3f}" for c in vals))
    if len(used) < H.MIN_TARGETS:
        return {"no_data": True, "n_used": len(used)}
    out = {"n_used": len(used)}
    for k in per:
        out[k] = stats_of([st.fmean(per[k][t]) for t in used], seed, k)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "m1_indexset_bookkeeping.json"))
    a = ap.parse_args()
    PRIMARY = "panel_f1_fresh.json"
    POPS = (("样本2(主)", PRIMARY), ("基准4560", "panel_f1_designs.json"),
            ("样本3", "panel_f1_third.json"), ("第四批", "panel_f1_fourth.json"))
    R = {}
    KEEP = {}
    print("=== 移动一:在交叉配对指标集上按类记账,全部在 GC 精确配平的对上评估 ===", flush=True)
    print("=== 先折叠四个人群,汇总填补比例,执行闸门 V7(>30% 的类作废),再跑 ===", flush=True)
    for nm, fn in POPS:
        KEEP[fn] = build(fn, a.workers)
        print(f"  折叠完 {nm}", flush=True)
    impc = collections.Counter(); ntot = 0
    for _n, fn in POPS:
        for m in KEEP[fn][1]:
            if "error" in m:
                continue
            ntot += 1
            for x in m["imp"]:
                impc[x] += 1
    rates = {c: impc[c] / ntot for c in CLASSES}
    keepc = [i for i, c in enumerate(CLASSES) if rates[c] <= 0.30]
    voided = [c for c in CLASSES if rates[c] > 0.30]
    print("  闸门 V7 汇总填补比例(四人群 %d 条):" % ntot
          + ", ".join(f"{c} {rates[c]*100:.1f}%" for c in CLASSES))
    print(f"  **V7 作废 {len(voided)} 根:{voided or '无'} ⇒ 类数 {len(CLASSES)} -> {len(keepc)};"
          f"塌缩臂配平到 {len(keepc)} 列,每根噪声臂 {len(keepc)} 列**")
    R["V7"] = {"rates": {c: round(rates[c], 4) for c in CLASSES}, "voided": voided,
               "n_classes_used": len(keepc), "classes_used": [CLASSES[i] for i in keepc]}
    if len(keepc) < 3:
        R["verdict"] = "void_too_few_classes_survive_V7"
        print("  **存活的类少于 3 根 ⇒ 本次作废**")
        json.dump(R, open(a.out, "w"), indent=1, ensure_ascii=False); return 0
    for j, (nm, fn) in enumerate(POPS):
        keep, M = KEEP[fn]
        R[fn] = run_pop(nm, keep, M, a.splits, a.seed, 7000 + 149 * j, keepc)
        json.dump(R, open(a.out, "w"), indent=1, ensure_ascii=False)

    print("\n=== 三个必响探针(第 47 条对偶义务;任一不响则本次作废,第 54 条)===")
    keep, M = KEEP[PRIMARY]
    pr = {}
    pr["K0"] = run_pop("探针 完全不配平(K0 必响)", keep, M, a.splits, a.seed, 7777, keepc, "real", "none")
    pr["K1"] = run_pop("探针 分类列换噪声(K1 必响)", keep, M, a.splits, a.seed, 7778, keepc, "noiseC8")
    pr["K2"] = run_pop("探针 分类退化为 p_all(K2 必响)", keep, M, a.splits, a.seed, 7779, keepc, "degenerate")
    p0 = pr["K0"]["A_G"]["excl"]
    p1 = not pr["K1"]["A_C8"]["excl"]
    p2 = not pr["K2"]["C8_minus_s1pad"]["excl"]
    print(f"  K0 探针 A_G−A {pr['K0']['A_G']['mean']:+.4f} {pr['K0']['A_G']['ci']}"
          f"  {'**响 ✓**' if p0 else '✗ 没响'}")
    print(f"  K1 探针 A_C8−A {pr['K1']['A_C8']['mean']:+.4f} {pr['K1']['A_C8']['ci']}"
          f"  {'**响 ✓**' if p1 else '✗ 没响'}")
    print(f"  K2 探针 C8−s1pad {pr['K2']['C8_minus_s1pad']['mean']:+.4f} {pr['K2']['C8_minus_s1pad']['ci']}"
          f"  {'**响 ✓**' if p2 else '✗ 没响'}")
    R["probes"] = {"K0": p0, "K1": p1, "K2": p2}
    if not (p0 and p1 and p2):
        R["verdict"] = "void_probe_did_not_fire"
        print("\n  **第 54 条:必响探针没响 ⇒ 本次运行作废。不许事后换探针给这一跑背书。**")
        json.dump(R, open(a.out, "w"), indent=1, ensure_ascii=False)
        return 0

    print("\n=== 结果(主人群)===")
    P = R[PRIMARY]
    for k in ("A_G", "A_C8", "A_s1pad", "C8_minus_s1pad", "A_skelC8", "A_ed"):
        o = P[k]
        lbl = {"A_G": "内建必须为零", "A_C8": "**分类记账(主臂)**", "A_s1pad": "塌缩 p_all(宽度配平)",
               "C8_minus_s1pad": "**分类 − 塌缩(移动一的主判据)**", "A_skelC8": "指标集安慰剂(骨架)",
               "A_ed": "M3 的真列(接续用)"}[k]
        print(f"  {k:16s} {o['mean']:+.4f}  95% {o['ci']}  MDE +{o['MDE']:.4f}  效应/MDE {o['eff_over_mde']:.2f}"
              f"  正号 {o['pos']}/{o['n']}  {'排除 0' if o['excl'] else '含 0'}  [种子翻转 {o['seed_flip']}] {lbl}")

    print(f"\n=== 零点与逐人群假阳率(第 45/46/53 条,{R_NOISE} 根 × {NC} 列)===")
    fp = {}
    for nm, fn in POPS:
        d = R[fn]
        if d.get("no_data"):
            print(f"  {nm:10s} 无数据"); continue
        nz = [d[f"A_n{j}"]["mean"] for j in range(R_NOISE)]
        nex = sum(1 for j in range(R_NOISE) if d[f"A_n{j}"]["excl"])
        fp[fn] = nex
        print(f"  {nm:10s} 零点中心 {st.fmean(nz):+.5f}  95分位 {sorted(nz)[int(.95*R_NOISE)]:+.5f}"
              f"  排除 0 的根数 **{nex}/{R_NOISE}**  | A_C8−A {d['A_C8']['mean']:+.4f}"
              f" {'排除0' if d['A_C8']['excl'] else '含0 '}  分类−塌缩 {d['C8_minus_s1pad']['mean']:+.4f}"
              f" {'排除0' if d['C8_minus_s1pad']['excl'] else '含0'}")

    print("\n=== 判定(冻结的打死条件)===")
    ex = lambda d, k: d[k]["excl"]
    K0 = ex(P, "A_G")
    nz = [P[f"A_n{j}"]["mean"] for j in range(R_NOISE)]
    K1 = not ex(P, "A_C8")
    K2 = not ex(P, "C8_minus_s1pad")
    K3 = P["A_C8"]["mean"] <= st.fmean(nz) + sorted(nz)[int(.95 * R_NOISE)]
    K4 = ex(P, "A_skelC8") and abs(P["A_skelC8"]["mean"]) >= 0.5 * abs(P["A_C8"]["mean"])
    reps = 0; dropped = []
    for nm, fn in POPS:
        if fn == PRIMARY or R[fn].get("no_data"):
            continue
        if fp.get(fn, 0) >= 2:
            dropped.append(nm); continue
        reps += 1 if ex(R[fn], "A_C8") else 0
    K5 = (not K1) and reps < 2
    print(f"  K0 GC 相消失效:{'FIRES -> 主人群作废' if K0 else '不触发 ✓ 相消生效'}")
    print(f"  K1 分类记账无增量({P['A_C8']['mean']:+.4f}):{'FIRES -> unresolved' if K1 else '不触发 ⇒ 排除 0'}")
    print(f"  K2 **分类比塌缩没多买到东西**({P['C8_minus_s1pad']['mean']:+.4f}):"
          f"{'FIRES -> unresolved,不是 refuted' if K2 else '**不触发 ⇒ 排除 0,分类确实多买到了**'}")
    print(f"  K3 宽度混杂:{'FIRES' if K3 else '不触发'}")
    print(f"  K4 指标集不特异(骨架安慰剂 {P['A_skelC8']['mean']:+.4f}):{'FIRES' if K4 else '不触发'}")
    print(f"  K5 复现不足(复现 {reps}/{3-len(dropped)}"
          f"{';按第 53 条作废的人群:' + ','.join(dropped) if dropped else ''}):{'FIRES' if K5 else '不触发'}")
    if K0:
        v = "void_cancellation_failed"
    elif K1:
        v = "unresolved_classified_bookkeeping"
    elif K4:
        v = "index_set_not_specific"
    elif K3:
        v = "width_confounded"
    else:
        v = "classification_beats_collapse" if not K2 else "gain_but_collapse_matches_it"
        if K5:
            v += "__underreplicated"
    print(f"  裁定:{v}")
    print("  **口径:区间含 0 一律写 `unresolved`,绝不写 `refuted`;`underreplicated` = 主人群成立、需要复现。**")
    print("  边界:没有盲人群;只用骨架可形成子集;仍在 Turner 族内;只量信息量,不判上线。")
    R["verdict"] = v
    json.dump(R, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
