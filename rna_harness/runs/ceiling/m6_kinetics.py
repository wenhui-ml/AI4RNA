#!/usr/bin/env python3
"""M6: the fifth family -- kinetics. Pre-registered in m6_kinetics_declaration.json.

Every column this project has ever used comes from the equilibrium partition function (bpp,
ensemble_defect, pr_structure) or from the target's static structure. Two sequences can share an ensemble
and fold along completely different paths, so kinetics is the one degree of freedom never touched. Four
families are now measured empty while the invariant demonstrably exists (the empirical form splits all four
populations at permutation p from 0.00005 to 0.00085).

Two arms off one folding pass: the kinetic block as a NEW COLUMN on exactly-GC-matched pairs (M3's
construction, with the pre-registered ed_mfe placebo and twelve width-matched noise arms), and its
target-level aggregates as an INVARIANT for g_t (M5's construction, leave-one-population-out).

Rule 55 is honoured here: both the interval and the permutation null preserve the dependence between
rotations -- resample TARGETS and recompute all four, and permute the target->g map GLOBALLY. M5 shuffled
inside each rotation independently and that made its null about twice too narrow.

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
from m2_invariant_split import spearman, half_stats                                # noqa: E402
from m2b_review_audit import acc_pair_of_blocks, perm_diff_p                       # noqa: E402

KIN = ["barrier", "dG_skel_mfe", "trap_dG", "cotr_delay", "cotr_conflict"]
NK = len(KIN)
R_NOISE = 12
MIN_PAIRS = 30
ALPHA_B = 0.05 / NK
BENCH = "__BENCHMARK_gt_other_pops"
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
        mfe_s, mfe_e = fc.mfe()
        fc.pf()
        e_skel = fc.eval_structure(skel)
        # --- kinetics -------------------------------------------------------------------------------
        sad = fc.path_findpath_saddle(mfe_s, skel) / 100.0        # dekacal -> kcal
        barrier = sad - max(mfe_e, e_skel)
        fc2 = RNA.fold_compound(seq)
        for i in sorted({x for p in sp for x in p}):
            fc2.hc_add_up(i + 1)
        _s3, e3 = fc2.mfe()
        trap = e3 - mfe_e
        S = set(sp)
        part = {x: y for (x, y) in sp}
        part.update({y: x for (x, y) in sp})
        need, delay, confl = max(1, len(sp) // 2), 1.0, []
        for f in range(2, 11):
            p = seq[:max(4, int(n * f / 10))]
            ps, _pe = RNA.fold(p)
            pp = pk_pairs(ps)
            got = len(pp & S)
            bad = sum(1 for (i, j) in pp if (i in part and part[i] != j) or (j in part and part[j] != i))
            confl.append(bad / max(1, len(pp)))
            if delay == 1.0 and got >= need:
                delay = f / 10.0
        # --- GC and the two ensemble columns (placebo + the real one) ---------------------------------
        cs = set(cross); gc = 0
        for (i, j) in sorted(cs):
            if (i + 1, j - 1) in cs:
                for (x, y) in ((i, j), (i + 1, j - 1)):
                    if {seq[x], seq[y]} == {"G", "C"}:
                        gc += 1
        return {"puzzle": puz,
                "K": [barrier, e_skel - mfe_e, trap, delay, st.fmean(confl)],
                "gc": [float(gc), (seq.count("G") + seq.count("C")) / n],
                "ed_skel": fc.ensemble_defect(skel), "ed_mfe": fc.ensemble_defect(mfe_s)}
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


def arm1(nm, keep, M, splits, seed, noise_seed, variant="real", mode="exact", verbose=False):
    cells, cols, key = collections.defaultdict(list), {}, {}
    rngn = random.Random(noise_seed)
    for r, m in zip(keep, M):
        if "error" in m:
            continue
        rr = dict(r)
        K = list(m["K"])
        if variant == "noiseK":
            K = [rngn.gauss(0, 1.0) for _ in range(NK)]
        ed_pad = [m["ed_skel"]] + [rngn.gauss(0, 1.0) for _ in range(NK - 1)]
        mf_pad = [m["ed_mfe"]] + [rngn.gauss(0, 1.0) for _ in range(NK - 1)]
        noise = [rngn.gauss(0, 1.0) for _ in range(NK * R_NOISE)]
        cols[id(rr)] = base_block(r) + m["gc"] + K + ed_pad + mf_pad + noise
        key[id(rr)] = (m["gc"][0], round(m["gc"][1], 3))
        cells[(r["puzzle"], r["method"])].append(rr)
    rws = [(c, g) for c, g in cells.items() if len(g) >= 2]
    if not rws:
        return {"no_data": True}
    nB = len(base_block(keep[0]))
    o_gc, o_k = nB, nB + 2
    o_ed, o_mf, o_nz = o_k + NK, o_k + 2 * NK, o_k + 3 * NK
    iA = list(range(nB))
    idx = {"A": iA, "A_G": iA + [o_gc, o_gc + 1],
           "A_kin": iA + list(range(o_k, o_k + NK)),
           "A_ed_pad": iA + list(range(o_ed, o_ed + NK)),
           "A_mfe_pad": iA + list(range(o_mf, o_mf + NK))}
    for j in range(R_NOISE):
        idx[f"A_n{j}"] = iA + list(range(o_nz + j * NK, o_nz + (j + 1) * NK))
    keys = list(idx)
    by_t = collections.defaultdict(list)
    for c, gp in rws:
        by_t[c[0]].append((c, gp))
    tg = sorted(by_t)
    per = {k: {t: [] for t in tg} for k in keys if k != "A"}
    per["kin_minus_edpad"] = {t: [] for t in tg}
    rng = random.Random(seed); order = list(tg); leak = 0; npairs = 0
    for _s in range(splits):
        rng.shuffle(order)
        cut = max(1, int(0.7 * len(order)))
        tr, te = set(order[:cut]), set(order[cut:])
        TR = [x for t in tr for x in by_t[t]]; TE = [x for t in te for x in by_t[t]]
        if not TR or not TE:
            continue
        if {id(r) for _c, g in TR for r in g} & {id(r) for _c, g in TE for r in g}:
            leak += 1
        P = {k: fit_predict(TR, TE, cols, ix) for k, ix in idx.items()}
        for t in te:
            acc, n, _tt, _sk = acc_multi(by_t[t], P, keys, key, mode)
            npairs += n
            if n < MIN_PAIRS or any(v is None for v in acc.values()):
                continue
            for k in keys:
                if k != "A":
                    per[k][t].append(acc[k] - acc["A"])
            per["kin_minus_edpad"][t].append(acc["A_kin"] - acc["A_ed_pad"])
    used = [t for t in tg if len(per["A_kin"][t]) >= H.MIN_SPLITS_PER_TARGET]
    print(f"  {nm:26s} 格 {len(rws)} 靶标 {len(used)} 配平可判对 {npairs}"
          f" | 泄漏 {leak}/{splits} {'✓' if not leak else '✗'}", flush=True)
    if verbose and rws:
        V = {KIN[i]: [cols[id(r)][o_k + i] for _c, g in rws for r in g] for i in range(NK)}
        gv = [cols[id(r)][o_gc] for _c, g in rws for r in g]
        print("    V2 每根与 gc_cross 的相关:"
              + ", ".join(f"{k} {np.corrcoef(v, gv)[0,1]:+.3f}" for k, v in V.items()))
        print("    V2 每根的方差:" + ", ".join(f"{k} {np.var(v):.4g}" for k, v in V.items()))
    if len(used) < H.MIN_TARGETS:
        return {"no_data": True, "n_used": len(used)}
    return {"n_used": len(used), **{k: stats_of([st.fmean(per[k][t]) for t in used], seed, k) for k in per}}


def g_only(keep, M, splits, seed):
    cells, cols = collections.defaultdict(list), {}
    for r, m in zip(keep, M):
        if "error" in m:
            continue
        rr = dict(r)
        cols[id(rr)] = base_block(r) + m["gc"] + [m["ed_skel"], m["ed_mfe"]]
        cells[(r["puzzle"], r["method"])].append(rr)
    rws = [(c, g) for c, g in cells.items() if len(g) >= 2]
    nB = len(base_block(keep[0]))
    iBC = list(range(nB + 2)); iED = iBC + [nB + 2]
    by_t = collections.defaultdict(list)
    for c, gp in rws:
        by_t[c[0]].append((c, gp))
    tg = sorted(by_t)
    per = {t: [] for t in tg}
    rng = random.Random(seed); order = list(tg)
    for _s in range(splits):
        rng.shuffle(order)
        cut = max(1, int(0.7 * len(order)))
        tr, te = set(order[:cut]), set(order[cut:])
        TR = [x for t in tr for x in by_t[t]]; TE = [x for t in te for x in by_t[t]]
        if not TR or not TE:
            continue
        PA = fit_predict(TR, TE, cols, iBC); PB = fit_predict(TR, TE, cols, iED)
        for t in te:
            aA, aB, nA, _n, _t2, _a, _b = acc_pair_of_blocks(by_t[t], PA, PB, "common05")
            if aA is None or aB is None or nA < H.MIN_PAIRS_PER_TARGET:
                continue
            per[t].append(aB - aA)
    g = {t: st.fmean(v) for t, v in per.items() if len(v) >= H.MIN_SPLITS_PER_TARGET}
    kin = {}
    for t in g:
        rows = [m for r, m in zip(keep, M) if "error" not in m and r["puzzle"] == t]
        if rows:
            kin[t] = {KIN[i]: st.fmean([m["K"][i] for m in rows]) for i in range(NK)}
    return g, {t: kin[t] for t in g if t in kin}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "m6_kinetics.json"))
    a = ap.parse_args()
    R = {}; KEEP = {}; G = {}; KINT = {}
    print("=== M6 第五族:动力学。臂1 = 新列(GC 精确配平);臂2 = 解释 g_t 的不变量 ===", flush=True)
    for nm, fn in POPS:
        keep, M = build(fn, a.workers)
        bad = sum(1 for m in M if "error" in m and "not formable" not in m.get("error", ""))
        tot = len(M)
        print(f"  折叠完 {nm}  非骨架原因的失败 {bad}/{tot} = {bad/max(1,tot)*100:.2f}%"
              f"  {'✓' if bad / max(1, tot) <= 0.10 else '✗ 闸门 V7 -> 本人群作废'}", flush=True)
        KEEP[fn] = (keep, M)
        G[fn], KINT[fn] = g_only(keep, M, a.splits, a.seed)

    print("\n=== 臂1 必响探针 ===", flush=True)
    keep, M = KEEP[PRIMARY]
    p0 = arm1("探针 完全不配平(K0 必响)", keep, M, a.splits, a.seed, 9001, "real", "none")
    p1 = arm1("探针 动力学列换噪声(K1 必响)", keep, M, a.splits, a.seed, 9002, "noiseK")
    ok0 = p0["A_G"]["excl"]; ok1 = not p1["A_kin"]["excl"]
    print(f"  K0 探针 A_G−A {p0['A_G']['mean']:+.4f} {p0['A_G']['ci']}  {'**响 ✓**' if ok0 else '✗'}")
    print(f"  K1 探针 A_kin−A {p1['A_kin']['mean']:+.4f} {p1['A_kin']['ci']}  {'**响 ✓**' if ok1 else '✗'}")
    R["probes_arm1"] = {"K0": bool(ok0), "K1": bool(ok1)}
    if not (ok0 and ok1):
        R["verdict_arm1"] = "void_probe_did_not_fire"
        print("  **第 54 条:臂1 作废,不许事后换探针。**")
    else:
        print("\n=== 臂1 主运行 ===", flush=True)
        for j, (nm, fn) in enumerate(POPS):
            k2, m2 = KEEP[fn]
            R[fn] = arm1(nm, k2, m2, a.splits, a.seed, 9100 + 97 * j, verbose=(fn == PRIMARY))
            json.dump(R, open(a.out, "w"), indent=1, ensure_ascii=False)
        P = R[PRIMARY]
        print("\n  主人群:")
        LB = {"A_G": "内建必须为零", "A_kin": "**动力学块(主臂)**", "A_ed_pad": "ensemble_defect(宽度配平)",
              "A_mfe_pad": "**安慰剂 ed_mfe(宽度配平)**", "kin_minus_edpad": "动力学 − ensemble_defect"}
        for k in ("A_G", "A_kin", "A_ed_pad", "A_mfe_pad", "kin_minus_edpad"):
            o = P[k]
            print(f"    {k:16s} {o['mean']:+.4f}  95% {o['ci']}  效应/MDE {o['eff_over_mde']:.2f}"
                  f"  {'排除 0' if o['excl'] else '含 0'}  [种子 {o['seed_flip']}]  {LB[k]}")
        print(f"\n  逐人群同侧假阳(第 53 条改写版,{R_NOISE} 根 × {NK} 列):")
        drop = []
        for nm, fn in POPS:
            d = R[fn]
            if d.get("no_data"):
                continue
            pos = sum(1 for j in range(R_NOISE) if d[f"A_n{j}"]["excl"] and d[f"A_n{j}"]["ci"][0] > 0)
            neg = sum(1 for j in range(R_NOISE) if d[f"A_n{j}"]["excl"] and d[f"A_n{j}"]["ci"][1] < 0)
            nz = [d[f"A_n{j}"]["mean"] for j in range(R_NOISE)]
            if fn != PRIMARY and pos >= 2:
                drop.append(nm)
            print(f"    {nm:10s} 零点 {st.fmean(nz):+.5f} | **同侧(正) {pos}/{R_NOISE}**,负 {neg}/{R_NOISE}"
                  f" | A_kin−A {d['A_kin']['mean']:+.4f} {'排除0' if d['A_kin']['excl'] else '含0'}")
        nz = [P[f"A_n{j}"]["mean"] for j in range(R_NOISE)]
        K0 = P["A_G"]["excl"]; K1 = not P["A_kin"]["excl"]
        K2 = P["A_mfe_pad"]["excl"] and abs(P["A_mfe_pad"]["mean"]) >= 0.5 * abs(P["A_kin"]["mean"])
        K3 = P["A_kin"]["mean"] <= st.fmean(nz) + sorted(nz)[int(.95 * R_NOISE)]
        reps = 0; tot_r = 0
        for nm, fn in POPS:
            if fn == PRIMARY or R[fn].get("no_data") or nm in drop:
                continue
            tot_r += 1; reps += 1 if R[fn]["A_kin"]["excl"] else 0
        K5 = (not K1) and reps < 2
        print(f"\n  K0 相消失效:{'FIRES -> 作废' if K0 else '不触发 ✓'}")
        print(f"  K1 动力学块无增量({P['A_kin']['mean']:+.4f}):{'FIRES -> unresolved' if K1 else '不触发 ⇒ 排除 0'}")
        print(f"  K2 安慰剂也解释得了({P['A_mfe_pad']['mean']:+.4f}):{'FIRES' if K2 else '不触发'}")
        print(f"  K3 宽度混杂:{'FIRES' if K3 else '不触发'}")
        print(f"  K5 复现不足({reps}/{tot_r}{';作废:' + ','.join(drop) if drop else ''}):{'FIRES' if K5 else '不触发'}")
        v1 = ("void_cancellation_failed" if K0 else "unresolved_kinetics_as_column" if K1 else
              "explained_by_generic_concentration" if K2 else "width_confounded" if K3 else
              "kinetics_adds" + ("__underreplicated" if K5 else ""))
        print(f"  臂1 裁定:{v1}")
        R["verdict_arm1"] = v1

    # ---------------- arm 2 -----------------------------------------------------------------------
    print(f"\n=== 臂2:动力学的靶标级汇总能不能解释 g_t(留一交叉拟合,Bonferroni {ALPHA_B:.3f})===")
    rot = {}
    for c in KIN + [BENCH]:
        rot[c] = []
        for nm, fn in POPS:
            others = [f for _n, f in POPS if f != fn]
            col = {}
            for t in G[fn]:
                v = ([G[o][t] for o in others if t in G[o]] if c == BENCH
                     else [KINT[o][t][c] for o in others if t in KINT[o]])
                if len(v) >= 2:
                    col[t] = st.fmean(v)
            ts = sorted(col)
            if len(ts) >= 30:
                rot[c].append((ts, col, G[fn]))
    allt = sorted(set().union(*[set(ts) for ts, _c, _g in rot[BENCH]]))
    scr = {}
    for c in KIN + [BENCH]:
        if not rot[c]:
            scr[c] = {"skipped": True}; continue
        obs = st.fmean([spearman([col[t] for t in ts], [gg[t] for t in ts]) for ts, col, gg in rot[c]])
        # rule 55: bootstrap resamples TARGETS and recomputes ALL rotations from the same draw
        rng = random.Random(a.seed + 909); bs = []
        for _ in range(3000):
            cnt = collections.Counter(allt[rng.randrange(len(allt))] for _ in range(len(allt)))
            rs = []
            for ts, col, gg in rot[c]:
                xs, ys = [], []
                for t in ts:
                    k = cnt[t]
                    if k:
                        xs.extend([col[t]] * k); ys.extend([gg[t]] * k)
                if len(xs) >= 10 and len(set(xs)) > 1:
                    rs.append(spearman(xs, ys))
            if rs:
                bs.append(st.fmean(rs))
        bs.sort()
        lb = bs[int(ALPHA_B / 2 * len(bs))]; hb = bs[int((1 - ALPHA_B / 2) * len(bs))]
        # rule 55: permutation uses ONE GLOBAL target relabelling applied to every rotation
        rngp = random.Random(a.seed + 55); NP = 3000; hits = 0
        for _ in range(NP):
            perm = list(allt); rngp.shuffle(perm)
            mp = dict(zip(allt, perm))
            rs = []
            for ts, col, gg in rot[c]:
                xs = [col[t] for t in ts if mp[t] in gg]
                ys = [gg[mp[t]] for t in ts if mp[t] in gg]
                if len(xs) >= 10:
                    rs.append(spearman(xs, ys))
            if rs and abs(st.fmean(rs)) >= abs(obs):
                hits += 1
        pv = (hits + 1) / (NP + 1)
        ok = bool((lb > 0 or hb < 0) and pv < ALPHA_B)
        scr[c] = {"rho_mean": round(obs, 4), "ci_bonf": [round(lb, 4), round(hb, 4)],
                  "perm_p": round(pv, 5), "passes": ok}
        print(f"  {c:24s} rho均 {obs:+.4f}  {100*(1-ALPHA_B):.0f}% [{lb:+.4f},{hb:+.4f}]"
              f"  置换 p {pv:.5f}  {'**通过**' if ok else '不通过'}"
              f"  {'**基准(必响)**' if c == BENCH else ''}")
    R["arm2"] = scr
    if not scr[BENCH].get("passes"):
        R["verdict_arm2"] = "void_benchmark_did_not_fire"
        print("  **基准列没响 ⇒ 臂2 作废(第 54 条)。**")
    else:
        hw = (scr[BENCH]["ci_bonf"][1] - scr[BENCH]["ci_bonf"][0]) / 2
        print(f"  分辨力:基准 rho {scr[BENCH]['rho_mean']},区间半宽 {hw:.3f} ⇒ 分辨得了 |rho| ≳ {hw:.2f}")
        passing = [c for c in KIN if scr.get(c, {}).get("passes")]
        K4 = not passing
        print(f"  K4 五个候选无一通过:{'FIRES' if K4 else f'不触发(通过 {passing})'}")
        if K4:
            print("  **强制标签:`no_invariant_found_in_kinetics` —— 五个不是全空间。**")
        R["verdict_arm2"] = "no_invariant_found_in_kinetics" if K4 else "kinetic_invariant_found"
    print("\n  **口径:区间含 0 一律写 `unresolved`,绝不写 `refuted`。**")
    print("  边界:**没有盲人群**;只用骨架可形成子集;**动力学量与平衡量共享 Turner 参数** ——"
          "『新自由度』指泛函形式新,不是能量模型新;不判上线。")
    json.dump(R, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
