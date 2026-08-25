#!/usr/bin/env python3
"""M2: Deng move two -- split the index set by an INVARIANT into an exactly-computable main part and a
remainder that only needs a bound.

Pre-registered in m2_invariant_split_declaration.json.

THE MISTAKE THIS EXISTS TO REPAIR. I averaged g_t over every target, got +0.006, compared it to a pooled MDE
of +0.0096, and wrote down "57 targets lock out everything below +0.02 in this library". That sentence is
true of the AVERAGE and I promoted it to a theorem about the library. The per-target spread is 0.037-0.071 --
two to six times the effect -- so the averaging is throwing away more than it is measuring. This run asks
whether an invariant computable from the TARGET ALONE separates the targets that carry the effect from the
ones that do not.

WHAT IT MAY NOT DO. It does not re-open H2's frozen verdict (`does_not_generalise` stands at its own pooling).
It only asks whether a main part exists.

RULE 51 (new, from m3_boot_stability): every gate that fires on "does the interval contain 0" reports its
flip rate across bootstrap seeds. A gate with a non-zero flip rate is an instrument reading, not a conclusion.

COST: $0, offline.
"""
from __future__ import annotations

import argparse, collections, json, math, os, random, statistics as st, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from e1_ensemble_marginals import base_block, load_pop                            # noqa: E402
from e3_scrambled_reference import work                                           # noqa: E402
from h1_heldout_ensemble_defect import fit_predict                                # noqa: E402
import h2_heldout_cluster_ci as H                                                 # noqa: E402

N_CAND = 10                 # frozen candidate count -> Bonferroni 0.05/10 = 0.005
ALPHA_B = 0.05 / N_CAND
NPERM = 5000
BOOT_RHO = 2000
R_SEED = 50                 # rule 51: seed-flip rate on every deciding interval
MIN_HALF = 15               # void gate 3
RAND_SPLITS = 200


# ----------------------------------------------------------------------------- rank statistics
def _ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = _ranks(x), _ranks(y)
    n = len(x)
    mx, my = st.fmean(rx), st.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx)); dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx > 0 and dy > 0 else 0.0


def rho_ci(x, y, seed, level):
    """Cluster bootstrap over TARGETS (the elements of x/y are per-target)."""
    rng = random.Random(seed)
    n = len(x)
    vs = []
    for _ in range(BOOT_RHO):
        idx = [rng.randrange(n) for _ in range(n)]
        vs.append(spearman([x[i] for i in idx], [y[i] for i in idx]))
    vs.sort()
    lo = (1.0 - level) / 2.0
    return [round(vs[int(lo * BOOT_RHO)], 4), round(vs[min(BOOT_RHO - 1, int((1 - lo) * BOOT_RHO))], 4)]


def rho_perm_p(x, y, seed):
    obs = abs(spearman(x, y))
    rng = random.Random(seed + 991)
    yy = list(y)
    hits = 0
    for _ in range(NPERM):
        rng.shuffle(yy)
        if abs(spearman(x, yy)) >= obs:
            hits += 1
    return (hits + 1) / (NPERM + 1)


def flip_rate_rho(x, y, level, seeds=R_SEED):
    """Rule 51: how often does the excludes-zero verdict flip when only the bootstrap seed changes?"""
    ex = 0
    for s in range(seeds):
        c = rho_ci(x, y, 7000 + 13 * s, level)
        ex += 1 if (c[0] > 0 or c[1] < 0) else 0
    return ex, seeds


def flip_rate_mean(vec, seeds=R_SEED):
    ex = 0
    for s in range(seeds):
        c, _n = H.boot_ci(vec, 8000 + 13 * s)
        ex += 1 if (c[0] > 0 or c[1] < 0) else 0
    return ex, seeds


# ----------------------------------------------------------------------------- the estimand: g_t
def per_target_g(nm, fn, splits, seed, workers):
    """H2's construction, but returning the per-target vector instead of only its mean."""
    from leaderboard_design import targets
    T = targets()
    print(f"\n=== {nm} ===", flush=True)
    rec, _s, _m = load_pop(fn)
    keep = [r for r in rec if base_block(r) is not None]
    jobs = [(r["puzzle"], r["seq"], T[r["puzzle"]]["target"]) for r in keep]
    from concurrent.futures import ProcessPoolExecutor
    M = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for m in ex.map(work, jobs, chunksize=8):
            M.append(m)
    # formability bookkeeping over ALL designs (an invariant input; no wet, no oracle)
    tot = collections.Counter(); form = collections.Counter(); gc_cross_rate = collections.defaultdict(list)
    for r, m in zip(keep, M):
        tot[r["puzzle"]] += 1
        if "error" not in m:
            form[r["puzzle"]] += 1
            if m.get("n_skel_pairs"):
                pass
    cells, cols = collections.defaultdict(list), {}
    for r, m in zip(keep, M):
        if "error" in m:
            continue
        rr = dict(r)
        cols[id(rr)] = base_block(r) + m["C"] + [m["ed_skel"]] + [m["ed_mfe"]]
        cells[(r["puzzle"], r["method"])].append(rr)
        gc_cross_rate[r["puzzle"]].append(m["C"][0])
    rws = [(c, g) for c, g in cells.items() if len(g) >= 2]
    tg = sorted({c[0] for c, _g in rws})
    nB = len(base_block(keep[0]))
    iBC = list(range(nB + 2)); iED = iBC + [nB + 2]
    by_t = collections.defaultdict(list)
    for c, gp in rws:
        by_t[c[0]].append((c, gp))
    per_t = {t: [] for t in tg}
    rng = random.Random(seed)
    order = list(tg)
    leak = 0
    for _s in range(splits):
        rng.shuffle(order)
        cut = max(1, int(0.7 * len(order)))
        tr, te = set(order[:cut]), set(order[cut:])
        TR = [x for t in tr for x in by_t[t]]; TE = [x for t in te for x in by_t[t]]
        if not TR or not TE:
            continue
        if {id(r) for _c, g in TR for r in g} & {id(r) for _c, g in TE for r in g}:
            leak += 1
        P = {k: fit_predict(TR, TE, cols, ix) for k, ix in (("BC", iBC), ("BC_ed", iED))}
        for t in te:
            a = {}
            nt = None
            for k in P:
                v, n, _ti = H.pair_accuracy_rows(by_t[t], P[k])
                a[k] = v; nt = n
            if nt is None or nt < H.MIN_PAIRS_PER_TARGET or any(v is None for v in a.values()):
                continue
            per_t[t].append(a["BC_ed"] - a["BC"])
    g = {t: st.fmean(per_t[t]) for t in tg if len(per_t[t]) >= H.MIN_SPLITS_PER_TARGET}
    print(f"  分子 {sum(len(x) for _c, x in rws)} 格 {len(rws)} 靶标 {len(tg)};产生 g_t 的靶标 {len(g)}"
          f";分子级泄漏 {leak}/{splits} {'✓' if not leak else '✗ 作废'}")
    if g:
        vv = list(g.values())
        print(f"  g_t 均值 {st.fmean(vv):+.4f}  sd {st.stdev(vv):.4f}  正号 {sum(1 for x in vv if x>0)}/{len(vv)}"
              f"  跨度 {min(vv):+.4f} ~ {max(vv):+.4f}")
    aux = {"formable_rate": {t: form[t] / tot[t] for t in tot if tot[t]},
           "gc_cross_mean": {t: st.fmean(v) for t, v in gc_cross_rate.items() if v}}
    return g, aux, leak


# ----------------------------------------------------------------------------- the ten invariants
TOPO_ORD = {"H-type": 0, "kissing / chain": 1, "higher-order": 2}


def invariants(tg, aux):
    """Every column is a function of (target structure, the population's design sequences). No wet, no oracle."""
    from leaderboard_design import targets
    from rna_harness.benchmarks.rnainvbench import _round_only, pk_pairs
    from rna_harness.structure_utils import classify_topology
    from gamma_star import crossing_set
    T = targets()
    out = collections.defaultdict(dict)
    for t in tg:
        tgt = T[t]["target"]
        S = set(crossing_set(tgt)); K = set(pk_pairs(_round_only(tgt)))
        heads = [(i, j) for (i, j) in S if (i - 1, j + 1) not in S]
        lens = []
        for (i, j) in heads:
            n = 1
            while (i + n, j - n) in S:
                n += 1
            lens.append(n)
        out["L"][t] = float(len(tgt))
        out["n_cross"][t] = float(len(S))
        out["n_skel"][t] = float(len(K))
        out["cross_frac"][t] = len(S) / max(1, len(S) + len(K))
        out["n_stems"][t] = float(len(heads))
        out["max_stem_len"][t] = float(max(lens) if lens else 0)
        out["span"][t] = float(st.median([j - i for (i, j) in S])) if S else 0.0
        out["topo_ord"][t] = float(TOPO_ORD[classify_topology(tgt)["topology"]])
        out["formable_rate"][t] = aux["formable_rate"].get(t, float("nan"))
        # the frozen name `gc_headroom` is DEGENERATE: crossing and skeleton positions are disjoint, so the
        # upper bound on G-C at crossing positions is 1.0 for every target. Reported as such and replaced by
        # the realised rate, which is still target+sequence only. The substitution is counted in the ten.
        out["gc_headroom_LITERAL_constant"][t] = 1.0
        out["gc_cross_realised"][t] = aux["gc_cross_mean"].get(t, float("nan"))
    return out


CANDIDATES = ["L", "n_cross", "n_skel", "cross_frac", "n_stems", "max_stem_len", "span",
              "topo_ord", "formable_rate", "gc_cross_realised"]


# ----------------------------------------------------------------------------- split arithmetic
def half_stats(vec):
    if len(vec) < 2:
        return None
    m = st.fmean(vec); sd = st.stdev(vec)
    mde = 1.96 * sd / math.sqrt(len(vec))
    return {"n": len(vec), "mean": round(m, 4), "sd": round(sd, 4), "MDE": round(mde, 4),
            "eff_over_mde": round(abs(m) / mde, 2) if mde > 0 else 0.0}


def split_by(g, col, thr):
    hi = [g[t] for t in g if t in col and not math.isnan(col[t]) and col[t] > thr]
    lo = [g[t] for t in g if t in col and not math.isnan(col[t]) and col[t] <= thr]
    return lo, hi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "m2_invariant_split.json"))
    a = ap.parse_args()
    PRIMARY = "panel_f1_fresh.json"
    POPS = (("样本2(主)", PRIMARY), ("基准4560", "panel_f1_designs.json"),
            ("样本3", "panel_f1_third.json"), ("第四批", "panel_f1_fourth.json"))
    G, AUX, res = {}, {}, {}
    for nm, fn in POPS:
        G[fn], AUX[fn], leak = per_target_g(nm, fn, a.splits, a.seed, a.workers)
        res[fn] = {"n_targets_g": len(G[fn]), "leak": leak,
                   "g_mean": round(st.fmean(list(G[fn].values())), 4) if G[fn] else None}
        json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)

    # merged object (declaration: average g_t across populations, cluster cap 57)
    allt = sorted(set().union(*[set(G[fn]) for _n, fn in POPS]))
    merged = {t: st.fmean([G[fn][t] for _n, fn in POPS if t in G[fn]]) for t in allt}
    print(f"\n=== 合并对象:{len(merged)} 个靶标(上限 57);均值 {st.fmean(list(merged.values())):+.4f}"
          f"  正号 {sum(1 for v in merged.values() if v>0)}/{len(merged)} ===")
    res["merged_n"] = len(merged)

    gp = G[PRIMARY]
    tgP = sorted(gp)
    INV = invariants(sorted(set(allt) | set(tgP)), AUX[PRIMARY])
    # merge formability/gc from every population for targets the primary lacks (merged view only)
    AUXALL = {"formable_rate": {}, "gc_cross_mean": {}}
    for k in AUXALL:
        for _n, fn in POPS:
            for t, v in AUX[fn][k].items():
                AUXALL[k].setdefault(t, []).append(v)
        AUXALL[k] = {t: st.fmean(v) for t, v in AUXALL[k].items()}
    INV_M = invariants(allt, AUXALL)

    print(f"\n=== step1 筛选(仅主人群,{len(tgP)} 个靶标);候选 {N_CAND} 个,"
          f"Bonferroni 阈值 {ALPHA_B:.4f},区间取 {100*(1-ALPHA_B):.1f}% ===")
    print(f"  退化列检查:gc_headroom 字面定义 = 交叉位与骨架位不相交 ⇒ 上界恒为 1.0,方差 0 ⇒ "
          f"作废并由 gc_cross_realised 替代(计入十个之中)")
    yP = [gp[t] for t in tgP]
    scr = {}
    for c in CANDIDATES:
        col = INV[c]
        ts = [t for t in tgP if t in col and not math.isnan(col[t])]
        if len(ts) < MIN_HALF * 2:
            scr[c] = {"skipped": f"可用靶标 {len(ts)} 太少"}; continue
        x = [col[t] for t in ts]; y = [gp[t] for t in ts]
        if st.pstdev(x) == 0:
            scr[c] = {"skipped": "方差为 0"}; continue
        r = spearman(x, y)
        ci95 = rho_ci(x, y, a.seed + 31, 0.95)
        cib = rho_ci(x, y, a.seed + 31, 1 - ALPHA_B)
        p = rho_perm_p(x, y, a.seed)
        ex_b = cib[0] > 0 or cib[1] < 0
        fr, fs = flip_rate_rho(x, y, 1 - ALPHA_B)
        scr[c] = {"n": len(ts), "rho": round(r, 4), "ci95": ci95, "ci_bonf": cib,
                  "perm_p": round(p, 5), "passes_bonf": bool(ex_b and p < ALPHA_B),
                  "seed_flip": f"{fr}/{fs}"}
        print(f"  {c:22s} rho {r:+.4f}  95% {ci95}  {100*(1-ALPHA_B):.1f}% {cib}"
              f"  置换 p {p:.4f}  {'**通过 Bonferroni**' if scr[c]['passes_bonf'] else '不通过'}"
              f"  [种子翻转 {fr}/{fs}]")
    res["step1_primary"] = scr

    # --- void gate 1: the screen must have detection power (rule 47 dual obligation) --------------------
    print("\n  [闸门1 / 第47条] 检出力探针 —— 必须响的输入")
    probes = {}
    # (a) declared probe: mean tube score per target. NOT guaranteed correlated -> reported, not relied on.
    from leaderboard_design import targets as _tg
    wet_mean = {}
    for _n, fn in POPS:
        pass
    rec, _s, _m = load_pop(PRIMARY)
    wm = collections.defaultdict(list)
    for r in rec:
        wm[r["puzzle"]].append(r["wet"])
    wet_mean = {t: st.fmean(v) for t, v in wm.items()}
    tsw = [t for t in tgP if t in wet_mean]
    xw = [wet_mean[t] for t in tsw]; yw = [gp[t] for t in tsw]
    pw = rho_perm_p(xw, yw, a.seed)
    probes["declared_mean_wet"] = {"rho": round(spearman(xw, yw), 4), "perm_p": round(pw, 5),
                                   "fires": bool(pw < ALPHA_B)}
    print(f"    (a) 声明里的探针 mean(wet):rho {probes['declared_mean_wet']['rho']:+.4f}"
          f" p {pw:.4f} {'响' if probes['declared_mean_wet']['fires'] else '不响'}"
          f"  —— **它本来就不保证相关,不能当检出力证据**")
    # (b) added probe that MUST fire by construction: g_t itself plus noise at rho ~ 0.5
    rngp = random.Random(4242)
    sdg = st.stdev(yP)
    xs = [gp[t] + rngp.gauss(0, 1.6 * sdg) for t in tgP]
    ps = rho_perm_p(xs, yP, a.seed)
    probes["synthetic_rho_half"] = {"rho": round(spearman(xs, yP), 4), "perm_p": round(ps, 5),
                                    "fires": bool(ps < ALPHA_B)}
    print(f"    (b) 补加的构造性探针(g_t + 噪声):rho {probes['synthetic_rho_half']['rho']:+.4f}"
          f" p {ps:.5f} {'**响 ✓ 筛选流程有检出力**' if probes['synthetic_rho_half']['fires'] else '✗ 不响 -> 筛选本身无检出力,本次作废'}")
    res["gate1_probes"] = probes

    passing = [c for c in CANDIDATES if scr.get(c, {}).get("passes_bonf")]
    K1 = not passing
    res["passing"] = passing
    verdict_parts = []
    if K1:
        print(f"\n  K1 触发:十个候选里没有一个在 Bonferroni 阈值下排除 0")
        print(f"  **强制标签:`no_invariant_found_among_these_ten` —— 十个候选不是全空间**")
        res["verdict"] = "no_invariant_found_among_these_ten"
        # still report the best three, as leads for the next declaration (not as findings)
        rank = sorted([c for c in CANDIDATES if "rho" in scr.get(c, {})],
                      key=lambda c: -abs(scr[c]["rho"]))[:3]
        print("  下一份声明的线索(不是发现):" + ", ".join(f"{c} rho {scr[c]['rho']:+.3f} p {scr[c]['perm_p']:.3f}"
                                                          for c in rank))
        res["leads"] = rank
        json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
        print(f"  -> {a.out}")
        return 0

    best = max(passing, key=lambda c: abs(scr[c]["rho"]))
    col = INV[best]
    ts = [t for t in tgP if t in col and not math.isnan(col[t])]
    thr = st.median([col[t] for t in ts])
    print(f"\n=== step2 切分:选中 `{best}`(rho {scr[best]['rho']:+.4f}),中位阈值 {thr:.4f} ===")
    lo, hi = split_by(gp, col, thr)
    sl, sh = half_stats(lo), half_stats(hi)
    for nmh, s in (("<=thr", sl), (">thr", sh)):
        print(f"  {nmh:6s} n {s['n']:3d}  均值 {s['mean']:+.4f}  sd {s['sd']:.4f}"
              f"  MDE +{s['MDE']:.4f}  效应/MDE {s['eff_over_mde']:.2f}")
    main_vec, main_name = (lo, "<=thr") if abs(sl["mean"]) >= abs(sh["mean"]) else (hi, ">thr")
    ms = half_stats(main_vec)
    g3 = min(sl["n"], sh["n"]) >= MIN_HALF
    print(f"  闸门3 每半 >= {MIN_HALF} 个靶标:{'OK' if g3 else '✗ 切分作废'}")
    fr, fs = flip_rate_mean(main_vec)
    print(f"  [第51条] 主项均值区间的种子翻转:排除 0 的种子 {fr}/{fs}"
          f"  {'裁定稳定' if fr in (0, fs) else '**裁定由 RNG 决定**'}")
    K2 = ms["eff_over_mde"] < 2.0
    res["step2"] = {"invariant": best, "threshold": thr, "low": sl, "high": sh,
                    "main": main_name, "main_stats": ms, "gate3": g3, "seed_flip": f"{fr}/{fs}"}

    # --- step4: random-split null ----------------------------------------------------------------------
    rngr = random.Random(a.seed + 77)
    vals = [gp[t] for t in ts]
    nlo = sl["n"]
    nulls = []
    for _ in range(RAND_SPLITS):
        v = list(vals); rngr.shuffle(v)
        A, B = v[:nlo], v[nlo:]
        sa, sb = half_stats(A), half_stats(B)
        nulls.append(max(sa["eff_over_mde"], sb["eff_over_mde"]))
    nulls.sort()
    q95 = nulls[int(0.95 * len(nulls))]
    K3 = ms["eff_over_mde"] <= q95
    print(f"\n=== step4 随机切分零点({RAND_SPLITS} 次):效应/MDE 的 95 分位 {q95:.2f};"
          f"真实切分 {ms['eff_over_mde']:.2f} ⇒ {'落在零点内' if K3 else '**超出零点**'} ===")
    res["step4_null"] = {"q95": round(q95, 3), "real": ms["eff_over_mde"], "median_null": round(nulls[len(nulls)//2], 3)}

    # --- K4: replication, EXECUTE the frozen split only ------------------------------------------------
    print(f"\n=== K4 复现(只执行 `{best}` @ {thr:.4f},不重选)===")
    reps = 0; repd = {}
    for nm, fn in POPS:
        if fn == PRIMARY:
            continue
        gr = G[fn]
        colr = invariants(sorted(gr), AUX[fn])[best]
        l2, h2 = split_by(gr, colr, thr)
        s2 = half_stats(l2 if main_name == "<=thr" else h2)
        if s2 is None:
            repd[fn] = {"no_data": True}; print(f"  {nm}: 主项为空"); continue
        okn = s2["n"] >= MIN_HALF
        good = okn and s2["eff_over_mde"] >= 2.0
        reps += 1 if good else 0
        repd[fn] = {**s2, "counts": bool(good), "half_floor_ok": okn}
        print(f"  {nm:10s} 主项 n {s2['n']:3d}  均值 {s2['mean']:+.4f}  效应/MDE {s2['eff_over_mde']:.2f}"
              f"  {'✓ 计入' if good else ('半边不足 15 -> 不计' if not okn else '不计入')}")
    K4 = reps < 2
    res["k4"] = {"reps": reps, "detail": repd}

    print("\n=== 判定(冻结的打死条件)===")
    print(f"  K1 十个候选里没有一个排除 0:不触发(通过 {len(passing)} 个:{', '.join(passing)})")
    print(f"  K2 主项效应/MDE < 2({ms['eff_over_mde']:.2f}):{'FIRES' if K2 else '不触发'}")
    print(f"  K3 真实切分落在随机切分 95% 以内:{'FIRES ⇒ 这个不变量什么也没切开' if K3 else '不触发'}")
    print(f"  K4 复现 {reps}/3 < 2:{'FIRES' if K4 else '不触发'}")
    v = ("invariant_splits_nothing" if K3 else
         "main_part_below_resolution" if K2 else
         "main_part_found" + ("__underreplicated" if K4 else ""))
    print(f"  裁定:{v}")
    print("  **口径:区间含 0 一律写 `unresolved`,绝不写 `refuted`;`underreplicated` = 主人群成立、需要复现。**")
    print("  边界:没有盲人群;不改 H2 在其自身汇总口径下的 `does_not_generalise`;只判有没有主项,不判上线。")
    res["verdict"] = v
    res["merged_g"] = {t: round(v2, 5) for t, v2 in merged.items()}
    res["primary_g"] = {t: round(v2, 5) for t, v2 in gp.items()}
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
