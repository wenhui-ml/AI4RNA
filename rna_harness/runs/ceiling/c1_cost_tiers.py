#!/usr/bin/env python3
"""C1: split the shipped criterion bar by COMPUTE COST and ask, in both directions, what the judge panel buys.

Pre-registered in c1_cost_tiers_declaration.json (rule 38/39/40/41/42/43/44/45/46/47/48 spec).

WHY THIS AND NOT ANOTHER NEW COLUMN. All six populations draw designs from the SAME 57 targets (union 57,
intersection 40), and the cluster is the target, so cluster count is capped at 57 no matter how many designs
are sampled. With sd(g_t)=0.0368 the library's held-out MDE floor is +0.0096. ensemble_defect averages +0.0137
= 1.43x that floor, with effect/MDE ratios of 0.54/1.12/1.21/1.56/1.80/1.81 across populations -- the exact
shape of an effect sitting on the detection floor. More designs cannot fix it because designs are not clusters.

TIERS. T0 is O(L) sequence and complementarity only -- no folding, no oracle, no partition function. T1 adds
partition-function scalars that do not reference the target (this is where mean_bp_dist finally gets a test; it
fell through the gap between E2, which killed the positional story, and E3, which narrowed to the two
skeleton-referencing columns). T2 is the shipped panel, which costs three oracle runs per design.

UNITS. 7.3's observations (0.1001 vs 0.1651, +0.049, 0.0837) are IN-SAMPLE NET R^2. The +0.0096 floor is in
HELD-OUT PAIRWISE ACCURACY. Those are different scales and must not be compared; the "5.1x/8.8x the floor"
claim is withdrawn. Whether these effects clear the floor in the metric that HAS a floor is unknown in
advance, which is why K4 makes "was this measurement capable of answering" a gate rather than an afterthought.

COST: $0, offline.
"""
from __future__ import annotations

import argparse, collections, json, math, os, random, re, statistics as st, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from e1_ensemble_marginals import base_block, load_pop                           # noqa: E402
from h1_heldout_ensemble_defect import fit_predict                               # noqa: E402
import h2_heldout_cluster_ci as H                                                # noqa: E402

CANON = {frozenset("GC"), frozenset("AU"), frozenset("GU")}
R_NOISE = 24
SPLITS = 200
T0_NAMES = ("gc_cross", "gc_frac", "formable_skel", "formable_cross",
            "n_noncanon_skel", "n_noncanon_cross")
T1_EXTRA = ("mean_bp_dist", "entropy_mean")


FORBIDDEN_IN_T0 = ("fold_compound", "pf(", "per_oracle", "ensemble_defect",
                    "positional_entropy", "mfe(", "bpp(")


def assert_tier_purity():
    """Void gate 7: T0's builder must not fold, must not call an oracle, must not build a partition function.

    Scans the FUNCTION BODY with the docstring stripped -- the first version scanned the whole source and
    flagged `per_oracle` out of its own docstring, which is a gate reporting a violation that does not exist.
    Rule 47's dual obligation is exercised below: a spec that really does read an oracle column must trip it.
    """
    src = open(os.path.abspath(__file__)).read()
    body = src[src.index("def t0_columns"):src.index("def work")]
    body = re.sub(r'""".*?"""', "", body, flags=re.S)          # drop the docstring, keep the code
    hit = [p for p in FORBIDDEN_IN_T0 if p in body]
    if hit:
        raise AssertionError(f"闸门7:T0 构造器里出现禁用调用 {hit}")
    probe = "def fake_t0(r):\n    return r['per_oracle']['pkiss']\n"
    if not [p for p in FORBIDDEN_IN_T0 if p in probe]:
        raise AssertionError("闸门7 空转:连一个真的读判官列的构造器都测不出来")
    return True


def t0_columns(seq, target):
    """Tier 0: O(L) base counting and complementarity ONLY.

    Void gate 7 (tier purity): this function must not fold, must not call an oracle, must not build a
    partition function. It imports nothing from RNA and reads no `per_oracle` field; the audit asserts that
    by source inspection.
    """
    from rna_harness.benchmarks.rnainvbench import _round_only, pk_pairs
    skel = pk_pairs(_round_only(target))
    cross = pk_pairs(target) - skel
    def bad(ps):
        return sum(1 for (i, j) in ps if frozenset((seq[i], seq[j])) not in CANON)
    bs, bc = bad(skel), bad(cross)
    gc_cross = sum(1 for (i, j) in cross if frozenset((seq[i], seq[j])) == frozenset("GC"))
    return {"gc_cross": float(gc_cross),
            "gc_frac": (seq.count("G") + seq.count("C")) / len(seq),
            "formable_skel": 0.0 if bs else 1.0,
            "formable_cross": 0.0 if bc else 1.0,
            "n_noncanon_skel": float(bs),
            "n_noncanon_cross": float(bc)}


def work(item):
    """T0 (free) + the two target-free partition-function scalars of T1."""
    puz, seq, target = item
    try:
        out = t0_columns(seq, target)
        import RNA
        fc = RNA.fold_compound(seq)
        fc.pf()
        ent = fc.positional_entropy()
        out["mean_bp_dist"] = fc.mean_bp_distance()
        out["entropy_mean"] = st.fmean(ent[1:len(seq) + 1])
        return {"puzzle": puz, **out}
    except Exception as exc:                                                     # noqa: BLE001
        return {"puzzle": puz, "error": repr(exc)[:140]}


def pair_acc_common(rows, pred):
    """Within-cell pairwise ordering accuracy on the COMMON pair set.

    Amendment (see the declaration): a PREDICTION tie scores 0.5 -- chance -- instead of being dropped. Only
    a TUBE tie is dropped, because there the ground truth is undecidable. Dropping prediction ties let a
    coarse block be scored only on the pairs it could separate, and T0's tie rate was 16.67% against T2's
    1.45%, which is what manufactured "the cheap tier beats the panel" in the smoke. Every block now shares
    an identical denominator.
    """
    ok = 0.0
    n = ties_tube = ties_pred = 0
    for _c, g in rows:
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                dw = g[a]["wet"] - g[b]["wet"]
                if dw == 0:
                    ties_tube += 1
                    continue
                n += 1
                dp = pred[id(g[a])] - pred[id(g[b])]
                if dp == 0:
                    ties_pred += 1
                    ok += 0.5
                else:
                    ok += 1.0 if ((dw > 0) == (dp > 0)) else 0.0
    return (ok / n if n else None), n, ties_tube, ties_pred


def mde(gvec):
    """Rule 48: the smallest effect this population could have resolved."""
    return 1.96 * st.stdev(gvec) / math.sqrt(len(gvec)) if len(gvec) > 2 else float("nan")


def heldout(rws, cols, idx_of, splits, seed, ref=None):
    """Per-target pairwise ranking accuracy for every block in `idx_of`, plus the noise-column calibration."""
    by_t = collections.defaultdict(list)
    for c, gp in rws:
        by_t[c[0]].append((c, gp))
    tg = sorted(by_t); order = list(tg)
    per = {k: {t: [] for t in tg} for k in idx_of}
    REF = ref if ref is not None else next(iter(idx_of))   # the block whose pair count sets the floor
    rng = random.Random(seed)
    ties_tot = pairs_tot = skipped = 0
    leak_mol = leak_tg = 0
    predtie = {}
    for _s in range(splits):
        rng.shuffle(order)
        cut = max(1, int(0.7 * len(order)))
        tr, te = set(order[:cut]), set(order[cut:])
        TR = [x for t in tr for x in by_t[t]]
        TE = [x for t in te for x in by_t[t]]
        if not TR or not TE:
            continue
        # void gate 1, the NON-VACUOUS version: molecule level AND target level
        if {id(r) for _c, g in TR for r in g} & {id(r) for _c, g in TE for r in g}:
            leak_mol += 1
        if {c[0] for c, _g in TR} & {c[0] for c, _g in TE}:
            leak_tg += 1
        preds = {k: fit_predict(TR, TE, cols, ix) for k, ix in idx_of.items()}
        for t in te:
            acc = {}
            nt = None
            for k in idx_of:
                v, n, ti_tube, ti_pred = pair_acc_common(by_t[t], preds[k])
                acc[k] = v
                predtie[k] = predtie.get(k, 0) + ti_pred
                if k == REF:
                    nt = n; pairs_tot += n; ties_tot += ti_tube
            if nt is None or nt < H.MIN_PAIRS_PER_TARGET or any(v is None for v in acc.values()):
                skipped += 1
                continue
            for k in idx_of:
                per[k][t].append(acc[k])
    used = [t for t in tg if len(per[REF][t]) >= H.MIN_SPLITS_PER_TARGET]
    return per, used, tg, dict(ties_tube=ties_tot, pairs=pairs_tot, skipped=skipped,
                               leak_mol=leak_mol, leak_tg=leak_tg,
                               pred_tie_rate={k: round(v / max(1, pairs_tot), 4)
                                              for k, v in predtie.items()})


def run_pop(nm, fn, splits, seed, workers):
    from leaderboard_design import targets
    T = targets()
    print(f"\n=== {nm} ===", flush=True)
    rec, _s, _m = load_pop(fn)
    keep = [r for r in rec if base_block(r) is not None]
    jobs = [(r["puzzle"], r["seq"], T[r["puzzle"]]["target"]) for r in keep]
    from concurrent.futures import ProcessPoolExecutor
    M = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, m in enumerate(ex.map(work, jobs, chunksize=8), 1):
            M.append(m)
            if i % 3000 == 0:
                print(f"    {i}/{len(jobs)}", flush=True)
    cells, cols = collections.defaultdict(list), {}
    err = 0
    for r, m in zip(keep, M):
        if "error" in m:
            err += 1
            continue
        rr = dict(r)
        cols[id(rr)] = ([m[k] for k in T0_NAMES] + [m[k] for k in T1_EXTRA] + base_block(r))
        cells[(r["puzzle"], r["method"])].append(rr)
    rws = [(c, g) for c, g in cells.items() if len(g) >= 2]
    dropped_cells = sum(1 for c, g in cells.items() if len(g) < 2)
    n0, n1e = len(T0_NAMES), len(T1_EXTRA)
    nB = len(base_block(keep[0]))
    iT0 = list(range(n0))
    iT1 = list(range(n0 + n1e))
    iT2 = list(range(n0 + n1e, n0 + n1e + nB))
    idx_of = {"T0": iT0, "T1": iT1, "T2": iT2,
              "T0uT2": iT0 + iT2, "T1uT2": iT1 + iT2}
    # void gate 5: the union index sets must literally be the unions
    assert set(idx_of["T0uT2"]) == set(iT0) | set(iT2), "闸门5:T0∪T2 的列索引不是并集"
    assert set(idx_of["T1uT2"]) == set(iT1) | set(iT2), "闸门5:T1∪T2 的列索引不是并集"
    mols = [r for _c, g in rws for r in g]
    print(f"  失败 {err};可用分子 {len(mols)},格 {len(rws)}(剔除 {dropped_cells}),"
          f"靶标 {len({c[0] for c, _g in rws})};列 T0={n0} T1={n0+n1e} T2={nB}")

    # ---- noise calibration, rule 45/46: R independent no-information columns ------------------------
    rngn = random.Random(999)
    noise = {k: [rngn.gauss(0, 1.0) for _ in range(R_NOISE)] for k in cols}
    per, used, tg, stats = heldout(rws, cols, idx_of, splits, seed)
    if len(used) < H.MIN_TARGETS:
        print(f"  产生 g_t 的靶标 {len(used)} < {H.MIN_TARGETS} -> 无数据(cluster 地板)")
        return {"no_data": True, "n_used": len(used)}
    print(f"  泄漏(分子级/靶标级)自查:{stats['leak_mol']}/{splits} 与 {stats['leak_tg']}/{splits}"
          f"  {'✓' if stats['leak_mol'] == 0 and stats['leak_tg'] == 0 else '✗ 作废'}")
    print(f"  地板剔除:靶标 {len(tg)-len(used)},(靶标,划分) 跳过 {stats['skipped']};"
          f"试管并列剔除 {stats['ties_tube']}/{stats['ties_tube']+stats['pairs']}"
          f" = {stats['ties_tube']/max(1,stats['ties_tube']+stats['pairs'])*100:.2f}%")
    print(f"  各块【预测并列】率(记 0.5,不剔除,分母相同):"
          + "  ".join(f"{k} {v*100:.2f}%" for k, v in stats["pred_tie_rate"].items()))

    acc = {k: st.fmean([st.fmean(per[k][t]) for t in used]) for k in idx_of}
    gT0 = [st.fmean(per["T0"][t]) for t in used]
    MDE = mde(gT0)
    print(f"  留出格内成对排序准确率(靶标均值,cluster={len(used)}),本人群 MDE = +{MDE:.4f}")
    for k in ("T0", "T1", "T2", "T0uT2", "T1uT2"):
        print(f"    {k:7s} {acc[k]:.4f}   超过掷硬币 {acc[k]-0.5:+.4f}")

    def contrast(a, b):
        """Per-target paired difference acc(a) - acc(b), with the target cluster bootstrap."""
        g = [st.fmean(per[a][t]) - st.fmean(per[b][t]) for t in used]
        ci, n = H.boot_ci(g, seed + 17)
        assert n == len(used) and n != len(rws) and n != len(mols), "闸门6:重抽单位不是靶标"
        return {"mean": round(st.fmean(g), 4), "ci": ci, "n_clusters": n,
                "pos_targets": sum(1 for x in g if x > 0),
                "excl": bool(ci[0] > 0 or ci[1] < 0)}

    C = {"T2_minus_T0": contrast("T2", "T0"), "T2_minus_T1": contrast("T2", "T1"),
         "T0uT2_minus_T2": contrast("T0uT2", "T2"), "T0uT2_minus_T0": contrast("T0uT2", "T0"),
         "T1uT2_minus_T2": contrast("T1uT2", "T2"), "T1_minus_T0": contrast("T1", "T0")}
    print("  对比(靶标 cluster bootstrap):")
    for k, v in C.items():
        print(f"    {k:18s} {v['mean']:+.4f}  95% {v['ci']}"
              f"  {'排除 0' if v['excl'] else '含 0'}  正号靶标 {v['pos_targets']}/{v['n_clusters']}"
              f"  {'(< MDE,测不出)' if abs(v['mean']) < MDE else ''}")

    # ---- noise columns: the null for "one more column" (rule 45/46) --------------------------------
    # One pass over the splits computing ALL R noise variants, instead of R passes. The baseline predictions
    # do not depend on the noise column, so BC is fitted once per split and reused by every variant.
    ncol = len(cols[id(mols[0])])
    augs = [{k: v + [noise[k][j]] for k, v in cols.items()} for j in range(R_NOISE)]
    by_t = collections.defaultdict(list)
    for c, gp in rws:
        by_t[c[0]].append((c, gp))
    order = list(sorted(by_t))
    pern = [{t: [] for t in order} for _ in range(R_NOISE)]
    rngs = random.Random(seed)
    for _s in range(splits):
        rngs.shuffle(order)
        cut = max(1, int(0.7 * len(order)))
        tr, te = set(order[:cut]), set(order[cut:])
        TR = [x for t in tr for x in by_t[t]]
        TE = [x for t in te for x in by_t[t]]
        if not TR or not TE:
            continue
        pB = fit_predict(TR, TE, cols, iT0)
        accB = {}
        for t in te:
            v, n, _ = H.pair_accuracy_rows(by_t[t], pB)
            accB[t] = (v, n)
        for j in range(R_NOISE):
            pX = fit_predict(TR, TE, augs[j], iT0 + [ncol])
            for t in te:
                vB, n = accB[t]
                if vB is None or n < H.MIN_PAIRS_PER_TARGET:
                    continue
                vX, _n2, _ = H.pair_accuracy_rows(by_t[t], pX)
                if vX is None:
                    continue
                pern[j][t].append(vX - vB)
    nz = []
    for j in range(R_NOISE):
        uu = [t for t in order if len(pern[j][t]) >= H.MIN_SPLITS_PER_TARGET]
        if len(uu) < H.MIN_TARGETS:
            continue
        g = [st.fmean(pern[j][t]) for t in uu]
        ci, _n = H.boot_ci(g, seed + 17)
        nz.append({"mean": round(st.fmean(g), 5), "ci": ci,
                   "excl": bool(ci[0] > 0 or ci[1] < 0), "pos": bool(ci[0] > 0)})
    zc = st.fmean([x["mean"] for x in nz]) if nz else float("nan")
    npos = sum(1 for x in nz if x["excl"] and x["pos"])
    print(f"  无信息列标定({len(nz)} 根):零点中心 {zc:+.5f}"
          f"  sd {st.stdev([x['mean'] for x in nz]) if len(nz) > 2 else float('nan'):.5f}"
          f"  排除0 {sum(1 for x in nz if x['excl'])} 根(偏正 {npos})")
    print(f"  ⇒ 所有增量的【相对无信息列】校正值 = 增量 − ({zc:+.5f})")
    if npos:
        print("  ⚠ 闸门6:出现【正侧】假阳性 -> 结论里所有正向判定必须降级")

    # ---- column-count-matched control (amendment, rule 39 + 46): pad T0/T1 up to T2's width -------
    N_PAD = 8
    rngp = random.Random(4242)
    pad_res = {"T0pad_minus_T2": [], "T1pad_minus_T2": []}
    for _q in range(N_PAD):
        k0, k1 = nB - n0, nB - (n0 + n1e)
        pv = {k: [rngp.gauss(0, 1.0) for _ in range(max(k0, k1))] for k in cols}
        aug = {k: v + pv[k] for k, v in cols.items()}
        base = len(cols[id(mols[0])])
        ipad = {"T0pad": iT0 + list(range(base, base + k0)),
                "T1pad": iT1 + list(range(base, base + k1)), "T2": iT2}
        assert len(ipad["T0pad"]) == nB and len(ipad["T1pad"]) == nB, "列数匹配没做到 T2 的宽度"
        pp, up, _tg2, _s3 = heldout(rws, aug, ipad, splits, seed, ref="T2")
        uu = [t for t in up if len(pp["T2"][t]) >= H.MIN_SPLITS_PER_TARGET]
        if len(uu) < H.MIN_TARGETS:
            continue
        for tag, blk in (("T0pad_minus_T2", "T0pad"), ("T1pad_minus_T2", "T1pad")):
            g = [st.fmean(pp[blk][t]) - st.fmean(pp["T2"][t]) for t in uu]
            ci, _n = H.boot_ci(g, seed + 17)
            pad_res[tag].append({"mean": round(st.fmean(g), 4), "ci": ci,
                                 "excl": bool(ci[0] > 0 or ci[1] < 0)})
    print(f"  列数匹配对照(T0/T1 补噪声列到 {nB} 列,独立 padding {N_PAD} 次):")
    for tag, L in pad_res.items():
        if not L:
            print(f"    {tag:18s} 无数据"); continue
        ms = [x["mean"] for x in L]
        print(f"    {tag:18s} 均值 {st.fmean(ms):+.4f}  范围 {min(ms):+.4f}~{max(ms):+.4f}"
              f"  {sum(1 for x in L if x['excl'])}/{len(L)} 次排除 0")

    ratio = (acc["T0"] - 0.5) / (acc["T2"] - 0.5) if acc["T2"] > 0.5 else float("nan")
    print(f"  **T0 达到 T2 超硬币技巧的 {ratio*100:.1f}%**"
          f"(样本内 7.3 的同类比值是 0.1001/0.1651 = 60.6%)")
    return {"n_cells": len(rws), "n_mol": len(mols), "n_used": len(used), "MDE": round(MDE, 4),
            "acc": {k: round(v, 4) for k, v in acc.items()}, "contrasts": C,
            "ratio_T0_over_T2": round(ratio, 4), "pad": pad_res, "noise": {"center": round(zc, 5), "n": len(nz),
            "n_excl": sum(1 for x in nz if x["excl"]), "n_excl_pos": npos, "detail": nz},
            "stats": stats, "dropped_cells": dropped_cells, "errors": err}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=SPLITS)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="", help="smoke: run just this population file")
    ap.add_argument("--out", default=os.path.join(HERE, "c1_cost_tiers.json"))
    a = ap.parse_args()
    assert_tier_purity()
    print("闸门7 层纯度:T0 构造器无禁用调用,且探针(真读判官列)能被检出  OK")
    PRIMARY = "panel_f1_fresh.json"
    POPS = ((f"样本2(主)", PRIMARY), ("基准4560(复现)", "panel_f1_designs.json"),
            ("样本3(复现)", "panel_f1_third.json"), ("第四批(复现)", "panel_f1_fourth.json"),
            ("第五批(附报)", "panel_f1_fifth.json"), ("第六批(附报)", "panel_f1_sixth.json"))
    if a.only:
        POPS = tuple((n, f) for n, f in POPS if f == a.only)
    res = {}
    for nm, fn in POPS:
        res[fn] = run_pop(nm, fn, a.splits, a.seed, a.workers)
        json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    if a.only:
        print("\n(--only 冒烟模式,不做裁定)")
        return 0

    print("\n=== 判定(冻结的打死条件)===")
    P = res[PRIMARY]
    if P.get("no_data"):
        v = "no_data"
    else:
        small = min(abs(P["contrasts"]["T2_minus_T0"]["mean"]),
                    abs(P["contrasts"]["T0uT2_minus_T2"]["mean"]))
        K4 = small < P["MDE"]
        K1 = P["ratio_T0_over_T2"] < 0.5
        c = P["contrasts"]["T0uT2_minus_T0"]
        K2 = c["excl"] and c["mean"] > 2 * P["MDE"]
        reps = 0
        for fn in ("panel_f1_designs.json", "panel_f1_third.json", "panel_f1_fourth.json"):
            r = res[fn]
            if r.get("no_data"):
                continue
            rc = r["contrasts"]["T0uT2_minus_T0"]
            if (r["ratio_T0_over_T2"] < 0.5) == K1 and (rc["excl"] and rc["mean"] > 2 * r["MDE"]) == K2:
                reps += 1
        pad = P.get("pad", {}).get("T0pad_minus_T2", [])
        K5 = (not pad) or (sum(1 for x in pad if x["excl"]) < len(pad) / 2)
        K3 = reps < 2
        print(f"  K4 测量够不够格(较小的量 {small:+.4f} vs MDE +{P['MDE']:.4f}):"
              f"{'FIRES -> 本人群报【测不出】,不许报无差别' if K4 else '不触发'}")
        print(f"  K1 便宜层没达到大半(比值 {P['ratio_T0_over_T2']*100:.1f}% < 50%):"
              f"{'FIRES' if K1 else '不触发'}")
        print(f"  K2 面板加了非边缘的量({c['mean']:+.4f} vs 2×MDE +{2*P['MDE']:.4f}):"
              f"{'FIRES' if K2 else '不触发'}")
        print(f"  K3 复现不足(复现 {reps}/3):{'FIRES' if K3 else '不触发'}")
        print(f"  K5 T0 的优势只是列数(列数匹配后 {sum(1 for x in pad if x['excl'])}/{len(pad)} 次排除 0):"
              f"{'FIRES' if K5 else '不触发'}")
        v = ("not_measurable" if K4 else
             "cheap_tier_advantage_is_column_count" if K5 else
             "panel_earns_its_cost" if (K1 and K2) else
             "cheap_tier_reaches_most_and_panel_adds_little" if (not K1 and not K2) else
             "cheap_tier_reaches_most_but_panel_still_adds" if (not K1 and K2) else
             "cheap_tier_falls_short_but_panel_adds_little")
        if K3:
            v += "__underreplicated"
        P["kills"] = {"K1": K1, "K2": K2, "K3": K3, "K4": K4, "K5": K5, "replications": reps}
    print(f"  裁定:{v}")
    print("  边界(声明里写好的):① 没有盲人群可用(H2 读过全部六个),被测的【量】是新的而【测试】不是盲的;"
          "② T0 若达到大半,那是关于面板【花的钱】的负面结论,抬的是目标②不是④;"
          "③ 只量信息量,不判上线 —— 改动出厂决定要先过重新校准。")
    res["verdict"] = v
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
