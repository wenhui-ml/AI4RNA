#!/usr/bin/env python3
"""H2: H1's frozen K1 was never computed. Recompute it with the interval the declaration actually names,
on two populations H1 never touched.

Pre-registered in h2_heldout_cluster_ci_declaration.json (rule 38/39/40/41/42 spec).

THE DEFECT H2 REPAIRS. H1's declaration froze "the interval OF THE MEAN of the 200 split gains". Its code
computed `sorted(v)[int(.025n)], sorted(v)[int(.975n)]` -- the 2.5/97.5 percentiles of the 200 split values
themselves. That is a single-split PREDICTION interval whose width contains the Monte-Carlo variance of which
targets happened to land in the held-out set. It is not an interval on the mean. So H1's K1/K2 are VOID, not
failed -- the frozen quantity was never evaluated.

Declaring a void after seeing an unfavourable number is exactly the move that launders failures, so the void
only counts paired with a re-test on populations I have not looked at: panel_f1_sixth.json (primary, 49
targets / 130 cells) and panel_f1_fifth.json (replication). Chosen on cluster count alone, written down in the
declaration before running.

THE CORRECTED INTERVAL. Per target t, g_t = mean over the splits that held t out of the pairwise ranking
accuracy gain computed on t's own held-out cells. Then cluster bootstrap over TARGETS (B=5000) on mean_t g_t.
Clusters = targets = the sampling unit. This removes the "which targets got held out" Monte-Carlo variance and
keeps between-target heterogeneity, which IS the estimand's uncertainty. It may come out WIDER than H1's
percentile interval; the declaration says so in advance, and if K1 still fires the void rescues nothing.

COST: $0, offline.
"""
from __future__ import annotations

import argparse, collections, json, os, random, statistics as st, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from e1_ensemble_marginals import base_block, load_pop                           # noqa: E402
from e3_scrambled_reference import work                                          # noqa: E402
from h1_heldout_ensemble_defect import fit_predict                               # noqa: E402

MIN_PAIRS_PER_TARGET = 30          # void gate 3(a)
MIN_SPLITS_PER_TARGET = 20         # void gate 3(b)
MIN_TARGETS = 20                   # void gate 3(c) -- the cluster floor
BOOT = 5000


def pair_accuracy_rows(rows, pred):
    """Within-cell pairwise ordering accuracy over `rows` (a list of (cell, group)). Ties dropped, counted."""
    ok = n = ties = 0
    for _c, g in rows:
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                dw = g[a]["wet"] - g[b]["wet"]
                if dw == 0:
                    ties += 1; continue
                dp = pred[id(g[a])] - pred[id(g[b])]
                if dp == 0:
                    ties += 1; continue
                n += 1
                ok += (dw > 0) == (dp > 0)
    return (ok / n if n else None), n, ties


def boot_ci(vec, seed):
    """Cluster bootstrap over the ELEMENTS OF `vec`, which are per-target statistics (void gate 6)."""
    rng = random.Random(seed)
    n = len(vec)
    means = []
    for _ in range(BOOT):
        s = 0.0
        for _i in range(n):
            s += vec[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    return [round(means[int(0.025 * BOOT)], 4), round(means[int(0.975 * BOOT)], 4)], n


def pct_ci(v):
    """H1's interval, reproduced verbatim and LABELLED as a single-split prediction interval."""
    s2 = sorted(v)
    return [round(s2[int(0.025 * len(s2))], 4), round(s2[min(len(s2) - 1, int(0.975 * len(s2)))], 4)]


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
    cells = collections.defaultdict(list)
    cols = {}
    for r, m in zip(keep, M):
        if "error" in m:
            continue
        rr = dict(r)
        cols[id(rr)] = base_block(r) + m["C"] + [m["ed_skel"]] + [m["ed_mfe"]]
        cells[(r["puzzle"], r["method"])].append(rr)
    rws = [(c, g) for c, g in cells.items() if len(g) >= 2]
    dropped_cells = sum(1 for c, g in cells.items() if len(g) < 2)
    tg = sorted({c[0] for c, _g in rws})
    nB = len(base_block(keep[0]))
    iBC = list(range(nB + 2)); iED = iBC + [nB + 2]; iMFE = iBC + [nB + 3]
    print(f"  可用分子 {sum(len(g) for _c, g in rws)},格 {len(rws)}"
          f"(成对需 >=2,剔除 {dropped_cells}),靶标 {len(tg)}")

    # ---- void gate 7: the two added columns are actually wired and are not the same column -----------
    ed = np.array([cols[id(r)][nB + 2] for _c, g in rws for r in g], dtype=float)
    mf = np.array([cols[id(r)][nB + 3] for _c, g in rws for r in g], dtype=float)
    v_ed, v_mf = float(ed.var()), float(mf.var())
    rho = float(np.corrcoef(ed, mf)[0, 1]) if v_ed > 0 and v_mf > 0 else 1.0
    g7 = (v_ed > 0) and (v_mf > 0) and (abs(rho) < 0.999)
    print(f"  闸门7 两列接上了:var(ed_skel)={v_ed:.4g} var(ed_mfe)={v_mf:.4g} 相关={rho:+.4f}"
          f"  {'OK' if g7 else '✗ 列没接上 -> 本人群作废'}")
    if not g7:
        return {"no_data": True, "reason": "void gate 7: columns not wired", "rho": rho}

    by_t = collections.defaultdict(list)
    for c, gp in rws:
        by_t[c[0]].append((c, gp))
    rng = random.Random(seed)
    per_t = {t: {"ed": [], "mfe": [], "pair": []} for t in tg}
    skipped_ts = 0          # (target, split) pairs below the pair floor
    ties_tot = pairs_tot = 0
    d_ed_split, d_mfe_split, d_pair_split = [], [], []      # H1's split-level view, for the labelled column
    order = list(tg)
    for _s in range(splits):
        rng.shuffle(order)
        cut = max(1, int(0.7 * len(order)))
        tr, te = set(order[:cut]), set(order[cut:])
        assert not (tr & te), "void gate 1: target leakage"
        TR = [x for t in tr for x in by_t[t]]
        TE = [x for t in te for x in by_t[t]]
        if not TR or not TE:
            continue
        preds = {}
        for k, idx in (("BC", iBC), ("BC_ed", iED), ("BC_mfe", iMFE)):
            preds[k] = fit_predict(TR, TE, cols, idx)
        # split-level (H1's view)
        a_s = {}
        for k in preds:
            acc, n, ti = pair_accuracy_rows(TE, preds[k])
            a_s[k] = acc
            if k == "BC":
                pairs_tot += n; ties_tot += ti
        if all(v is not None for v in a_s.values()):
            d_ed_split.append(a_s["BC_ed"] - a_s["BC"])
            d_mfe_split.append(a_s["BC_mfe"] - a_s["BC"])
            d_pair_split.append((a_s["BC_ed"] - a_s["BC"]) - (a_s["BC_mfe"] - a_s["BC"]))
        # per-target (H2's view)
        for t in te:
            rows_t = by_t[t]
            acc = {}
            nt = None
            for k in preds:
                v, n, _ti = pair_accuracy_rows(rows_t, preds[k])
                acc[k] = v; nt = n
            if nt is None or nt < MIN_PAIRS_PER_TARGET or any(v is None for v in acc.values()):
                skipped_ts += 1
                continue
            per_t[t]["ed"].append(acc["BC_ed"] - acc["BC"])
            per_t[t]["mfe"].append(acc["BC_mfe"] - acc["BC"])
            per_t[t]["pair"].append((acc["BC_ed"] - acc["BC"]) - (acc["BC_mfe"] - acc["BC"]))

    used = [t for t in tg if len(per_t[t]["ed"]) >= MIN_SPLITS_PER_TARGET]
    dropped_t = len(tg) - len(used)
    print(f"  (靶标,划分) 低于成对地板({MIN_PAIRS_PER_TARGET})跳过 {skipped_ts} 次;"
          f"可用划分数 < {MIN_SPLITS_PER_TARGET} 的靶标剔除 {dropped_t},剩 {len(used)}")
    print(f"  累计成对 {pairs_tot},并列剔除 {ties_tot} = {ties_tot/max(1,ties_tot+pairs_tot)*100:.2f}%")
    if len(used) < MIN_TARGETS:
        print(f"  产生 g_t 的靶标 {len(used)} < {MIN_TARGETS} -> 无数据(cluster 地板)")
        return {"no_data": True, "reason": "void gate 3(c): cluster floor", "n_targets_used": len(used)}

    if not d_ed_split:
        print("  划分级统计为空 -> 无数据"); return {"no_data": True, "reason": "no usable split"}
    g = {k: [st.fmean(per_t[t][k]) for t in used] for k in ("ed", "mfe", "pair")}
    out = {"n_cells": len(rws), "n_targets": len(tg), "n_targets_used": len(used),
           "dropped_targets": dropped_t, "dropped_cells": dropped_cells,
           "skipped_target_splits": skipped_ts, "pairs": pairs_tot, "ties": ties_tot,
           "gate7": {"var_ed": v_ed, "var_mfe": v_mf, "corr": rho}}
    for k, label in (("ed", "d_ed"), ("mfe", "d_mfe"), ("pair", "d_paired")):
        ci, nb = boot_ci(g[k], seed + 17)
        assert nb == len(used), "void gate 6: bootstrap unit is not the target"
        out[label] = {"mean": round(st.fmean(g[k]), 4), "ci_cluster_boot": ci,
                      "targets_positive": sum(1 for x in g[k] if x > 0), "n_clusters": nb}
    out["h1_style_split_percentile"] = {
        "note": "单次划分预测区间,不是均值区间 —— 只为与 h1_audit.md 的表对得上",
        "d_ed": {"mean": round(st.fmean(d_ed_split), 4), "ci": pct_ci(d_ed_split),
                 "wins": sum(1 for x in d_ed_split if x > 0), "n_splits": len(d_ed_split)},
        "d_mfe": {"mean": round(st.fmean(d_mfe_split), 4), "ci": pct_ci(d_mfe_split),
                  "wins": sum(1 for x in d_mfe_split if x > 0)},
        "d_paired": {"mean": round(st.fmean(d_pair_split), 4), "ci": pct_ci(d_pair_split),
                     "wins": sum(1 for x in d_pair_split if x > 0)}}
    print(f"  写对的区间(靶标 cluster bootstrap,B={BOOT},cluster={len(used)})")
    for label, nm2 in (("d_ed", "ed_skel 增量"), ("d_mfe", "ed_mfe 安慰剂"), ("d_paired", "K2 配对差")):
        o = out[label]
        print(f"    {nm2:16s} {o['mean']:+.4f}  95% {o['ci_cluster_boot']}"
              f"  {'排除 0' if (o['ci_cluster_boot'][0] > 0 or o['ci_cluster_boot'][1] < 0) else '含 0'}"
              f"  正号靶标 {o['targets_positive']}/{o['n_clusters']}")
    h = out["h1_style_split_percentile"]
    print(f"  (对照 H1 口径的分位区间:ed {h['d_ed']['mean']:+.4f} {h['d_ed']['ci']} 胜 "
          f"{h['d_ed']['wins']}/{h['d_ed']['n_splits']};配对差 {h['d_paired']['mean']:+.4f} "
          f"{h['d_paired']['ci']})", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "h2_heldout_cluster_ci.json"))
    a = ap.parse_args()
    PRIMARY, REPL = "panel_f1_sixth.json", "panel_f1_fifth.json"
    POPS = ((f"第六批(主,盲)", PRIMARY), (f"第五批(复现,盲)", REPL),
            ("基准4560(非盲参考)", "panel_f1_designs.json"), ("样本3(非盲参考)", "panel_f1_third.json"),
            ("第四批(非盲参考)", "panel_f1_fourth.json"), ("样本2(非盲参考)", "panel_f1_fresh.json"))
    res = {}
    for nm, fn in POPS:
        res[fn] = run_pop(nm, fn, a.splits, a.seed, a.workers)
        json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)

    print("\n=== 判定(冻结的打死条件;只用两个盲人群)===")
    P = res[PRIMARY]
    if P.get("no_data"):
        v = "no_data"
    else:
        def ex(d, k):
            c = d[k]["ci_cluster_boot"]; return c[0] > 0 or c[1] < 0
        K1 = not ex(P, "d_ed")
        K2 = not ex(P, "d_paired")
        K3 = ex(P, "d_ed") and not (not res[REPL].get("no_data") and ex(res[REPL], "d_ed"))
        print(f"  K1 留出上无增量(不外推)         :{'FIRES' if K1 else '不触发'}")
        print(f"  K2 外推靠集合集中度而非引用靶标 :{'FIRES' if K2 else '不触发'}")
        print(f"  K3 盲复现不足                   :{'FIRES' if K3 else '不触发'}")
        v = ("does_not_generalise" if K1 else "generalises_but_not_target_specific" if K2 else
             "underreplicated" if K3 else "generalises_and_target_specific")
        P["kills"] = {"K1": K1, "K2": K2, "K3": K3}
    print(f"  裁定:{v}")
    print("  非盲的四个人群只作参考,不参与任何 K(声明冻结)。")
    print("  提醒(声明里写好的):『能外推』不等于『该上线』。接进出厂判据会改动决定,"
          "那需要重新校准(与 k1_audit.md 同一条)。本实验只判外推。")
    res["verdict"] = v
    res["boundary"] = ("人群是骨架可形成子集;仍在 Turner 族内,不修判官独立性;不许与全人群的 0.196 混引。")
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
