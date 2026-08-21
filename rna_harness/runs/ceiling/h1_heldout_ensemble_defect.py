#!/usr/bin/env python3
"""H1: does ensemble_defect(target skeleton) hold up on targets the model never saw?

Pre-registered in h1_heldout_ensemble_defect_declaration.json (rule 38/39/40/41 spec).

E3 MEASURED SOMETHING ELSE. Its +0.024 is in-sample R^2 minus a within-cell permutation null. That null
subtracts what feature COUNT buys; it does not subtract fitting and evaluating on the same molecules. Whether
the column extrapolates to a new target, E3 did not ask.

THE PRIMARY METRIC IS LEAK-FREE BY CONSTRUCTION. This project's estimator demeans within cell, so a naive
held-out R^2 needs the held-out cell's own y mean -- one leaked number per cell. Instead: PAIRWISE RANKING
ACCURACY inside held-out cells. For every pair (i, j) in a held-out cell, does the model order them the way
the tube does? Cell effects cancel exactly in the difference, so no held-out mean is ever touched. It is also
the estimand the product actually needs -- pick one design for a given target, i.e. rank within cell.

Splits are by TARGET, not by molecule: another design for a target already seen is far easier than a new
target, and splitting by molecule would wash the difficulty out. 200 splits, 70/30, seeded.

THE E3 PLACEBO COMES ALONG. BC_mfe carries ensemble_defect(nested MFE) into the held-out setting. If it
extrapolates as well as the target-referencing column, then what generalises is ensemble concentration, not
distance to the intended structure.

COST: $0, offline. One fold per molecule; the splits are linear algebra.
"""
from __future__ import annotations

import argparse, collections, json, os, random, statistics as st, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from e1_ensemble_marginals import base_block, load_pop                           # noqa: E402
from e3_scrambled_reference import work                                          # noqa: E402

MIN_PAIRS_PER_SPLIT = 500


def fit_predict(train_rows, test_rows, cols, idx):
    """Fit on within-cell-demeaned TRAIN data only; return raw predictions for test rows.

    The test predictions are used only through WITHIN-CELL DIFFERENCES downstream, so any intercept or cell
    effect cancels and no held-out y is consulted at any point (void gate 2).
    """
    ys, Xs = [], []
    for _c, g in train_rows:
        y = np.array([r["wet"] for r in g], dtype=float)
        X = np.array([[cols[id(r)][k] for k in idx] for r in g], dtype=float)
        ys.append(y - y.mean()); Xs.append(X - X.mean(axis=0))
    y = np.concatenate(ys); X = np.vstack(Xs)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    out = {}
    for _c, g in test_rows:
        for r in g:
            out[id(r)] = float(np.dot([cols[id(r)][k] for k in idx], beta))
    return out


def pair_accuracy(test_rows, pred):
    """Within-cell pairwise ordering accuracy. Ties in the tube score are dropped and counted."""
    ok = n = ties = 0
    for _c, g in test_rows:
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                dw = g[a]["wet"] - g[b]["wet"]
                if dw == 0:
                    ties += 1
                    continue
                dp = pred[id(g[a])] - pred[id(g[b])]
                if dp == 0:
                    ties += 1
                    continue
                n += 1
                ok += (dw > 0) == (dp > 0)
    return (ok / n if n else None), n, ties


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
    rws = [(c, g) for c, g in cells.items() if len(g) >= 2]        # gate 3: a cell needs >= 2 to pair
    dropped = sum(1 for c, g in cells.items() if len(g) < 2)
    tg = sorted({c[0] for c, _g in rws})
    nB = len(base_block(keep[0]))
    iBC = list(range(nB + 2))
    iED = iBC + [nB + 2]
    iMFE = iBC + [nB + 3]
    print(f"  可用分子 {sum(len(g) for _c, g in rws)},格 {len(rws)}(成对需 >=2,剔除 {dropped}),靶标 {len(tg)}")
    rng = random.Random(seed)
    accs = {"BC": [], "BC_ed": [], "BC_mfe": []}
    d_ed, d_mfe, d_pair = [], [], []
    voided = 0
    ties_tot = pairs_tot = 0
    for s in range(splits):
        rng.shuffle(tg)
        cut = max(1, int(0.7 * len(tg)))
        tr, te = set(tg[:cut]), set(tg[cut:])
        assert not (tr & te), "void gate 1: target leakage"
        TR = [(c, g) for c, g in rws if c[0] in tr]
        TE = [(c, g) for c, g in rws if c[0] in te]
        if not TR or not TE:
            voided += 1
            continue
        a = {}
        for k, idx in (("BC", iBC), ("BC_ed", iED), ("BC_mfe", iMFE)):
            p = fit_predict(TR, TE, cols, idx)
            acc, n, ties = pair_accuracy(TE, p)
            a[k] = acc
            if k == "BC":
                pairs_tot += n; ties_tot += ties
                if n < MIN_PAIRS_PER_SPLIT:
                    voided += 1
                    a = None
                    break
        if a is None or any(v is None for v in a.values()):
            continue
        for k in accs:
            accs[k].append(a[k])
        d_ed.append(a["BC_ed"] - a["BC"])
        d_mfe.append(a["BC_mfe"] - a["BC"])
        d_pair.append((a["BC_ed"] - a["BC"]) - (a["BC_mfe"] - a["BC"]))
    n_ok = len(d_ed)
    print(f"  有效划分 {n_ok}/{splits}(作废 {voided});累计成对 {pairs_tot},并列剔除 {ties_tot}"
          f" = {ties_tot/max(1,ties_tot+pairs_tot)*100:.2f}%")
    if n_ok < 20:
        print("  有效划分 < 20 -> 无数据"); return {"no_data": True, "n_splits_ok": n_ok}

    def q(v):
        s2 = sorted(v)
        return [round(s2[int(0.025 * len(s2))], 4), round(s2[min(len(s2) - 1, int(0.975 * len(s2)))], 4)]
    out = {"n_splits_ok": n_ok, "voided": voided, "pairs": pairs_tot, "ties": ties_tot,
           "n_cells": len(rws), "n_targets": len(tg),
           "acc": {k: round(st.fmean(v), 4) for k, v in accs.items()},
           "d_ed": {"mean": round(st.fmean(d_ed), 4), "ci": q(d_ed),
                    "wins": sum(1 for x in d_ed if x > 0)},
           "d_mfe": {"mean": round(st.fmean(d_mfe), 4), "ci": q(d_mfe),
                     "wins": sum(1 for x in d_mfe if x > 0)},
           "d_paired": {"mean": round(st.fmean(d_pair), 4), "ci": q(d_pair),
                        "wins": sum(1 for x in d_pair if x > 0)}}
    print(f"  留出格内成对排序准确率(200 次划分均值)")
    print(f"    BC(基线)        {out['acc']['BC']:.4f}")
    print(f"    BC + ed_skel     {out['acc']['BC_ed']:.4f}   增量 {out['d_ed']['mean']:+.4f}"
          f"  95% {out['d_ed']['ci']}  胜 {out['d_ed']['wins']}/{n_ok}")
    print(f"    BC + ed_mfe(安慰剂) {out['acc']['BC_mfe']:.4f}   增量 {out['d_mfe']['mean']:+.4f}"
          f"  95% {out['d_mfe']['ci']}  胜 {out['d_mfe']['wins']}/{n_ok}")
    print(f"    K2 配对差(靶标−MFE)  {out['d_paired']['mean']:+.4f}  95% {out['d_paired']['ci']}"
          f"  胜 {out['d_paired']['wins']}/{n_ok}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "h1_heldout_ensemble_defect.json"))
    a = ap.parse_args()
    POPS = (("基准4560(主)", "panel_f1_designs.json"), ("样本3", "panel_f1_third.json"),
            ("第四批", "panel_f1_fourth.json"), ("样本2", "panel_f1_fresh.json"))
    res = {}
    for nm, fn in POPS:
        res[fn] = run_pop(nm, fn, a.splits, a.seed, a.workers)
        json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("\n=== 判定(冻结的打死条件)===")
    P = res["panel_f1_designs.json"]
    if P.get("no_data"):
        v = "no_data"
    else:
        def ex(d, k):
            c = d[k]["ci"]; return c[0] > 0 or c[1] < 0
        K1 = not ex(P, "d_ed")
        K2 = not ex(P, "d_paired")
        reps = sum(1 for fn in ("panel_f1_third.json", "panel_f1_fourth.json", "panel_f1_fresh.json")
                   if not res[fn].get("no_data") and ex(res[fn], "d_ed"))
        K3 = ex(P, "d_ed") and reps < 2
        print(f"  K1 留出上无增量(不外推)         :{'FIRES' if K1 else '不触发'}")
        print(f"  K2 外推靠集合集中度而非引用靶标 :{'FIRES' if K2 else '不触发'}")
        print(f"  K3 复现不足                     :{'FIRES' if K3 else '不触发'} (复现 {reps}/3)")
        v = ("does_not_generalise" if K1 else "generalises_but_not_target_specific" if K2 else
             "underreplicated" if K3 else "generalises_and_target_specific")
        P["kills"] = {"K1": K1, "K2": K2, "K3": K3, "replications": reps}
    print(f"  裁定:{v}")
    print("  提醒(声明里写好的):『能外推』不等于『该上线』。接进出厂判据会改动决定,"
          "那需要重新校准(与 k1_audit.md 同一条)。本实验只判外推。")
    res["verdict"] = v
    res["boundary"] = ("人群是骨架可形成的 77% 子集;仍在 Turner 族内,不修判官独立性;"
                       "不许与全人群的 0.196 混引。")
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
