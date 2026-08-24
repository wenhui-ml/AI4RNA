#!/usr/bin/env python3
"""The width-matched null C1 needed and did not have.

C1 calibrated the cost of adding ONE useless column (rule 45). But the contrast that survived --
acc(T0uT2) - acc(T2) -- adds SIX columns to T2. The calibration and the contrast are different widths, so
the corrected value could not be quoted. This measures the matched null: T2 plus k USELESS columns, minus T2,
for k = 1 and k = 6, R draws each.

Direction is known in advance and is unfavourable to a naive reading of the raw number: more columns cost
held-out accuracy, so a +0.030 increment that ALREADY paid the six-column toll understates the information the
six free columns carry. Measuring it turns "the direction favours us" into a number.

$0, offline.
"""
from __future__ import annotations
import collections, json, os, random, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/data/wenhui/openclaude_rna")
import h2_heldout_cluster_ci as H
from h1_heldout_ensemble_defect import fit_predict
from e1_ensemble_marginals import base_block, load_pop
from c1_cost_tiers import t0_columns, T0_NAMES, T1_EXTRA, work, heldout

R = 24
SPLITS = 200
KS = (1, 6)


def build(fn, workers=14):
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
    cells, cols = collections.defaultdict(list), {}
    for r, m in zip(keep, M):
        if "error" in m:
            continue
        rr = dict(r)
        cols[id(rr)] = [m[k] for k in T0_NAMES] + [m[k] for k in T1_EXTRA] + base_block(r)
        cells[(r["puzzle"], r["method"])].append(rr)
    rws = [(c, g) for c, g in cells.items() if len(g) >= 2]
    nB = len(base_block(keep[0]))
    return rws, cols, len(T0_NAMES), len(T1_EXTRA), nB


out = {}
for nm, fn in (("样本2(主)", "panel_f1_fresh.json"), ("基准4560", "panel_f1_designs.json")):
    rws, cols, n0, n1e, nB = build(fn)
    mols = [r for _c, g in rws for r in g]
    iT2 = list(range(n0 + n1e, n0 + n1e + nB))
    base = len(cols[id(mols[0])])
    print(f"\n=== {nm} 分子 {len(mols)} 格 {len(rws)} T2={nB} 列 ===", flush=True)
    row = {}
    for k in KS:
        rng = random.Random(777 + k)
        noise = {key: [[rng.gauss(0, 1.0) for _ in range(k)] for _ in range(R)] for key in cols}
        ms = []
        for j in range(R):
            aug = {key: v + noise[key][j] for key, v in cols.items()}
            idx = {"T2": iT2, "T2n": iT2 + list(range(base, base + k))}
            per, used, _tg, _s = heldout(rws, aug, idx, SPLITS, 0, ref="T2")
            uu = [t for t in used if len(per["T2"][t]) >= H.MIN_SPLITS_PER_TARGET]
            if len(uu) < H.MIN_TARGETS:
                continue
            g = [st.fmean(per["T2n"][t]) - st.fmean(per["T2"][t]) for t in uu]
            ms.append(st.fmean(g))
        row[k] = {"center": round(st.fmean(ms), 5), "sd": round(st.stdev(ms), 5), "n": len(ms)}
        print(f"  在 T2 上加 {k} 根【无信息】列:零点 {st.fmean(ms):+.5f}  sd {st.stdev(ms):.5f}"
              f"  ({len(ms)} 次抽样)", flush=True)
    obs = {"panel_f1_fresh.json": 0.0303, "panel_f1_designs.json": 0.0388}[fn]
    z6 = row[6]["center"]
    print(f"  实测 T0∪T2 − T2 = {obs:+.4f};宽度匹配零点 = {z6:+.5f}"
          f"  ⇒ **校正后 = {obs - z6:+.4f}**")
    print(f"  (每列平均代价:k=1 时 {row[1]['center']:+.5f},k=6 时 {z6/6:+.5f}/列"
          f" ⇒ {'次线性,列间共享过拟合' if abs(z6) < 6*abs(row[1]['center']) else '大致线性'})")
    row["observed"] = obs
    row["corrected"] = round(obs - z6, 4)
    out[nm] = row
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "c1_width_null.json"), "w"), indent=1, ensure_ascii=False)
print("\n写出 c1_width_null.json")
