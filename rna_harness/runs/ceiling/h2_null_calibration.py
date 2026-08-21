#!/usr/bin/env python3
"""Calibrate the false-positive rate of H2's frozen test.

THE ALARM. h2_review_diag drew one pure-noise column and got -0.0068 95% [-0.0127,-0.0013] -- EXCLUDING 0.
h2_noise_baseline drew a different pure-noise column on the SAME population with the SAME machinery and got
+0.0028 [-0.0031,+0.0090] -- containing 0. Two useless columns, two different answers, and at least one of
them "significant". If a no-information column clears the gate more often than 5% of the time, then every
"excludes 0" H2 printed is uninterpretable at face value, including the five populations that passed.

WHAT THIS MEASURES. R independent no-information columns against the SAME baseline, SAME 200 splits, SAME
cluster bootstrap. The fraction whose interval excludes 0 is the empirical size of the test. It should be
about 5%. Reported per population, two-sided and split by direction.

Efficiency: the baseline predictions do not depend on the noise column, so BC is fitted once per split and
all R noise variants reuse it.

$0, offline.
"""
from __future__ import annotations
import collections, json, os, random, statistics as st, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h2_heldout_cluster_ci as H
from h1_heldout_ensemble_defect import fit_predict
from h2_review_diag import build

R = 24          # independent no-information columns
SPLITS = 200


def calibrate(nm, fn, workers=14):
    rws, cols, nB = build(fn, workers)
    mols = [r for _c, g in rws for r in g]
    rng = random.Random(999)
    # noise for EVERY key in cols, not just the molecules that survived the >=2 cell gate -- the earlier
    # version keyed on `mols` and blew up on the dropped ones (KeyError). Precompute the R augmented column
    # tables once instead of rebuilding them inside the split loop.
    noise = {k: [rng.gauss(0, 1.0) for _ in range(R)] for k in cols}
    augs = [{k: v + [noise[k][j]] for k, v in cols.items()} for j in range(R)]
    by_t = collections.defaultdict(list)
    for c, gp in rws:
        by_t[c[0]].append((c, gp))
    tg = sorted(by_t); order = list(tg)
    iBC = list(range(nB + 2))
    # per-noise-column, per-target list of gains
    per = [{t: [] for t in tg} for _ in range(R)]
    srng = random.Random(0)                       # same split stream as H2
    for _s in range(SPLITS):
        srng.shuffle(order)
        cut = max(1, int(0.7 * len(order)))
        tr, te = set(order[:cut]), set(order[cut:])
        TR = [x for t in tr for x in by_t[t]]
        TE = [x for t in te for x in by_t[t]]
        if not TR or not TE:
            continue
        pB = fit_predict(TR, TE, cols, iBC)
        accB = {}
        for t in te:
            a, n, _ = H.pair_accuracy_rows(by_t[t], pB)
            accB[t] = (a, n)
        for j in range(R):
            pX = fit_predict(TR, TE, augs[j], iBC + [nB + 4])
            for t in te:
                aB, n = accB[t]
                if n < H.MIN_PAIRS_PER_TARGET or aB is None:
                    continue
                aX, _n2, _ = H.pair_accuracy_rows(by_t[t], pX)
                if aX is None:
                    continue
                per[j][t].append(aX - aB)
    res = []
    for j in range(R):
        used = [t for t in tg if len(per[j][t]) >= H.MIN_SPLITS_PER_TARGET]
        if len(used) < H.MIN_TARGETS:
            continue
        g = [st.fmean(per[j][t]) for t in used]
        ci, _n = H.boot_ci(g, 17)
        res.append({"mean": round(st.fmean(g), 4), "ci": ci,
                    "excl": bool(ci[0] > 0 or ci[1] < 0), "pos": bool(ci[0] > 0)})
    ex = [r for r in res if r["excl"]]
    print(f"\n=== {nm} 分子 {len(mols)} 格 {len(rws)} 靶标 {len(tg)} ===")
    print(f"  {len(res)} 根【无信息】列,区间排除 0 的有 {len(ex)} 根 "
          f"= {len(ex)/len(res)*100:.1f}%(名义 5%)")
    print(f"    其中偏正 {sum(1 for r in ex if r['pos'])},偏负 {sum(1 for r in ex if not r['pos'])}")
    ms = [r["mean"] for r in res]
    print(f"  无信息列点估计:均值 {st.fmean(ms):+.4f} 标准差 {st.stdev(ms):.4f} "
          f"范围 {min(ms):+.4f} ~ {max(ms):+.4f}")
    ws = [r["ci"][1] - r["ci"][0] for r in res]
    print(f"  区间平均宽 {st.fmean(ws):.4f};而点估计在各次抽样间的标准差 {st.stdev(ms):.4f} "
          f"⇒ 覆盖 {'不足(区间太窄)' if st.stdev(ms)*3.92 > st.fmean(ws) else '大致合理'}")
    print(f"  参考:H2 报的 ed_skel 点估计在这个人群上是 "
          f"{ {'panel_f1_sixth.json':'+0.0060','panel_f1_fifth.json':'+0.0143','panel_f1_designs.json':'+0.0087'}[fn] }"
          f",而无信息列能走到 {max(ms):+.4f}", flush=True)
    return {"n": len(res), "n_excl": len(ex), "rate": round(len(ex)/len(res), 4),
            "mean_of_means": round(st.fmean(ms), 5), "sd_of_means": round(st.stdev(ms), 5),
            "mean_width": round(st.fmean(ws), 5), "detail": res}


out = {}
for nm, fn in (("第六批(主,盲)", "panel_f1_sixth.json"),
               ("基准4560(非盲,最大)", "panel_f1_designs.json")):
    out[nm] = calibrate(nm, fn)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "h2_null_calibration.json"), "w"), indent=1, ensure_ascii=False)
print("\n写出 h2_null_calibration.json")
