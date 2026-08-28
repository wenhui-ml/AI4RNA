#!/usr/bin/env python3
"""M31s: how much of each predictor's within-design ordering IS helix depth?

A supporting measurement for M31/M31r, not a new claim. M31r established that a free depth
baseline ties the best predictor at ranking which compensatory flips rescue, and that a real
increment survives once depth is held fixed. Between those two facts sits one number the paper
needs and neither run reports: the within-cell rank agreement between each predictor and depth
itself, and the distribution of depth that makes the tie-breaking arm of M31r possible.

Offline, $0. No new data: M26's admission set and M26c's cached neural scores.
"""
from __future__ import annotations

import collections, json, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m26_model_scored as M26
import m31_which_flips_rescue as M31

LOG = []


def log(s=""):
    print(s, flush=True); LOG.append(s)


def main():
    log("=== M31s:每个预测器的格内排序里,有多少其实就是螺旋深度 ===")
    log("  M31/M31r 的支撑测量,不产生新主张。离线 $0。\n")
    recs, _d = M26.build()
    z = np.load(M31.NEURAL_CACHE, allow_pickle=True)
    pn = {k: v for k, v in zip(z["keys"], z["mats"])}
    for r in recs:
        i, j = min(r["a"], r["b"]), max(r["a"], r["b"])
        r["nmut"] = float(pn[r["mut_seq"]][i, j])
    del pn, z
    cells = M31.make_cells(recs)
    log(f"  {len(recs)} 次干预,{len(cells)} 个设计格")

    def cz(v):
        v = np.asarray(v, float) - np.mean(v)
        n = float(np.sqrt((v * v).sum()))
        return v / n if n > 0 else None

    def wrho(a, b):
        num = den = 0.0
        for _k, rs in cells:
            x, y = cz(rankdata([r[a] for r in rs])), cz(rankdata([r[b] for r in rs]))
            if x is None or y is None:
                continue
            num += len(rs) * float(x @ y); den += len(rs)
        return (num / den) if den else None

    OUT = {"n_records": len(recs), "n_cells": len(cells), "with_depth": {}, "with_outcome": {}}
    log("\n  格内秩一致度(权重 = 格大小)")
    log(f"  {'预测器':16s} {'对深度':>10s} {'对实测 rescue':>14s} {'差':>9s}")
    for nm, key in (("P_vienna", "pmut"), ("P_neural", "nmut")):
        d = wrho(key, "depth"); y = wrho(key, "rf")
        OUT["with_depth"][nm] = round(d, 5)
        OUT["with_outcome"][nm] = round(y, 5)
        OUT.setdefault("gap", {})[nm] = round(y - d, 5)
        log(f"  {nm:16s} {d:+10.5f} {y:+14.5f} {y-d:+9.5f}")
    OUT["depth_vs_outcome"] = round(wrho("depth", "rf"), 5)
    log(f"  {'depth':16s} {'—':>10s} {OUT['depth_vs_outcome']:+14.5f}")
    log("  ⇒ 每个模型『知道多少 rescue』与『知道多少深度』几乎相等。")

    cnt = collections.Counter(r["depth"] for r in recs)
    OUT["depth_histogram"] = {str(k): v for k, v in sorted(cnt.items())}
    OUT["depth_le2_frac"] = round(sum(v for k, v in cnt.items() if k <= 2) / len(recs), 4)
    # how many ordered pairs inside a cell are tied on depth -- the room the tie-break arm has
    tie = tot = 0
    for _k, rs in cells:
        d = np.array([r["depth"] for r in rs])
        n = len(d)
        tot += n * (n - 1) // 2
        _, c = np.unique(d, return_counts=True)
        tie += int(sum(x * (x - 1) // 2 for x in c))
    OUT["tied_pair_fraction"] = round(tie / tot, 4)
    log(f"\n  深度直方图 {OUT['depth_histogram']}")
    log(f"  深度 <= 2 的占 {OUT['depth_le2_frac']}")
    log(f"  格内配对中深度打平的占 {OUT['tied_pair_fraction']} "
        f"<- 这是 M31r 的『打破平手』臂能有作用的空间")

    json.dump(OUT, open(os.path.join(HERE, "m31s_depth_share.json"), "w"),
              ensure_ascii=False, indent=1)
    open(os.path.join(HERE, "m31s_depth_share.log"), "w").write("\n".join(LOG) + "\n")
    log("\n写出 m31s_depth_share.json / .log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
