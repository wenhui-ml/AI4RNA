#!/usr/bin/env python3
"""M31t: the recovery-versus-budget curve behind Claim 3's figure.

A display quantity for M31/M31r, not a new claim. M31r reported recovery at one budget (the top
quartile of a design's flips). The figure needs the whole curve, and a figure must be drawn from a
committed artifact rather than recomputed at build time.

For every budget k as a fraction of a design cell's flips, this reports the share of that cell's
rescuing flips captured by: ranking on helix depth alone; on each predictor alone; on depth with
the predictor used only to break its ties; and at random. Same cells, same threshold, same
weighting as M31r. Offline, $0.
"""
from __future__ import annotations

import json, math, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m26_model_scored as M26
import m31_which_flips_rescue as M31

FRACS = tuple(round(0.05 * i, 2) for i in range(1, 21))       # 5% .. 100%
LOG = []


def log(s=""):
    print(s, flush=True); LOG.append(s)


def main():
    log("=== M31t:捞回率随预算的曲线(Claim 3 的图)===")
    log("  M31/M31r 的展示量,不产生新主张。离线 $0。\n")
    recs, _d = M26.build()
    z = np.load(M31.NEURAL_CACHE, allow_pickle=True)
    pn = {k: v for k, v in zip(z["keys"], z["mats"])}
    for r in recs:
        i, j = min(r["a"], r["b"]), max(r["a"], r["b"])
        r["nmut"] = float(pn[r["mut_seq"]][i, j])
    del pn, z
    cells = M31.make_cells(recs)
    th = M31.THRESH_MAIN
    log(f"  {len(recs)} 次干预,{len(cells)} 个格,阈值 {th}")

    def order_plain(rs, key):
        return np.argsort(-np.array([r[key] for r in rs], float), kind="stable")

    def order_tie(rs, key):
        d = -np.array([r["depth"] for r in rs], float)
        m = -np.array([r[key] for r in rs], float)
        return np.lexsort((m, d))

    rankers = {
        "depth_only": lambda rs: order_plain(rs, "depth"),
        "P_vienna": lambda rs: order_plain(rs, "pmut"),
        "P_neural": lambda rs: order_plain(rs, "nmut"),
        "depth_then_P_vienna": lambda rs: order_tie(rs, "pmut"),
        "depth_then_P_neural": lambda rs: order_tie(rs, "nmut"),
    }
    curves = {k: [] for k in rankers}
    curves["random"] = []
    usable = 0
    for f in FRACS:
        acc = {k: [0.0, 0.0] for k in list(rankers) + ["random"]}
        for _k, rs in cells:
            y = np.array([r["rf"] >= th for r in rs])
            npos, n = int(y.sum()), len(rs)
            if npos == 0 or npos == n:
                continue
            kk = max(1, math.ceil(f * n))
            for nm, fn in rankers.items():
                acc[nm][0] += npos * float(y[fn(rs)[:kk]].sum()) / npos
                acc[nm][1] += npos
            acc["random"][0] += npos * (kk / n); acc["random"][1] += npos
        for nm in acc:
            curves[nm].append(round(acc[nm][0] / acc[nm][1], 5) if acc[nm][1] else None)
        usable = int(acc["depth_only"][1])
    log(f"  可评分的正例权重 {usable}")
    log(f"\n  {'预算':>6s} " + "".join(f"{k[:14]:>16s}" for k in curves))
    for i, f in enumerate(FRACS):
        log(f"  {100*f:5.0f}% " + "".join(f"{curves[k][i]:16.5f}" for k in curves))

    OUT = {"fracs": list(FRACS), "curves": curves, "threshold": th,
           "n_cells": len(cells), "n_records": len(recs), "pos_weight": usable}
    json.dump(OUT, open(os.path.join(HERE, "m31t_recovery_curve.json"), "w"),
              ensure_ascii=False, indent=1)
    open(os.path.join(HERE, "m31t_recovery_curve.log"), "w").write("\n".join(LOG) + "\n")
    log("\n写出 m31t_recovery_curve.json / .log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
