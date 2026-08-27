#!/usr/bin/env python3
"""M19: clear the M13-era null defect out of M13c, M13d and M15.

Declaration: m19_null_sweep_declaration.json, frozen and pushed at 987a327c BEFORE this file
existed (the twentieth done that way).

The repair is M18's, word for word: treatment and every caliper column stay real, y is permuted
within cell, so the matched set is fixed and the permutation distribution is exact. M18 measured
that null's type-I error at 1/240 against a nominal 0.005; the unrepaired one gave 13/240.

This file does NOT reimplement the three experiments. It imports them and replaces exactly one
function in each -- the null -- then calls their own main(). That is what makes K0 (point
estimates and pair counts identical to the committed JSON) a property of the construction rather
than something I have to hope I reproduced.

COST: $0, offline.
"""
from __future__ import annotations

import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", ".."))); sys.path.insert(0, HERE)


# ---------------------------------------------------------------- the two shapes of "permute y"
def perm_y_celldicts(D, rng):
    """for m13c: D['cells'] is a list of dicts each holding its own y."""
    cells = []
    for c in D["cells"]:
        y = c["y"][rng.permutation(len(c["y"]))]
        cells.append({**c, "y": y - y.mean()})
    return {**D, "cells": cells}


def perm_y_flat(D, rng):
    """for m13/m15: D['y'] is flat and D['cells'] is a list of index arrays."""
    y = D["y"].copy()
    for idx in D["cells"]:
        v = y[idx][rng.permutation(len(idx))]
        y[idx] = v - v.mean()
    return {**D, "y": y}


def patch_m13c():
    import m13c_symmetric as S, m13c_pairing_vs_composition as C3

    def null_dist_repaired(D, key, delta, hold, seed, nperm):
        rng = np.random.default_rng(seed)
        treat = [c[key] for c in D["cells"]]                 # REAL treatment, never shuffled
        out = []
        for _ in range(nperm):
            M = C3.match(perm_y_celldicts(D, rng), treat, delta, hold)
            out.append(C3.point(M) if M else 0.0)
        return np.sort(np.array(out))

    S.null_dist = null_dist_repaired
    return S


def patch_m15():
    import m15_structure_filter as M15
    from m13_gc_placement import match as g1match

    def null_p_repaired(D, treat, delta, seed, obs, nperm=M15.NPERM):
        rng = np.random.default_rng(seed); n = []
        for _ in range(nperm):
            M = g1match(perm_y_flat(D, rng), treat, delta, use_panel=True)
            n.append(float(sum(M["sums"].values()) / sum(M["cnts"].values())) if M else 0.0)
        n = np.sort(np.array(n))
        return (float((n >= obs).sum()) + 1) / (nperm + 1), float(n.std()), n

    M15.null_p = null_p_repaired
    return M15


def run_m13d(seed, nperm, out):
    """m13d inlines its null, so its loop is repeated here with the design held fixed.
    It carries no ledger claim (K4 of the declaration); the numbers are updated, not re-judged."""
    from m13_cache import POPS
    from m13b_gc_placement import assemble, match, point, pval, DELTA
    from m13d_stability_caliper import within_sd, KS
    C = json.load(open(os.path.join(HERE, "m13_features.json")))
    E = json.load(open(os.path.join(HERE, "m13b_extra.json")))
    OLD = json.load(open(os.path.join(HERE, "m13d_stability_caliper.json")))
    OUT, K0bad = {}, []
    for nm, fn in POPS:
        d = assemble(C[fn]["rows"], E[fn], seed); w = within_sd(d, "dg")
        treat = [c["gcp"] for c in d["cells"]]
        base = point(match(d, treat, DELTA))
        print(f"=== {nm} ===  格内 dG SD {w:.2f}  无 dG 配平 {base:+.3f}")
        rows = []
        for k in KS:
            cal = k * w; d2 = dict(d); d2["sdg"] = cal / 0.20
            M = match(d2, treat, DELTA, dg=True)
            if M is None or M["npair"] < 30:
                rows.append({"k": k, "cal": round(cal, 3), "npair": 0 if M is None else M["npair"]})
                continue
            obs = point(M)
            rng = np.random.default_rng(seed)
            null = np.sort(np.array([point(match(perm_y_celldicts(d2, rng), treat, DELTA, dg=True)
                                           or {"sums": {0: 0.0}, "cnts": {0: 1}})
                                     for _ in range(nperm)]))
            p = pval(obs, null)
            o = next((r for r in OLD.get(fn, {}).get("rungs", []) if r.get("k") == k), None)
            if o and o.get("effect") is not None and abs(o["effect"] - round(obs, 3)) > 1e-3:
                K0bad.append(f"m13d/{nm}/k={k}: 旧{o['effect']:+.3f} 新{obs:+.3f}")
            rows.append({"k": k, "cal": round(cal, 3), "effect": round(obs, 3),
                         "p_REPAIRED": round(p, 5), "p_OLD": (o or {}).get("p"),
                         "npair": M["npair"], "retain": round(obs / (base or 1e-9), 3)})
            print(f"   caliper {cal:6.2f} kcal  效应 {obs:+7.3f}  保留 {obs/(base or 1e-9):6.1%}  "
                  f"p {p:.4f} (旧 {(o or {}).get('p')})  配对 {M['npair']:5d}")
        OUT[fn] = {"pop": nm, "within_sd_dg": round(w, 3), "base": round(base, 3), "rungs": rows}
    json.dump({"K0_moved": K0bad, "pops": OUT}, open(out, "w"), ensure_ascii=False, indent=1)
    return K0bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nperm", type=int, default=2000)
    ap.add_argument("--only", default="all", choices=("all", "m13c", "m15", "m13d"))
    a = ap.parse_args()
    print("=== M19:把 M18 的修法推到 M13c / M13d / M15 ===")
    print("  声明 987a327c(代码存在之前已推送)。离线 $0。")
    print("  **不重写实验 —— 只把每个实验里的【零分布那一个函数】换掉,再调它自己的 main()。**\n")

    if a.only in ("all", "m13c"):
        print("########## M13c(账本 D2)——修好的零分布 ##########")
        S = patch_m13c()
        sys.argv = ["m13c_symmetric", "--seed", str(a.seed), "--nperm", str(a.nperm),
                    "--out", os.path.join(HERE, "m19_m13c_symmetric_repaired.json")]
        S.main()

    if a.only in ("all", "m15"):
        print("\n########## M15(账本 C5)——修好的零分布 ##########")
        M15 = patch_m15()
        sys.argv = ["m15_structure_filter", "--seed", str(a.seed),
                    "--out", os.path.join(HERE, "m19_m15_structure_filter_repaired.json")]
        M15.main()

    if a.only in ("all", "m13d"):
        print("\n########## M13d(无账本主张,只更新数字)——修好的零分布 ##########")
        bad = run_m13d(a.seed, max(1000, a.nperm // 2),
                       os.path.join(HERE, "m19_m13d_stability_repaired.json"))
        print(f"  m13d K0 {'通过' if not bad else '**触发:' + '; '.join(bad[:3])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
