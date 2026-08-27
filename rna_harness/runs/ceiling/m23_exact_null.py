#!/usr/bin/env python3
"""M23: which reference distribution is the right one for a calipered matched design?

Declaration m23_exact_null_declaration.json, frozen and pushed at e8d4907f BEFORE this file
existed. Offline, $0.

Proposition 1 as written says "under the null y perp t within cell". An external review pointed
out that this is too weak: the statistic depends on a matched set built jointly from
(t, gamma, pi), and gamma and pi predict y, so permuting y within cell breaks the y-gamma
relation as well and the test is not exact. This measures four reference distributions against
two data-generating processes whose true effect is zero.

  N1  permute the treatment column                 -- the one rule 80 forbids
  N2  permute y within cell                        -- what the paper currently uses
  N3  Freedman-Lane: regress y on (gamma, pi) within cell, permute the residuals, add the
      fitted values back                            -- preserves the covariate-outcome relation
  N4  permute y within (gamma, pi) caliper blocks   -- the non-parametric analogue of N3

  DGP1  y permuted within cell        -- y independent of EVERYTHING; N2 is exact here by
                                        construction, and this is the regime the committed
                                        13/240 and 1/240 were measured in
  DGP2  t permuted within gamma blocks -- y keeps its real relation to gamma and pi, and only
                                        the t-y relation is cut. This is the regime the review
                                        is about, and it matches none of the four nulls'
                                        constructions.

Point estimates and matched-set sizes must be identical across the four nulls: a null changes
what the statistic is compared against, never the statistic.
"""
from __future__ import annotations

import argparse, collections, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", ".."))); sys.path.insert(0, HERE)
from m13b_gc_placement import (assemble, match, point, perm_cols, pval, POPS, DELTA,
                               CAL_GC, CAL_P)                                   # noqa: E402

NPERM, DRAWS, P_GATE = 400, 40, 0.005
NULLS = ("N1_permute_t", "N2_permute_y_cell", "N3_freedman_lane", "N4_blocked")


def _D(fn, seed):
    C = json.load(open(os.path.join(HERE, "m13_features.json")))
    E = json.load(open(os.path.join(HERE, "m13b_extra.json")))
    return assemble(C[fn]["rows"], E[fn], seed)


# ---------------------------------------------------------------- the four reference draws
def _cells(d, ys):
    return {**d, "cells": [{**c, "y": y - y.mean()} for c, y in zip(d["cells"], ys)]}


def draw_N2(d, rng):
    return _cells(d, [c["y"][rng.permutation(len(c["y"]))] for c in d["cells"]])


def draw_N3(d, rng):
    """Freedman-Lane. The working model is linear in (gamma, pi); anything non-linear stays in
    the residual and is exchanged, exactly as N2 exchanges it."""
    out = []
    for c in d["cells"]:
        n = len(c["y"])
        X = np.column_stack([np.ones(n), c["gcf"], c["pB"]])
        b, *_ = np.linalg.lstsq(X, c["y"], rcond=None)
        fit = X @ b
        out.append(fit + (c["y"] - fit)[rng.permutation(n)])
    return _cells(d, out)


def _blocks(c, sp):
    gb = np.floor(c["gcf"] / CAL_GC).astype(np.int64)
    pb = np.floor(c["pB"] / (CAL_P * (sp or 1.0))).astype(np.int64)
    return gb * 1000003 + pb


def draw_N4(d, rng):
    out = []
    for c in d["cells"]:
        key = _blocks(c, d["sp"]); y = c["y"].copy()
        for k in np.unique(key):
            ix = np.nonzero(key == k)[0]
            if len(ix) > 1:
                y[ix] = y[ix[rng.permutation(len(ix))]]
        out.append(y)
    return _cells(d, out)


def block_sizes(d):
    s = []
    for c in d["cells"]:
        key = _blocks(c, d["sp"])
        s += list(collections.Counter(key.tolist()).values())
    return s


def null_dist(d, kind, seed, treat, nperm=NPERM):
    rng = np.random.default_rng(seed); out = []
    for _ in range(nperm):
        if kind == "N1_permute_t":
            M = match(d, perm_cols(d, rng), DELTA, panel=False)
        else:
            dd = {"N2_permute_y_cell": draw_N2, "N3_freedman_lane": draw_N3,
                  "N4_blocked": draw_N4}[kind](d, rng)
            M = match(dd, treat, DELTA, panel=False)
        out.append(point(M) if M else 0.0)
    return np.sort(np.array(out))


# ---------------------------------------------------------------- the two zero-effect DGPs
def dgp1(d, rng):
    """y permuted within cell: y independent of every covariate."""
    return draw_N2(d, rng)


def dgp2(d, rng):
    """t permuted within gamma blocks: y keeps its real relation to gamma and pi; t does not."""
    cells = []
    for c in d["cells"]:
        gb = np.floor(c["gcf"] / CAL_GC).astype(np.int64)
        t = c["gcp"].copy()
        for k in np.unique(gb):
            ix = np.nonzero(gb == k)[0]
            if len(ix) > 1:
                t[ix] = t[ix[rng.permutation(len(ix))]]
        cells.append({**c, "gcp": t})
    return {**d, "cells": cells}


def job(spec):
    fn, seed, dgp, lo, hi = spec
    d = _D(fn, seed)
    acc = {k: {"n": 0, "hits": 0, "p": [], "nullsd": []} for k in NULLS}
    npairs, effs = set(), []
    for k in range(lo, hi):
        rng = np.random.default_rng(600000 + 7919 * seed + 131 * k + (1 if dgp == "DGP1" else 2))
        d0 = (dgp1 if dgp == "DGP1" else dgp2)(d, rng)
        treat = [c["gcp"] for c in d0["cells"]]
        M = match(d0, treat, DELTA, panel=False)
        if M is None or M["npair"] < 30:
            continue
        obs = point(M); npairs.add(M["npair"]); effs.append(round(obs, 4))
        for kind in NULLS:
            nl = null_dist(d0, kind, 900000 + 131 * k + hash(kind) % 9973, treat)
            p = pval(obs, nl)
            s = acc[kind]
            s["n"] += 1; s["hits"] += int(p <= P_GATE); s["p"].append(round(p, 5))
            s["nullsd"].append(round(float(nl.std()), 5))
    return fn, dgp, acc, effs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(HERE, "m23_exact_null.json"))
    a = ap.parse_args()
    print("=== M23:哪一个参照分布才是对的 ===")
    print(f"  声明 e8d4907f。四个零分布 × 两个 δ=0 数据生成过程 × {DRAWS} 抽样 × {NPERM} 置换。")
    print(f"  名义 {P_GATE}。离线 $0。\n")

    d0 = _D(POPS[0][1], a.seed)
    bs = block_sizes(d0)
    med = float(np.median(bs))
    print(f"=== K4 检查:N4 的块大小(主人群)中位 {med:.1f},均值 {np.mean(bs):.2f},"
          f"最大 {max(bs)},块数 {len(bs)}")
    print(f"  {'块中位 >= 3,N4 有置换自由度' if med >= 3 else '**块中位 < 3 ⇒ N4 功率不足,记 blocked_underpowered**'}\n")

    CH = [(0, 10), (10, 20), (20, 30), (30, 40)]
    specs = [(fn, a.seed, g, lo, hi) for _n, fn in POPS for g in ("DGP1", "DGP2")
             for lo, hi in CH]
    from concurrent.futures import ProcessPoolExecutor
    ACC = collections.defaultdict(lambda: collections.defaultdict(
        lambda: {k: {"n": 0, "hits": 0, "p": [], "nullsd": []} for k in NULLS}))
    EFF = collections.defaultdict(lambda: collections.defaultdict(list))
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for fn, dgp, acc, effs in ex.map(job, specs):
            for k in NULLS:
                t = ACC[fn][dgp][k]
                for f in ("n", "hits"):
                    t[f] += acc[k][f]
                t["p"] += acc[k]["p"]; t["nullsd"] += acc[k]["nullsd"]
            EFF[fn][dgp] += effs

    OUT = {"nominal": P_GATE, "draws": DRAWS, "nperm": NPERM,
           "N4_block_median": med, "N4_underpowered": bool(med < 3)}
    for dgp in ("DGP1", "DGP2"):
        print(f"=== {dgp} " + ("(y 与一切协变量独立)" if dgp == "DGP1"
                               else "(y 保留与 γ、π 的真实关系,只切断 t)") + " ===")
        print(f"  {'人群':12s} " + " ".join(f"{k.split('_',1)[0]:>13s}" for k in NULLS))
        tot = {k: [0, 0] for k in NULLS}
        for nm, fn in POPS:
            row = []
            for k in NULLS:
                s = ACC[fn][dgp][k]
                tot[k][0] += s["hits"]; tot[k][1] += s["n"]
                row.append(f"{s['hits']:3d}/{s['n']:<3d}={s['hits']/max(s['n'],1):5.3f}")
            print(f"  {nm:12s} " + " ".join(f"{x:>13s}" for x in row))
        print(f"  {'合计':12s} " + " ".join(
            f"{tot[k][0]:3d}/{tot[k][1]:<3d}={tot[k][0]/max(tot[k][1],1):5.3f}" for k in NULLS))
        sds = {k: float(np.mean(sum((ACC[fn][dgp][k]["nullsd"] for _n, fn in POPS), [])))
               for k in NULLS}
        print("  零分布 SD 均值 " + "  ".join(f"{k.split('_',1)[0]}:{v:.4f}" for k, v in sds.items()))
        OUT[dgp] = {"total": {k: tot[k] for k in NULLS},
                    "rate": {k: round(tot[k][0] / max(tot[k][1], 1), 4) for k in NULLS},
                    "null_sd_mean": {k: round(v, 5) for k, v in sds.items()},
                    "per_pop": {fn: {k: {"n": ACC[fn][dgp][k]["n"], "hits": ACC[fn][dgp][k]["hits"]}
                                     for k in NULLS} for _n, fn in POPS}}
        print()

    r2 = OUT["DGP2"]["rate"]; s2 = OUT["DGP2"]["null_sd_mean"]
    OUT["staked"] = {
        "1_N3_nullsd_le_N2": bool(s2["N3_freedman_lane"] <= s2["N2_permute_y_cell"]),
        "2_N3_rate_in_0002_0010": bool(0.002 <= r2["N3_freedman_lane"] <= 0.010),
        "3_N2_conservative_under_DGP2": bool(r2["N2_permute_y_cell"] <= r2["N3_freedman_lane"]),
        "4_N4_agrees_with_N3": bool(abs(r2["N4_blocked"] - r2["N3_freedman_lane"]) <= 0.006),
        "5_N1_inflated": bool(r2["N1_permute_t"] > 0.010),
    }
    if r2["N3_freedman_lane"] > 0.010:
        v = "freedman_lane_anticonservative__keep_N2_and_state_it_is_conservative"
    elif r2["N3_freedman_lane"] < 0.002:
        v = "freedman_lane_still_conservative__report_both"
    else:
        v = "freedman_lane_is_the_exact_reference__N2_is_conservative"
    OUT["verdict"] = v
    json.dump(OUT, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"=== 裁定: {v} ===")
    print("押注逐条: " + json.dumps(OUT["staked"], ensure_ascii=False))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
