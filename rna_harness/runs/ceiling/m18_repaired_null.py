#!/usr/bin/env python3
"""M18: hold the matched design fixed and permute the outcome instead.

Declaration: m18_repaired_null_declaration.json, frozen and pushed at ac0bea62 BEFORE this file
existed (the nineteenth done that way).

Since M13 the permutation null has shuffled gc_paired. Matching is calipered on global GC too,
and the two columns are jointly structured, so shuffling the treatment rebuilds the matched set
instead of relabelling it -- 2.2-2.5x more pairs panel-free, 2.5-3.1x more panel-matched. The
statistic was judged against a null computed on a larger, easier design.

Here the treatment columns stay real, the matched set is therefore fixed, and y is permuted
within cell. Under the null that y is independent of gc_paired given the cell, that is exact.

Point estimates must come out bit-identical to M17/M13b. If they move, the repair is in the
wrong place and the run voids.

COST: $0, offline.
"""
from __future__ import annotations

import argparse, collections, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", ".."))); sys.path.insert(0, HERE)
from m13b_gc_placement import (assemble, match, point, cluster_ci, pval, qdelta, shifted_gc,
                               synth, POPS, DELTA, MAX_SMD, MIN_PAIRS, MIN_TGT, NSHIFT)  # noqa: E402

NPERM_MAIN, NPERM_PROBE = 2000, 400
NOISE_DRAWS, LAD_DRAWS, SELF_DRAWS = 12, 8, 40
RUNGS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0)
P_GATE, LAD_SELFCHECK_MAX = 0.005, 0.15


def _D(fn, seed):
    C = json.load(open(os.path.join(HERE, "m13_features.json")))
    E = json.load(open(os.path.join(HERE, "m13b_extra.json")))
    return assemble(C[fn]["rows"], E[fn], seed)


def permute_y(d, rng):
    """the repair: y shuffled within cell, every matching column left real."""
    out = []
    for c in d["cells"]:
        y = c["y"][rng.permutation(len(c["y"]))]
        out.append({**c, "y": y - y.mean()})
    return {**d, "cells": out}


def null_y(d, seed, nperm, treat=None, delta=DELTA, panel=False):
    rng = np.random.default_rng(seed)
    t = treat if treat is not None else [c["gcp"] for c in d["cells"]]
    out = []
    for _ in range(nperm):
        M = match(permute_y(d, rng), t, delta, panel=panel)
        out.append(point(M) if M else 0.0)
    return np.sort(np.array(out))


def smds(d, M):
    sg = float(np.std(np.concatenate([c["gcf"] for c in d["cells"]]))) or 1.0
    return abs(float(M["dgcf"].mean())) / sg, abs(float(M["dpB"].mean())) / (d["sp"] or 1.0)


def arm(d, seed, nperm, panel):
    M = match(d, [c["gcp"] for c in d["cells"]], DELTA, panel=panel)
    if M is None:
        return None
    obs = point(M); null = null_y(d, seed, nperm, panel=panel)
    sg, sp = smds(d, M)
    return {"effect": round(obs, 4), "p_perm_REPAIRED": round(pval(obs, null), 5),
            "null_mean": round(float(null.mean()), 4), "null_sd": round(float(null.std()), 4),
            "z_vs_null_REPAIRED": round((obs - float(null.mean())) / (float(null.std()) or 1), 2),
            "boot_ci_secondary": cluster_ci(M, seed), "npair": M["npair"], "ntgt": M["ntgt"],
            "smd_gcf": round(sg, 4), "smd_pB": round(sp, 4),
            "usable": bool(M["npair"] >= MIN_PAIRS and M["ntgt"] >= MIN_TGT and sg <= MAX_SMD)}


def p3_selfcheck(d, seed, k0, k1):
    """the repair's own type-I check: a permuted-y draw judged by the repaired null."""
    hits, n = 0, 0
    for k in range(k0, k1):
        rng = np.random.default_rng(600000 + 7919 * seed + 131 * k)
        d2 = permute_y(d, rng)
        M = match(d2, [c["gcp"] for c in d2["cells"]], DELTA, panel=False)
        if M is None or M["npair"] < 30:
            continue
        nl = null_y(d2, 950000 + 7919 * seed + 131 * k, NPERM_PROBE)
        hits += int(pval(point(M), nl) <= P_GATE); n += 1
    return {"n": n, "hits": hits}


def p1_noise(d, seed):
    hits, effs = 0, []
    for k in range(NOISE_DRAWS):
        rng = np.random.default_rng(9000 + 97 * seed + k)
        cols = synth(d, 0.0, rng); dl = qdelta(d, cols)
        M = match(d, cols, dl, panel=False)
        if M is None or M["npair"] < 30:
            continue
        nl = null_y(d, 31000 + 97 * seed + k, NPERM_PROBE, treat=cols, delta=dl)
        effs.append(round(point(M), 4)); hits += int(pval(point(M), nl) <= P_GATE)
    return {"draws": NOISE_DRAWS, "hits": hits, "effects": effs}


def p4_shift(d, real, seed):
    rng = np.random.default_rng(4000 + seed); eff, leak = [], []
    for _ in range(NSHIFT):
        cols, _cr = shifted_gc(d, float(rng.uniform(0.10, 0.90)))
        M = match(d, cols, qdelta(d, cols), panel=False)
        if M is None:
            continue
        eff.append(point(M)); leak.append(float(M["dgcp"].mean()))
    if not eff:
        return None
    eff = np.array(eff); q95 = float(np.quantile(eff, .95))
    return {"n_shift": len(eff), "shift_mean": round(float(eff.mean()), 4),
            "shift_q95": round(q95, 4), "real": real, "pass": bool(real > q95),
            "leaked_gcp_gap": round(float(np.mean(leak)), 5)}


def p2_rung(d, delta_pts, seed):
    hits, recov = 0, []
    for k in range(LAD_DRAWS):
        rng = np.random.default_rng(77000 + 997 * seed + 31 * k + int(delta_pts * 100))
        d2 = permute_y(d, rng); cells = []
        for c in d2["cells"]:
            y = c["y"] + (c["gcp"] > np.median(c["gcp"])).astype(float) * delta_pts
            cells.append({**c, "y": y - y.mean()})
        d2 = {**d2, "cells": cells}
        M = match(d2, [c["gcp"] for c in d2["cells"]], DELTA, panel=False)
        if M is None or M["npair"] < 30:
            continue
        recov.append(point(M))
        nl = null_y(d2, 88000 + 997 * seed + 31 * k + int(delta_pts * 100), NPERM_PROBE)
        hits += int(pval(point(M), nl) <= P_GATE)
    n = max(1, len(recov))
    return {"delta_pts": delta_pts, "draws": len(recov), "hit_rate": round(hits / n, 3),
            "recovered_mean": round(float(np.mean(recov)) if recov else 0.0, 4)}


def job(spec):
    fn, seed, what = spec
    d = _D(fn, seed)
    if what == "arms":
        A = arm(d, seed, NPERM_MAIN, panel=False); B = arm(d, seed, NPERM_MAIN, panel=True)
        return fn, "arms", {"A_oracle_free": A, "B_panel_matched": B, "n": d["n"],
                            "ratio": round(A["effect"] / (B["effect"] or 1e-9), 3),
                            "P1": p1_noise(d, seed), "P4": p4_shift(d, A["effect"], seed),
                            "P0": {"effect": round(point(match(d, [np.zeros(len(c["gcp"]))
                                   for c in d["cells"]], DELTA, panel=False) or
                                   {"sums": {0: 0.0}, "cnts": {0: 1}}), 10)}}
    if what.startswith("self:"):
        lo, hi = (int(x) for x in what.split(":")[1].split("-"))
        return fn, what, p3_selfcheck(d, seed, lo, hi)
    return fn, what, p2_rung(d, float(what.split(":")[1]), seed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--out", default=os.path.join(HERE, "m18_repaired_null.json"))
    a = ap.parse_args()
    print("=== M18:钉死配对设计,改打乱 y ===")
    print("  声明 ac0bea62(代码存在之前已推送)。离线 $0。")
    print("  点估计必须与 M17/M13b 逐字相同 —— 修的是零分布,不是估计量。\n")
    OLD = json.load(open(os.path.join(HERE, "m17_oracle_free_VOID_ladder_selfcheck_failed.json")))

    specs = [(fn, a.seed, "arms") for _n, fn in POPS]
    specs += [(fn, a.seed, f"lad:{r}") for _n, fn in POPS for r in RUNGS]
    specs += [(fn, a.seed, f"self:{lo}-{lo+10}") for _n, fn in POPS for lo in (0, 10, 20, 30)]
    from concurrent.futures import ProcessPoolExecutor
    RES = collections.defaultdict(dict)
    print(f"--- 并行 {len(specs)} 任务 / {a.workers} 进程 ---", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for fn, what, r in ex.map(job, specs):
            RES[fn][what] = r; done += 1
            if done % 12 == 0:
                print(f"  {done}/{len(specs)}", flush=True)

    print("\n--- K0:点估计与旧版逐字比对 ---")
    K0bad = []
    for nm, fn in POPS:
        o = OLD["pops"][fn]["arms"]; n_ = RES[fn]["arms"]
        for k in ("A_oracle_free", "B_panel_matched"):
            if o[k]["effect"] != n_[k]["effect"] or o[k]["npair"] != n_[k]["npair"]:
                K0bad.append(f"{nm}/{k}")
        print(f"  {nm:12s} oracle-free {n_['A_oracle_free']['effect']:+7.3f} "
              f"(旧 {o['A_oracle_free']['effect']:+7.3f})  面板配平 "
              f"{n_['B_panel_matched']['effect']:+7.3f} (旧 {o['B_panel_matched']['effect']:+7.3f})"
              f"  {'一致' if not K0bad or nm not in ' '.join(K0bad) else '**不一致**'}")
    print(f"  K0 {'通过 —— 估计量一位没动' if not K0bad else '**触发:' + ','.join(K0bad) + '**'}")

    print("\n--- 主结果:修好的零分布,新旧 z / p 并排 ---")
    print(f"  {'人群':12s} {'效应':>8s} | {'旧z':>6s} {'新z':>6s} {'缩到':>6s} {'新p':>8s} | "
          f"{'面板配平 效应':>12s} {'旧z':>6s} {'新z':>6s} {'新p':>8s}")
    for nm, fn in POPS:
        o = OLD["pops"][fn]["arms"]; v = RES[fn]["arms"]
        A, B = v["A_oracle_free"], v["B_panel_matched"]
        oa, ob = o["A_oracle_free"]["z_vs_null"], o["B_panel_matched"]["z_vs_null"]
        print(f"  {nm:12s} {A['effect']:+8.3f} | {oa:6.1f} {A['z_vs_null_REPAIRED']:6.1f} "
              f"{A['z_vs_null_REPAIRED']/oa:6.2f} {A['p_perm_REPAIRED']:8.5f} | "
              f"{B['effect']:+12.3f} {ob:6.1f} {B['z_vs_null_REPAIRED']:6.1f} "
              f"{B['p_perm_REPAIRED']:8.5f}")

    print("\n--- P3:修理自检(打乱 y 当观测,用修好的零分布判) ---")
    st = {}
    for nm, fn in POPS:
        h = sum(RES[fn][f"self:{lo}-{lo+10}"]["hits"] for lo in (0, 10, 20, 30))
        n = sum(RES[fn][f"self:{lo}-{lo+10}"]["n"] for lo in (0, 10, 20, 30))
        st[fn] = (h, n); print(f"  {nm:12s} {h}/{n} = {h/max(n,1):.4f}   (M17d 旧构造 同项见日志)")
    H = sum(v[0] for v in st.values()); N = sum(v[1] for v in st.values())
    print(f"  合计 {H}/{N} = {H/N:.4f}   名义 0.005   M17d 未修时 13/240 = 0.0542")

    print("\n--- 探针 ---")
    for nm, fn in POPS:
        v = RES[fn]["arms"]
        print(f"  {nm:12s} P0 {v['P0']['effect']:+.8f} | P1 {v['P1']['hits']}/12 | "
              f"P4 真实{v['P4']['real']:+.2f} vs 95分位{v['P4']['shift_q95']:+.2f} "
              f"{'通过' if v['P4']['pass'] else '**不通过**'}")

    print("\n--- P2 梯级(建在零上) ---")
    print(f"  {'人群':12s} " + "  ".join(f"δ={r:<4g}" for r in RUNGS))
    for nm, fn in POPS:
        row = [RES[fn][f"lad:{r}"] for r in RUNGS]
        print(f"  {nm:12s} " + "  ".join(f"{x['hit_rate']:5.2f} " for x in row))

    self0 = max(RES[fn]["lad:0.0"]["hit_rate"] for _n, fn in POPS)
    P0ok = all(RES[fn]["arms"]["P0"]["effect"] == 0.0 for _n, fn in POPS)
    P1ok = sum(RES[fn]["arms"]["P1"]["hits"] for _n, fn in POPS) <= 3
    P3ok = H <= 3
    sig = sum(1 for _n, fn in POPS if RES[fn]["arms"]["A_oracle_free"]["p_perm_REPAIRED"] <= P_GATE
              and RES[fn]["arms"]["A_oracle_free"]["effect"] > 0
              and RES[fn]["arms"]["A_oracle_free"]["usable"])
    sigB = sum(1 for _n, fn in POPS if RES[fn]["arms"]["B_panel_matched"]["p_perm_REPAIRED"] <= P_GATE
               and RES[fn]["arms"]["B_panel_matched"]["effect"] > 0)
    K4 = self0 > LAD_SELFCHECK_MAX
    verdict = ("void_estimates_moved" if K0bad else "void_repair_failed_selfcheck" if not P3ok
               else "void_ladder_selfcheck_failed" if K4
               else "void_probe_failed" if not (P0ok and P1ok)
               else "placement_survives_repaired_null" if sig >= 5
               else "placement_dies_under_repaired_null" if sig <= 2 else "partially_survives")
    print("\n=== 裁定 ===")
    print(f"  K0 {'通过' if not K0bad else '触发'} | P0 {'合格' if P0ok else '不合格'} | "
          f"P1 {sum(RES[fn]['arms']['P1']['hits'] for _n,fn in POPS)}/72 {'合格' if P1ok else '不合格'} | "
          f"P3 修理自检 {H}/{N} {'合格' if P3ok else '**不合格**'} | "
          f"梯级 δ=0 {self0:.2f} {'合格' if not K4 else '**不合格**'}")
    print(f"  oracle-free 显著 {sig}/6   面板配平显著 {sigB}/6")
    print(f"  **裁定:{verdict}**")
    json.dump({"declaration": "m18_repaired_null_declaration.json @ ac0bea62",
               "pops": {fn: RES[fn] for _n, fn in POPS}, "verdict": verdict,
               "K0_estimates_moved": K0bad, "P0_ok": bool(P0ok), "P1_ok": bool(P1ok),
               "P3_selfcheck": {"hits": H, "n": N, "rate": round(H / N, 5),
                                "unrepaired_rate_m17d": 0.0542},
               "ladder_selfcheck_max": self0, "sig_oracle_free": sig, "sig_panel_matched": sigB},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
