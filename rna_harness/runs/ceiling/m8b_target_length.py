#!/usr/bin/env python3
"""M8b: do long targets starve the GC criterion?

M8's most consistent row was target_len against GC capture -- -0.3139, -0.2781, -0.2810, -0.3531, all
four populations negative -- and my counter recorded it as 0/4 because I had hard-coded "same side" as
"positive" (rule 60). This run takes that one quantity, freezes its direction as NEGATIVE, drops the
Bonferroni penalty that widened M8's intervals to +-0.34, and -- the part that matters -- puts the
question to the fifth and sixth batches, which have never been touched on capture_t.

The four core populations are the DISCOVERY SAMPLE, not a replication: the direction came from looking
at them. Only the blind pair tests the pre-registered direction. Labelled `discovery` in the verdict.

Why the blind pair may survive here when it died in M3c: M3c's exact-GC-matched construction spends
86-93% of all pairs, leaving the newer batches under the per-split pair floor. capture@10 matches
nothing; it only needs cells holding at least 10 designs.

COST: $0, offline.
"""
from __future__ import annotations

import argparse, collections, json, math, os, random, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from e1_ensemble_marginals import base_block                                      # noqa: E402
from h1_heldout_ensemble_defect import fit_predict                                # noqa: E402
import h2_heldout_cluster_ci as H                                                 # noqa: E402
from m1c_skeleton_bookkeeping import build                                        # noqa: E402
from m2_invariant_split import spearman                                           # noqa: E402
from m4c_error_persistence import cap_cell, N_SEL                                  # noqa: E402

CORE = (("样本2(主)", "panel_f1_fresh.json"), ("基准4560", "panel_f1_designs.json"),
        ("样本3", "panel_f1_third.json"), ("第四批", "panel_f1_fourth.json"))
BLIND = (("第五批(盲)", "panel_f1_fifth.json"), ("第六批(盲)", "panel_f1_sixth.json"))
POPS = CORE + BLIND
R_NOISE, MIN_CELL, NBOOT = 12, 2, 2000
M8_REF = {"panel_f1_fresh.json": -0.3139, "panel_f1_designs.json": -0.2781,
          "panel_f1_third.json": -0.2810, "panel_f1_fourth.json": -0.3531}


def rho_ci(x, y, seed):
    """95% -- single hypothesis, no Bonferroni (that is what widened M8 to +-0.34)."""
    rng = random.Random(seed); n = len(x); vs = []
    for _ in range(NBOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        vs.append(spearman([x[i] for i in idx], [y[i] for i in idx]))
    vs.sort()
    return [round(vs[int(0.025 * NBOOT)], 4), round(vs[int(0.975 * NBOOT)], 4)], n


def run_pop(nm, keep, M, splits, seed, noise_seed):
    cells, cols, meta = collections.defaultdict(list), {}, {}
    rngn = random.Random(noise_seed)
    for r, m in zip(keep, M):
        if "error" in m:
            continue
        rr = dict(r)
        cols[id(rr)] = base_block(r) + m["gc"] + [rngn.gauss(0, 1.0) for _ in range(R_NOISE)]
        meta[id(rr)] = float(len(r["seq"]))                                       # V1: structure only
        cells[(r["puzzle"], r["method"])].append(rr)
    rws = [(c, g) for c, g in cells.items() if len(g) >= MIN_CELL]
    if not rws:
        print(f"  {nm:14s} 无可用格 -> 作废", flush=True); return None
    nB = len(base_block(keep[0]))
    idx = {"A": list(range(nB)), "GC_only": [nB, nB + 1]}
    keys = list(idx)
    by_t = collections.defaultdict(list)
    for c, gp in rws:
        by_t[c[0]].append((c, gp))
    tg = sorted(by_t)
    per = {k: {t: [] for t in tg} for k in keys}
    wsd = {t: [] for t in tg}
    rng = random.Random(seed); order = list(tg)
    S = dict(leak_mol=0, leak_tg=0, small=0)
    for _s in range(splits):
        rng.shuffle(order)
        cut = max(1, int(0.7 * len(order)))
        tr, te = set(order[:cut]), set(order[cut:])
        TR = [x for t in tr for x in by_t[t]]; TE = [x for t in te for x in by_t[t]]
        if not TR or not TE:
            continue
        if {id(r) for _c, g in TR for r in g} & {id(r) for _c, g in TE for r in g}:
            S["leak_mol"] += 1
        if {c[0] for c, _g in TR} & {c[0] for c, _g in TE}:
            S["leak_tg"] += 1
        P = {k: fit_predict(TR, TE, cols, ix) for k, ix in idx.items()}
        for t in te:
            for c, g in by_t[t]:
                wet = [r["wet"] for r in g]
                if len(wet) < N_SEL:
                    S["small"] += 1; continue
                for k in keys:
                    v = cap_cell(wet, [P[k][id(r)] for r in g], N_SEL)
                    if v is not None:
                        per[k][t].append(v)
    used = [t for t in tg if len(per["A"][t]) >= H.MIN_SPLITS_PER_TARGET]
    print(f"  {nm:14s} 格 {len(rws)} 靶标 {len(used)} | 泄漏 分子 {S['leak_mol']}/{splits}"
          f" 靶标 {S['leak_tg']}/{splits} {'✓' if not S['leak_mol'] and not S['leak_tg'] else '✗'}"
          f" | 格内数<{N_SEL} 跳过 {S['small']}", flush=True)
    if len(used) < 20:                                                            # V7
        print(f"    ↳ 有效靶标 {len(used)} < 20 ⇒ **该人群作废,不进任何分母**", flush=True)
        return None
    for t in used:
        for _c, g in by_t[t]:
            w = [r["wet"] for r in g]
            if len(w) >= 2:
                wsd[t].append(st.stdev(w))
    L = {t: st.fmean([meta[id(r)] for _c, g in by_t[t] for r in g]) for t in used}
    ND = {t: st.fmean([float(len(g)) for _c, g in by_t[t]]) for t in used}
    WS = {t: (st.fmean(wsd[t]) if wsd[t] else 0.0) for t in used}
    rt = random.Random(noise_seed + 313)
    NZ = {f"TN{j}": {t: rt.gauss(0, 1) for t in used} for j in range(R_NOISE)}
    rp = random.Random(noise_seed + 707)
    shuffled = list(L.values()); rp.shuffle(shuffled)
    PL = {t: v for t, v in zip(used, shuffled)}                                   # placebo: permuted length
    return {"vec": {k: {t: st.fmean(per[k][t]) for t in used} for k in keys},
            "used": used, "len": L, "nd": ND, "wsd": WS, "noise": NZ, "plac": PL}


def one(D, ykey, xmap, seed):
    u = D["used"]
    x = [xmap[t] for t in u]; y = [D["vec"][ykey][t] for t in u]
    if len(set(x)) < 3:
        return None
    r = spearman(x, y); ci, n = rho_ci(x, y, seed)
    fl = sum(1 for s in range(50) if (lambda c: c[0] > 0 or c[1] < 0)(rho_ci(x, y, seed + 991 + 13 * s)[0]))
    return {"rho": round(r, 4), "ci": ci, "n": n, "flip": fl,
            "neg": bool(ci[1] < 0)}                                               # V5: direction is NEGATIVE


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "m8b_target_length.json"))
    a = ap.parse_args()
    D, NM, FN = {}, {}, {}
    print("=== M8b:长靶标是不是饿死了 GC 判据(四人群=发现样本,第五/六批=盲测)===", flush=True)
    for j, (nm, fn) in enumerate(POPS):
        keep, M = build(fn, a.workers)
        d = run_pop(nm, keep, M, a.splits, a.seed, 9500 + 157 * j)
        if d:
            D[fn] = d; NM[fn] = nm
    live_core = [fn for _n, fn in CORE if fn in D]
    live_blind = [fn for _n, fn in BLIND if fn in D]

    print("\n=== 必响探针(第 54 条)===")
    dev = []
    for fn in live_core:
        r = one(D[fn], "GC_only", D[fn]["len"], a.seed + 17)
        dev.append(abs(r["rho"] - M8_REF[fn]))
    p2 = bool(dev) and max(dev) < 0.02
    print(f"  P2 逐字复现 M8 的发现样本  最大差 {max(dev):.4f}  {'✓ 响' if p2 else '✗ 没响'}")
    sds = [st.stdev(list(D[fn]["len"].values())) for fn in D]
    p1 = all(s > 0 for s in sds)
    print(f"  P1 长度有分辨力  各人群标准差 {[round(s,1) for s in sds]}  {'✓ 响' if p1 else '✗ 没响'}")
    lv = {fn: st.fmean(list(D[fn]["vec"]["GC_only"].values())) for fn in live_core}
    p0 = True
    print(f"  P0 GC capture 水平(四核心) {[round(v,4) for v in lv.values()]}  ✓ 响")
    probes = {"P2_max_dev": round(max(dev), 4) if dev else None, "fired": {"P0": p0, "P1": p1, "P2": p2}}
    if not (p0 and p1 and p2):
        json.dump({"probes": probes, "verdict": "void_probe_did_not_fire"},
                  open(a.out, "w"), indent=1, ensure_ascii=False)
        print("\n**必响探针没有全响 ⇒ 本次运行作废(第 54 条)。**"); return 0

    print("\n=== 主表:rho(target_len, capture^GC),95% 区间,方向预注册为【负】 ===")
    print(f"  {'人群':<16}{'rho':>9}{'95%':>24}{'翻转':>8}{'上界<0':>9}{'标注':>10}")
    R = {}
    for fn in D:
        r = one(D[fn], "GC_only", D[fn]["len"], a.seed + 17); R[fn] = r
        tag = "discovery" if fn in live_core else "**BLIND**"
        print(f"  {NM[fn]:<16}{r['rho']:+9.4f}   [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}]{r['flip']:>6}/50"
              f"{'✓' if r['neg'] else '✗':>9}{tag:>10}")
    disc = sum(1 for fn in live_core if R[fn]["neg"])
    blind_neg = sum(1 for fn in live_blind if R[fn]["neg"])

    print("\n=== 对照臂 ===")
    RA, RP, RN = {}, {}, []
    for fn in D:
        RA[fn] = one(D[fn], "A", D[fn]["len"], a.seed + 17)
        RP[fn] = one(D[fn], "GC_only", D[fn]["plac"], a.seed + 17)
    print(f"  {'人群':<16}{'面板 A':>22}{'安慰剂(打乱长度)':>26}")
    for fn in D:
        print(f"  {NM[fn]:<16}{RA[fn]['rho']:+.4f} {'负' if RA[fn]['neg'] else '  '}"
              f"          {RP[fn]['rho']:+.4f} {'负' if RP[fn]['neg'] else '  '}")
    nfire = 0
    for j in range(R_NOISE):
        hits = sum(1 for fn in live_blind if one(D[fn], "GC_only", D[fn]["noise"][f"TN{j}"], a.seed + 17)["neg"])
        RN.append(hits); nfire += 1 if hits >= max(1, len(live_blind)) else 0
    print(f"  12 根靶标级噪声在盲人群上的负向命中:{RN}  **全中的根数 {nfire}/12**")

    print("\n=== 混淆报出(不进判据)===")
    for fn in D:
        cl = spearman([D[fn]["len"][t] for t in D[fn]["used"]], [D[fn]["nd"][t] for t in D[fn]["used"]])
        cw = spearman([D[fn]["len"][t] for t in D[fn]["used"]], [D[fn]["wsd"][t] for t in D[fn]["used"]])
        print(f"  {NM[fn]:<16}rho(len, n_designs) {cl:+.4f}   rho(len, 格内 wet 标准差) {cw:+.4f}")

    K = {"K0": nfire >= 2,
         "K1": bool(live_blind) and blind_neg == 0,
         "K2": bool(live_blind) and sum(1 for fn in live_blind if RA[fn]["neg"]) >= 1
               and abs(st.fmean([RA[fn]["rho"] for fn in live_blind])) >= 0.5 * abs(st.fmean([R[fn]["rho"] for fn in live_blind])),
         "K3": sum(1 for fn in live_blind if RP[fn]["neg"]) >= 1,
         "K4": not live_blind}
    print("\n=== 判决条件 ===")
    for k, v in K.items():
        print(f"  {k}: {'**触发**' if v else '未触发'}")
    if K["K0"]:
        verdict = "void_instrument_manufactures_negative_relation"
    elif K["K3"]:
        verdict = "void_any_target_level_column_would_do"
    elif K["K4"]:
        verdict = f"discovery_{disc}_of_{len(live_core)}__blind_unavailable"
    elif K["K1"]:
        verdict = f"discovery_{disc}_of_{len(live_core)}__blind_null"
    else:
        verdict = f"long_targets_starve_gc__blind_{blind_neg}_of_{len(live_blind)}__discovery_{disc}_of_{len(live_core)}"
        if K["K2"]:
            verdict += "__not_gc_specific"
    print(f"\n=== 裁定 ===\n  **{verdict}**")
    print(f"  发现样本 {disc}/{len(live_core)}(**不是复现**);盲测 {blind_neg}/{len(live_blind)}")
    print("  本实验**不解释**为什么长靶标会这样(设计空间/信息密度/湿实验噪声三者分不开),见审计边界节。")
    json.dump({"names": NM, "GC": R, "A": RA, "placebo": RP, "noise_blind_hits": RN,
               "discovery": disc, "blind_neg": blind_neg, "n_core": len(live_core),
               "n_blind": len(live_blind), "probes": probes, "kill": K, "verdict": verdict},
              open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
