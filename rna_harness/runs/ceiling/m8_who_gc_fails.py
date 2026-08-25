#!/usr/bin/env python3
"""M8: which targets does the GC criterion fail on every batch?

M4c/M4d established that GC_only's per-target capture persists across batches (6/6 pairs, rho +0.4951,
width-matched pure noise -0.0123, reliability 0.9997). That makes the failure set a REAL, STABLE object.
Move two says: split the index set by an invariant into a main part and a boundable remainder.

The mechanism under test is one sentence a single column can refute:
  two GC columns can only rank designs BY THE DIFFERENCE IN THEIR GC. On a target whose designs barely
  differ in gc_cross, the GC criterion has nothing to work with, so it fails -- and because "how much do
  this target's designs differ in GC" is a property of the target and the design process, not of the
  batch, it fails again next batch.

Why this is not a sixth empty family: the five empty families were asked to explain g_t, whose own
cross-population reproducibility was knocked down by R6's placebo. The quantity here was confirmed
stable against a width-matched null yesterday-in-run-time. And the hypothesis is mechanistic, not a
35-column sweep.

BOUNDARY: this judges whether the failure set is characterised by the invariant. It does NOT propose a
better criterion, and it is not a product recommendation.

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
from m4c_error_persistence import cap_cell, boot_ci_pairs, N_SEL                   # noqa: E402

POPS = (("样本2(主)", "panel_f1_fresh.json"), ("基准4560", "panel_f1_designs.json"),
        ("样本3", "panel_f1_third.json"), ("第四批", "panel_f1_fourth.json"))
R_NOISE, MIN_CELL, NBOOT = 12, 2, 2000
REAL_INV = ["spread_gc", "spread_gcfrac", "mean_gcfrac", "n_designs", "target_len"]
ALPHA = 0.05 / len(REAL_INV)                                                      # V8 Bonferroni


def rho_ci(x, y, seed):
    rng = random.Random(seed); n = len(x); vs = []
    for _ in range(NBOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        vs.append(spearman([x[i] for i in idx], [y[i] for i in idx]))
    vs.sort()
    lo = int((ALPHA / 2) * NBOOT); hi = int((1 - ALPHA / 2) * NBOOT) - 1
    return [round(vs[lo], 4), round(vs[hi], 4)], n


def flip_rate(x, y, seed0):
    ex = 0
    for s in range(50):
        c, _n = rho_ci(x, y, seed0 + 13 * s)
        ex += 1 if (c[0] > 0 or c[1] < 0) else 0
    return ex


def run_pop(nm, keep, M, splits, seed, noise_seed):
    cells, cols, seqs = collections.defaultdict(list), {}, {}
    rngn = random.Random(noise_seed)
    for r, m in zip(keep, M):
        if "error" in m:
            continue
        rr = dict(r)
        cols[id(rr)] = base_block(r) + m["gc"] + [rngn.gauss(0, 1.0) for _ in range(R_NOISE)]
        # V1: the invariants are built from design sequence / structure only. `wet` is never read here.
        seqs[id(rr)] = {"gc_cross": m["gc"][0], "gc_frac": m["gc"][1],
                        "len": float(len(r["seq"])) if "seq" in r else float("nan"),
                        "dnoise": rngn.gauss(0, 1.0)}
        cells[(r["puzzle"], r["method"])].append(rr)
    rws = [(c, g) for c, g in cells.items() if len(g) >= MIN_CELL]
    nB = len(base_block(keep[0]))
    idx = {"A": list(range(nB)), "GC_only": [nB, nB + 1]}
    for j in range(R_NOISE):
        idx[f"N{j}"] = list(range(nB)) + [nB + 2 + j]
    keys = list(idx)
    by_t = collections.defaultdict(list)
    for c, gp in rws:
        by_t[c[0]].append((c, gp))
    tg = sorted(by_t)
    per = {k: {t: [] for t in tg} for k in keys}
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
    # invariants: per cell, then averaged to target. Never touches wet (V1).
    INV = {nm_: {} for nm_ in REAL_INV + ["spread_noise"]}
    for t in used:
        sg, sf, mf, nd, tl, sn = [], [], [], [], [], []
        for _c, g in by_t[t]:
            v = [seqs[id(r)] for r in g]
            if len(v) >= 2:
                sg.append(st.stdev([q["gc_cross"] for q in v]))
                sf.append(st.stdev([q["gc_frac"] for q in v]))
                sn.append(st.stdev([q["dnoise"] for q in v]))
            mf.append(st.fmean([q["gc_frac"] for q in v]))
            nd.append(float(len(v)))
            tl.append(st.fmean([q["len"] for q in v]))
        INV["spread_gc"][t] = st.fmean(sg) if sg else 0.0
        INV["spread_gcfrac"][t] = st.fmean(sf) if sf else 0.0
        INV["spread_noise"][t] = st.fmean(sn) if sn else 0.0
        INV["mean_gcfrac"][t] = st.fmean(mf)
        INV["n_designs"][t] = st.fmean(nd)
        INV["target_len"][t] = st.fmean(tl)
    print(f"  {nm:14s} 格 {len(rws)} 靶标 {len(used)} | 泄漏 分子 {S['leak_mol']}/{splits}"
          f" 靶标 {S['leak_tg']}/{splits} {'✓' if not S['leak_mol'] and not S['leak_tg'] else '✗ 作废'}"
          f" | 格内数<{N_SEL} 跳过 {S['small']}", flush=True)
    vec = {k: {t: st.fmean(per[k][t]) for t in used} for k in keys}
    # 12 target-level noise invariants (distinct from the design-level placebo)
    rt = random.Random(noise_seed + 313)
    for j in range(R_NOISE):
        INV[f"TN{j}"] = {t: rt.gauss(0, 1) for t in used}
    return vec, used, INV


def judge_inv(NAMES, VECS, USEDS, INVS, ykey, ikey, seed):
    rows, rep = [], 0
    for i in range(len(NAMES)):
        u = sorted(USEDS[i])
        x = [INVS[i][ikey][t] for t in u]; y = [VECS[i][ykey][t] for t in u]
        if len(set(x)) < 3 or any(v != v for v in x) or any(v != v for v in y):
            rows.append(None); continue   # 常数列或 NaN(某人群缺 seq 时 target_len 会是 NaN)
        r = spearman(x, y); ci, n = rho_ci(x, y, seed + 7 * i)
        fl = flip_rate(x, y, seed + 900 + 7 * i)
        ex = ci[0] > 0 or ci[1] < 0
        rep += 1 if ci[0] > 0 else 0                                              # V5: same side (positive)
        rows.append({"rho": round(r, 4), "ci": ci, "n": n, "excl": bool(ex), "flip": fl})
    return rows, rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "m8_who_gc_fails.json"))
    a = ap.parse_args()
    NAMES, VECS, USEDS, INVS = [], [], [], []
    print("=== M8:GC 判据每批都失败的那批靶标,是哪些 ===", flush=True)
    for j, (nm, fn) in enumerate(POPS):
        keep, M = build(fn, a.workers)
        v, u, inv = run_pop(nm, keep, M, a.splits, a.seed, 9000 + 151 * j)
        NAMES.append(nm); VECS.append(v); USEDS.append(set(u)); INVS.append(inv)

    print("\n=== 必响探针(第 54 条)===")
    core = sorted(set.intersection(*USEDS))
    VD = [{"GC_only": V["GC_only"]} for V in VECS]
    obs, _ci, _p = boot_ci_pairs(core, VD, "GC_only", a.seed + 17)
    d0 = abs(st.fmean(obs) - 0.4951)
    p0 = d0 < 0.02
    print(f"  P0 与 M4d 同一个量  本次六对均值 {st.fmean(obs):+.4f} vs M4d +0.4951,差 {d0:.4f}"
          f"  {'✓ 响' if p0 else '✗ 没响'}")
    sds = [st.stdev([INVS[i]["spread_gc"][t] for t in sorted(USEDS[i])]) for i in range(len(NAMES))]
    p1 = all(s > 0 for s in sds)
    print(f"  P1 spread_gc 有分辨力  四人群标准差 {[round(s,3) for s in sds]}  {'✓ 响' if p1 else '✗ 没响'}")
    p2 = True   # V1 asserted structurally: the invariant dict is built from gc/len/noise only
    print(f"  P2 不变量计算路径不接触 wet(V1 结构性断言)  {'✓ 响' if p2 else '✗ 没响'}")
    probes = {"P0_dev_from_m4d": round(d0, 4), "P1_sds": [round(s, 4) for s in sds],
              "fired": {"P0": p0, "P1": p1, "P2": p2}}
    if not (p0 and p1 and p2):
        json.dump({"probes": probes, "verdict": "void_probe_did_not_fire"},
                  open(a.out, "w"), indent=1, ensure_ascii=False)
        print("\n**必响探针没有全响 ⇒ 本次运行作废(第 54 条)。**"); return 0

    print(f"\n=== 主臂:rho(capture_t^GC, 不变量),逐人群,Bonferroni {ALPHA:.4f} ===")
    R = {}
    print(f"  {'不变量':<18}" + "".join(f"{n:<20}" for n in NAMES) + "复现")
    for ik in REAL_INV + ["spread_noise"]:
        rows, rep = judge_inv(NAMES, VECS, USEDS, INVS, "GC_only", ik, a.seed)
        R[ik] = {"rows": rows, "rep": rep}
        line = f"  {ik:<18}"
        for r in rows:
            line += ("—".ljust(20) if r is None else
                     f"{r['rho']:+.3f}{'*' if r['ci'][0]>0 else ' '}[{r['ci'][0]:+.2f},{r['ci'][1]:+.2f}]".ljust(20))
        print(line + f" **{rep}/4**")

    print(f"\n=== 归属臂:同一分析跑在面板 capture_t^A 上 ===")
    RA = {}
    for ik in REAL_INV:
        rows, rep = judge_inv(NAMES, VECS, USEDS, INVS, "A", ik, a.seed)
        RA[ik] = {"rows": rows, "rep": rep}
        print(f"  {ik:<18}" + "".join(("—".ljust(20) if r is None else f"{r['rho']:+.3f}".ljust(20)) for r in rows)
              + f" **{rep}/4**")

    print(f"\n=== 12 根靶标级噪声不变量(第 46/52/53 条)===")
    nrep = []
    for j in range(R_NOISE):
        _rows, rep = judge_inv(NAMES, VECS, USEDS, INVS, "GC_only", f"TN{j}", a.seed)
        nrep.append(rep)
    print(f"  逐根的同侧复现计数:{nrep}  **>=3/4 的根数 {sum(1 for x in nrep if x>=3)}/{R_NOISE}**")

    main_rep = R["spread_gc"]["rep"]
    gm = st.fmean([r["rho"] for r in R["spread_gc"]["rows"] if r])
    am = st.fmean([r["rho"] for r in RA["spread_gc"]["rows"] if r])
    K = {"K0": sum(1 for x in nrep if x >= 3) >= 2,
         "K1": main_rep < 3,
         "K2": bool(RA["spread_gc"]["rep"] >= 3 and abs(am) >= 0.5 * abs(gm)),
         "K3": R["spread_noise"]["rep"] >= 3,
         "K4": R["mean_gcfrac"]["rep"] > main_rep}
    print("\n=== 判决条件 ===")
    for k, v in K.items():
        print(f"  {k}: {'**触发**' if v else '未触发'}")
    if K["K0"]:
        verdict = "void_instrument_manufactures_relation"
    elif K["K3"]:
        verdict = "void_any_within_cell_spread_would_do"
    elif K["K1"]:
        verdict = "spread_does_not_explain_the_failure_set"
    else:
        verdict = f"gc_spread_characterises_the_failure_set__{main_rep}_of_4"
        if K["K2"]:
            verdict += "__not_gc_specific"
        if K["K4"]:
            verdict += "__level_not_spread"
    print(f"\n=== 裁定 ===\n  **{verdict}**")
    print("  边界(声明里冻死的):**本实验判失败集是不是由这个不变量刻画的,不判怎么修。**"
          "即使成立,它给出的也只是一个可界的余项,不是一个更好的判据。**不许读成产品建议。**")
    json.dump({"names": NAMES, "GC": R, "A": RA, "noise_rep": nrep, "probes": probes,
               "kill": K, "verdict": verdict, "alpha": ALPHA},
              open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
