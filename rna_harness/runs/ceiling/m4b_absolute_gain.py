#!/usr/bin/env python3
"""M4b: the absolute gain, not the ratio.

`m4_audit.md` section 5 recorded this as an untested item in its own words: M4 reported only the RATIO
(capture rate) and never recorded `model - random` on its own. capture(n) = (E_model - E_rand)/(E_oracle -
E_rand); both parts move with n, and M4's explanation for the ratio falling ("n grows, the oracle gets
stronger") constrains the DENOMINATOR only. The product currency is the numerator.

PRE-RUN AMENDMENT A (in the declaration, before any data): M4's `capture()` voids cells where
oracle == random because the ratio is undefined there. That gate is a property of the RATIO, not of the
quantity -- the absolute gain is perfectly well defined on those cells and equals 0. Copying it would
systematically drop cells whose gain must be zero and bias the primary object UPWARD. M4b keeps them
(primary) and reports the M4-matched subset alongside. The fix makes the primary object SMALLER.

BOUNDARY: this measures the shape of the gain curve. It is NOT the answer to "how big should n be" --
that is the gain curve intersected with the per-candidate cost, and there is no cost model here.

COST: $0, offline.
"""
from __future__ import annotations

import argparse, collections, itertools, json, math, os, random, statistics as st, sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from e1_ensemble_marginals import base_block                                      # noqa: E402
from h1_heldout_ensemble_defect import fit_predict                                # noqa: E402
import h2_heldout_cluster_ci as H                                                 # noqa: E402
from m2b_gt_target_property import fold                                           # noqa: E402
from m4_selection_currency import e_oracle, e_model                               # noqa: E402

POPS = (("样本2(主)", "panel_f1_fresh.json"), ("基准4560", "panel_f1_designs.json"),
        ("样本3", "panel_f1_third.json"), ("第四批", "panel_f1_fourth.json"))
PRIMARY = "panel_f1_fresh.json"
NS = (2, 5, 10, 20, 50)
MIN_CELL = 2
BLOCKS = ("A", "A_G", "A_ed", "GC_only")


def selfcheck(rounds=300, seed=4242, tol=1e-9):
    """V3: the closed form against full enumeration, with ties. Voids the run if it fails."""
    rng = random.Random(seed); worst = 0.0
    for _ in range(rounds):
        m = rng.randint(2, 8)
        n = rng.randint(2, m)
        wet = [rng.choice([0.0, 0.5, 1.0, 2.0, 2.0]) for _ in range(m)]           # ties on purpose
        pred = [rng.choice([-1.0, 0.0, 0.0, 1.0]) for _ in range(m)]              # prediction ties too
        tot = comb(m, n)
        bo = bm = 0.0
        for sub in itertools.combinations(range(m), n):
            bo += max(wet[i] for i in sub) / tot
            best = max(pred[i] for i in sub)
            arg = [i for i in sub if pred[i] == best]
            bm += st.fmean([wet[i] for i in arg]) / tot                            # uniform tie-break
        co = e_oracle(wet, n, m, tot)
        cm, _t = e_model(wet, pred, n, m)
        worst = max(worst, abs(co - bo), abs(cm - bm))
    return worst, worst < tol


def run_pop(nm, keep, M, splits, seed, probe=None):
    """probe: None | 'oracle' (pred := wet) | 'anti' (pred := -wet) | 'noise' (pred := N(0,1))."""
    cells, cols = collections.defaultdict(list), {}
    for r, m in zip(keep, M):
        if "error" in m:
            continue
        rr = dict(r)
        cols[id(rr)] = base_block(r) + m["gc"] + [m["ed_skel"]]
        cells[(r["puzzle"], r["method"])].append(rr)
    rws = [(c, g) for c, g in cells.items() if len(g) >= MIN_CELL]
    nB = len(base_block(keep[0]))
    iA = list(range(nB))
    idx = {"A": iA, "A_G": iA + [nB, nB + 1], "A_ed": iA + [nB + 2], "GC_only": [nB, nB + 1]}
    by_t = collections.defaultdict(list)
    for c, gp in rws:
        by_t[c[0]].append((c, gp))
    tg = sorted(by_t)
    # AG = primary (degenerate cells kept, gain 0); AGm = M4-matched subset; ORC = oracle - rand
    AG = {k: {n: {t: [] for t in tg} for n in NS} for k in BLOCKS}
    AGm = {k: {n: {t: [] for t in tg} for n in NS} for k in BLOCKS}
    ORC = {n: {t: [] for t in tg} for n in NS}
    WSD = []
    rng = random.Random(seed); rngp = random.Random(seed + 991); order = list(tg)
    S = dict(leak_mol=0, leak_tg=0, small=0, degen=0, ties=0, cells=0)
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
                m = len(wet)
                S["cells"] += 1
                if len(wet) >= 2:
                    WSD.append(st.stdev(wet))
                r_ = st.fmean(wet)
                for n in NS:
                    if m < n:
                        S["small"] += 1
                        continue
                    tot = comb(m, n)
                    o = e_oracle(wet, n, m, tot)
                    degen = (o - r_ <= 0)
                    if degen:
                        S["degen"] += 1
                    ORC[n][t].append(o - r_)
                    for k in BLOCKS:
                        if probe == "oracle":
                            pred = wet
                        elif probe == "anti":
                            pred = [-w for w in wet]
                        elif probe == "noise":
                            pred = [rngp.gauss(0, 1) for _ in wet]
                        else:
                            pred = [P[k][id(r)] for r in g]
                        em, ti = e_model(wet, pred, n, m)
                        S["ties"] += ti
                        AG[k][n][t].append(em - r_)                                # amendment A: kept
                        if not degen:
                            AGm[k][n][t].append(em - r_)
    used_by_n = {n: [t for t in tg if len(AG["A"][n][t]) >= H.MIN_SPLITS_PER_TARGET] for n in NS}
    used10 = used_by_n[10]
    print(f"  {nm:14s} 格 {len(rws)} 靶标(n=10) {len(used10)} | 泄漏 分子 {S['leak_mol']}/{splits}"
          f" 靶标 {S['leak_tg']}/{splits} {'✓' if not S['leak_mol'] and not S['leak_tg'] else '✗ 作废'}"
          f" | 格内数<n 跳过 {S['small']},退化格(保留,记 0)**{S['degen']}**,预测并列 {S['ties']}", flush=True)
    print("    每个 n 的可用靶标数:" + ", ".join(f"n={n} {len(used_by_n[n])}" for n in NS), flush=True)
    if len(used10) < H.MIN_TARGETS:
        return {"no_data": True, "n_used": len(used10)}
    out = {"n_used": len(used10), "stats": dict(S), "targets_by_n": {n: len(used_by_n[n]) for n in NS},
           "wet_sd_mean": round(st.fmean(WSD), 4) if WSD else None}               # V6

    def summarise(vec, tag):
        ci, nb = H.boot_ci(vec, seed + 17)
        assert nb == len(vec), "闸门 V5:重抽单位不是靶标"
        mde = 1.96 * st.stdev(vec) / math.sqrt(len(vec)) if len(vec) > 1 else 0.0
        ex = sum(1 for s2 in range(50)
                 if (lambda c: c[0] > 0 or c[1] < 0)(H.boot_ci(vec, 9900 + 13 * s2)[0]))
        return {"tag": tag, "mean": round(st.fmean(vec), 4), "ci": ci, "n": nb, "MDE": round(mde, 4),
                "excl": bool(ci[0] > 0 or ci[1] < 0), "flip": ex,
                "eff_over_mde": round(abs(st.fmean(vec)) / mde, 2) if mde > 0 else 0.0}

    for n in NS:
        un = used_by_n[n]
        if len(un) < H.MIN_TARGETS:
            out[f"skipped@{n}"] = len(un)
            continue
        out[f"ORACLE_minus_rand@{n}"] = summarise([st.fmean(ORC[n][t]) for t in un], f"O-R@{n}")
        for k in BLOCKS:
            out[f"{k}@{n}"] = summarise([st.fmean(AG[k][n][t]) for t in un], f"{k}@{n}")
            um = [t for t in un if AGm[k][n][t]]
            if len(um) >= H.MIN_TARGETS:
                out[f"{k}@{n}__m4set"] = summarise([st.fmean(AGm[k][n][t]) for t in um], f"{k}@{n}m4")
        for k in ("A_G", "A_ed", "GC_only"):
            out[f"{k}_minus_A@{n}"] = summarise(
                [st.fmean(AG[k][n][t]) - st.fmean(AG["A"][n][t]) for t in un], f"{k}-A@{n}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "m4b_absolute_gain.json"))
    a = ap.parse_args()

    print("=== V3 闸门:闭式期望 vs 全枚举(300 组,含并列)===", flush=True)
    worst, ok = selfcheck()
    print(f"  最大绝对差 {worst:.3e}  {'✓ 通过' if ok else '✗ 超过 1e-9 ⇒ 整轮作废'}", flush=True)
    if not ok:
        json.dump({"verdict": "void_closed_form_failed", "worst": worst},
                  open(a.out, "w"), indent=1, ensure_ascii=False)
        return 0

    R, FOLDS = {}, {}
    print("\n=== M4b 主臂:绝对增益 model − random(wet 原单位),退化格保留 ===", flush=True)
    for _j, (nm, fn) in enumerate(POPS):
        keep, M = fold(fn, a.workers); FOLDS[fn] = (keep, M)
        R[fn] = run_pop(nm, keep, M, a.splits, a.seed)
        json.dump({"selfcheck": worst, "main": R}, open(a.out, "w"), indent=1, ensure_ascii=False)

    P = R[PRIMARY]
    print("\n=== 必响探针(第 54 条)===", flush=True)
    keep, M = FOLDS[PRIMARY]
    sp = max(30, a.splits // 4)
    PR = {p: run_pop(f"探针 {p}", keep, M, sp, a.seed, probe=p) for p in ("oracle", "anti", "noise")}
    ns_ok = [n for n in NS if f"A@{n}" in PR["oracle"] and f"ORACLE_minus_rand@{n}" in PR["oracle"]]
    d0 = max(abs(PR["oracle"][f"A@{n}"]["mean"] - PR["oracle"][f"ORACLE_minus_rand@{n}"]["mean"])
             for n in ns_ok) if ns_ok else 1.0
    p0 = d0 < 1e-3                       # 靶标级均值四舍五入到 1e-4,恒等式检验的容差按报出精度定
    p1 = all(PR["anti"][f"A@{n}"]["mean"] <= 0 for n in ns_ok)                     # rule 56: one-sided
    p2 = all(not PR["noise"][f"A@{n}"]["excl"] for n in ns_ok)                     # two-sided: no known dir
    print(f"  P0 oracle 恒等式  最大差 {d0:.2e}  {'✓ 响' if p0 else '✗ 没响'}")
    print(f"  P1 反向(单侧)     " + " ".join(f"n={n}:{PR['anti'][f'A@{n}']['mean']:+.4f}" for n in ns_ok)
          + f"  {'✓ 响' if p1 else '✗ 没响'}")
    print(f"  P2 噪声(双侧)     " + " ".join(f"n={n}:{PR['noise'][f'A@{n}']['mean']:+.4f}" for n in ns_ok)
          + f"  {'✓ 响' if p2 else '✗ 没响'}")
    probes = {"P0_max_dev": d0, "fired": {"P0": p0, "P1": p1, "P2": p2},
              "anti": {n: PR["anti"].get(f"A@{n}") for n in ns_ok},
              "noise": {n: PR["noise"].get(f"A@{n}") for n in ns_ok}}
    if not (p0 and p1 and p2):
        json.dump({"selfcheck": worst, "main": R, "probes": probes,
                   "verdict": "void_probe_did_not_fire"}, open(a.out, "w"), indent=1, ensure_ascii=False)
        print("\n**必响探针没有全响 ⇒ 本次运行作废(第 54 条)。不在本份声明里换探针。**")
        return 0

    print(f"\n=== 主人群:绝对增益 AG(n)(wet 原单位;格内 wet 标准差均值 {P['wet_sd_mean']})===")
    hdr = "  " + f"{'块':<10}" + "".join(f"{'n='+str(n):<26}" for n in NS)
    print(hdr)
    for k in BLOCKS:
        row = f"  {k:<10}"
        for n in NS:
            o = P.get(f"{k}@{n}")
            row += (f"{o['mean']:+.4f}[{o['ci'][0]:+.4f},{o['ci'][1]:+.4f}]".ljust(26)
                    if o else "—".ljust(26))
        print(row)
    row = f"  {'ORACLE−R':<10}"
    for n in NS:
        o = P.get(f"ORACLE_minus_rand@{n}")
        row += (f"{o['mean']:+.4f}".ljust(26) if o else "—".ljust(26))
    print(row)
    print("\n  M4 口径(剔除退化格)对照 —— 两个数都要引:")
    for k in BLOCKS:
        row = f"  {k:<10}"
        for n in NS:
            o = P.get(f"{k}@{n}__m4set")
            row += (f"{o['mean']:+.4f}".ljust(26) if o else "—".ljust(26))
        print(row)

    ags = [(n, P[f"A@{n}"]["mean"]) for n in NS if f"A@{n}" in P]
    mono = all(ags[i][1] <= ags[i + 1][1] + 1e-12 for i in range(len(ags) - 1))
    peak = max(ags, key=lambda x: x[1])[0] if ags else None
    a20 = P.get("A@20")
    K = {"K1": bool(a20 and not a20["excl"]),
         "K2": not mono,
         "K3": sum(1 for _n, fn in POPS[1:]
                   if R.get(fn) and R[fn].get("A@10") and R[fn]["A@10"]["excl"]) < 2,
         "K4": any(P.get(f"GC_only@{n}") and P.get(f"A@{n}")
                   and P[f"GC_only@{n}"]["mean"] >= P[f"A@{n}"]["mean"]
                   and (P[f"GC_only@{n}"]["ci"][0] > P[f"A@{n}"]["ci"][1]
                        or P[f"A@{n}"]["ci"][0] > P[f"GC_only@{n}"]["ci"][1]) for n in NS)}
    print("\n=== 判决条件 ===")
    for k, v in K.items():
        print(f"  {k}: {'**触发**' if v else '未触发'}")
    verdict = "absolute_gain_" + ("monotone_in_n" if mono else f"peaks_at_n={peak}")
    if K["K1"]:
        verdict += "__null_at_n20"
    if K["K3"]:
        verdict += "__underreplicated"
    if K["K4"]:
        verdict += "__gc_only_beats_the_panel_in_absolute_units"
    print(f"\n=== 裁定 ===\n  **{verdict}**")
    print("  边界(声明里冻死的):**本实验测的是增益曲线的形状,不是『n 该开多大』的答案。**"
          "后者 = 增益曲线 ∩ 每条候选的成本,成本模型不在本实验里。")
    json.dump({"selfcheck": worst, "main": R, "probes": probes, "kill": K,
               "monotone": mono, "peak_n": peak, "verdict": verdict},
              open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
