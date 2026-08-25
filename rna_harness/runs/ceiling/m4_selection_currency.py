#!/usr/bin/env python3
"""M4: Deng move four -- long horizon = short horizon + control of error propagation.

Pre-registered in m4_selection_currency_declaration.json (amended before any deciding run).

THE MISMATCH THIS EXISTS TO MEASURE. Every experiment on this line scores a criterion by net R^2 or by
pairwise ranking accuracy. Both are TOTAL-ORDER measures. The product does top-1-of-n: it keeps the single
best candidate and throws the rest away. A column that adds +0.03 to pairwise accuracy can change zero top-1
choices, and the reverse is equally possible. None of the fifteen experiments has measured this.

THE CURRENCY. Per held-out cell and per n, capture rate = (model - random) / (oracle - random): the fraction
of the ATTAINABLE gain the criterion actually collects. It is dimensionless inside the cell, so target
difficulty cancels.

EXACT, NOT SAMPLED (amendment A). The declaration said draw n designs 200 times; this computes the same
expectation in closed form from hypergeometric combinatorics, which is that Monte Carlo in the limit. It
removes sampling noise -- a dangerous direction -- so the justification is purely that it is the same
estimand computed exactly.

THE HALF THIS DOES NOT COVER. Move four's second half is error propagation across rounds: on one target the
criterion's mistakes are CORRELATED across design rounds (same failure mode), so they do not average out the
way independent noise would. This run measures the SINGLE-round selection value only, and says so.

COST: $0, offline.
"""
from __future__ import annotations

import argparse, collections, json, math, os, random, statistics as st, sys
from math import comb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from e1_ensemble_marginals import base_block                                      # noqa: E402
from h1_heldout_ensemble_defect import fit_predict                                # noqa: E402
import h2_heldout_cluster_ci as H                                                 # noqa: E402
from m1c_skeleton_bookkeeping import build                                        # noqa: E402

NS = (2, 5, 10, 20)
MIN_CELL = 2


def e_oracle(wet, n, m, tot):
    """E[max wet over a uniform n-subset] = sum_v v * (C(c<=v,n) - C(c<v,n)) / C(m,n)."""
    out = 0.0
    for v in sorted(set(wet)):
        le = sum(1 for w in wet if w <= v)
        lt = sum(1 for w in wet if w < v)
        out += v * (comb(le, n) - comb(lt, n)) / tot
    return out


def e_model(wet, pred, n, m):
    """E[wet of the argmax-prediction item in a uniform n-subset]; prediction ties broken uniformly."""
    den = comb(m - 1, n - 1)
    out = 0.0
    ties = 0
    for i in range(m):
        a = 0; b = 0
        for j in range(m):
            if pred[j] > pred[i]:
                a += 1
            elif pred[j] == pred[i]:
                b += 1
        if b > 1:
            ties += 1
        rem = m - a - b
        s = 0.0
        for k in range(0, min(b - 1, n - 1) + 1):
            need = n - 1 - k
            if need < 0 or need > rem:
                continue
            s += comb(b - 1, k) * comb(rem, need) / den / (k + 1)
        out += wet[i] * (n / m) * s
    return out, ties


def capture(wet, pred, n, cache):
    m = len(wet)
    if m < n:
        return None, 0
    tot = comb(m, n)
    r = st.fmean(wet)
    if n not in cache:
        cache[n] = e_oracle(wet, n, m, tot)
    o = cache[n]
    if o - r <= 0:                       # void gate 4/5: degenerate cell, capture undefined
        return None, 0
    em, ties = e_model(wet, pred, n, m)
    return (em - r) / (o - r), ties


def run_pop(nm, keep, M, splits, seed, probe=None):
    """probe: None | 'oracle' (model := true wet, capture must be 1.0) | 'anti' (model := -wet, must be <=0)."""
    cells, cols = collections.defaultdict(list), {}
    for r, m in zip(keep, M):
        if "error" in m:
            continue
        rr = dict(r)
        cols[id(rr)] = base_block(r) + m["gc"] + [m["ed_skel"], m["p_all_skel"]]
        cells[(r["puzzle"], r["method"])].append(rr)
    rws = [(c, g) for c, g in cells.items() if len(g) >= MIN_CELL]
    nB = len(base_block(keep[0]))
    iA = list(range(nB))
    idx = {"A": iA, "A_G": iA + [nB, nB + 1], "A_ed": iA + [nB + 2],
           "A_pall": iA + [nB + 3], "GC_only": [nB, nB + 1]}
    keys = list(idx)
    by_t = collections.defaultdict(list)
    for c, gp in rws:
        by_t[c[0]].append((c, gp))
    tg = sorted(by_t)
    per = {k: {n: {t: [] for t in tg} for n in NS} for k in keys}
    rng = random.Random(seed); order = list(tg)
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
                oc = {}
                for n in NS:
                    if m < n:
                        S["small"] += 1
                        continue
                    for k in keys:
                        if probe == "oracle":
                            pred = wet
                        elif probe == "anti":
                            pred = [-w for w in wet]
                        else:
                            pred = [P[k][id(r)] for r in g]
                        v, ti = capture(wet, pred, n, oc)
                        S["ties"] += ti
                        if v is None:
                            if k == "A":
                                S["degen"] += 1
                            continue
                        per[k][n][t].append(v)
    # A target can have data at n=10 and none at n=20 (all its cells hold fewer than 20 designs), so the
    # usable target set is PER n. capture() returns None only for reasons shared by every block (cell too
    # small, or oracle == random), so the set derived from block A is the same for all blocks.
    used_by_n = {n: [t for t in tg if len(per["A"][n][t]) >= H.MIN_SPLITS_PER_TARGET] for n in NS}
    used = used_by_n[10]
    print(f"  {nm:22s} 格 {len(rws)} 靶标 {len(used)} | 泄漏 分子 {S['leak_mol']}/{splits}"
          f" 靶标 {S['leak_tg']}/{splits} {'✓' if not S['leak_mol'] and not S['leak_tg'] else '✗ 作废'}"
          f" | 格内数<n 跳过 {S['small']},退化格(oracle==random)作废 {S['degen']}"
          f",预测并列条目 {S['ties']}", flush=True)
    if len(used) < H.MIN_TARGETS:
        return {"no_data": True, "n_used": len(used)}
    out = {"n_used": len(used), "stats": {k: v for k, v in S.items()}}
    print("    每个 n 的可用靶标数:" + ", ".join(f"n={n} {len(used_by_n[n])}" for n in NS))
    def summarise(vec, tag):
        ci, nb = H.boot_ci(vec, seed + 17)
        assert nb == len(vec) and nb != len(rws), "闸门:重抽单位不是靶标"
        mde = 1.96 * st.stdev(vec) / math.sqrt(len(vec))
        ex = 0
        for s2 in range(50):
            c, _n = H.boot_ci(vec, 9900 + 13 * s2)
            ex += 1 if (c[0] > 0 or c[1] < 0) else 0
        return {"tag": tag, "mean": round(st.fmean(vec), 4), "ci": ci, "n": nb,
                "MDE": round(mde, 4), "excl": bool(ci[0] > 0 or ci[1] < 0),
                "eff_over_mde": round(abs(st.fmean(vec)) / mde, 2) if mde > 0 else 0.0,
                "seed_flip": f"{ex}/50"}
    out["targets_by_n"] = {n: len(used_by_n[n]) for n in NS}
    for n in NS:
        un = used_by_n[n]
        if len(un) < H.MIN_TARGETS:
            out[f"skipped@{n}"] = len(un)
            continue
        for k in keys:
            out[f"{k}@{n}"] = summarise([st.fmean(per[k][n][t]) for t in un], f"{k}@{n}")
        for k in ("A_G", "A_ed", "A_pall", "GC_only"):
            d = [st.fmean(per[k][n][t]) - st.fmean(per["A"][n][t]) for t in un]
            out[f"{k}_minus_A@{n}"] = summarise(d, f"{k}-A@{n}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "m4_selection_currency.json"))
    a = ap.parse_args()
    PRIMARY = "panel_f1_fresh.json"
    POPS = (("样本2(主)", PRIMARY), ("基准4560", "panel_f1_designs.json"),
            ("样本3", "panel_f1_third.json"), ("第四批", "panel_f1_fourth.json"))
    R = {}; KEEP = {}
    print("=== M4 移动四:把成对准确率翻译成产品货币(top-1-of-n 的捕获率)===", flush=True)
    for nm, fn in POPS:
        KEEP[fn] = build(fn, a.workers)
        print(f"  折叠完 {nm}", flush=True)

    print("\n=== 闸门 5 的两个必响探针(模型换成真实 wet / 负 wet)===", flush=True)
    keep, M = KEEP[PRIMARY]
    po = run_pop("探针 model=真实wet", keep, M, 30, a.seed, probe="oracle")
    pa = run_pop("探针 model=−真实wet", keep, M, 30, a.seed, probe="anti")
    NSP = [n for n in NS if f"A@{n}" in po and f"A@{n}" in pa]      # void gate 3 may skip an n at 30 splits
    ok_o = bool(NSP) and all(abs(po[f"A@{n}"]["mean"] - 1.0) < 1e-9 for n in NSP)
    ok_a = bool(NSP) and all(pa[f"A@{n}"]["mean"] <= 0.0 for n in NSP)
    print(f"  (探针只跑 30 次划分,n 的靶标数不足 20 者按闸门 3 跳过;实际检查 n={NSP})")
    print("  model=真实wet 的捕获率:" + ", ".join(f"n={n} {po[f'A@{n}']['mean']:.6f}" for n in NSP)
          + f"  {'**恒等于 1.0 ✓**' if ok_o else '✗ 不等于 1 -> 整个实验作废'}")
    print("  model=−真实wet 的捕获率:" + ", ".join(f"n={n} {pa[f'A@{n}']['mean']:+.4f}" for n in NSP)
          + f"  {'**<= 0 ✓**' if ok_a else '✗ > 0 -> 整个实验作废'}")
    R["probes"] = {"oracle_is_one": ok_o, "anti_is_nonpositive": ok_a}
    if not (ok_o and ok_a):
        R["verdict"] = "void_probe_failed"
        json.dump(R, open(a.out, "w"), indent=1, ensure_ascii=False)
        print("  **闸门 5 探针不过 ⇒ 整个实验作废(声明原文)。**"); return 0

    print("\n=== 主运行 ===", flush=True)
    for nm, fn in POPS:
        keep, M = KEEP[fn]
        R[fn] = run_pop(nm, keep, M, a.splits, a.seed)
        json.dump(R, open(a.out, "w"), indent=1, ensure_ascii=False)

    P = R[PRIMARY]
    print("\n=== 主人群:捕获率 = 拿到了【可得增益】的百分之几 ===")
    print(f"  {'块':10s}" + "".join(f"{'n='+str(n):>22s}" for n in NS))
    print(f"  (每个 n 的靶标数:{P['targets_by_n']})")
    for k in ("A", "A_G", "A_ed", "A_pall", "GC_only"):
        row = f"  {k:10s}"
        for n in NS:
            if f"{k}@{n}" not in P:
                row += f"{'靶标不足':>22s}"; continue
            o = P[f"{k}@{n}"]
            row += f"{o['mean']*100:>8.1f}% {str(o['ci']):>13s}"
        print(row)
    print("\n=== 相对出厂面板 A 的增量 ===")
    for k in ("A_G", "A_ed", "A_pall", "GC_only"):
        for n in NS:
            if f"{k}_minus_A@{n}" not in P:
                print(f"  {k+'−A':12s} n={n:<3d} 靶标不足 {P['targets_by_n'][n]} < 20,跳过"); continue
            o = P[f"{k}_minus_A@{n}"]
            print(f"  {k+'−A':12s} n={n:<3d} {o['mean']*100:+7.2f} 个百分点  95% {o['ci']}"
                  f"  MDE {o['MDE']*100:.2f}  {'排除 0' if o['excl'] else '含 0'}  [种子 {o['seed_flip']}]")

    print("\n=== 判定(冻结的打死条件)===")
    if "A@10" not in P:
        R["verdict"] = "no_data_at_n10"; json.dump(R, open(a.out, "w"), indent=1, ensure_ascii=False)
        print("  n=10 的靶标数不足 cluster 地板 -> 无数据"); return 0
    K1 = not P["A@10"]["excl"]
    K2 = (not P["A_G_minus_A@10"]["excl"]) or (abs(P["A_G_minus_A@10"]["mean"]) < P["A_G_minus_A@10"]["MDE"])
    NS_OK = [n for n in NS if f"A@{n}" in P]
    caps = [P[f"A@{n}"]["mean"] for n in NS_OK]
    K3 = all(caps[i + 1] <= caps[i] for i in range(len(caps) - 1))
    reps = 0
    for nm, fn in POPS:
        if fn == PRIMARY or R[fn].get("no_data"):
            continue
        if "A@10" in R[fn] and (not K1) == R[fn]["A@10"]["excl"]:
            reps += 1
    K4 = reps < 2
    print(f"  K1 出厂判据在产品货币下无价值(A@10 = {P['A@10']['mean']*100:.1f}%):"
          f"{'FIRES -> unresolved,不是 refuted' if K1 else '不触发 ⇒ 排除 0'}")
    print(f"  K2 两根 GC 列加不动(A_G−A@10 = {P['A_G_minus_A@10']['mean']*100:+.2f} 点,"
          f"MDE {P['A_G_minus_A@10']['MDE']*100:.2f}):{'FIRES' if K2 else '不触发'}")
    print(f"  K3 捕获率随 n 单调不增(n={NS_OK}: {', '.join(f'{c*100:.1f}%' for c in caps)}):"
          f"{'FIRES ⇒ 【top-1-of-n 会放大小优势】不成立' if K3 else '不触发 ⇒ 随 n 放大'}")
    print(f"  K4 复现不足(复现 {reps}/3):{'FIRES' if K4 else '不触发'}")
    v = ("panel_has_no_selection_value" if K1 else
         "selection_value_but_gc_adds_nothing" if K2 else
         "selection_value_and_gc_adds")
    if K3: v += "__no_amplification_with_n"
    if K4: v += "__underreplicated"
    print(f"  裁定:{v}")
    print("  **口径:区间含 0 一律写 `unresolved`,绝不写 `refuted`。**")
    print("  **移动四只做了一半:本实验测【单轮】选择价值。误差跨轮相关那一半是写明的未测项,"
          "不许说本实验覆盖了移动四的全部。**")
    print("  边界:没有盲人群;只用骨架可形成子集;A_pall 是声明后补的臂,不进任何打死条件;不判上线。")
    R["verdict"] = v
    json.dump(R, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
