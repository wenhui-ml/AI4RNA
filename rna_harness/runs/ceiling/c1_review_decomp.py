#!/usr/bin/env python3
"""C1 review: verify the surviving contrast before believing it, and decompose it.

The unexpected result is that six O(L) columns add +0.029..+0.039 held-out (a +26..+54% relative gain in
above-chance skill) on top of a 22-column three-oracle panel, on 6/6 populations. Four checks, all about the
CODE and the DECOMPOSITION, not about the hypothesis:

  D1  denominator identity -- after the common-pair-set fix every block must be scored on the SAME pairs.
      Asserted empirically per (target, split), not argued.
  D2  WHICH COLUMNS. T0 splits into three pairs: G = composition (gc_cross, gc_frac), F = formability
      indicators (formable_skel, formable_cross), N = non-canonical counts (a graded F). Each pair added to
      T2 separately, plus F∪N. Width-matched against the 2-column and 4-column noise nulls measured here.
      Without this, "cheap complementarity beats the panel" and "cheap composition beats the panel" are
      indistinguishable, and they are different findings.
  D3  per-target distribution of the surviving contrast -- a few targets or the whole population?
  D4  the reverse direction, restated inside K4's discipline: T2 adds how much over T0, and is it above MDE?

$0, offline.
"""
from __future__ import annotations
import collections, json, math, os, random, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/data/wenhui/openclaude_rna")
import h2_heldout_cluster_ci as H
from h1_heldout_ensemble_defect import fit_predict
from e1_ensemble_marginals import base_block, load_pop
from c1_cost_tiers import T0_NAMES, T1_EXTRA, work, pair_acc_common, mde

SPLITS = 200
R_NOISE = 12                      # noise draws per width, rule 46
GROUPS = {"G_组成": ("gc_cross", "gc_frac"),
          "F_可形成指示": ("formable_skel", "formable_cross"),
          "N_非标准计数": ("n_noncanon_skel", "n_noncanon_cross")}


def build(fn, workers=14):
    from leaderboard_design import targets
    T = targets()
    rec, _s, _m = load_pop(fn)
    keep = [r for r in rec if base_block(r) is not None]
    jobs = [(r["puzzle"], r["seq"], T[r["puzzle"]]["target"]) for r in keep]
    from concurrent.futures import ProcessPoolExecutor
    M = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for m in ex.map(work, jobs, chunksize=8):
            M.append(m)
    cells, cols = collections.defaultdict(list), {}
    for r, m in zip(keep, M):
        if "error" in m:
            continue
        rr = dict(r)
        cols[id(rr)] = [m[k] for k in T0_NAMES] + [m[k] for k in T1_EXTRA] + base_block(r)
        cells[(r["puzzle"], r["method"])].append(rr)
    rws = [(c, g) for c, g in cells.items() if len(g) >= 2]
    return rws, cols, len(T0_NAMES), len(T1_EXTRA), len(base_block(keep[0]))


def run(nm, fn, seed=0):
    rws, cols, n0, n1e, nB = build(fn)
    mols = [r for _c, g in rws for r in g]
    base = len(cols[id(mols[0])])
    iT0 = list(range(n0)); iT2 = list(range(n0 + n1e, n0 + n1e + nB))
    pos = {c: i for i, c in enumerate(T0_NAMES)}
    print(f"\n=== {nm} 分子 {len(mols)} 格 {len(rws)} ===", flush=True)

    # noise columns for the width-matched controls: 4 of them (2-wide and 4-wide nulls)
    rngn = random.Random(31337)
    noise = {k: [[rngn.gauss(0, 1.0) for _ in range(4)] for _ in range(R_NOISE)] for k in cols}

    blocks = {"T0": iT0, "T2": iT2, "T0uT2": iT0 + iT2}
    for g, names in GROUPS.items():
        blocks[f"T2u{g}"] = iT2 + [pos[c] for c in names]
    blocks["T2uF_N"] = iT2 + [pos[c] for c in GROUPS["F_可形成指示"] + GROUPS["N_非标准计数"]]

    by_t = collections.defaultdict(list)
    for c, gp in rws:
        by_t[c[0]].append((c, gp))
    order = list(sorted(by_t))
    per = {k: {t: [] for t in order} for k in blocks}
    pern = {(w, j): {t: [] for t in order} for w in (2, 4) for j in range(R_NOISE)}
    denom_mismatch = 0
    rng = random.Random(seed)
    augs = {j: {k: v + noise[k][j] for k, v in cols.items()} for j in range(R_NOISE)}
    for _s in range(SPLITS):
        rng.shuffle(order)
        cut = max(1, int(0.7 * len(order)))
        tr, te = set(order[:cut]), set(order[cut:])
        TR = [x for t in tr for x in by_t[t]]; TE = [x for t in te for x in by_t[t]]
        if not TR or not TE:
            continue
        P = {k: fit_predict(TR, TE, cols, ix) for k, ix in blocks.items()}
        Pn = {}
        for j in range(R_NOISE):
            for w in (2, 4):
                Pn[(w, j)] = fit_predict(TR, TE, augs[j], iT2 + list(range(base, base + w)))
        for t in te:
            ns = set(); vals = {}
            for k in blocks:
                v, n, _tt, _tp = pair_acc_common(by_t[t], P[k])
                vals[k] = v; ns.add(n)
            if len(ns) != 1:                       # D1: identical denominator across blocks
                denom_mismatch += 1
            n = ns.pop() if len(ns) == 1 else 0
            if n < H.MIN_PAIRS_PER_TARGET or any(v is None for v in vals.values()):
                continue
            for k in blocks:
                per[k][t].append(vals[k])
            for key, p in Pn.items():
                v, _n, _a, _b = pair_acc_common(by_t[t], p)
                if v is not None:
                    pern[key][t].append(v)
    used = [t for t in order if len(per["T2"][t]) >= H.MIN_SPLITS_PER_TARGET]
    print(f"  D1 分母一致性:各块成对数不一致的 (靶标,划分) = {denom_mismatch}"
          f"  {'✓ 完全一致' if denom_mismatch == 0 else '✗ 仍在不同对集合上打分'}")
    if len(used) < H.MIN_TARGETS:
        print("  cluster 地板 -> 无数据"); return {"no_data": True}
    M = mde([st.fmean(per["T2"][t]) for t in used])
    print(f"  cluster {len(used)},本人群 MDE = +{M:.4f}")

    def C(a, b):
        g = [st.fmean(per[a][t]) - st.fmean(per[b][t]) for t in used]
        ci, n = H.boot_ci(g, seed + 17)
        return {"mean": round(st.fmean(g), 4), "ci": ci, "n": n,
                "pos": sum(1 for x in g if x > 0), "excl": bool(ci[0] > 0 or ci[1] < 0),
                "over_mde": abs(st.fmean(g)) > M, "g": [round(x, 4) for x in g]}

    # width-matched nulls
    nulls = {}
    for w in (2, 4):
        ms = []
        for j in range(R_NOISE):
            gg = [st.fmean(pern[(w, j)][t]) - st.fmean(per["T2"][t]) for t in used
                  if len(pern[(w, j)][t]) >= H.MIN_SPLITS_PER_TARGET]
            if len(gg) >= H.MIN_TARGETS:
                ms.append(st.fmean(gg))
        nulls[w] = {"center": round(st.fmean(ms), 5), "sd": round(st.stdev(ms), 5), "n": len(ms)}
        print(f"  宽度 {w} 的无信息零点:{st.fmean(ms):+.5f}  sd {st.stdev(ms):.5f}  ({len(ms)} 次)")

    print("  D2 分解 —— 哪几根列在起作用(每项都报【相对同宽度无信息列】的校正值):")
    out = {"MDE": round(M, 4), "n_used": len(used), "nulls": nulls, "contrasts": {}}
    for tag, w in (("T2uG_组成", 2), ("T2uF_可形成指示", 2), ("T2uN_非标准计数", 2), ("T2uF_N", 4)):
        key = tag if tag in blocks else tag.replace("T2u", "T2u")
        key = {"T2uG_组成": "T2uG_组成", "T2uF_可形成指示": "T2uF_可形成指示",
               "T2uN_非标准计数": "T2uN_非标准计数", "T2uF_N": "T2uF_N"}[tag]
        c = C(key, "T2"); z = nulls[w]["center"]
        c["corrected"] = round(c["mean"] - z, 4)
        out["contrasts"][tag] = c
        print(f"    {tag:20s} 生 {c['mean']:+.4f} 95%{c['ci']} "
              f"{'排除0' if c['excl'] else '含 0'}  校正后 {c['corrected']:+.4f}"
              f"  {'>MDE' if c['over_mde'] else '<MDE 测不出'}  正号 {c['pos']}/{c['n']}")
    c6 = C("T0uT2", "T2")
    z6 = None
    out["contrasts"]["T0uT2_minus_T2_全6根"] = c6
    print(f"    {'全 6 根(参照)':20s} 生 {c6['mean']:+.4f} 95%{c6['ci']} "
          f"{'排除0' if c6['excl'] else '含 0'}  {'>MDE' if c6['over_mde'] else '<MDE'}"
          f"  正号 {c6['pos']}/{c6['n']}")
    rev = C("T0uT2", "T0")
    out["contrasts"]["T0uT2_minus_T0_面板在便宜层之上"] = rev
    print(f"  D4 反向:面板在便宜层之上 {rev['mean']:+.4f} 95%{rev['ci']}"
          f"  {'排除0' if rev['excl'] else '含 0'}  {'>MDE' if rev['over_mde'] else '<MDE ⇒ 测不出,不许说是零'}")
    gg = sorted(c6["g"])
    print(f"  D3 逐靶标分布(全 6 根那条):中位 {st.median(gg):+.4f} sd {st.stdev(gg):.4f}"
          f" 范围 {min(gg):+.4f}~{max(gg):+.4f};去两端各 3 个后均值 "
          f"{st.fmean(gg[3:-3]):+.4f}(全体 {st.fmean(gg):+.4f})"
          f" ⇒ {'不是少数靶标' if abs(st.fmean(gg[3:-3])-st.fmean(gg))<0.005 else '受少数靶标影响'}")
    return out


res = {}
for nm, fn in (("样本2(主)", "panel_f1_fresh.json"), ("样本3", "panel_f1_third.json"),
               ("第四批", "panel_f1_fourth.json")):
    res[nm] = run(nm, fn)
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "c1_review_decomp.json"), "w"), indent=1, ensure_ascii=False)
print("\n写出 c1_review_decomp.json")
