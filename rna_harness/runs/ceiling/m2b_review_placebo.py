#!/usr/bin/env python3
"""R6, the control that decides whether M2b is about ensemble_defect at all.

M2b showed the per-target gain g_t reproduces across four disjoint design populations at rho ~ 0.44. That is
only a finding about ensemble_defect if a USELESS column does NOT do the same thing. If it does, what
reproduces is a property of the leave-targets-out procedure -- which targets happen to be helped by adding
any column at all -- and the ensemble_defect story evaporates.

Three arms, all scored with C1's fixed rule (common pair set, prediction tie = 0.5):
  ed     the real column, ensemble_defect of the skeleton
  mfe    H2's PRE-REGISTERED placebo: ensemble_defect of the MFE structure. A real, design-dependent,
         ensemble-concentration column that is not about the target.
  noise  three independent N(0,1) columns -- the width-matched null of rule 45.

Every block is fit on the SAME splits in one pass, so the arms differ only in which column was added.

COST: $0, offline.
"""
from __future__ import annotations

import argparse, collections, json, math, os, random, statistics as st, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from e1_ensemble_marginals import base_block                                      # noqa: E402
from h1_heldout_ensemble_defect import fit_predict                                # noqa: E402
import h2_heldout_cluster_ci as H                                                 # noqa: E402
from m2_invariant_split import spearman, rho_ci, half_stats                       # noqa: E402
from m2b_gt_target_property import fold                                           # noqa: E402
from m2b_review_audit import perm_diff_p                                          # noqa: E402

POPS = (("样本2(主)", "panel_f1_fresh.json"), ("基准4560", "panel_f1_designs.json"),
        ("样本3", "panel_f1_third.json"), ("第四批", "panel_f1_fourth.json"))
PRIMARY = "panel_f1_fresh.json"
R_NOISE = int(os.environ.get("R_NOISE", "3"))   # rule 46: one control is not a calibration


def acc_common(rows, preds, keys):
    """One pass, every block scored on the SAME pair set; a prediction tie is worth 0.5 (C1's fixed rule)."""
    ok = {k: 0.0 for k in keys}
    n = 0
    for _c, g in rows:
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                dw = g[a]["wet"] - g[b]["wet"]
                if dw == 0:
                    continue
                n += 1
                for k in keys:
                    d = preds[k][id(g[a])] - preds[k][id(g[b])]
                    ok[k] += 0.5 if d == 0 else (1.0 if ((dw > 0) == (d > 0)) else 0.0)
    return ({k: (ok[k] / n if n else None) for k in keys}, n)


def all_arms(keep, M, splits, seed, noise_seed):
    cells, cols = collections.defaultdict(list), {}
    for r, m in zip(keep, M):
        if "error" in m:
            continue
        rr = dict(r)
        cols[id(rr)] = base_block(r) + m["C"] + [m["ed_skel"], m["ed_mfe"]]
        cells[(r["puzzle"], r["method"])].append(rr)
    rws = [(c, g) for c, g in cells.items() if len(g) >= 2]
    nB = len(base_block(keep[0]))
    rngn = random.Random(noise_seed)
    for k in cols:
        cols[k] = cols[k] + [rngn.gauss(0, 1.0) for _ in range(R_NOISE)]
    base = nB + 2
    idx = {"A": list(range(base)),
           "ed": list(range(base)) + [base],
           "mfe": list(range(base)) + [base + 1]}
    for j in range(R_NOISE):
        idx[f"noise{j}"] = list(range(base)) + [base + 2 + j]
    keys = list(idx)
    by_t = collections.defaultdict(list)
    for c, gp in rws:
        by_t[c[0]].append((c, gp))
    tg = sorted(by_t)
    per = {k: {t: [] for t in tg} for k in keys if k != "A"}
    rng = random.Random(seed); order = list(tg)
    for _s in range(splits):
        rng.shuffle(order)
        cut = max(1, int(0.7 * len(order)))
        tr, te = set(order[:cut]), set(order[cut:])
        TR = [x for t in tr for x in by_t[t]]; TE = [x for t in te for x in by_t[t]]
        if not TR or not TE:
            continue
        P = {k: fit_predict(TR, TE, cols, ix) for k, ix in idx.items()}
        for t in te:
            acc, n = acc_common(by_t[t], P, keys)
            if n < H.MIN_PAIRS_PER_TARGET or any(v is None for v in acc.values()):
                continue
            for k in per:
                per[k][t].append(acc[k] - acc["A"])
    # the PAIRED contrast, H2's pre-registered K2 object: ed gain minus placebo gain, split by split, so
    # generic ensemble concentration cancels and what is left is skeleton-specific.
    per["paired"] = {t: [x - y for x, y in zip(per["ed"][t], per["mfe"][t])] for t in tg}
    return {k: {t: st.fmean(v) for t, v in per[k].items() if len(v) >= H.MIN_SPLITS_PER_TARGET}
            for k in per}


def cross_rho(G, arm):
    rs = []
    det = []
    for i, (n1, f1) in enumerate(POPS):
        for n2, f2 in POPS[i + 1:]:
            ts = sorted(set(G[f1][arm]) & set(G[f2][arm]))
            x = [G[f1][arm][t] for t in ts]; y = [G[f2][arm][t] for t in ts]
            r = spearman(x, y); ci = rho_ci(x, y, 11, 0.95)
            rs.append(r); det.append({"pair": f"{n1}|{n2}", "n": len(ts), "rho": round(r, 4), "ci": ci,
                                      "excl": bool(ci[0] > 0 or ci[1] < 0)})
    return st.fmean(rs), det


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "m2b_review_placebo.json"))
    a = ap.parse_args()
    G, res = {}, {}
    print("=== 折叠 + 一次跑完全部臂(同一批划分,只差加了哪一列)===", flush=True)
    for j, (nm, fn) in enumerate(POPS):
        keep, M = fold(fn, a.workers)
        G[fn] = all_arms(keep, M, a.splits, a.seed, 5000 + 97 * j)
        line = "  ".join(f"{k} {st.fmean(list(v.values())):+.5f}" for k, v in G[fn].items())
        print(f"  {nm:10s} 靶标 {len(G[fn]['ed'])}  g_t 均值:{line}", flush=True)
        res.setdefault("means", {})[fn] = {k: round(st.fmean(list(v.values())), 5) for k, v in G[fn].items()}
        json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)

    print("\n=== 跨人群可复现性:真列 vs 安慰剂 vs 噪声 ===")
    for arm in ["ed", "mfe", "paired"] + [f"noise{j}" for j in range(R_NOISE)]:
        m, det = cross_rho(G, arm)
        nex = sum(1 for d in det if d["excl"])
        tag = {"ed": "**真列 ensemble_defect**", "mfe": "安慰剂 ed_mfe(H2 预注册)",
               "paired": "**配对差 ed−mfe(相消掉一般集合集中度)**"}.get(arm, "纯噪声列")
        rl = ", ".join("%+.2f" % d["rho"] for d in det)
        print(f"  {arm:8s} {tag:28s} 六对均值 rho {m:+.4f}  区间排除 0 的对数 {nex}/6"
              f"  [{rl}]")
        res.setdefault("cross_rho", {})[arm] = {"mean": round(m, 4), "n_excl": nex, "detail": det}
    nz = [res["cross_rho"][f"noise{j}"]["mean"] for j in range(R_NOISE)]
    print(f"\n  噪声列的零点:{st.fmean(nz):+.4f}(3 次:{', '.join(f'{x:+.4f}' for x in nz)})")
    print(f"  安慰剂 {res['cross_rho']['mfe']['mean']:+.4f}   真列 {res['cross_rho']['ed']['mean']:+.4f}")

    print("\n=== arm2 切分:每条臂各用自己的选择集,看能不能切开主人群 ===")
    for arm in ["ed", "mfe", "paired"] + [f"noise{j}" for j in range(R_NOISE)]:
        SEL = [f for _n, f in POPS if f != PRIMARY]
        sel = {}
        for t in set().union(*[set(G[f][arm]) for f in SEL]):
            v = [G[f][arm][t] for f in SEL if t in G[f][arm]]
            if len(v) >= 2:
                sel[t] = st.fmean(v)
        thr = st.median(list(sel.values()))
        gp = G[PRIMARY][arm]
        hi = [gp[t] for t in gp if t in sel and sel[t] > thr]
        lo = [gp[t] for t in gp if t in sel and sel[t] <= thr]
        sh, sl = half_stats(hi), half_stats(lo)
        d, p = perm_diff_p(list(hi), list(lo))
        print(f"  {arm:8s} 高半 {sh['mean']:+.4f}(n {sh['n']}, 效应/MDE {sh['eff_over_mde']:.2f})"
              f"  低半 {sl['mean']:+.4f}  半间差 {d:+.4f}  置换 p {p:.5f}"
              f"  {'**切得开**' if p < 0.05 else '切不开'}")
        res.setdefault("arm2", {})[arm] = {"thr": thr, "high": sh, "low": sl, "diff": round(d, 4), "p": p}
    res["g_per_target"] = {fn: {k: {t: round(x, 5) for t, x in v.items()} for k, v in G[fn].items()}
                           for _n, fn in POPS}
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)

    ed = res["cross_rho"]["ed"]["mean"]; mfe = res["cross_rho"]["mfe"]["mean"]
    print("\n=== 这条控制怎么读 ===")
    if mfe >= 0.7 * ed or st.fmean(nz) >= 0.7 * ed:
        print("  **安慰剂或噪声也复现到同一量级 ⇒ 复现的是【留靶标法本身】,不是 ensemble_defect。"
              "M2b 的臂1 该按这个读法重写。**")
    elif mfe > 0.15:
        print("  安慰剂部分复现 ⇒ 一部分可复现性来自【加任意一列都会帮到哪些靶标】,"
              "真列高出的那部分才是 ensemble_defect 特有的。两个数都要引。")
    else:
        print("  **安慰剂与噪声都不复现,真列复现 ⇒ 可复现性是 ensemble_defect 特有的,不是程序的性质。**")
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
