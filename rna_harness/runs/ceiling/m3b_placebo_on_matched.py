#!/usr/bin/env python3
"""M3b: the control M3 never ran.

M3's positive reads "on GC-matched pairs, ensemble_defect still ranks". Its built-in control (A_G - A) proves
the GC CANCELLATION works, and its noise null proves a useless column does nothing. Neither of those tests
the ATTRIBUTION: it never asked whether a real, design-dependent, ensemble-concentration column that is NOT
about the skeleton would do the same thing on those same matched pairs.

R6 just showed that on the unmatched construction the pre-registered placebo ed_mfe reproduces across
populations almost as well as the real column. So the same question has to be put to M3, on M3's own matched
pairs, or M3's attribution rests on an assumption rather than a measurement -- which is the exact failure the
never-retreat memory was written about, pointed the other way.

Arms, all on the SAME exactly-GC-matched pairs and the SAME splits:
  A_ed    base + ensemble_defect(skeleton)      -- M3's real column
  A_mfe   base + ensemble_defect(MFE)           -- H2/E3's pre-registered placebo
  A_pair  the split-by-split difference of the two -- generic concentration cancelled
  A_nj    base + one N(0,1) column, j = 0..11   -- rule 46 calibration
  A_G     base + the two GC columns             -- M3's built-in must-be-zero control

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
from m2b_gt_target_property import fold                                           # noqa: E402

POPS = (("样本2(主)", "panel_f1_fresh.json"), ("基准4560", "panel_f1_designs.json"),
        ("样本3", "panel_f1_third.json"), ("第四批", "panel_f1_fourth.json"))
PRIMARY = "panel_f1_fresh.json"
R_NOISE = 12
SPLITS_DEFAULT, MIN_PAIRS = 200, 30


def acc_matched_multi(rows, preds, keys, key, mode="exact"):
    """Every block scored on the SAME set of GC-matched, tube-decidable pairs. Prediction tie = 0.5."""
    ok = {k: 0.0 for k in keys}
    n = tt = skipped = 0
    for _c, g in rows:
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                ka, kb = key[id(g[a])], key[id(g[b])]
                same = True if mode == "none" else (ka == kb)
                if not same:
                    skipped += 1; continue
                dw = g[a]["wet"] - g[b]["wet"]
                if dw == 0:
                    tt += 1; continue
                n += 1
                for k in keys:
                    d = preds[k][id(g[a])] - preds[k][id(g[b])]
                    ok[k] += 0.5 if d == 0 else (1.0 if ((dw > 0) == (d > 0)) else 0.0)
    return ({k: (ok[k] / n if n else None) for k in keys}, n, tt, skipped)


def run_pop(nm, keep, M, splits, seed, noise_seed, mode="exact"):
    cells, cols, key = collections.defaultdict(list), {}, {}
    rngn = random.Random(noise_seed)
    for r, m in zip(keep, M):
        if "error" in m:                      # e3's work errors when the skeleton is not formable
            continue
        rr = dict(r)
        cols[id(rr)] = (base_block(r) + m["C"] + [m["ed_skel"], m["ed_mfe"]]
                        + [rngn.gauss(0, 1.0) for _ in range(R_NOISE)])
        key[id(rr)] = (m["C"][0], round(m["C"][1], 3))
        cells[(r["puzzle"], r["method"])].append(rr)
    rws = [(c, g) for c, g in cells.items() if len(g) >= 2]
    nB = len(base_block(keep[0]))
    iA = list(range(nB))
    idx = {"A": iA, "A_G": iA + [nB, nB + 1],
           "A_ed": iA + [nB + 2], "A_mfe": iA + [nB + 3]}
    for j in range(R_NOISE):
        idx[f"A_n{j}"] = iA + [nB + 4 + j]
    keys = list(idx)
    by_t = collections.defaultdict(list)
    for c, gp in rws:
        by_t[c[0]].append((c, gp))
    tg = sorted(by_t)
    per = {k: {t: [] for t in tg} for k in keys if k != "A"}
    per["A_pair"] = {t: [] for t in tg}
    rng = random.Random(seed); order = list(tg)
    S = dict(pairs=0, tie_tube=0, skipped=0, leak=0)
    for _s in range(splits):
        rng.shuffle(order)
        cut = max(1, int(0.7 * len(order)))
        tr, te = set(order[:cut]), set(order[cut:])
        TR = [x for t in tr for x in by_t[t]]; TE = [x for t in te for x in by_t[t]]
        if not TR or not TE:
            continue
        if {id(r) for _c, g in TR for r in g} & {id(r) for _c, g in TE for r in g}:
            S["leak"] += 1
        P = {k: fit_predict(TR, TE, cols, ix) for k, ix in idx.items()}
        for t in te:
            acc, n, tt2, sk = acc_matched_multi(by_t[t], P, keys, key, mode)
            S["pairs"] += n; S["tie_tube"] += tt2; S["skipped"] += sk
            if n < MIN_PAIRS or any(v is None for v in acc.values()):
                continue
            for k in per:
                if k == "A_pair":
                    per[k][t].append((acc["A_ed"] - acc["A"]) - (acc["A_mfe"] - acc["A"]))
                else:
                    per[k][t].append(acc[k] - acc["A"])
    used = [t for t in tg if len(per["A_ed"][t]) >= H.MIN_SPLITS_PER_TARGET]
    tot = S["pairs"] + S["tie_tube"] + S["skipped"]
    print(f"  {nm:10s} 格 {len(rws)} 靶标 {len(used)} | GC 配平且可判的对 {S['pairs']}"
          f"(被配平剔除 {S['skipped']/max(1,tot)*100:.1f}%,试管并列 {S['tie_tube']})"
          f" | 分子级泄漏 {S['leak']}/{splits} {'✓' if not S['leak'] else '✗'}", flush=True)
    out = {}
    for k in per:
        if len(used) < H.MIN_TARGETS:
            continue
        g = [st.fmean(per[k][t]) for t in used]
        ci, nb = H.boot_ci(g, seed + 17)
        assert nb == len(used), "闸门:重抽单位不是靶标"
        mde = 1.96 * st.stdev(g) / math.sqrt(len(g))
        out[k] = {"mean": round(st.fmean(g), 4), "ci": ci, "n": nb, "MDE": round(mde, 4),
                  "excl": bool(ci[0] > 0 or ci[1] < 0), "pos": sum(1 for x in g if x > 0),
                  "eff_over_mde": round(abs(st.fmean(g)) / mde, 2) if mde > 0 else 0.0}
        # rule 51 seed-flip
        ex = 0
        for s2 in range(50):
            c, _n = H.boot_ci(g, 9100 + 13 * s2)
            ex += 1 if (c[0] > 0 or c[1] < 0) else 0
        out[k]["seed_flip"] = f"{ex}/50"
    return out, len(used)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=SPLITS_DEFAULT)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "m3b_placebo_on_matched.json"))
    a = ap.parse_args()
    R = {}
    print("=== M3 的构造(精确 GC 配平),加上 M3 从来没跑过的安慰剂臂 ===", flush=True)
    for j, (nm, fn) in enumerate(POPS):
        keep, M = fold(fn, a.workers)
        R[fn], nu = run_pop(nm, keep, M, a.splits, a.seed, 6000 + 131 * j)
        json.dump(R, open(a.out, "w"), indent=1, ensure_ascii=False)

    print("\n=== 结果:GC 已按构造相消,谁还在排序 ===")
    hdr = f"  {'人群':<12}{'A_G−A(必须为零)':<26}{'A_ed−A(M3 的真列)':<28}{'A_mfe−A(安慰剂)':<26}{'A_pair(相消后)':<24}"
    print(hdr)
    for nm, fn in POPS:
        d = R[fn]
        if not d:
            print(f"  {nm:<12}无数据"); continue
        def f(k):
            o = d[k]
            return f"{o['mean']:+.4f} {'排除0' if o['excl'] else '含0 '}"
        print(f"  {nm:<12}{f('A_G'):<26}{f('A_ed'):<28}{f('A_mfe'):<26}{f('A_pair'):<24}")
    print("\n  详细(主人群):")
    for k in ("A_G", "A_ed", "A_mfe", "A_pair"):
        o = R[PRIMARY][k]
        print(f"    {k:8s} {o['mean']:+.4f}  95% {o['ci']}  MDE +{o['MDE']:.4f}"
              f"  效应/MDE {o['eff_over_mde']:.2f}  正号 {o['pos']}/{o['n']}"
              f"  {'排除 0' if o['excl'] else '含 0'}  [种子翻转 {o['seed_flip']}]")

    print(f"\n=== 噪声零点(第 46 条,{R_NOISE} 根,与主臂同批划分)===")
    for nm, fn in POPS:
        d = R[fn]
        if not d:
            continue
        nz = [d[f"A_n{j}"]["mean"] for j in range(R_NOISE)]
        nex = sum(1 for j in range(R_NOISE) if d[f"A_n{j}"]["excl"])
        print(f"  {nm:10s} 零点中心 {st.fmean(nz):+.5f}  范围 [{min(nz):+.5f}, {max(nz):+.5f}]"
              f"  区间排除 0 的根数 **{nex}/{R_NOISE}**"
              f"  ⇒ A_ed 校正后 {d['A_ed']['mean'] - st.fmean(nz):+.4f}")

    print("\n=== 归属怎么读 ===")
    P = R[PRIMARY]
    ed, mfe, pr = P["A_ed"]["mean"], P["A_mfe"]["mean"], P["A_pair"]["mean"]
    print(f"  主人群:真列 {ed:+.4f} | 安慰剂 {mfe:+.4f}({mfe/ed*100:+.0f}% of 真列) | 相消后 {pr:+.4f}")
    if P["A_mfe"]["excl"] and abs(mfe) >= 0.5 * abs(ed):
        print("  **安慰剂在配平对上也排除 0 且量级过半 ⇒ M3 的归属不成立:"
              "那个信号是一般集合集中度,不是骨架特异。M3 的裁定字符串不变,但它的【读法】必须改。**")
    elif P["A_mfe"]["excl"]:
        print("  安慰剂也排除 0 但量级小 ⇒ 一部分归一般集合集中度,相消后那部分才是骨架特异的。两个数都要引。")
    else:
        print("  **安慰剂在配平对上不排除 0 ⇒ M3 的归属经受住了这次对照:"
              "GC 相消之后还在排序的那个量,是骨架特异的,不是一般集合集中度。**")
    print("  注:本次不重读 M3 的冻结裁定(第 42 条)。这是一次事后对照,"
          "它能改的是【归属的讲法】与下一份声明,改不了 `gc_free_signal_in_ensemble_defect__underreplicated`。")
    json.dump(R, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
