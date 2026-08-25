#!/usr/bin/env python3
"""Diagnostic on M5, NOT a re-decision.

M5's `passes` needed both a permutation p under Bonferroni AND a 99.5% interval excluding zero. Two
candidates cleared the p (ed_range 0.00100, ed_sd 0.00400, against a line of 0.005) and failed the interval.
The interval is the part I built wrong: I AVERAGED THE FOUR ROTATIONS' ENDPOINTS. Averaging endpoints of four
separate intervals gives roughly the width of ONE rotation's interval -- it is not an interval on the mean of
the four rotation rhos at all. The permutation test, by contrast, permutes inside each rotation and averages,
so it IS a correct null for the frozen statistic.

So the correct interval is built here: resample TARGETS with replacement, recompute all four rotations from
the resampled target set, average, repeat. Plus the split test for the two leads.

This changes nothing about M5's frozen verdict (`no_invariant_found_in_measurement_geometry`, K1 fired). It
exists to be the input to M5b's declaration -- go measure it, do not quote it as a caveat.

COST: $0, offline.
"""
from __future__ import annotations
import argparse, collections, json, math, os, random, statistics as st, sys
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from m2_invariant_split import spearman, half_stats                              # noqa: E402
from m2b_gt_target_property import fold                                          # noqa: E402
from m2b_review_audit import perm_diff_p                                         # noqa: E402
from m5_measurement_geometry import g_and_geometry, POPS, CAND, ALPHA_B, BENCH   # noqa: E402

B = 4000


def col_for(G, GEO, fn, c, others):
    out = {}
    for t in G[fn]:
        v = ([G[o][t] for o in others if t in G[o]] if c == BENCH
             else [GEO[o][t][c] for o in others if t in GEO[o] and c in GEO[o][t]])
        if len(v) >= 2:
            out[t] = st.fmean(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    G, GEO = {}, {}
    print("=== 重算(同种子,与 M5 完全一致)===", flush=True)
    for nm, fn in POPS:
        keep, M = fold(fn, a.workers)
        G[fn], GEO[fn], _lk, _md = g_and_geometry(keep, M, a.splits, a.seed)
        print(f"  {nm:10s} 靶标 {len(G[fn])}", flush=True)

    rot = {}
    for c in CAND + [BENCH]:
        rot[c] = []
        for nm, fn in POPS:
            others = [f for _n, f in POPS if f != fn]
            col = col_for(G, GEO, fn, c, others)
            ts = sorted(col)
            if len(ts) >= 30:
                rot[c].append((ts, [col[t] for t in ts], [G[fn][t] for t in ts], col))

    print(f"\n=== 正确构造的区间:重抽【靶标】,四个轮换全部用同一批重抽靶标重算再平均(B={B})===")
    print("   (M5 里那个是把四个区间的端点【平均】—— 那给的是单个轮换区间的宽度,不是均值的区间)")
    out = {}
    for c in CAND + [BENCH]:
        if not rot[c]:
            continue
        obs = st.fmean([spearman(x, y) for _ts, x, y, _col in rot[c]])
        allt = sorted(set().union(*[set(ts) for ts, _x, _y, _c in rot[c]]))
        rng = random.Random(a.seed + 909)
        bs = []
        for _ in range(B):
            pick = collections.Counter(allt[rng.randrange(len(allt))] for _ in range(len(allt)))
            rs = []
            for ts, _x, _y, col in rot[c]:
                xs, ys = [], []
                for t in ts:
                    for _k in range(pick[t]):
                        xs.append(col[t]); ys.append(G[[f for _n, f in POPS
                                                        if set(G[f]) >= {t} and t in G[f]][0]][t])
                if len(set(xs)) > 1 and len(xs) >= 10:
                    rs.append(None)
            bs.append(None)
        # the loop above is a placeholder guard; do it properly per rotation with its own g
        bs = []
        gmap = []
        for ts, _x, _y, col in rot[c]:
            fn = [f for _n, f in POPS if all(t in G[f] for t in ts[:3])][0]
            gmap.append((ts, col, G[fn]))
        rng = random.Random(a.seed + 909)
        for _ in range(B):
            idx = [allt[rng.randrange(len(allt))] for _ in range(len(allt))]
            cnt = collections.Counter(idx)
            rs = []
            for ts, col, gg in gmap:
                xs, ys = [], []
                for t in ts:
                    k = cnt[t]
                    if k:
                        xs.extend([col[t]] * k); ys.extend([gg[t]] * k)
                if len(xs) >= 10 and len(set(xs)) > 1:
                    rs.append(spearman(xs, ys))
            if rs:
                bs.append(st.fmean(rs))
        bs.sort()
        lo95, hi95 = bs[int(.025 * len(bs))], bs[int(.975 * len(bs))]
        lb, hb = bs[int(ALPHA_B / 2 * len(bs))], bs[int((1 - ALPHA_B / 2) * len(bs))]
        ex = (lb > 0 or hb < 0)
        out[c] = {"rho_mean": round(obs, 4), "ci95": [round(lo95, 4), round(hi95, 4)],
                  "ci_bonf": [round(lb, 4), round(hb, 4)], "excl_bonf": bool(ex)}
        print(f"  {c:24s} rho均 {obs:+.4f}  95% [{lo95:+.4f},{hi95:+.4f}]"
              f"  {100*(1-ALPHA_B):.1f}% [{lb:+.4f},{hb:+.4f}]  {'**排除 0**' if ex else '含 0'}")

    print("\n=== 两条线索的切分检验(基准列给的是 +0.0344,p=0.00020)===")
    sp = {}
    for c in ("ed_range", "ed_sd", BENCH):
        rows = []
        for nm, fn in POPS:
            others = [f for _n, f in POPS if f != fn]
            col = col_for(G, GEO, fn, c, others)
            if len(col) < 30:
                continue
            thr = st.median(list(col.values()))
            hi = [G[fn][t] for t in col if col[t] > thr]
            lo = [G[fn][t] for t in col if col[t] <= thr]
            d, p = perm_diff_p(list(hi), list(lo))
            rb = random.Random(77); bb = []
            for _ in range(4000):
                A = [hi[rb.randrange(len(hi))] for _ in range(len(hi))]
                Bv = [lo[rb.randrange(len(lo))] for _ in range(len(lo))]
                bb.append(st.fmean(A) - st.fmean(Bv))
            bb.sort(); ci = [round(bb[100], 4), round(bb[3899], 4)]
            rows.append({"pop": nm, "diff": round(d, 4), "p": p, "ci": ci,
                         "excl": bool(ci[0] > 0 or ci[1] < 0),
                         "hi": round(st.fmean(hi), 4), "lo": round(st.fmean(lo), 4)})
            print(f"  {c:24s} {nm:10s} 高半 {st.fmean(hi):+.4f} 低半 {st.fmean(lo):+.4f}"
                  f"  半间差 {d:+.4f} 95% {ci} 置换 p {p:.5f}"
                  f"  {'排除 0' if rows[-1]['excl'] else '含 0'}")
        sp[c] = rows
    json.dump({"intervals": out, "splits": sp}, open(os.path.join(HERE, "m5_review_interval.json"), "w"),
              indent=1, ensure_ascii=False)
    print("\n**这是诊断,不是改判。M5 的冻结裁定仍是 `no_invariant_found_in_measurement_geometry`(K1 触发)。**")
    print("**它的位置是 M5b 声明的输入:去测它,不是把它当保留意见引用。**")
    return 0


if __name__ == "__main__":
    sys.exit(main())
