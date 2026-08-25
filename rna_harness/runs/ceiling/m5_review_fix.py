#!/usr/bin/env python3
"""Bug fix + re-measurement of M5's screen. Diagnostic, NOT a re-decision.

THE BUG. `m5_review_interval.py` rebuilt its bootstrap input as

    fn = [f for _n, f in POPS if all(t in G[f] for t in ts[:3])][0]

which picks the FIRST population whose g contains the rotation's first three targets. Since every population
holds nearly every target, that is essentially always 样本2 -- so all four rotations were bootstrapped
against 样本2's g instead of their own. The point estimates were computed correctly from the rotation arrays;
only the intervals used the wrong mapping. A tell was visible and I did not chase it: the benchmark's 95%
interval came out as [+0.5754,+0.8556] with a point estimate of +0.5729 sitting BELOW its lower bound. A
bootstrap interval that excludes its own point estimate is not a coincidence.

M6 stores (ts, col, G[fn]) directly and its benchmark interval behaves (99% [+0.3486,+0.7144] around
+0.5729). This script re-measures M5's ten candidates with that construction, plus the rule-55 global
permutation null (one target relabelling applied to every rotation, not an independent shuffle per rotation).

M5's frozen verdict does not change. This is the input to M5b's declaration.

COST: $0, offline.
"""
from __future__ import annotations
import argparse, collections, json, os, random, statistics as st, sys
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from m2_invariant_split import spearman                                          # noqa: E402
from m2b_gt_target_property import fold                                          # noqa: E402
from m5_measurement_geometry import g_and_geometry, POPS, CAND, ALPHA_B, BENCH   # noqa: E402

B = 3000
NP = 3000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    G, GEO = {}, {}
    for nm, fn in POPS:
        keep, M = fold(fn, a.workers)
        G[fn], GEO[fn], _l, _m = g_and_geometry(keep, M, a.splits, a.seed)
        print(f"  折叠+g 完成 {nm} 靶标 {len(G[fn])}", flush=True)

    rot = {}
    for c in CAND + [BENCH]:
        rot[c] = []
        for nm, fn in POPS:
            others = [f for _n, f in POPS if f != fn]
            col = {}
            for t in G[fn]:
                v = ([G[o][t] for o in others if t in G[o]] if c == BENCH
                     else [GEO[o][t][c] for o in others if t in GEO[o] and c in GEO[o][t]])
                if len(v) >= 2:
                    col[t] = st.fmean(v)
            ts = sorted(col)
            if len(ts) >= 30:
                rot[c].append((ts, col, G[fn]))          # <- the fix: this rotation's OWN g
    allt = sorted(set().union(*[set(ts) for ts, _c, _g in rot[BENCH]]))

    print(f"\n=== 修正后的筛选({len(CAND)} 个候选,Bonferroni {ALPHA_B:.4f},"
          f"bootstrap 与置换都按第 55 条保留轮换相关)===")
    out = {}
    for c in CAND + [BENCH]:
        obs = st.fmean([spearman([col[t] for t in ts], [gg[t] for t in ts]) for ts, col, gg in rot[c]])
        rng = random.Random(a.seed + 909); bs = []
        for _ in range(B):
            cnt = collections.Counter(allt[rng.randrange(len(allt))] for _ in range(len(allt)))
            rs = []
            for ts, col, gg in rot[c]:
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
        rngp = random.Random(a.seed + 55); hits = 0
        for _ in range(NP):
            perm = list(allt); rngp.shuffle(perm)
            mp = dict(zip(allt, perm))
            rs = []
            for ts, col, gg in rot[c]:
                xs = [col[t] for t in ts if mp[t] in gg]
                ys = [gg[mp[t]] for t in ts if mp[t] in gg]
                if len(xs) >= 10:
                    rs.append(spearman(xs, ys))
            if rs and abs(st.fmean(rs)) >= abs(obs):
                hits += 1
        pv = (hits + 1) / (NP + 1)
        ok = bool((lb > 0 or hb < 0) and pv < ALPHA_B)
        inside = lo95 <= obs <= hi95
        out[c] = {"rho": round(obs, 4), "ci95": [round(lo95, 4), round(hi95, 4)],
                  "ci_bonf": [round(lb, 4), round(hb, 4)], "perm_p": round(pv, 5),
                  "passes": ok, "point_inside_ci95": bool(inside)}
        print(f"  {c:24s} rho {obs:+.4f}  95% [{lo95:+.4f},{hi95:+.4f}]"
              f"  {100*(1-ALPHA_B):.1f}% [{lb:+.4f},{hb:+.4f}]  置换 p {pv:.5f}"
              f"  {'**通过**' if ok else '不通过'}  {'' if inside else '**点估计落在 95% 区间外 -> 有问题**'}")
    json.dump(out, open(os.path.join(HERE, "m5_review_fix.json"), "w"), indent=1, ensure_ascii=False)
    hw = (out[BENCH]["ci_bonf"][1] - out[BENCH]["ci_bonf"][0]) / 2
    print(f"\n  分辨力(修正后):基准 rho {out[BENCH]['rho']},Bonferroni 区间半宽 {hw:.3f}"
          f" ⇒ 分辨得了 |rho| ≳ {hw:.2f}")
    passing = [c for c in CAND if out[c]["passes"]]
    print(f"  修正后通过的候选:{passing or '无'}")
    print("\n**这是诊断,不是改判。M5 的冻结裁定仍是 `no_invariant_found_in_measurement_geometry`。**")
    return 0


if __name__ == "__main__":
    sys.exit(main())
