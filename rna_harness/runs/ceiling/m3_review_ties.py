#!/usr/bin/env python3
"""M3 review: is the A_ed gain just breaking ties that A cannot break?

Block A ties on 4.79% of GC-matched pairs and scores 0.5 there; A_ed never ties. If A_ed won every one of
those, the gain would be 0.0479*0.5 = +0.024, and the observed gain is +0.0140 -- the same order. So the
whole effect could be "ensemble_defect only helps where the panel is completely indifferent", which is a
much narrower claim than "it improves ranking". Decompose the gain over the two disjoint pair sets:
  TIED   -- pairs where A's prediction difference is exactly 0
  OPEN   -- pairs where A already has an opinion
Same splits, same seed, same GC matching as the run. $0, offline.
"""
from __future__ import annotations
import collections, math, random, statistics as st, sys, os, json
sys.path.insert(0, "/data/wenhui/openclaude_rna"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e1_ensemble_marginals import base_block, load_pop, marginals, E_FREE
from h1_heldout_ensemble_defect import fit_predict
import h2_heldout_cluster_ci as H

SPLITS = 200


def run(fn, formable, label):
    from leaderboard_design import targets
    T = targets()
    rec, _s, _m = load_pop(fn)
    keep = [r for r in rec if base_block(r) is not None]
    from concurrent.futures import ProcessPoolExecutor
    M = []
    with ProcessPoolExecutor(max_workers=14) as ex:
        for m in ex.map(marginals, [(r["puzzle"], r["seq"], T[r["puzzle"]]["target"]) for r in keep],
                        chunksize=8):
            M.append(m)
    cells, cols, key = collections.defaultdict(list), {}, {}
    for r, m in zip(keep, M):
        if "error" in m or (formable and "ens_defect_skel" not in m):
            continue
        rr = dict(r)
        ext = [m["ens_defect_skel"]] if formable else [m[k] for k in E_FREE]
        cols[id(rr)] = base_block(r) + [m["gc_cross"], m["gc_frac"]] + ext
        key[id(rr)] = (m["gc_cross"], round(m["gc_frac"], 3))
        cells[(r["puzzle"], r["method"])].append(rr)
    rws = [(c, g) for c, g in cells.items() if len(g) >= 2]
    nB = len(base_block(keep[0])); nE = 1 if formable else len(E_FREE)
    iA = list(range(nB)); iX = iA + list(range(nB + 2, nB + 2 + nE))
    by_t = collections.defaultdict(list)
    for c, gp in rws:
        by_t[c[0]].append((c, gp))
    order = list(sorted(by_t)); rng = random.Random(0)
    # per target: [n_tied, okA_tied, okX_tied, n_open, okA_open, okX_open]
    acc = {t: [0, 0.0, 0.0, 0, 0.0, 0.0] for t in order}
    for _s in range(SPLITS):
        rng.shuffle(order)
        cut = max(1, int(0.7 * len(order)))
        tr, te = set(order[:cut]), set(order[cut:])
        TR = [x for t in tr for x in by_t[t]]; TE = [x for t in te for x in by_t[t]]
        if not TR or not TE:
            continue
        pA = fit_predict(TR, TE, cols, iA); pX = fit_predict(TR, TE, cols, iX)
        for t in te:
            for _c, g in by_t[t]:
                for a in range(len(g)):
                    for b in range(a + 1, len(g)):
                        if key[id(g[a])] != key[id(g[b])]:
                            continue
                        dw = g[a]["wet"] - g[b]["wet"]
                        if dw == 0:
                            continue
                        dA = pA[id(g[a])] - pA[id(g[b])]
                        dX = pX[id(g[a])] - pX[id(g[b])]
                        sA = 0.5 if dA == 0 else (1.0 if (dw > 0) == (dA > 0) else 0.0)
                        sX = 0.5 if dX == 0 else (1.0 if (dw > 0) == (dX > 0) else 0.0)
                        o = 0 if dA == 0 else 3
                        acc[t][o] += 1; acc[t][o + 1] += sA; acc[t][o + 2] += sX
    used = [t for t in order if acc[t][0] + acc[t][3] > 0 and acc[t][3] > 0]
    nt = sum(acc[t][0] for t in used); no = sum(acc[t][3] for t in used)
    print(f"\n=== {label} ===")
    print(f"  靶标 {len(used)};A 并列对 {nt} ({nt/(nt+no)*100:.2f}%),A 有意见的对 {no}")
    def block(idx, name):
        g = [(acc[t][idx+2] - acc[t][idx+1]) / acc[t][idx] for t in used if acc[t][idx] > 0]
        if len(g) < H.MIN_TARGETS:
            print(f"  {name}:靶标 {len(g)} < 20,无数据"); return None
        ci, n = H.boot_ci(g, 17)
        mde = 1.96 * st.stdev(g) / math.sqrt(len(g))
        print(f"  {name:22s} 每对增益 {st.fmean(g):+.4f}  95% {ci}  (MDE +{mde:.4f})"
              f"  {'排除 0' if ci[0] > 0 or ci[1] < 0 else '含 0'}  cluster {n}")
        return st.fmean(g)
    gt = block(0, "A 并列的对上")
    go = block(3, "A 有意见的对上")
    if gt is not None and go is not None:
        share_t = gt * nt / (nt + no); share_o = go * no / (nt + no)
        print(f"  合成:并列贡献 {share_t:+.4f} + 有意见贡献 {share_o:+.4f} = {share_t+share_o:+.4f}"
              f"  (运行报的是 +0.0140)")
        print(f"  **并列部分占总增益的 {share_t/(share_t+share_o)*100:.1f}%**")
    return {"n_tied": nt, "n_open": no}


run("panel_f1_fresh.json", True, "样本2(主) 臂1 ensemble_defect")
run("panel_f1_fresh.json", False, "样本2(主) 臂2 五根自由集合边际")
