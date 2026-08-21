#!/usr/bin/env python3
"""H2 review diagnostics: verify the machinery before believing the verdict.

The unexpected result is that the BLIND primary (sixth batch) contains 0 while five of six populations
exclude 0. Before treating that as a finding, four things have to be checked, and the checks must be about
the CODE, not about the hypothesis:

  D1  positive control -- inject a column that is a deliberate function of the tube score. The machinery
      must report a large gain. If it does not, the estimator is broken and every H2 number is void.
  D2  negative control -- inject pure noise. Must land on ~0 with an interval containing 0.
  D3  void gate 6 is VACUOUS as written (`nb` is returned by boot_ci as len(vec), so `nb == len(used)`
      cannot fail). Build the non-vacuous version and show it FIRES on a deliberately wrong unit.
  D4  describe how g_t is distributed on the sixth batch and how many splits each target got, to see
      whether the small point estimate is a few targets or the whole population.

D1/D2 run on the sixth batch, i.e. the population with the unexpected result. $0, offline.
"""
from __future__ import annotations
import collections, json, os, random, statistics as st, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from e1_ensemble_marginals import base_block, load_pop
from e3_scrambled_reference import work
from h1_heldout_ensemble_defect import fit_predict
import h2_heldout_cluster_ci as H

FN = "panel_f1_sixth.json"


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
        cols[id(rr)] = base_block(r) + m["C"] + [m["ed_skel"]] + [m["ed_mfe"]]
        cells[(r["puzzle"], r["method"])].append(rr)
    rws = [(c, g) for c, g in cells.items() if len(g) >= 2]
    return rws, cols, len(base_block(keep[0]))


def per_target(rws, cols, iBC, iX, splits=200, seed=0):
    """Return {target: [gain per split]} for block iX over baseline iBC. Same flow as H2.run_pop."""
    by_t = collections.defaultdict(list)
    for c, gp in rws:
        by_t[c[0]].append((c, gp))
    tg = sorted(by_t); order = list(tg)
    rng = random.Random(seed)
    per = {t: [] for t in tg}
    nsplits = {t: 0 for t in tg}
    for _s in range(splits):
        rng.shuffle(order)
        cut = max(1, int(0.7 * len(order)))
        tr, te = set(order[:cut]), set(order[cut:])
        TR = [x for t in tr for x in by_t[t]]
        TE = [x for t in te for x in by_t[t]]
        if not TR or not TE:
            continue
        pB = fit_predict(TR, TE, cols, iBC)
        pX = fit_predict(TR, TE, cols, iX)
        for t in te:
            nsplits[t] += 1
            aB, nB, _ = H.pair_accuracy_rows(by_t[t], pB)
            aX, nX, _ = H.pair_accuracy_rows(by_t[t], pX)
            if nB < H.MIN_PAIRS_PER_TARGET or aB is None or aX is None:
                continue
            per[t].append(aX - aB)
    used = [t for t in tg if len(per[t]) >= H.MIN_SPLITS_PER_TARGET]
    return per, used, nsplits


def main() -> int:
    print("=== H2 复核诊断(第六批 = 出现意外结果的那个人群)===", flush=True)
    rws, cols, nB = build(FN)
    mols = [r for _c, g in rws for r in g]
    print(f"  分子 {len(mols)},格 {len(rws)},靶标 {len({c[0] for c,_g in rws})},基础列 {nB}")
    iBC = list(range(nB + 2)); iED = iBC + [nB + 2]

    # ---- D1 / D2: inject a leak column and a noise column at index nB+4 / nB+5 --------------------
    rng = random.Random(12345)
    for r in mols:
        v = cols[id(r)]
        assert len(v) == nB + 4, "列布局与 H2 不一致 -> 诊断作废"
        cols[id(r)] = v + [r["wet"] + rng.gauss(0, 0.05), rng.gauss(0, 1.0)]
    iLEAK = iBC + [nB + 4]
    iNOISE = iBC + [nB + 5]

    res = {}
    for tag, idx in (("D1 正对照 wet+噪声(故意泄漏)", iLEAK),
                     ("D2 负对照 纯噪声", iNOISE),
                     ("参照 ed_skel(H2 的处理组)", iED)):
        per, used, _ns = per_target(rws, cols, iBC, idx, splits=200, seed=0)
        g = [st.fmean(per[t]) for t in used]
        ci, n = H.boot_ci(g, 17)
        res[tag] = (round(st.fmean(g), 4), ci, sum(1 for x in g if x > 0), n)
        print(f"  {tag:34s} 增量 {st.fmean(g):+.4f}  95% {ci}"
              f"  {'排除 0' if ci[0] > 0 or ci[1] < 0 else '含 0'}  正号靶标 {sum(1 for x in g if x>0)}/{n}",
              flush=True)

    ok_d1 = res["D1 正对照 wet+噪声(故意泄漏)"][1][0] > 0.10
    ok_d2 = not (res["D2 负对照 纯噪声"][1][0] > 0 or res["D2 负对照 纯噪声"][1][1] < 0)
    print(f"  D1 判定(正对照必须给出 >+0.10 的下界):{'通过' if ok_d1 else '不通过 -> 估计量坏了'}")
    print(f"  D2 判定(负对照必须含 0):{'通过' if ok_d2 else '不通过 -> 估计量坏了'}")

    # ---- D3: void gate 6 is vacuous; build the non-vacuous version --------------------------------
    print("\n=== D3 闸门 6 是否空转 ===")
    print("  H2 的写法:`ci, nb = boot_ci(g[k], ...)` 然后 `assert nb == len(used)`。")
    print("  boot_ci 的返回值就是 len(vec),而 vec 就是 g[k],而 g[k] 的长度就是 len(used)。")
    print("  ⇒ 这个断言【永远为真】,与重抽单位是靶标/格/分子无关。**空转闸门,第三次。**")

    per, used, nsplits = per_target(rws, cols, iBC, iED, splits=200, seed=0)
    n_tg = len({c[0] for c, _g in rws})
    n_cells, n_mol = len(rws), len(mols)

    def gate6(vec_len):
        """Non-vacuous: the resampled unit count must be the TARGET count, and must differ from the
        cell count and the molecule count -- so passing a per-cell or per-molecule vector FAILS."""
        if vec_len == n_cells:
            raise AssertionError(f"闸门6:重抽单位是【格】({n_cells}),不是靶标")
        if vec_len == n_mol:
            raise AssertionError(f"闸门6:重抽单位是【分子】({n_mol}),不是靶标")
        if vec_len > n_tg:
            raise AssertionError(f"闸门6:重抽单位 {vec_len} > 靶标总数 {n_tg}")
        return True

    print(f"  非空转版本:靶标 {n_tg} / 格 {n_cells} / 分子 {n_mol} —— 三者互不相等,所以长度可判别单位。")
    print(f"  用真实向量(len={len(used)}):", "通过" if gate6(len(used)) else "?")
    for bad_name, bad_len in (("每格一个值", n_cells), ("每分子一个值", n_mol)):
        try:
            gate6(bad_len)
            print(f"  用【{bad_name}】(len={bad_len}):**没有报错 -> 闸门仍然没用**")
        except AssertionError as e:
            print(f"  用【{bad_name}】(len={bad_len}):正确报错 -> {e}")

    # ---- D4: how is g_t distributed on the sixth batch? -------------------------------------------
    print("\n=== D4 第六批的 g_t 分布(描述性,不改裁定)===")
    g = sorted((st.fmean(per[t]), t, len(per[t])) for t in used)
    vals = [x[0] for x in g]
    print(f"  产生 g_t 的靶标 {len(used)}/{n_tg};每靶标可用划分数 中位 {st.median([len(per[t]) for t in used])}"
          f",最小 {min(len(per[t]) for t in used)},最大 {max(len(per[t]) for t in used)}")
    print(f"  g_t: 中位 {st.median(vals):+.4f}  均值 {st.fmean(vals):+.4f}  标准差 {st.stdev(vals):.4f}"
          f"  最小 {min(vals):+.4f}  最大 {max(vals):+.4f}")
    print(f"  最负 5 个: {[f'{v:+.3f}({t})' for v,t,_n in g[:5]]}")
    print(f"  最正 5 个: {[f'{v:+.3f}({t})' for v,t,_n in g[-5:]]}")
    trimmed = vals[3:-3]
    print(f"  去掉两端各 3 个靶标后的均值 {st.fmean(trimmed):+.4f}(全体 {st.fmean(vals):+.4f})"
          f" ⇒ {'不是少数靶标拉低的' if abs(st.fmean(trimmed)-st.fmean(vals))<0.004 else '受少数靶标影响'}")
    dropped = [t for t in sorted({c[0] for c, _g in rws}) if t not in used]
    print(f"  被地板剔除的靶标 {len(dropped)} 个:{dropped}")
    print(f"    它们各自被划入留出集的次数:{[nsplits[t] for t in dropped]}"
          f"  ⇒ {'不是抽样偏差,是成对数不足' if all(nsplits[t] >= 20 for t in dropped) else '抽样次数本身就不够'}")

    json.dump({"D1_D2": {k: v for k, v in res.items()}, "n_targets": n_tg, "n_cells": n_cells,
               "n_mol": n_mol, "n_used": len(used), "g_t": [(t, round(v, 4)) for v, t, _n in g],
               "dropped_targets": dropped},
              open(os.path.join(HERE, "h2_review_diag.json"), "w"), indent=1, ensure_ascii=False)
    return 0 if (ok_d1 and ok_d2) else 3


if __name__ == "__main__":
    sys.exit(main())
