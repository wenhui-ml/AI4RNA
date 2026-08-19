#!/usr/bin/env python3
"""A11: at the top of a ruler -- where the score has no gradient -- does a NON-SCORE quantity read anything?

Declared in `top_signal_declaration.json`, committed and pushed before a single entropy was computed. The
three things that make this run interpretable:

  THE QUANTITY IS FIXED IN ADVANCE, not selected. certify.py's shipped tie-breaker already picks among
  equal-grade candidates by ensemble entropy and raised outside agreement (IPknot 21.8% -> 27.0%, sign
  p=0.036), while picking purely on entropy scored 34-34. Entropy works on ties and not over the whole range,
  and ties are the saturated top. So there is no select-a-quantity step, and the DIRECTION is declared too:
  higher entropy should mean a better tube score.

  THE OUTCOME IS PHYSICAL AND SHARES NO ERROR PATH WITH THE PREDICTOR. Entropy comes from ViennaRNA's
  partition function; the outcome is the measured OpenKnot score. The tie-breaker has never been tested
  against measurement -- it was judged by IPknot/Knotty/RNAPKplex on our own designs.

  SUCCESS NEEDS BOTH SAMPLES. Samples 2 and 3 are disjoint molecules and each is a full test; a pass on one is
  not a result. That is the discipline A7 lacked this morning -- it held on one sample and went into a commit
  message as a conclusion, and the third sample took it back the same day.

The entropy is of the NESTED ensemble: ViennaRNA does not see pseudoknots, so this quantity says nothing
directly about the knot. That is both why it is interesting as a non-score reading and where its meaning stops.

COST: $0. Partition functions only inside the tie blocks.
"""
from __future__ import annotations

import argparse, collections, json, os, random, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)

from arbitration_confirm import load_fresh                                     # noqa: E402

MIN_BLOCK = 4          # a rank correlation over fewer than four points is not a measurement
MIN_DISTINCT_WET = 3


def spearman(xs, ys):
    """Rank correlation with average ranks for ties; None when either side is constant."""
    def rk(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v); i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2.0 + 1
            i = j + 1
        return out
    a, b = rk(xs), rk(ys)
    n = len(a)
    ma, mb = st.mean(a), st.mean(b)
    va = sum((x - ma) ** 2 for x in a); vb = sum((y - mb) ** 2 for y in b)
    if va == 0 or vb == 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va * vb) ** 0.5


def ent_worker(seq):
    from rna_harness.cognitive.certify import _ensemble_entropy
    return _ensemble_entropy(seq)


def blocks(rec, key):
    """Per cell, the ruler's top tie block -- the designs it literally cannot tell apart."""
    g = collections.defaultdict(list)
    for r in rec:
        g[(r["puzzle"], r["method"])].append(r)
    out = {}
    for k, rows in g.items():
        top = max(x[key] for x in rows)
        blk = [x for x in rows if x[key] == top]
        if len(blk) >= MIN_BLOCK and len({round(x["wet"], 6) for x in blk}) >= MIN_DISTINCT_WET:
            out[k] = blk
    return out


def boot_median(vals, draws, rng):
    """Cluster bootstrap over cells of the MEDIAN per-cell correlation; the unit is the cell."""
    v = list(vals)
    if len(v) < 5:
        return None, None, None
    n = len(v)
    out = sorted(st.median([v[rng.randrange(n)] for _ in range(n)]) for _ in range(draws))
    return st.median(v), out[int(0.025 * draws)], out[min(draws - 1, int(0.975 * draws))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "top_signal.json"))
    a = ap.parse_args()
    D = json.load(open(os.path.join(HERE, "top_signal_declaration.json")))
    res = {"declaration": D, "samples": {}}

    SAMPLES = (("样本2 (160/靶标)", os.path.join(HERE, "panel_f1_fresh.json")),
               ("样本3 (80/靶标)", os.path.join(HERE, "panel_f1_third.json")))
    from concurrent.futures import ProcessPoolExecutor

    for nm, path in SAMPLES:
        rec, _sp, meta = load_fresh(path, "panel_recall_product")
        prim = blocks(rec, "f1c")                  # declared primary population
        seco = blocks(rec, "panel")                # declared secondary population
        need = sorted({x["seq"] for blk in list(prim.values()) + list(seco.values()) for x in blk})
        print(f"\n=== {nm}:{len(rec)} 条分子;RNet 顶端并列块 {len(prim)} 格,"
              f"面板顶端并列块 {len(seco)} 格;要算配分函数的唯一序列 {len(need)} 条 ===", flush=True)
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            E = dict(zip(need, ex.map(ent_worker, need, chunksize=8)))
        bad = sum(1 for v in E.values() if v is None or v == float("-inf"))
        print(f"  熵算完;失败 {bad} 条(失败的序列整块不用)")
        row = {"n_designs": len(rec), "n_entropy": len(E), "n_entropy_failed": bad}

        for pname, pop in (("RNet 顶端并列块(主)", prim), ("面板顶端并列块(次)", seco)):
            print(f"  --- {pname} ---")
            sub = {}
            for k, blk in pop.items():
                v = [(E[x["seq"]], x["wet"], x) for x in blk
                     if E.get(x["seq"]) not in (None, float("-inf"))]
                if len(v) >= MIN_BLOCK and len({round(y, 6) for _e, y, _x in v}) >= MIN_DISTINCT_WET:
                    sub[k] = v
            out_p = {"n_cells": len(sub)}
            # `k` is not a stored column in these files -- the first version of this script read
            # r.get("k"), got None everywhere, and printed "0 usable cells", which reads like a measurement
            # and is not one. It is DERIVED here from per_oracle instead: k = how many oracles reach full
            # crossing recall, which is the shipped confirm rung's own count.
            def _k(t):
                po = t[2].get("per_oracle") or {}
                v = [x for x in po.values() if x]
                if not v:
                    return None
                return sum(1 for x in v if x.get("recall_product", 0) >= 1.0)
            quantities = (("熵(主检验)", lambda t: t[0]),
                          ("n_pred_cross(次)", lambda t: t[2].get("n_pred_cross")),
                          ("面板 k(次,推导)", _k),
                          ("|两尺子分歧|(次)", lambda t: abs(t[2]["f1c"] - t[2]["panel"])))
            for qn, f in quantities:
                rhos = []
                for k, v in sub.items():
                    xs = [f(t) for t in v]
                    if any(x is None for x in xs):
                        continue
                    r = spearman(xs, [t[1] for t in v])
                    if r is not None:
                        rhos.append(r)
                pt, lo, hi = boot_median(rhos, a.draws, random.Random(a.seed))
                if pt is None:
                    print(f"      {qn:20s} 可用格 {len(rhos)} —— 太少,不给数")
                    out_p[qn] = {"n_cells": len(rhos)}; continue
                pos = lo > 0
                print(f"      {qn:20s} 格 {len(rhos):3d}  中位 rho {pt:+.3f} "
                      f"聚类 CI [{lo:+.3f},{hi:+.3f}]"
                      f"{'  ✓ 排除 0 且为正(预测方向)' if pos else ('  ✓ 排除 0 但为负(反方向)' if hi < 0 else '  跨过 0')}")
                out_p[qn] = {"n_cells": len(rhos), "median_rho": pt, "ci": [lo, hi],
                             "excludes_zero_positive": bool(pos), "excludes_zero_negative": bool(hi < 0)}
            row[pname] = out_p
        res["samples"][nm] = row

    k = "RNet 顶端并列块(主)"
    a2 = res["samples"]["样本2 (160/靶标)"][k].get("熵(主检验)", {})
    a3 = res["samples"]["样本3 (80/靶标)"][k].get("熵(主检验)", {})
    both = bool(a2.get("excludes_zero_positive") and a3.get("excludes_zero_positive"))
    print(f"\n=== 判定(按声明:主检验 = RNet 顶端并列块内 熵 vs 试管分,【两批都要】在正方向排除 0)===")
    for nm, r in (("样本2", a2), ("样本3", a3)):
        if "median_rho" in r:
            print(f"  {nm}: rho {r['median_rho']:+.3f} [{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]"
                  f"  {'✓' if r.get('excludes_zero_positive') else '✗'}")
        else:
            print(f"  {nm}: 无数据")
    if both:
        print("  ✓ 两批都成立 → 顶端【有】可读信号。但按声明这【仍不算】能力表一格:")
        print("    还需要第四批全新分子复现 + 一次 A10 式的『能否改变一个决定』检验。")
    else:
        print("  ✗ 不成立 → 先验最好的那个非分数量在顶端也读不出东西。")
        print("    按声明写成:选择的天花板不是『选哪把尺子』的问题 —— 这条直接落到 §三之三 那条线上。")
    res["A11_BOTH_SAMPLES_PASS"] = both
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n写出 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
