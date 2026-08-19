#!/usr/bin/env python3
"""Why did A10 fail? EXPLORATORY -- this explains a pre-registered failure, it does not rescue it.

A10's declared primary came back null and its licensed group was 50.0% [37.0, 63.0] -- a coin flip. Before
that becomes a story, one control has to be run and one hypothesis has to be measured rather than told.

  THE CONTROL. Does the criterion replicate AT ALL on this third sample? A6/A7 confirmed it on samples 1-2
  over general discordant pairs. If sample 3's general discordant pairs also show it, then A10's failure is
  about the DECISION, not about the sample. If they do not, something is wrong with the sample or the join and
  no explanation of A10 is worth hearing.

  THE HYPOTHESIS, testable. A10 does not ask about a random discordant pair. It asks about the pair
  (panel's argmax, learned ruler's argmax) -- both candidates sitting at the TOP of their own ruler. This
  project has measured, five separate times, that a bounded verifier loses its gradient at its own ceiling:
  the panel's rank correlation with quality decays as its score rises (+0.279 / +0.158 / +0.040), and 36.1% of
  measured designs sit at RNet crossed-pair F1 = 1.000 exactly. If that is what happened, the criterion should
  work on discordant pairs LOW in both rulers and fail on pairs HIGH in both -- and every decision this
  product makes is taken at the top. Stratified by how high the pair sits, on the same frozen cut.

Nothing here is pre-registered and nothing here may enter the capability table. Its only job is to say whether
A10's null is "the criterion is weak" or "the criterion is measured in a place where no decision is taken",
because those two have different consequences and the difference is measurable.

COST: $0.
"""
from __future__ import annotations

import argparse, collections, json, os, random, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", ".."))); sys.path.insert(0, HERE)

from arbitration_audit import cell_pairs                                       # noqa: E402
from arbitration_confirm import load_fresh                                     # noqa: E402
from arbitration_absolute import ranks01                                       # noqa: E402

MARGIN = 3.0
CUTS = {"panel_recall_product": 0.16406, "panel_f1": 0.2313432835820895}


def cluster(percell, draws, rng):
    fs = [sum(1 for p in v if p["learned_right"]) / len(v) for v in percell.values()]
    if not fs:
        return None, None, None
    n = len(fs)
    out = sorted(sum(fs[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws))
    return sum(fs) / n, out[int(0.025 * draws)], out[min(draws - 1, int(0.975 * draws))]


def pairs_with_height(rec, margin):
    """Discordant pairs per cell, each also carrying HOW HIGH it sits in each ruler.

    height = the lower of the two candidates' in-cell ranks, per ruler. A pair of two top candidates has a
    high `h_learned` and `h_panel`; a pair near the bottom has low ones. `top_contest` marks the single pair
    A10 actually decided on: both candidates are their own ruler's argmax.
    """
    g = collections.defaultdict(list)
    for r in rec:
        g[(r["puzzle"], r["method"])].append(r)
    cells = {}
    for k, rows in g.items():
        if len(rows) < 3:
            continue
        vf, vp = [x["f1c"] for x in rows], [x["panel"] for x in rows]
        if st.pstdev(vf) == 0 or st.pstdev(vp) == 0:
            continue
        rf, rp = ranks01(vf), ranks01(vp)
        mf, mp = max(vf), max(vp)
        for x, a_, b_ in zip(rows, rf, rp):
            x["rf"], x["rp"] = a_, b_
        ps = []
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                x, y = rows[i], rows[j]
                dr, dp, dw = x["f1c"] - y["f1c"], x["panel"] - y["panel"], x["wet"] - y["wet"]
                if dr == 0 or dp == 0 or (dr > 0) == (dp > 0) or abs(dw) <= margin:
                    continue
                ps.append({"learned_right": (dr > 0) == (dw > 0),
                           "d": abs(x["rf"] - y["rf"]) - abs(x["rp"] - y["rp"]),
                           "h_learned": min(x["rf"], y["rf"]), "h_panel": min(x["rp"], y["rp"]),
                           "top_contest": ((x["f1c"] == mf and y["panel"] == mp)
                                           or (y["f1c"] == mf and x["panel"] == mp))})
        if ps:
            cells[k] = ps
    return cells


def region(cells, cut, draws, seed, keep=None, label=""):
    sub = {}
    for k, v in cells.items():
        w = [p for p in v if (keep is None or keep(p))]
        if w:
            sub[k] = w
    top = {k: [p for p in v if p["d"] > cut] for k, v in sub.items()}
    top = {k: v for k, v in top.items() if v}
    n = sum(len(v) for v in top.values())
    if n < 20:
        print(f"  {label:26s} 顶档 n={n} —— 太少,不给数")
        return {"n": n}
    pt, lo, hi = cluster(top, draws, random.Random(seed))
    npair = sum(len(v) for v in sub.values())
    print(f"  {label:26s} 相悖 {npair:5d}  顶档 n={n:4d} 格{len(top):3d}  "
          f"跟学习型 {pt:.1%} [{lo:.1%},{hi:.1%}]{'  ✓ 上越 0.5' if lo > 0.5 else '  跨 0.5'}")
    return {"n_pairs_stratum": npair, "n": n, "n_cells": len(top), "frac": pt, "ci": [lo, hi],
            "clears_above": bool(lo > 0.5)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--third", default=os.path.join(HERE, "panel_f1_third.json"))
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "arbitration_decision_why.json"))
    a = ap.parse_args()
    res = {"exploratory": True, "pre_registered": False}

    for col in ("panel_recall_product", "panel_f1"):
        cut = CUTS[col]
        rec, _sp, meta = load_fresh(a.third, col)
        cells = pairs_with_height(rec, MARGIN)
        tag = "出厂列" if col == "panel_recall_product" else "观测列"
        print(f"\n=== {tag} {col};切点 {cut:.5f} ===")
        row = {}
        # THE CONTROL FIRST. If the criterion does not replicate on this sample's general discordant pairs,
        # nothing below explains anything.
        row["control_all_pairs"] = region(cells, cut, a.draws, a.seed, None, "对照:全部相悖配对")

        # height terciles, boundaries from THIS sample's own distribution (exploratory, so stated not hidden)
        hs = sorted(min(p["h_learned"], p["h_panel"]) for v in cells.values() for p in v)
        q1, q2 = hs[len(hs) // 3], hs[2 * len(hs) // 3]
        print(f"  高度三分位(两把尺子里较低的那个格内秩):{q1:.3f} / {q2:.3f}")
        row["height_cuts"] = [q1, q2]
        for nm, f in (("低位对", lambda p: min(p["h_learned"], p["h_panel"]) <= q1),
                      ("中位对", lambda p: q1 < min(p["h_learned"], p["h_panel"]) <= q2),
                      ("高位对", lambda p: min(p["h_learned"], p["h_panel"]) > q2)):
            row[nm] = region(cells, cut, a.draws, a.seed, f, nm)
        row["top_contest"] = region(cells, cut, a.draws, a.seed,
                                    lambda p: p["top_contest"], "A10 决定的那个配对")
        res[col] = row

    c = res["panel_recall_product"]
    print(f"\n=== 读法 ===")
    ok_ctrl = c["control_all_pairs"].get("clears_above")
    hi_ = c.get("高位对", {}).get("clears_above")
    lo_ = c.get("低位对", {}).get("clears_above")
    if not ok_ctrl:
        print("  对照都不成立 → 判据在第三批的一般相悖配对上就没复现,A10 的空结果【不能】归因于饱和,")
        print("  必须先查样本或连接是否有问题。")
    elif lo_ and not hi_:
        print("  对照成立、低位对成立、高位对不成立 → 信号在【尺子还有梯度】的地方,")
        print("  而产品的每一个决定都在顶端做。这是可测的解释,不是讲的故事。")
    else:
        print("  分层没有给出干净的高低差 → 饱和解释【没有被测到】,A10 的空结果暂时无机制。")
    res["control_replicates"] = bool(ok_ctrl)
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n写出 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
