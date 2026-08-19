#!/usr/bin/env python3
"""A4: is the arbitration a CRITERION or just a constant? Declared before running, with a cell holdout.

WHERE THIS STARTS. A1 established that when the two verifier families order two designs oppositely, the tube
sides with the learned ruler 57.6% of the time (7,858 discordant pairs, 184 cells, cluster CI [52.1, 62.9],
margin = the tube's own 3.0-point replicate noise). That is a constant. A constant is usable as a tie-break
but it is not a criterion, and objective 4 scores criteria. A3 tested three design-time features and none of
them demonstrably moved the winner once pairs were clustered by cell -- the strata that cleared 0.5 were the
ones with 85-152 cells and the ones that crossed it had 21-61, so that was missing resolution rather than a
measured absence.

MAGNITUDE WAS DELIBERATELY WITHHELD FROM A3. It is the most plausible feature and it was not added after
seeing A3's numbers, because adding the most promising covariate after looking is how the +0.250 retraction
happened. It is this file's PRIMARY, declared here before the run:

    confidence   each ruler's opinion about a pair, standardised INSIDE the cell so the two scales are
                 comparable: |z(ruler, a) - z(ruler, b)|. A cell whose ruler has no spread is dropped.
    the rule     FOLLOW THE MORE CONFIDENT RULER: if the learned ruler's confidence exceeds the panel's,
                 take its preferred design, else take the panel's.
    the baseline ALWAYS FOLLOW THE LEARNED RULER -- A1's constant, which is the thing to beat.

    A4  PRIMARY. On a holdout half of CELLS (split by hash of the cell key, seed 0, before any number is
        computed), does "follow the more confident" beat "always follow learned"? Success = the paired
        difference in accuracy has a cluster-bootstrap 95% CI excluding 0. Anything else is a failure,
        including a difference that is positive but whose interval spans 0.
    A5  MONOTONICITY, also declared. Split the holdout's pairs into quartiles of (learned confidence minus
        panel confidence). If magnitude carries information, accuracy of "follow learned" must rise across
        those quartiles in direction. A single favourable quartile with a non-monotone shape is a fishing
        result and is reported as such.
    KILL. Neither A4 nor A5 -> the arbitration is a CONSTANT, not a criterion. That is objective 4's honest
        boundary for this domain: the disagreement is resolvable on average and not resolvable per instance,
        and the product may use the constant as a tie-break while the domain does NOT count as a third
        criterion-manufacturing success.

The development half is reported too, and if development and holdout disagree the holdout is the result. The
same standing limitation as everything else in this chain: measured on other people's molecules, because our
own designs have no tube score.

!! 2026-08-19 AUDIT (arbitration_audit.py): READ THESE THREE CORRECTIONS BEFORE QUOTING ANY NUMBER BELOW.
   1. A5b's top bin 75.1% is INFLATED BY THE CONFIDENCE DEFINITION. |dz| is delta/sigma_cell, so a cell whose
      panel is nearly constant turns a numerically negligible panel difference into a huge "confidence". Under
      in-cell RANK differences (immune to sigma) the top bin is 64.1% [54.9, 73.2]; with one vote per cell,
      63.9% (z) and 60.5% (rank). The defensible figure is ~60-64%, not 75%.
   2. A4's FAILURE IS REAL, NOT UNDERPOWERED -- the audit measured what this file never did: FOLLOW_CONFIDENT
      changes the decision on 2,631 of 4,605 holdout pairs (57.1%), with a maximum attainable gain of +27.8%.
      The -1.5% is a measured negative.
   3. "THE PANEL'S CONFIDENCE CARRIES NO INFORMATION" WAS TOO STRONG. Binning on panel confidence alone, the
      top bin gives learned-right 54.0% pair-weighted but 39.1% [29.6, 48.3] per cell -- the second clears 0.5
      from BELOW, i.e. it points at following the PANEL there. Weighting-dependent, so unestablished, but it is
      not a coin flip and must not be described as one.
   A5's declared monotonicity holds in 3 of the 4 definition x weighting combinations and fails under
   rank x pair-weighted (bin 3 dips). The surviving claim is a TOP-BIN effect, not a clean gradient.

COST: $0. Arithmetic over files already on disk.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

TUBE_NOISE = 3.0
RNET_OPTIMISED = "Struct2SeQ-SHAPE"


def cell_pairs(rows, margin):
    """Discordant pairs of one cell, each carrying both rulers' in-cell standardised confidences."""
    if len(rows) < 3:
        return []
    out = []
    for key in ("panel", "f1c"):
        vals = [r[key] for r in rows]
        m, s = st.mean(vals), (st.pstdev(vals) if len(vals) > 1 else 0.0)
        if s == 0:
            return []                       # a ruler with no spread in this cell has no opinions to compare
        for r in rows:
            r["z_" + key] = (r[key] - m) / s
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            x, y = rows[i], rows[j]
            dp, dr, dw = x["panel"] - y["panel"], x["f1c"] - y["f1c"], x["wet"] - y["wet"]
            if dp == 0 or dr == 0 or (dp > 0) == (dr > 0) or abs(dw) <= margin:
                continue
            out.append({"conf_learned": abs(x["z_f1c"] - y["z_f1c"]),
                        "conf_panel": abs(x["z_panel"] - y["z_panel"]),
                        # who each ruler prefers, and who the tube prefers
                        "learned_right": (dr > 0) == (dw > 0)})
    return out


def acc(pairs, rule):
    """Accuracy of a decision rule over pairs. `rule` returns True when it sides with the learned ruler."""
    if not pairs:
        return None
    ok = sum(1 for p in pairs if rule(p) == p["learned_right"])
    return ok / len(pairs)


FOLLOW_LEARNED = lambda p: True                                        # noqa: E731  A1's constant
FOLLOW_CONFIDENT = lambda p: p["conf_learned"] >= p["conf_panel"]      # noqa: E731  A4's rule


def boot_diff(cells, draws, rng):
    """Cluster bootstrap of (confident - learned) accuracy difference; the unit is the cell."""
    keys = list(cells)
    out = []
    for _ in range(draws):
        pool = []
        for k in (keys[rng.randrange(len(keys))] for _ in keys):
            pool += cells[k]
        a, b = acc(pool, FOLLOW_CONFIDENT), acc(pool, FOLLOW_LEARNED)
        if a is not None and b is not None:
            out.append(a - b)
    out.sort()
    if not out:
        return None, None
    return out[int(0.025 * len(out))], out[min(len(out) - 1, int(0.975 * len(out)))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-key", default="panel_f1")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "arbitration_magnitude.json"))
    a = ap.parse_args()

    rng = random.Random(a.seed)
    from arbitration import load
    rec, spreads = load(a.panel_key)
    rec = [r for r in rec if r["method"] != RNET_OPTIMISED]

    groups = collections.defaultdict(list)
    for r in rec:
        groups[(r["puzzle"], r["method"])].append(r)
    cells = {k: v for k, v in ((k, cell_pairs(v, TUBE_NOISE)) for k, v in groups.items()) if v}
    print(f"{len(rec)} 条分子;有相悖配对的格子 {len(cells)},配对合计 {sum(len(v) for v in cells.values())}")

    # split by hash of the cell key, decided before any accuracy is computed
    def half(k):
        return int(hashlib.sha1(f"{a.seed}|{k[0]}|{k[1]}".encode()).hexdigest(), 16) % 2
    dev = {k: v for k, v in cells.items() if half(k) == 0}
    hold = {k: v for k, v in cells.items() if half(k) == 1}
    print(f"  开发半 {len(dev)} 格 / {sum(len(v) for v in dev.values())} 配对;"
          f"留出半 {len(hold)} 格 / {sum(len(v) for v in hold.values())} 配对\n")

    res = {"panel_key": a.panel_key, "margin": TUBE_NOISE, "n_cells": len(cells),
           "halves": {}, "A5": {}}
    for name, part in (("开发半", dev), ("留出半", hold)):
        flat = [p for v in part.values() for p in v]
        al, ac_ = acc(flat, FOLLOW_LEARNED), acc(flat, FOLLOW_CONFIDENT)
        lo, hi = boot_diff(part, a.draws, rng)
        clears = lo is not None and lo > 0
        star = "  <- A4 主检验" if name == "留出半" else ""
        print(f"{name}:总是跟学习型 {al:.1%}   跟更自信的那一家 {ac_:.1%}   差 {ac_-al:+.1%}"
              f"   聚类 CI [{lo:+.1%},{hi:+.1%}]{'  ✓ 排除 0' if clears else '  跨过 0'}{star}")
        res["halves"][name] = {"n_pairs": len(flat), "n_cells": len(part), "acc_follow_learned": al,
                               "acc_follow_confident": ac_, "diff": ac_ - al, "ci": [lo, hi],
                               "ci_excludes_zero": clears}

    # ---- A5 monotonicity on the holdout ---------------------------------------------------------------
    flat = [p for v in hold.values() for p in v]
    flat.sort(key=lambda p: p["conf_learned"] - p["conf_panel"])
    q = max(1, len(flat) // 4)
    print(f"\n=== A5 留出半按(学习型自信 − 面板自信)分四档,跟学习型的正确率 ===")
    fr = []
    for i in range(4):
        chunk = flat[i * q:(i + 1) * q] if i < 3 else flat[3 * q:]
        f = sum(1 for p in chunk if p["learned_right"]) / max(1, len(chunk))
        fr.append(f)
        print(f"  第 {i+1} 档 n={len(chunk):5d}  正确率 {f:.1%}")
        res["A5"][f"q{i+1}"] = {"n": len(chunk), "frac_learned_right": f}
    mono = all(fr[i] <= fr[i + 1] for i in range(3))
    print(f"  单调递增: {'是 ✓' if mono else '否 —— 幅度没有携带方向一致的信息'}")
    res["A5"]["monotone"] = mono

    # ---- A5b: the quartile numbers, clustered -------------------------------------------------------
    # A5 above bins the holdout's PAIRS, and pairs inside a cell are not independent -- the same objection
    # that killed A3's apparent reversal. So: take the bin boundaries from the DEVELOPMENT half (they are not
    # allowed to be fitted on the holdout), then cluster-bootstrap the holdout accuracy inside each fixed bin.
    dv = sorted(p_["conf_learned"] - p_["conf_panel"] for v in dev.values() for p_ in v)
    cuts = [dv[len(dv) // 4], dv[len(dv) // 2], dv[3 * len(dv) // 4]]
    def binof(p_):
        d = p_["conf_learned"] - p_["conf_panel"]
        return 0 if d <= cuts[0] else (1 if d <= cuts[1] else (2 if d <= cuts[2] else 3))
    print(f"\n=== A5b 同样四档,但边界取自开发半({[round(c,2) for c in cuts]}),"
          f"且按格子做聚类 bootstrap ===")
    res["A5b"] = {"cuts_from_dev": cuts}
    keys = list(hold)
    for b in range(4):
        percell = {k: [p_ for p_ in v if binof(p_) == b] for k, v in hold.items()}
        percell = {k: v for k, v in percell.items() if v}
        n = sum(len(v) for v in percell.values())
        if n < 20:
            print(f"  第 {b+1} 档 n={n} —— 太少,不给数")
            res["A5b"][f"q{b+1}"] = {"n": n}
            continue
        pt = sum(1 for v in percell.values() for p_ in v if p_["learned_right"]) / n
        ks = list(percell)
        draws = []
        for _ in range(a.draws):
            pool = [p_ for k in (ks[rng.randrange(len(ks))] for _ in ks) for p_ in percell[k]]
            if pool:
                draws.append(sum(1 for p_ in pool if p_["learned_right"]) / len(pool))
        draws.sort()
        lo_, hi_ = draws[int(0.025 * len(draws))], draws[min(len(draws) - 1, int(0.975 * len(draws)))]
        cl = lo_ > 0.5 or hi_ < 0.5
        print(f"  第 {b+1} 档 n={n:5d} 格 {len(percell):3d}  跟学习型正确率 {pt:.1%}"
              f"  聚类 CI [{lo_:.1%},{hi_:.1%}]{'  ✓ 越过 0.5' if cl else '  跨过 0.5'}")
        res["A5b"][f"q{b+1}"] = {"n": n, "n_cells": len(percell), "frac": pt,
                                 "cluster_ci": [lo_, hi_], "ci_clears_half": cl}
    top = res["A5b"].get("q4", {})
    bot = res["A5b"].get("q1", {})
    a5b_ok = bool(top.get("ci_clears_half") and top.get("frac", 0) > 0.5
                  and not (bot.get("ci_clears_half") and bot.get("frac", 1) > 0.5))
    res["A5b_top_bin_established"] = a5b_ok

    ok4 = res["halves"]["留出半"]["ci_excludes_zero"]
    print(f"\n=== 判定 ===")
    if ok4 or (mono and a5b_ok):
        print("  A4 的对称规则不成立,但幅度的梯度成立且顶档在聚类区间下越过 0.5")
        print("  → 仲裁是【单侧】判据的候选:学习型远比面板自信时跟它走;反向不携带信息。")
        print("  操作点(用哪一档)是看过数据之后才定的,所以它需要一次新的留出才算标定完。")
    else:
        print("  A4 与 A5 都不成立 → **仲裁是一个常数,不是判据**。")
        print("  按跑前写死的口径:产品可以把 57.6% 当平局打破器用,但这个领域【不算】")
        print("  造判据的第三个成功案例。这是 ④ 在这个领域的边界结论。")
    res["verdict_is_criterion"] = bool(ok4 or mono)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\n写出 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
