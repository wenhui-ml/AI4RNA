#!/usr/bin/env python3
"""A6, CONFIRMATORY: does the one-sided arbitration criterion hold on designs nobody has looked at?

WHAT WAS WRONG WITH THE EXPLORATORY VERSION, and it was only this. The top-quartile boundary that produced
"follow the learned ruler when it is much more confident than the panel" was chosen AFTER the bins were
printed. Everything else about it survived a full audit (`arbitration_audit.py`): the effect clears 0.5 under
both confidence definitions and both weightings. So the single missing thing is an operating point that was
fixed before the data existed.

WHAT MAKES THIS A REAL HOLDOUT. `criterion_vs_tube.json` sampled 80 designs per target (seed 0) out of 34,470
eligible OpenKnot designs; 29,910 were never touched, and all 57 targets can supply another 160. This run
scores a DISJOINT 160-per-target sample (9,120 designs, overlap asserted to be zero in
`novelty_stratified_ruler.replay_fresh`) and applies the cuts frozen in
`arbitration_confirm_declaration.json`, which was committed and pushed before a single fresh design was
folded. The cuts are absolute numbers and are NOT refitted here.

DECLARED, and reproduced from the declaration file at run time rather than retyped:

    primary        in-cell RANK difference of the two rulers' opinions about a pair, d = conf_learned -
                   conf_panel. Rank rather than z because |dz| is delta/sigma_cell, and a cell whose panel is
                   nearly constant turns a negligible difference into a huge "confidence" -- which is where
                   11 of the exploratory 75.1% came from.
    estimator      ONE VOTE PER CELL, cluster bootstrap 95% CI. The pair-weighted version is secondary,
                   because the ten largest cells held 45.3% of the exploratory pairs and pair-weighting is
                   what turned a 53.8% coin flip into a "57.6% constant".
    success        the TOP region's per-cell CI excludes 0.5 from above.
    one-sidedness  the BOTTOM region's per-cell CI must NOT exclude 0.5 from above. If it does, the criterion
                   is two-sided and the symmetric rule A4 rejected should be revisited.
    margin         pairs whose tube scores differ by <= 3.0 are dropped; the tube cannot referee inside its
                   own replicate noise. 3.0 is the exploratory estimate and is NOT re-fitted here, but this
                   sample yields an independent estimate of it, which is reported as a check.

ALSO REPORTED, not part of the test: the overall fraction (A1's "constant") on fresh data under both
weightings, since its retraction is the other half of today's audit and a replication is free.

THE STANDING LIMIT, unchanged: measured on other people's molecules, because our own designs have no tube
score. And the panel column is `panel_f1`, an observable added on 2026-08-19, not the shipped recall rung.

COST: $0. Reads two files on disk.
"""
from __future__ import annotations

import argparse, collections, json, os, random, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)

RNET_OPTIMISED = "Struct2SeQ-SHAPE"


def load_fresh(path, panel_key):
    """Rows of the fresh panel run, deduplicated by molecule the way ruler_gradient.load does it."""
    d = json.load(open(path))
    if d["n_failed"]:
        raise SystemExit(f"{d['n_failed']} designs failed to fold; fix before analysing")
    rows = [r for r in d["designs"] if r["method"] != RNET_OPTIMISED]
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["puzzle"], r["seq"])].append(r)
    out, spreads = [], []
    for _, cp in g.items():
        if len(cp) > 1:
            spreads.append(max(x["wet"] for x in cp) - min(x["wet"] for x in cp))
            assert len({round(x[panel_key], 9) for x in cp}) == 1, "same molecule, different panel score"
        r = dict(cp[0])
        r["wet"] = sum(x["wet"] for x in cp) / len(cp)
        r["panel"] = r[panel_key]
        out.append(r)
    return out, spreads, d


def cluster(percell, draws, rng, unweighted):
    ks = list(percell)
    if not ks:
        return None, None, None
    def est(sel):
        if unweighted:
            fs = [sum(1 for p in percell[k] if p["learned_right"]) / len(percell[k]) for k in sel]
            return st.mean(fs) if fs else None
        num = sum(1 for k in sel for p in percell[k] if p["learned_right"])
        den = sum(len(percell[k]) for k in sel)
        return num / den if den else None
    pt = est(ks)
    out = [v for v in (est([ks[rng.randrange(len(ks))] for _ in ks]) for _ in range(draws)) if v is not None]
    out.sort()
    return pt, out[int(0.025 * len(out))], out[min(len(out) - 1, int(0.975 * len(out)))]


def report(percell, draws, seed, label, res):
    """Both weightings for one region. The weaker of the two governs the claim."""
    n = sum(len(v) for v in percell.values())
    if n < 20:
        print(f"  {label:10s} n={n} —— 太少,不给数")
        res[label] = {"n": n}
        return res[label]
    pw, lw, hw = cluster(percell, draws, random.Random(seed), False)
    pu, lu, hu = cluster(percell, draws, random.Random(seed + 1), True)
    above_w, above_u = lw > 0.5, lu > 0.5
    below_u = hu < 0.5
    print(f"  {label:10s} n={n:5d} 格{len(percell):3d}   加权 {pw:.1%} [{lw:.1%},{hw:.1%}]"
          f"{'✓上越' if above_w else ('✓下越' if hw < 0.5 else '跨0.5')}"
          f"   每格一票 {pu:.1%} [{lu:.1%},{hu:.1%}]"
          f"{'✓上越' if above_u else ('✓下越' if below_u else '跨0.5')}")
    res[label] = {"n": n, "n_cells": len(percell), "weighted": [pw, lw, hw], "percell": [pu, lu, hu],
                  "percell_clears_half_above": bool(above_u), "percell_clears_half_below": bool(below_u),
                  "weighted_clears_half_above": bool(above_w)}
    return res[label]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", default=os.path.join(HERE, "panel_f1_fresh.json"))
    ap.add_argument("--declaration", default=os.path.join(HERE, "arbitration_confirm_declaration.json"))
    ap.add_argument("--panel-key", default="panel_f1")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "arbitration_confirm.json"))
    a = ap.parse_args()

    from arbitration_audit import cell_pairs
    D = json.load(open(a.declaration))
    margin = D["margin_tube_noise"]
    rec, spreads, meta = load_fresh(a.fresh, a.panel_key)
    print(f"新样本 {meta['n']} 条(每靶标 {meta.get('fresh_per_target')} 条,与旧 80 条互斥),"
          f"排除 {RNET_OPTIMISED} 后去重得 {len(rec)} 条分子")
    if spreads:
        print(f"独立复现的试管复测噪声:中位 {st.median(spreads):.1f} 分({len(spreads)} 条重复分子);"
              f"声明里钉死的边距是 {margin:.1f},不改")

    groups = collections.defaultdict(list)
    for r in rec:
        groups[(r["puzzle"], r["method"])].append(r)
    cells = {k: v for k, v in ((k, cell_pairs(v, margin)) for k, v in groups.items()) if v}
    npairs = sum(len(v) for v in cells.values())
    sizes = sorted((len(v) for v in cells.values()), reverse=True)
    print(f"有相悖配对的格子 {len(cells)},配对 {npairs}"
          f"(每格中位 {st.median(sizes):.0f},最大 {sizes[0]},前10格占 {sum(sizes[:10])/npairs:.1%})\n")

    res = {"declaration": D, "n_designs": len(rec), "n_cells": len(cells), "n_pairs": npairs,
           "tube_noise_median_fresh": (st.median(spreads) if spreads else None)}

    print("=== 参考:A1 的「常数」在新数据上（不是本次检验）===")
    res["A1_replication"] = {}
    report(cells, a.draws, a.seed, "全部配对", res["A1_replication"])

    for tag, kl, kp, cut_lo, cut_hi in (
            ("秩(主)", "rl", "rp", D["cut_rank_bottom_quartile"], D["cut_rank_top_quartile"]),
            ("z(次)", "cl", "cp", D["cut_z_bottom_quartile"], D["cut_z_top_quartile"])):
        print(f"\n=== {tag} 自信定义,切点冻结于 [{cut_lo:.4f}, {cut_hi:.4f}] ===")
        # DIAGNOSTIC, not part of the test: an operating point frozen as an absolute number only transfers if
        # the quantity's distribution transfers. The fresh cells are larger (median 7 designs vs 4), and rank
        # differences are computed inside a cell, so where the frozen cut LANDS on fresh data is worth seeing.
        # If it lands far from the quartiles it was taken from, the criterion's threshold is a property of how
        # many candidates the user happens to have -- which would be a usability finding, not a null.
        dd = sorted(p[kl] - p[kp] for v in cells.values() for p in v)
        def where(c):
            return sum(1 for x in dd if x <= c) / len(dd)
        print(f"  冻结切点落在新样本的第 {where(cut_lo):.1%} / {where(cut_hi):.1%} 分位"
              f"(取自旧样本时是 25% / 75%)")
        reg = {"cut_lands_at_fresh_quantile": [where(cut_lo), where(cut_hi)]}
        for name, lo, hi in (("底档", -1e9, cut_lo), ("中档", cut_lo, cut_hi), ("顶档", cut_hi, 1e9)):
            pc = {k: [p for p in v if lo < (p[kl] - p[kp]) <= hi] for k, v in cells.items()}
            pc = {k: v for k, v in pc.items() if v}
            report(pc, a.draws, a.seed, name, reg)
        res[tag] = reg

    top = res["秩(主)"]["顶档"]; bot = res["秩(主)"]["底档"]
    ok = bool(top.get("percell_clears_half_above"))
    one_sided = not bool(bot.get("percell_clears_half_above"))
    print(f"\n=== 判定(按声明,主口径 = 秩 × 每格一票)===")
    def show(label, r, yes, no):
        # A region can come back with fewer than 20 pairs -- the frozen cut is an absolute number and there is
        # no guarantee it splits fresh data into quarters. An unmeasurable region is NOT a pass and NOT a
        # failure, and it must not be silently read as either.
        if "percell" not in r:
            print(f"  {label} 配对 {r.get('n', 0)} 条 —— 冻结切点在新样本上没有切出可测的区域,"
                  f"这一侧无结论(既不算通过也不算失败)")
            return None
        pt, lo, hi = r["percell"]
        print(f"  {label} {pt:.1%} [{lo:.1%},{hi:.1%}]  {yes if lo > 0.5 else no}")
        return lo > 0.5
    got_top = show("顶档", top, "✓ 从上方越过 0.5 —— 声明的成功条件达成", "✗ 跨过 0.5 —— 判据没有确认")
    got_bot = show("底档", bot, "也从上方越过了 0.5 —— 判据是双侧的,A4 要重开", "仍未从上方越过 0.5 —— 单侧性成立")
    if got_top is None:
        ok, one_sided = False, False
    if got_bot is None:
        one_sided = False
    o = D["old_exploratory_values_being_tested"]
    print(f"  探索期的对应值:顶档 每格一票 {o['rank_top_bin_percell']:.1%},"
          f"底档 {o['rank_bottom_bin_percell']:.1%}")
    res["CONFIRMED"] = ok
    res["ONE_SIDED"] = one_sided
    if not ok:
        print("\n  按跑前写死的口径:确认失败 = 这个领域【不算】造判据的第三个成功案例,"
              "探索期的顶档效应记为未复现。")
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\n写出 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
