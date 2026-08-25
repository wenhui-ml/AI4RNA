#!/usr/bin/env python3
"""M2d: cash the two VOIDs. A void only counts paired with a re-test -- otherwise it launders a failure.

Pre-registered in m2d_void_repair_declaration.json.

VOID 1 -- M2b arm1, void by its own V2. The reliability estimator halved the CELLS, and a target holds 5.9
cells on average, so each half kept ~3. That was not low reliability, it was an unmeasurable one. Repair:
halve the DESIGNS INSIDE each cell instead. Every cell and every target survives; a cell of ~19.5 designs
gives halves of ~10, so within-cell pairs fall from ~190 to ~45, still far above the floor of 30.

VOID 2 -- M2's void gate 1. Its declared must-fire probe, mean(wet), did not fire, and I then added a
post-hoc probe that did and used it to certify the screen. That is a post-hoc criterion RESCUING my own
negative -- the mirror of the M3 mistake. Repair: freeze the synthetic probe in advance, and close the rule-50
gap by adding the data-quantity family (n_cells / n_designs / n_pairs), which was never among the ten.

Scoring uses C1's fixed rule throughout: common pair set, prediction tie worth 0.5.

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
from m2_invariant_split import spearman, rho_ci, rho_perm_p, CANDIDATES, invariants, R_SEED   # noqa: E402
from m2b_gt_target_property import fold                                           # noqa: E402
from m2b_review_audit import acc_pair_of_blocks, counts_per_target                # noqa: E402

POPS = (("样本2(主)", "panel_f1_fresh.json"), ("基准4560", "panel_f1_designs.json"),
        ("样本3", "panel_f1_third.json"), ("第四批", "panel_f1_fourth.json"))
PRIMARY = "panel_f1_fresh.json"
EXTRA = ["n_cells", "n_designs", "n_pairs"]                 # the rule-50 gap R3 exposed
ALL_CAND = CANDIDATES + EXTRA
ALPHA_B = 0.05 / len(ALL_CAND)


def g_from(keep, M, splits, seed, design_half=None, half_seed=131):
    """design_half in {None,0,1}: keep half the DESIGNS inside every cell (cells and targets all survive)."""
    cells, cols = collections.defaultdict(list), {}
    for r, m in zip(keep, M):
        if "error" in m:
            continue
        rr = dict(r)
        cols[id(rr)] = base_block(r) + m["C"] + [m["ed_skel"], m["ed_mfe"]]
        cells[(r["puzzle"], r["method"])].append(rr)
    if design_half is not None:
        hr = random.Random(half_seed)
        nc = {}
        for c, g in cells.items():
            gg = list(g); hr.shuffle(gg)
            nc[c] = [x for i, x in enumerate(gg) if i % 2 == design_half]
        cells = nc
    rws = [(c, g) for c, g in cells.items() if len(g) >= 2]
    tg = sorted({c[0] for c, _g in rws})
    nB = len(base_block(keep[0]))
    iBC = list(range(nB + 2)); iED = iBC + [nB + 2]
    by_t = collections.defaultdict(list)
    for c, gp in rws:
        by_t[c[0]].append((c, gp))
    per_t = {t: [] for t in tg}
    rng = random.Random(seed); order = list(tg); leak = 0
    for _s in range(splits):
        rng.shuffle(order)
        cut = max(1, int(0.7 * len(order)))
        tr, te = set(order[:cut]), set(order[cut:])
        TR = [x for t in tr for x in by_t[t]]; TE = [x for t in te for x in by_t[t]]
        if not TR or not TE:
            continue
        if {id(r) for _c, g in TR for r in g} & {id(r) for _c, g in TE for r in g}:
            leak += 1
        PA = fit_predict(TR, TE, cols, iBC); PB = fit_predict(TR, TE, cols, iED)
        for t in te:
            aA, aB, nA, nB2, _tt, _ta, _tb = acc_pair_of_blocks(by_t[t], PA, PB, "common05")
            if aA is None or aB is None or nA < H.MIN_PAIRS_PER_TARGET:
                continue
            per_t[t].append(aB - aA)
    g = {t: st.fmean(v) for t, v in per_t.items() if len(v) >= H.MIN_SPLITS_PER_TARGET}
    return g, by_t, leak


def rr(a, b, seed, level=0.95):
    ts = sorted(set(a) & set(b))
    if len(ts) < 15:
        return {"n": len(ts), "skipped": True}
    x = [a[t] for t in ts]; y = [b[t] for t in ts]
    ci = rho_ci(x, y, seed, level)
    ex = 0
    for s in range(R_SEED):
        c = rho_ci(x, y, 6600 + 13 * s, level)
        ex += 1 if (c[0] > 0 or c[1] < 0) else 0
    return {"n": len(ts), "rho": round(spearman(x, y), 4), "ci": ci,
            "excl": bool(ci[0] > 0 or ci[1] < 0), "seed_flip": f"{ex}/{R_SEED}"}


def sb(r):
    return 2 * r / (1 + r) if r > -1 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "m2d_void_repair.json"))
    a = ap.parse_args()
    R = {"repairs": ["M2b arm1 VOID by V2", "M2 VOID by void gate 1"]}
    G, GA, GB, BYT = {}, {}, {}, {}
    print("=== 兑现作废:信度换成【格内设计对半】,筛选补上数据量族与冻结的必响探针 ===", flush=True)
    for nm, fn in POPS:
        keep, M = fold(fn, a.workers)
        G[fn], BYT[fn], lk = g_from(keep, M, a.splits, a.seed)
        GA[fn], _b, _l = g_from(keep, M, a.splits, a.seed, design_half=0)
        GB[fn], _b, _l = g_from(keep, M, a.splits, a.seed, design_half=1)
        v = list(G[fn].values())
        print(f"  {nm:10s} 靶标 全量 {len(G[fn])} / 半A {len(GA[fn])} / 半B {len(GB[fn])}"
              f"  均值 {st.fmean(v):+.4f}  泄漏 {lk}/{a.splits} {'✓' if not lk else '✗'}", flush=True)

    print("\n=== 闸门 V2:管线本身的必响输入(不拆分,复制两份,rho 必须 > 0.99)===")
    p = rr(G[PRIMARY], dict(G[PRIMARY]), a.seed + 3)
    okp = p.get("rho", 0) > 0.99
    print(f"  复制两份 rho {p.get('rho')}  {'**> 0.99 ✓**' if okp else '✗ 管线不确定 -> 作废'}")
    R["V2_pipeline"] = {"rho": p.get("rho"), "ok": okp}
    if not okp:
        R["verdict"] = "void_pipeline_nondeterministic"
        json.dump(R, open(a.out, "w"), indent=1, ensure_ascii=False); return 0

    print("\n=== 修好的信度:格内【设计】对半(格数与靶标数全部保留)===")
    rel = {}
    for nm, fn in POPS:
        o = rr(GA[fn], GB[fn], a.seed + 6)
        full = round(sb(o["rho"]), 4) if "rho" in o else None
        rel[fn] = {"half": o, "full_SB": full}
        print(f"  {nm:10s} 半长 rho {o.get('rho')} 95% {o.get('ci')}"
              f" {'**排除 0**' if o.get('excl') else '含 0'}  [种子 {o.get('seed_flip')}]"
              f"  | Spearman-Brown 全长 {full}")
    R["reliability"] = rel
    K1 = not all(rel[fn]["half"].get("excl") for _n, fn in POPS)

    print("\n=== 跨人群 rho 对信度上限(M2b arm1 的那个分母,现在有了)===")
    pairs = []
    for i, (n1, f1) in enumerate(POPS):
        for n2, f2 in POPS[i + 1:]:
            o = rr(G[f1], G[f2], a.seed + 11)
            r1, r2 = rel[f1]["full_SB"], rel[f2]["full_SB"]
            ceil = round(math.sqrt(r1 * r2), 4) if (r1 and r2 and r1 > 0 and r2 > 0) else None
            pairs.append({"pair": f"{n1}|{n2}", **o, "ceiling": ceil})
            print(f"  {n1:10s} vs {n2:10s} rho {o.get('rho'):+.4f} {o.get('ci')}"
                  f" {'排除0' if o.get('excl') else '含0'}  衰减上限 sqrt(rel·rel) = {ceil}")
    R["cross"] = pairs
    mr = st.fmean([p2["rho"] for p2 in pairs if "rho" in p2])
    cs = [p2["ceiling"] for p2 in pairs if p2["ceiling"]]
    mc = st.fmean(cs) if cs else None
    K2 = bool(mc) and mr > mc * 1.05
    print(f"  六对均值 rho {mr:+.4f};信度上限均值 {mc};"
          f"{'**超过上限 -> 有共享混杂**' if K2 else '在上限之内 ✓'}")
    R["cross_mean_rho"] = round(mr, 4); R["ceiling_mean"] = mc

    print(f"\n=== 重做筛选:{len(ALL_CAND)} 个候选(原十个 + 数据量三根),"
          f"Bonferroni {ALPHA_B:.6f},区间取 {100*(1-ALPHA_B):.3f}% ===")
    gp = G[PRIMARY]
    tgP = sorted(gp)
    CN = counts_per_target(BYT[PRIMARY])
    AUX = {"formable_rate": {}, "gc_cross_mean": {}}
    from e1_ensemble_marginals import load_pop
    rec, _s, _m = load_pop(PRIMARY)
    keepr = [r for r in rec if base_block(r) is not None]
    INV = invariants(tgP, {"formable_rate": {t: 1.0 for t in tgP}, "gc_cross_mean": {t: 0.0 for t in tgP}})
    # formable_rate and gc_cross_mean rebuilt properly from the fold
    keep0, M0 = fold(PRIMARY, a.workers)
    tot = collections.Counter(); form = collections.Counter(); gcm = collections.defaultdict(list)
    for r, m in zip(keep0, M0):
        tot[r["puzzle"]] += 1
        if "error" not in m:
            form[r["puzzle"]] += 1; gcm[r["puzzle"]].append(m["C"][0])
    INV = invariants(tgP, {"formable_rate": {t: form[t] / tot[t] for t in tot if tot[t]},
                           "gc_cross_mean": {t: st.fmean(v) for t, v in gcm.items() if v}})
    for k in EXTRA:
        INV[k] = {t: float(CN[t][k]) for t in CN}
    # the FROZEN power probe (declared in advance)
    rngp = random.Random(4242)
    sdg = st.stdev([gp[t] for t in tgP])
    INV["__PROBE_synthetic"] = {t: gp[t] + rngp.gauss(0, 1.6 * sdg) for t in tgP}
    INV["__OLD_probe_mean_wet"] = {}
    wm = collections.defaultdict(list)
    for r in keepr:
        wm[r["puzzle"]].append(r["wet"])
    INV["__OLD_probe_mean_wet"] = {t: st.fmean(v) for t, v in wm.items()}

    scr = {}
    for c in ALL_CAND + ["__PROBE_synthetic", "__OLD_probe_mean_wet"]:
        col = INV.get(c, {})
        ts = [t for t in tgP if t in col and not math.isnan(col[t])]
        if len(ts) < 30 or st.pstdev([col[t] for t in ts]) == 0:
            scr[c] = {"skipped": True, "n": len(ts)}; continue
        x = [col[t] for t in ts]; y = [gp[t] for t in ts]
        cib = rho_ci(x, y, a.seed + 31, 1 - ALPHA_B)
        pv = rho_perm_p(x, y, a.seed)
        ok = bool((cib[0] > 0 or cib[1] < 0) and pv < ALPHA_B)
        scr[c] = {"n": len(ts), "rho": round(spearman(x, y), 4), "ci_bonf": cib,
                  "perm_p": round(pv, 5), "passes": ok}
        tag = ("**冻结的必响探针**" if c == "__PROBE_synthetic" else
               "(原声明里选错的探针,不作检出力证据)" if c == "__OLD_probe_mean_wet" else
               "**数据量族(第 50 条缺口)**" if c in EXTRA else "")
        print(f"  {c:24s} rho {scr[c]['rho']:+.4f}  {100*(1-ALPHA_B):.2f}% {cib}"
              f"  置换 p {pv:.5f}  {'**通过**' if ok else '不通过'}  {tag}")
    R["screen"] = scr
    R["alpha_bonf"] = ALPHA_B

    probe_ok = scr.get("__PROBE_synthetic", {}).get("passes", False)
    print(f"\n  闸门 V1 冻结的必响探针:{'**响 ✓ 筛选有检出力**' if probe_ok else '**✗ 没响 -> 本次作废(第 54 条)**'}")
    if not probe_ok:
        R["verdict"] = "void_frozen_probe_did_not_fire"
        json.dump(R, open(a.out, "w"), indent=1, ensure_ascii=False)
        print("  **不许再换探针给这一跑背书。**"); return 0

    passing = [c for c in ALL_CAND if scr.get(c, {}).get("passes")]
    K3 = not passing
    K4 = any(scr.get(c, {}).get("passes") for c in EXTRA)
    print("\n=== 判定(冻结的打死条件)===")
    print(f"  K1 信度仍测不出:{'FIRES' if K1 else '不触发 ⇒ 四个人群的信度区间全部排除 0'}")
    print(f"  K2 跨人群 rho 超过信度上限:{'FIRES' if K2 else '不触发'}")
    print(f"  K3 {len(ALL_CAND)} 个候选无一通过:{'FIRES' if K3 else f'不触发(通过:{passing})'}")
    print(f"  K4 数据量族里有通过的:{'FIRES ⇒ 之前是选错了族' if K4 else '不触发'}")
    v = []
    v.append("reliability_still_not_measurable" if K1 else "reliability_measured")
    if K2: v.append("cross_rho_exceeds_reliability_ceiling")
    v.append("no_invariant_found_among_these_thirteen" if K3 else "invariant_found")
    if K4: v.append("data_quantity_family_was_the_gap")
    print(f"  裁定:{'__'.join(v)}")
    if K3:
        print("  **强制标签照写:13 个候选不是全空间。**")
    if K1:
        print("  **M2b arm1 维持作废。**")
    else:
        print("  **M2b arm1 的作废已兑现:信度可测,跨人群 rho 现在有分母了。**")
    print("  **口径:区间含 0 一律写 `unresolved`,绝不写 `refuted`。**")
    print("  边界:**没有盲人群**;M2 与 M2b arm1 的原裁定仍是作废,本次是新的一跑,不是把它们改判;不判上线。")
    R["verdict"] = "__".join(v)
    R["g_primary"] = {t: round(x, 5) for t, x in gp.items()}
    json.dump(R, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
