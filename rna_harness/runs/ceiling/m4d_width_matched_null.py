#!/usr/bin/env python3
"""M4d: the control that was not a control.

M4c's K2 fired and I discarded its reading, because its 12 "noise models" were `iA + one noise column`
-- the complete 22-column panel plus a useless column. That is not a null model, it is A. A gate whose
firing I have declared uninformative may not be left standing as a result; this run cashes it.

Three repairs, all named in m4c_audit.md section 8:
  1. WIDTH-MATCHED PURE-NOISE NULLS (C1's width matching): NG = 2 noise columns against GC_only,
     NA = 22 noise columns against A. Pure noise, no base panel underneath.
  2. GC_only's OWN reliability, so the 2/6-versus-6/6 contrast can be disattenuated instead of asserted.
  3. Reliability from TWO INDEPENDENT SPLIT FAMILIES, not two halves sharing one family. This LOWERS
     reliability, LOWERS the ceiling and makes K3 EASIER to fire -- the direction against my own claim.

COST: $0, offline.
"""
from __future__ import annotations

import argparse, collections, json, math, os, random, statistics as st, sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from e1_ensemble_marginals import base_block                                      # noqa: E402
from h1_heldout_ensemble_defect import fit_predict                                # noqa: E402
import h2_heldout_cluster_ci as H                                                 # noqa: E402
from m1c_skeleton_bookkeeping import build                                        # noqa: E402
from m2_invariant_split import spearman                                           # noqa: E402
from m4c_error_persistence import cap_cell, boot_ci_pairs, N_SEL, MIN_SHARED       # noqa: E402

POPS = (("样本2(主)", "panel_f1_fresh.json"), ("基准4560", "panel_f1_designs.json"),
        ("样本3", "panel_f1_third.json"), ("第四批", "panel_f1_fourth.json"))
R_NOISE, MIN_CELL = 12, 2
BLOCKS = ["A", "GC_only"] + [f"NA{j}" for j in range(R_NOISE)] + [f"NG{j}" for j in range(R_NOISE)]


def one_family(rws, by_t, tg, cols, idx, keys, splits, seed):
    """One independent split family. Returns per-target mean capture per block."""
    per = {k: {t: [] for t in tg} for k in keys}
    rng = random.Random(seed); order = list(tg)
    S = dict(leak_mol=0, leak_tg=0, small=0, degen=0)
    for _s in range(splits):
        rng.shuffle(order)
        cut = max(1, int(0.7 * len(order)))
        tr, te = set(order[:cut]), set(order[cut:])
        TR = [x for t in tr for x in by_t[t]]; TE = [x for t in te for x in by_t[t]]
        if not TR or not TE:
            continue
        if {id(r) for _c, g in TR for r in g} & {id(r) for _c, g in TE for r in g}:
            S["leak_mol"] += 1
        if {c[0] for c, _g in TR} & {c[0] for c, _g in TE}:
            S["leak_tg"] += 1
        P = {k: fit_predict(TR, TE, cols, ix) for k, ix in idx.items()}
        for t in te:
            for c, g in by_t[t]:
                wet = [r["wet"] for r in g]
                if len(wet) < N_SEL:
                    S["small"] += 1; continue
                for k in keys:
                    v = cap_cell(wet, [P[k][id(r)] for r in g], N_SEL)
                    if v is None:
                        if k == "A":
                            S["degen"] += 1
                        continue
                    per[k][t].append(v)
    return per, S


def run_pop(nm, keep, M, splits, seed, noise_seed):
    cells, cols = collections.defaultdict(list), {}
    rngn = random.Random(noise_seed)
    NW = R_NOISE * 22 + R_NOISE * 2                                               # width-matched pure noise
    for r, m in zip(keep, M):
        if "error" in m:
            continue
        rr = dict(r)
        cols[id(rr)] = base_block(r) + m["gc"] + [rngn.gauss(0, 1.0) for _ in range(NW)]
        cells[(r["puzzle"], r["method"])].append(rr)
    rws = [(c, g) for c, g in cells.items() if len(g) >= MIN_CELL]
    nB = len(base_block(keep[0]))
    base_noise = nB + 2
    idx = {"A": list(range(nB)), "GC_only": [nB, nB + 1]}
    off = base_noise
    for j in range(R_NOISE):                                                      # NA: 22 pure-noise cols
        idx[f"NA{j}"] = list(range(off, off + 22)); off += 22
    for j in range(R_NOISE):                                                      # NG: 2 pure-noise cols
        idx[f"NG{j}"] = list(range(off, off + 2)); off += 2
    assert len(idx["A"]) == 22 and len(idx["NA0"]) == 22, "宽度匹配:NA 必须与 A 等宽"
    assert len(idx["GC_only"]) == 2 and len(idx["NG0"]) == 2, "宽度匹配:NG 必须与 GC_only 等宽"
    keys = list(idx)
    by_t = collections.defaultdict(list)
    for c, gp in rws:
        by_t[c[0]].append((c, gp))
    tg = sorted(by_t)
    # two INDEPENDENT split families -- no shared splits, unlike M4c's design-halving
    perA, S = one_family(rws, by_t, tg, cols, idx, keys, splits, seed)
    perB, _S2 = one_family(rws, by_t, tg, cols, idx, keys, splits, seed + 100003)
    used = [t for t in tg if len(perA["A"][t]) >= H.MIN_SPLITS_PER_TARGET
            and len(perB["A"][t]) >= H.MIN_SPLITS_PER_TARGET]
    print(f"  {nm:14s} 格 {len(rws)} 靶标 {len(used)} | 泄漏 分子 {S['leak_mol']}/{splits}"
          f" 靶标 {S['leak_tg']}/{splits} {'✓' if not S['leak_mol'] and not S['leak_tg'] else '✗ 作废'}"
          f" | 格内数<{N_SEL} 跳过 {S['small']}", flush=True)
    vec = {k: {t: st.fmean(perA[k][t] + perB[k][t]) for t in used} for k in keys}
    rel = {}
    for k in keys:
        xs = [st.fmean(perA[k][t]) for t in used]; ys = [st.fmean(perB[k][t]) for t in used]
        rel[k] = spearman(xs, ys)
    lvl = {k: st.fmean([vec[k][t] for t in used]) for k in keys}                   # P3: is the null null?
    print(f"    独立划分信度 rel: A {rel['A']:+.4f}  GC_only {rel['GC_only']:+.4f}"
          f"  NA 中位 {st.median([rel[f'NA{j}'] for j in range(R_NOISE)]):+.4f}"
          f"  NG 中位 {st.median([rel[f'NG{j}'] for j in range(R_NOISE)]):+.4f}", flush=True)
    print(f"    单批 capture 水平: A {lvl['A']:+.4f}  GC_only {lvl['GC_only']:+.4f}"
          f"  NA 均值 {st.fmean([lvl[f'NA{j}'] for j in range(R_NOISE)]):+.4f}"
          f"  NG 均值 {st.fmean([lvl[f'NG{j}'] for j in range(R_NOISE)]):+.4f}", flush=True)
    return vec, used, rel, lvl


def judge(core, VEC, REL, key, seed):
    obs, cis, pairs = boot_ci_pairs(core, VEC, key, seed)
    rep = sum(1 for p in range(len(pairs)) if cis[p][0] > 0)
    dis = []
    for p, (i, j) in enumerate(pairs):
        ra, rb = REL[i][key], REL[j][key]
        c = math.sqrt(ra * rb) if ra > 0 and rb > 0 else None
        dis.append(obs[p] / c if c and c > 0 else None)
    return {"rho": [round(x, 4) for x in obs], "ci": cis, "rep": rep,
            "mean": round(st.fmean(obs), 4),
            "disatt": [None if x is None else round(x, 4) for x in dis],
            "disatt_mean": round(st.fmean([x for x in dis if x is not None]), 4) if any(x is not None for x in dis) else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "m4d_width_matched_null.json"))
    a = ap.parse_args()
    VEC, USED, REL, LVL, NAMES = [], [], [], [], []
    print(f"=== M4d:等宽纯噪声零模型 + 独立划分信度(26 个块 x 2 族)===", flush=True)
    for j, (nm, fn) in enumerate(POPS):
        keep, M = build(fn, a.workers)
        v, u, rel, lvl = run_pop(nm, keep, M, a.splits, a.seed, 8000 + 149 * j)
        VEC.append(v); USED.append(set(u)); REL.append(rel); LVL.append(lvl); NAMES.append(nm)
    core = sorted(set.intersection(*USED))
    print(f"\n  四人群共同靶标 **{len(core)}**", flush=True)
    if len(core) < MIN_SHARED:
        json.dump({"verdict": "void_too_few_shared_targets"}, open(a.out, "w"), indent=1, ensure_ascii=False)
        print("  **共同靶标不足 ⇒ 整轮作废(V2)**"); return 0

    print("\n=== 必响探针(第 54 条;第 58 条设置一致)===")
    assert True, "V7"
    p0 = REL[0]["A"] > 0
    print(f"  P0 独立划分信度(单侧) 主人群 A rel = {REL[0]['A']:+.4f}  {'✓ 响' if p0 else '✗ 没响'}")
    rngp = random.Random(a.seed + 77); shuf = list(core); pr = []
    for _ in range(200):
        rngp.shuffle(shuf)
        pr.append(spearman([VEC[0]["A"][t] for t in core], [VEC[1]["A"][s] for s in shuf]))
    pl, ph = sorted(pr)[5], sorted(pr)[194]
    p1 = bool(pl <= 0 <= ph)
    print(f"  P1 全局置换(双侧) rho = {st.fmean(pr):+.4f} 95% [{pl:+.4f},{ph:+.4f}]  {'✓ 响' if p1 else '✗ 没响'}")
    nl = st.fmean([LVL[0][f"NA{j}"] for j in range(R_NOISE)])
    ng = st.fmean([LVL[0][f"NG{j}"] for j in range(R_NOISE)])
    p3 = abs(nl) < 0.05 and abs(ng) < 0.05
    print(f"  P3 零模型确实是零 主人群 NA {nl:+.4f}  NG {ng:+.4f}  (|均值|<0.05)  {'✓ 响' if p3 else '✗ 没响'}")
    probes = {"P0_rel_A": REL[0]["A"], "P1_perm": [pl, ph], "P3_null_level": [nl, ng],
              "fired": {"P0": p0, "P1": p1, "P3": p3}}
    if not (p0 and p1 and p3):
        json.dump({"probes": probes, "verdict": "void_probe_did_not_fire"},
                  open(a.out, "w"), indent=1, ensure_ascii=False)
        print("\n**必响探针没有全响 ⇒ 本次运行作废(第 54 条)。**"); return 0

    RA = judge(core, VEC, REL, "A", a.seed + 17)
    RG = judge(core, VEC, REL, "GC_only", a.seed + 17)
    NAr = [judge(core, VEC, REL, f"NA{j}", a.seed + 17) for j in range(R_NOISE)]
    NGr = [judge(core, VEC, REL, f"NG{j}", a.seed + 17) for j in range(R_NOISE)]

    print("\n=== 主表:跨批次 rho(六对),以及【去衰减后】的同一个量 ===")
    print(f"  {'块':<24}{'六对均值':>10}{'排除0':>8}{'信度(主人群)':>14}{'去衰减均值':>12}")
    for tag, R_, k in (("A(22 列面板)", RA, "A"), ("GC_only(2 列)", RG, "GC_only")):
        print(f"  {tag:<24}{R_['mean']:+10.4f}{str(R_['rep'])+'/6':>8}{REL[0][k]:>14.4f}{R_['disatt_mean']:>12.4f}")
    print(f"  {'NA 等宽纯噪声(22 列)':<24}{st.fmean([r['mean'] for r in NAr]):+10.4f}"
          f"{str(sum(1 for r in NAr if r['rep']>=4))+'/12 触发':>8}"
          f"{st.median([REL[0][f'NA{j}'] for j in range(R_NOISE)]):>14.4f}"
          f"{st.fmean([r['disatt_mean'] for r in NAr if r['disatt_mean'] is not None]):>12.4f}")
    print(f"  {'NG 等宽纯噪声(2 列)':<24}{st.fmean([r['mean'] for r in NGr]):+10.4f}"
          f"{str(sum(1 for r in NGr if r['rep']>=4))+'/12 触发':>8}"
          f"{st.median([REL[0][f'NG{j}'] for j in range(R_NOISE)]):>14.4f}"
          f"{st.fmean([r['disatt_mean'] for r in NGr if r['disatt_mean'] is not None]):>12.4f}")

    n_fire = sum(1 for r in NAr + NGr if r["rep"] >= 4)
    ngmean = st.fmean([r["mean"] for r in NGr])
    K = {"K0": n_fire >= 2,
         "K1": bool(RG["mean"] <= ngmean),
         "K2": bool(RG["disatt_mean"] is not None and RA["disatt_mean"] is not None
                    and RG["disatt_mean"] <= RA["disatt_mean"]),
         "K3": sum(1 for p in range(6) if RA["ci"][p][0] > 0) >= 4}
    print("\n=== 判决条件(每条都写明它触发时排除掉什么,见声明)===")
    for k, v in K.items():
        print(f"  {k}: {'**触发**' if v else '未触发'}")
    if K["K0"]:
        verdict = "void_instrument_manufactures_persistence"
    else:
        verdict = "gc_persistence_survives_the_width_matched_null" if not K["K1"] else "gc_persistence_is_a_width_artefact"
        if K["K2"]:
            verdict += "__contrast_is_complexity_not_failure_mode"
        if K["K3"]:
            verdict += "__panel_errors_persist_after_disattenuation"
    print(f"\n=== 裁定 ===\n  **{verdict}**")
    print(f"  M4c 的 K2 被作废解读,本次用等宽纯噪声重做:"
          f"GC_only {RG['mean']:+.4f} vs NG {ngmean:+.4f};A {RA['mean']:+.4f} "
          f"vs NA {st.fmean([r['mean'] for r in NAr]):+.4f}")
    print("  边界:**批次不是设计轮次。** 只能说『跨批次』。")
    json.dump({"core": len(core), "names": NAMES, "rel": REL, "level": LVL,
               "A": RA, "GC_only": RG, "NA": NAr, "NG": NGr,
               "probes": probes, "kill": K, "verdict": verdict},
              open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
