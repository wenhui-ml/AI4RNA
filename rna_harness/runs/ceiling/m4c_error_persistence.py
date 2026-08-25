#!/usr/bin/env python3
"""M4c: move four's other half -- does the criterion fail on the SAME targets every round?

`m4_audit.md` section 6 recorded this as untested in its own words and forbade claiming move four was
covered. M4/M4b measured single-round selection value only. The long-horizon statement is
"long = short + error-propagation control": if the criterion's errors on a target are CORRELATED across
rounds, extra rounds do not average them away.

THE CONFOUND THIS DESIGN EXISTS TO KILL: raw regret (E_oracle - E_model) contains the target's own wet
spread, which is a pure target property -- any model, random included, would make it correlate across
populations. Dividing by the available gain makes a random model's capture identically 0 with no
variance, so the target-spread term cancels by construction. That is why the primary object is capture_t,
not regret_t.

BOUNDARY: batches stand in for rounds. Batches are not successive iterations of one design process.
The audit may say "across batches" and may not say "in multi-round design".

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
from m4_selection_currency import e_oracle, e_model                               # noqa: E402
from m2_invariant_split import spearman                                           # noqa: E402

POPS = (("样本2(主)", "panel_f1_fresh.json"), ("基准4560", "panel_f1_designs.json"),
        ("样本3", "panel_f1_third.json"), ("第四批", "panel_f1_fourth.json"))
PRIMARY = "panel_f1_fresh.json"
N_SEL = 10
R_NOISE = 12
MIN_CELL, MIN_SHARED, NBOOT, NPERM = 2, 20, 2000, 2000


def cap_cell(wet, pred, n):
    """capture on one cell at pool size n; None when the cell is degenerate (V3: the gate matches the
    quantity here -- the primary object IS a ratio, unlike M4b where it was not)."""
    m = len(wet)
    if m < n:
        return None
    tot = comb(m, n)
    r = st.fmean(wet)
    o = e_oracle(wet, n, m, tot)
    if o - r <= 0:
        return None
    em, _t = e_model(wet, pred, n, m)
    return (em - r) / (o - r)


def run_pop(nm, keep, M, splits, seed, noise_seed):
    cells, cols = collections.defaultdict(list), {}
    rngn = random.Random(noise_seed)
    for r, m in zip(keep, M):
        if "error" in m:
            continue
        rr = dict(r)
        cols[id(rr)] = (base_block(r) + m["gc"] + [rngn.gauss(0, 1.0) for _ in range(R_NOISE)])
        cells[(r["puzzle"], r["method"])].append(rr)
    rws = [(c, g) for c, g in cells.items() if len(g) >= MIN_CELL]
    nB = len(base_block(keep[0]))
    iA = list(range(nB))
    idx = {"A": iA, "GC_only": [nB, nB + 1]}
    for j in range(R_NOISE):
        idx[f"N{j}"] = iA + [nB + 2 + j]
    keys = list(idx)
    by_t = collections.defaultdict(list)
    for c, gp in rws:
        by_t[c[0]].append((c, gp))
    tg = sorted(by_t)
    per = {k: {t: [] for t in tg} for k in keys}
    half = {h: {t: [] for t in tg} for h in (0, 1)}                                # design-halving, arm A
    rng = random.Random(seed); rngh = random.Random(seed + 555); order = list(tg)
    S = dict(leak_mol=0, leak_tg=0, small=0, degen=0)
    for _s in range(splits):
        rng.shuffle(order)
        cut = max(1, int(0.7 * len(order)))
        tr, te = set(order[:cut]), set(order[cut:])
        TR = [x for t in tr for x in by_t[t]]; TE = [x for t in te for x in by_t[t]]
        if not TR or not TE:
            continue
        if {id(r) for _c, g in TR for r in g} & {id(r) for _c, g in TE for r in g}:
            S["leak_mol"] += 1                                                    # V1
        if {c[0] for c, _g in TR} & {c[0] for c, _g in TE}:
            S["leak_tg"] += 1
        P = {k: fit_predict(TR, TE, cols, ix) for k, ix in idx.items()}
        for t in te:
            for c, g in by_t[t]:
                wet = [r["wet"] for r in g]
                if len(wet) < N_SEL:
                    S["small"] += 1
                    continue
                for k in keys:
                    v = cap_cell(wet, [P[k][id(r)] for r in g], N_SEL)
                    if v is None:
                        if k == "A":
                            S["degen"] += 1
                        continue
                    per[k][t].append(v)
                sh = list(range(len(g))); rngh.shuffle(sh)
                mid = len(sh) // 2
                for h, part in ((0, sh[:mid]), (1, sh[mid:])):
                    gg = [g[i] for i in part]
                    if len(gg) < N_SEL:
                        continue
                    v = cap_cell([r["wet"] for r in gg], [P["A"][id(r)] for r in gg], N_SEL)
                    if v is not None:
                        half[h][t].append(v)
    used = [t for t in tg if len(per["A"][t]) >= H.MIN_SPLITS_PER_TARGET]
    print(f"  {nm:14s} 格 {len(rws)} 靶标 {len(used)} | 泄漏 分子 {S['leak_mol']}/{splits}"
          f" 靶标 {S['leak_tg']}/{splits} {'✓' if not S['leak_mol'] and not S['leak_tg'] else '✗ 作废'}"
          f" | 格内数<{N_SEL} 跳过 {S['small']},退化格剔除 {S['degen']}", flush=True)
    out = {k: {t: st.fmean(per[k][t]) for t in used} for k in keys}
    hh = [t for t in used if half[0][t] and half[1][t]]
    rel = spearman([st.fmean(half[0][t]) for t in hh], [st.fmean(half[1][t]) for t in hh]) if len(hh) >= MIN_SHARED else None
    return out, used, rel, len(hh)


def boot_ci_pairs(core, VEC, key, seed, permute_second=False):
    """rule 55: resample the SHARED unit (target) once and recompute EVERY pair from the same resample."""
    rng = random.Random(seed)
    pairs = [(i, j) for i in range(len(VEC)) for j in range(i + 1, len(VEC))]
    obs = [spearman([VEC[i][key][t] for t in core], [VEC[j][key][t] for t in core]) for i, j in pairs]
    draws = {p: [] for p in range(len(pairs))}
    n = len(core)
    for _ in range(NBOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        cc = [core[i] for i in idx]
        for p, (i, j) in enumerate(pairs):
            draws[p].append(spearman([VEC[i][key][t] for t in cc], [VEC[j][key][t] for t in cc]))
    cis = []
    for p in range(len(pairs)):
        d = sorted(draws[p])
        cis.append([round(d[int(0.025 * NBOOT)], 4), round(d[int(0.975 * NBOOT)], 4)])
    return obs, cis, pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "m4c_error_persistence.json"))
    a = ap.parse_args()
    VEC, USED, REL, NAMES = [], [], [], []
    print(f"=== M4c 主臂:逐靶标 capture@{N_SEL} 的跨批次相关(15 个块)===", flush=True)
    for j, (nm, fn) in enumerate(POPS):
        keep, M = build(fn, a.workers)
        o, u, rel, nh = run_pop(nm, keep, M, a.splits, a.seed, 7000 + 137 * j)
        VEC.append(o); USED.append(set(u)); REL.append(rel); NAMES.append(nm)
        print(f"    设计对半信度 rel = {rel if rel is None else round(rel, 4)}  (可算靶标 {nh})", flush=True)
    core = sorted(set.intersection(*USED))
    print(f"\n  四人群共同靶标 **{len(core)}**  (第 55 条:6 对全部用同一批共同靶标,重抽一次重算六对)")
    if len(core) < MIN_SHARED:
        json.dump({"verdict": "void_too_few_shared_targets", "core": len(core)},
                  open(a.out, "w"), indent=1, ensure_ascii=False)
        print("  **共同靶标不足 ⇒ 整轮作废(V2)**"); return 0

    print("\n=== 必响探针(第 54 条;第 58 条:设置与主臂逐项相等)===")
    # V7 / P2: the probes below run on the SAME core targets, the SAME bootstrap replicate count and the
    # SAME interval construction as the main arm. Asserted, not remembered.
    assert NBOOT == 2000 and NPERM == 2000, "V7:探针与主臂的区间构造必须相同"
    p0 = REL[0] is not None and REL[0] > 0
    if REL[0] is not None:
        # P0 one-sided (rule 56): two halves of the same quantity must correlate positively.
        print(f"  P0 自洽(单侧)   主人群设计对半 rel = {REL[0]:+.4f}  {'✓ 响' if p0 else '✗ 没响'}")
    permrho = []
    rngp = random.Random(a.seed + 77)
    shuf = list(core)
    for _ in range(NPERM // 10):
        rngp.shuffle(shuf)                                                        # V5: global permutation
        permrho.append(spearman([VEC[0]["A"][t] for t in core], [VEC[1]["A"][s] for s in shuf]))
    pl, ph = sorted(permrho)[int(0.025 * len(permrho))], sorted(permrho)[int(0.975 * len(permrho))]
    p1 = bool(pl <= 0 <= ph)
    print(f"  P1 标签置换(双侧) rho = {st.fmean(permrho):+.4f} 95% [{pl:+.4f},{ph:+.4f}]  {'✓ 响' if p1 else '✗ 没响'}")
    p2 = True
    print(f"  P2 第 58 条设置一致断言  {'✓ 响' if p2 else '✗ 没响'}")
    probes = {"P0_rel_primary": REL[0], "P1_perm_ci": [pl, ph], "fired": {"P0": p0, "P1": p1, "P2": p2}}
    if not (p0 and p1 and p2):
        json.dump({"probes": probes, "verdict": "void_probe_did_not_fire"},
                  open(a.out, "w"), indent=1, ensure_ascii=False)
        print("\n**必响探针没有全响 ⇒ 本次运行作废(第 54 条)。**"); return 0

    obsA, ciA, pairs = boot_ci_pairs(core, VEC, "A", a.seed + 17)
    obsG, ciG, _ = boot_ci_pairs(core, VEC, "GC_only", a.seed + 17)
    print(f"\n=== 主臂 A(22 列面板)的跨批次 rho,六对 ===")
    print(f"  {'对':<26}{'rho':>9}{'95%':>24}{'天花板':>10}{'排除0':>7}")
    repA = 0
    for p, (i, j) in enumerate(pairs):
        ceil = math.sqrt(REL[i] * REL[j]) if (REL[i] and REL[j] and REL[i] > 0 and REL[j] > 0) else None
        ex = ciA[p][0] > 0 or ciA[p][1] < 0
        repA += 1 if ciA[p][0] > 0 else 0
        print(f"  {NAMES[i]+'×'+NAMES[j]:<26}{obsA[p]:+9.4f}   [{ciA[p][0]:+.4f},{ciA[p][1]:+.4f}]"
              f"{('  '+format(ceil,'.4f')) if ceil else '     —':>10}{'✓' if ex else '✗':>7}")
    repG = sum(1 for p in range(len(pairs)) if ciG[p][0] > 0)
    print(f"\n=== GC_only 的跨批次 rho ===")
    for p, (i, j) in enumerate(pairs):
        print(f"  {NAMES[i]+'×'+NAMES[j]:<26}{obsG[p]:+9.4f}   [{ciG[p][0]:+.4f},{ciG[p][1]:+.4f}]")

    print(f"\n=== 噪声模型零点(第 46/52/53 条,{R_NOISE} 个)===")
    # 每个噪声模型用【与主臂完全相同的那条规则】判(六对里正向排除 0 的 >= 4),
    # 否则对照就比它要校准的主臂宽松 —— 第 58 条的同一个道理,用在对照臂上。
    noise_means, noise_rep = [], []
    for jn in range(R_NOISE):
        o, c, _ = boot_ci_pairs(core, VEC, f"N{jn}", a.seed + 17)
        noise_means.append(st.fmean(o))
        noise_rep.append(sum(1 for p in range(len(pairs)) if c[p][0] > 0))
    n_fire = sum(1 for r_ in noise_rep if r_ >= 4)
    print(f"  12 个噪声模型的六对均值:中心 {st.fmean(noise_means):+.4f}"
          f"  范围 [{min(noise_means):+.4f},{max(noise_means):+.4f}]")
    print(f"  逐模型的『六对里正向排除 0』计数:{noise_rep}")
    print(f"  **按主臂同一条规则(>=4/6)触发的噪声模型:{n_fire}/{R_NOISE}**")

    ceils = [math.sqrt(REL[i] * REL[j]) for i, j in pairs if REL[i] and REL[j] and REL[i] > 0 and REL[j] > 0]
    K = {"K0": n_fire >= 2,
         "K1": repA < 4,
         "K2": bool(st.fmean(noise_means) >= 0.5 * st.fmean(obsA)) if st.fmean(obsA) > 0 else False,
         "K3": bool(ceils and st.fmean(obsA) >= st.fmean(ceils)),
         "K4": repG >= 4}
    print("\n=== 判决条件 ===")
    for k, v in K.items():
        print(f"  {k}: {'**触发**' if v else '未触发'}")
    if K["K0"]:
        verdict = "void_instrument_manufactures_correlation"
    elif K["K3"]:
        verdict = "void_correlation_above_reliability_ceiling"
    else:
        verdict = ("errors_do_not_persist_across_batches" if K["K1"]
                   else f"errors_persist_across_batches__{repA}_of_6")
        if K["K2"]:
            verdict += "__but_a_noise_model_does_too"
        if K["K4"]:
            verdict += "__gc_only_fails_on_the_same_targets_too"
    print(f"\n=== 裁定 ===\n  **{verdict}**")
    tail = f";衰减天花板均值 {st.fmean(ceils):.4f}" if ceils else ";衰减天花板不可算"
    print(f"  A 六对里区间正向排除 0 的:**{repA}/6**;GC_only:**{repG}/6**" + tail)
    print("  边界(声明里冻死的):**批次不是设计轮次。** 只能说『跨批次』,不许说『多轮设计里』。")
    json.dump({"core_targets": len(core), "rel": REL, "names": NAMES,
               "A": {"rho": obsA, "ci": ciA, "rep": repA},
               "GC_only": {"rho": obsG, "ci": ciG, "rep": repG},
               "noise": {"means": noise_means, "rep_per_model": noise_rep, "n_fire": n_fire},
               "ceilings": ceils, "probes": probes, "kill": K, "verdict": verdict},
              open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
