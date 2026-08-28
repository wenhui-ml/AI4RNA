#!/usr/bin/env python3
"""M35: the interval on the DIFFERENCE between predictors, which the manuscript never had.

Declared in m35_paired_predictor_differences_declaration.json, frozen and pushed before this file
existed. K1 is live: if the transformer-minus-EternaFold interval covers zero, the sentence "the
split follows the decomposition and not the training data" comes out of the abstract.

The manuscript reports three identity contrasts, each with its own puzzle-clustered interval, and
concludes about the differences between them. That is the fault it retracts twice elsewhere. The
existing artifacts cannot supply the repair: m32 bootstraps EternaFold and ViennaRNA in its own
resample and IMPORTS RibonanzaNet's point estimate from m26v, a different run. A paired difference
needs all four columns resampled over the same puzzles in the same draw.

COST: $0, offline. Both prediction caches are on disk; no GPU pass and no API call.
"""
from __future__ import annotations

import collections, json, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m26_model_scored as M26

NBOOT, NPERM, SEED = 2000, 2000, 3501
COLS = (("measured", "rf"), ("neural", "nmut"), ("vienna", "pmut"), ("eterna", "emut"))
PUBLISHED = {"rf": 0.05393, "pmut": -0.03039, "emut": -0.00569, "nmut": 0.06055}
PAIRS = (("neural", "vienna"), ("neural", "eterna"), ("vienna", "eterna"),
         ("measured", "neural"), ("measured", "vienna"), ("measured", "eterna"))
SB = lambda r: (r["design"], min(r["depth"], 3), r["crossed"])
LOG = []


def log(s=""):
    print(s, flush=True); LOG.append(s)


def ci(v, lo=0.025, hi=0.975):
    a = np.sort(np.asarray(v))
    return [round(float(a[int(lo * len(a))]), 5), round(float(a[int(hi * len(a))]), 5)]


def contrasts(recs):
    """All four contrasts on ONE record list -- the same records, so differences are paired."""
    out = {}
    for nm, key in COLS:
        v, w, _u = M26.matched(recs, "CG", "AU", SB, key=key)
        out[nm] = v
    return out


def main() -> int:
    OUT = {"nboot": NBOOT, "nperm": NPERM, "seed": SEED}
    log("=== M35:三个预测器之间【差值】的区间 ===\n")

    recs, _dropped = M26.build()
    log(f"纳入 {len(recs)} 次干预")

    # ---- join the two cached predictor columns. Index conventions differ and that is not cosmetic:
    # the neural cache is 0-based, EternaFold's posteriors are 1-based.
    z = np.load(os.path.join(HERE, "m26c_pneural_cache.npz"), allow_pickle=True)
    pn = {k: v for k, v in zip(z["keys"], z["mats"])}
    ze = np.load(os.path.join(HERE, "m32_eterna_cache.npz"), allow_pickle=True)
    pe = {str(k): (v.item() if hasattr(v, "item") else v)
          for k, v in zip(ze["keys"], ze["maps"])}

    miss = 0
    for r in recs:
        i, j = min(r["a"], r["b"]), max(r["a"], r["b"])
        if r["mut_seq"] not in pn or r["mut_seq"] not in pe:
            miss += 1; continue
        r["nmut"] = float(pn[r["mut_seq"]][i, j])
        r["emut"] = float(pe[r["mut_seq"]].get((i + 1, j + 1), 0.0))
    log(f"--- S1 连接 ---\n  四列齐备 {len(recs) - miss}/{len(recs)},缺 {miss}")
    OUT["S1_join_missing"] = miss
    if miss:
        log("  !! S1 未通过:有记录连不上,本次运行作废")
        OUT["VOID"] = "S1_join"
        json.dump(OUT, open(os.path.join(HERE, "m35_paired_predictor_differences.json"), "w"),
                  indent=2, ensure_ascii=False)
        return 1

    # ---- S2: every single-column contrast must reproduce its published value
    obs = contrasts(recs)
    log("\n--- S2 单列等价(必须对上已发表的数)---")
    s2 = {}
    for nm, key in COLS:
        d = abs(obs[nm] - PUBLISHED[key])
        s2[nm] = {"here": round(obs[nm], 5), "published": PUBLISHED[key], "abs_diff": d,
                  "passed": bool(d < 1e-4)}
        log(f"  {nm:9s} 本脚本 {obs[nm]:+.5f}   已发表 {PUBLISHED[key]:+.5f}   差 {d:.2e}"
            f"   {'OK' if d < 1e-4 else '!! 不符'}")
    OUT["S2_equivalence"] = s2
    if not all(v["passed"] for v in s2.values()):
        log("  !! K3 触发:本脚本的估计量不是已发表的那个,本次运行作废")
        OUT["VOID"] = "K3_estimator_drift"
        json.dump(OUT, open(os.path.join(HERE, "m35_paired_predictor_differences.json"), "w"),
                  indent=2, ensure_ascii=False)
        return 1

    # ---- paired cluster bootstrap: ONE puzzle index vector per draw, all four columns on it
    pz = sorted({r["puzzle"] for r in recs})
    byp = collections.defaultdict(list)
    for r in recs:
        byp[r["puzzle"]].append(r)
    log(f"\n--- S3 自助 ---\n  {len(pz)} 个 puzzle,每次抽样同一组下标喂给四列")
    rng = np.random.default_rng(SEED)
    draws = {nm: [] for nm, _k in COLS}
    for _ in range(NBOOT):
        ix = rng.integers(0, len(pz), len(pz))
        s = [r for q in ix for r in byp[pz[q]]]
        c = contrasts(s)
        if any(v is None for v in c.values()):
            continue
        for nm in draws:
            draws[nm].append(c[nm])
    n_ok = len(draws["measured"])
    log(f"  {n_ok}/{NBOOT} 次抽样四列都可估")
    OUT["S3_boot_usable"] = n_ok

    # ---- permutation on the DIFFERENCE, within stratum
    rp = np.random.default_rng(SEED + 1)
    perm_ge = {p: 0 for p in PAIRS}
    for _ in range(NPERM):
        by = collections.defaultdict(list)
        for r in recs:
            by[SB(r)].append(r)
        sh = []
        for _k, g in by.items():
            lab = [r["pt"] for r in g]
            rp.shuffle(lab)
            for r, t in zip(g, lab):
                q = dict(r); q["pt"] = t; sh.append(q)
        c = contrasts(sh)
        if any(v is None for v in c.values()):
            continue
        for a, b in PAIRS:
            if abs(c[a] - c[b]) >= abs(obs[a] - obs[b]):
                perm_ge[(a, b)] += 1

    log("\n=== 配对差(同一次抽样内的两列相减)===")
    res = {}
    for a, b in PAIRS:
        d = [x - y for x, y in zip(draws[a], draws[b])]
        cl = ci(d)
        pv = (perm_ge[(a, b)] + 1) / (NPERM + 1)
        covers = cl[0] <= 0.0 <= cl[1]
        res[f"{a}_minus_{b}"] = {"effect": round(obs[a] - obs[b], 5), "ci": cl,
                                 "p_perm": round(pv, 5), "covers_zero": bool(covers)}
        log(f"  {a:9s} - {b:9s}  {obs[a] - obs[b]:+.5f}   CI {cl}   置换 p {pv:.5f}"
            f"   {'含 0' if covers else '不含 0'}")
    OUT["paired_differences"] = res

    # ---- POST HOC, and marked as such. P3 was staked as covering zero and did not: the two
    # nearest-neighbour predictors are resolvably different from each other. That refutes the
    # manuscript's "and not the training data" as written, so the quantity the manuscript now
    # needs is the comparison of the two axes -- how far the architecture change moves the
    # contrast against how far the training-data change moves it. Same draws, same seed; this
    # adds no new data and is reported as post hoc because it was not staked.
    d_arch = [n - e for n, e in zip(draws["neural"], draws["eterna"])]      # architecture axis
    d_data = [e - v for e, v in zip(draws["eterna"], draws["vienna"])]      # training-data axis
    d_gap = [a - b for a, b in zip(d_arch, d_data)]
    OUT["post_hoc_axes"] = {
        "_why": "forced by P3 failing; not staked in the declaration",
        "architecture_axis_neural_minus_eterna": {
            "effect": round(obs["neural"] - obs["eterna"], 5), "ci": ci(d_arch)},
        "training_data_axis_eterna_minus_vienna": {
            "effect": round(obs["eterna"] - obs["vienna"], 5), "ci": ci(d_data)},
        "architecture_minus_training_data": {
            "effect": round((obs["neural"] - obs["eterna"]) - (obs["eterna"] - obs["vienna"]), 5),
            "ci": ci(d_gap), "covers_zero": bool(ci(d_gap)[0] <= 0.0 <= ci(d_gap)[1])},
        "both_axes_move_in_the_measured_direction": bool(
            (obs["neural"] - obs["eterna"]) > 0 and (obs["eterna"] - obs["vienna"]) > 0),
    }
    ph = OUT["post_hoc_axes"]
    log("\n=== 事后(P3 押输逼出来的):两条轴各自移动了多少 ===")
    log(f"  架构轴   neural - eterna  {ph['architecture_axis_neural_minus_eterna']['effect']:+.5f}"
        f"  CI {ph['architecture_axis_neural_minus_eterna']['ci']}")
    log(f"  训练数据轴 eterna - vienna {ph['training_data_axis_eterna_minus_vienna']['effect']:+.5f}"
        f"  CI {ph['training_data_axis_eterna_minus_vienna']['ci']}")
    log(f"  两轴之差                  {ph['architecture_minus_training_data']['effect']:+.5f}"
        f"  CI {ph['architecture_minus_training_data']['ci']}"
        f"  {'含 0' if ph['architecture_minus_training_data']['covers_zero'] else '不含 0'}")
    log(f"  两条轴都朝实测方向移动:{ph['both_axes_move_in_the_measured_direction']}")

    # ---- the staked predictions, graded in the declaration's own words
    P = {
        "P1": ("neural_minus_vienna", "EXCLUDES", not res["neural_minus_vienna"]["covers_zero"]),
        "P2": ("neural_minus_eterna", "EXCLUDES", not res["neural_minus_eterna"]["covers_zero"]),
        "P3": ("vienna_minus_eterna", "COVERS", res["vienna_minus_eterna"]["covers_zero"]),
        "P4": ("measured_minus_neural", "COVERS", res["measured_minus_neural"]["covers_zero"]),
    }
    log("\n=== 押注 ===")
    for k, (q, want, held) in P.items():
        log(f"  {k}  {q:24s} 押 {want:8s}  {'成立' if held else '**未成立**'}")
    OUT["staked"] = {k: {"quantity": q, "predicted": w, "held": bool(h)} for k, (q, w, h) in P.items()}

    # ---- kill conditions, read in the declaration's own wording
    K = {}
    K["K1_decomposition_sentence_unlicensed"] = bool(res["neural_minus_eterna"]["covers_zero"])
    K["K2_separation_withdrawn"] = bool(res["neural_minus_vienna"]["covers_zero"])
    K["K4_tracking_must_go"] = bool(not res["measured_minus_neural"]["covers_zero"])
    OUT["kill_conditions"] = K
    log("\n=== 杀条件 ===")
    for k, v in K.items():
        log(f"  {k}: {v}")
    OUT["verdict"] = ("difference_intervals_support_the_headline"
                      if not (K["K1_decomposition_sentence_unlicensed"]
                              or K["K2_separation_withdrawn"] or K["K4_tracking_must_go"])
                      else "MANUSCRIPT_MUST_BE_REWRITTEN_see_kill_conditions")
    log(f"\n  {OUT['verdict']}")

    OUT["log"] = LOG
    json.dump(OUT, open(os.path.join(HERE, "m35_paired_predictor_differences.json"), "w"),
              indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
