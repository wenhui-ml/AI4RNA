#!/usr/bin/env python3
"""M36: what the "conventional readout" is, and whether the two readouts can be subtracted.

Declared in m36_readout_scale_declaration.json, frozen and pushed before this file existed.

The manuscript's Claim 1 rests on paired_diff = R1 - R2, a rescue-factor contrast minus a
structure-frequency contrast. This file puts both on their own within-stratum scale before
differencing them, and names the CSV column the second one is read from -- which the manuscript
describes as a structure "inferred from the wild-type profile" and which is in fact
RibonanzaNet's own prediction, shipped in the benchmark.

Nothing here re-derives the pair set: it calls m25's own loader, m25's own strat() and m25's own
E4_matched, so a divergence between this file and the published numbers would be a bug in this
file and is checked as S2/P1 before anything is reported.

COST: $0, offline. Two CSV reads, no model call, no GPU.
"""
from __future__ import annotations

import collections, csv, json, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m25_estimator_benchmark as M25
import m16_rescue as M16

RNET_COLUMN = "RNet_structure"     # S1: named verbatim, and asserted to exist in the header
NBOOT, SEED = 2000, 3601
LOG = []


def log(s=""):
    print(s, flush=True); LOG.append(s)


def within_sd(recs, key):
    """Pooled within-stratum SD -- the scale the matched estimator actually works on."""
    by = collections.defaultdict(list)
    for r in recs:
        by[M25.strat(r)].append(float(r[key]))
    num = den = 0.0
    for _s, v in by.items():
        if len(v) < 2:
            continue
        a = np.asarray(v)
        num += float(((a - a.mean()) ** 2).sum()); den += len(a) - 1
    return float(np.sqrt(num / den)) if den else float("nan")


def build():
    """m25's arm-B pair set, rebuilt with m25's own pieces."""
    rnet = M25.load_rnet()
    csv.field_size_limit(10 ** 9)
    rows, wt = [], {}
    with open(M25.M2R) as f:
        for d in csv.DictReader(f):
            rows.append(d)
            if d["mutA"] in ("", "NA", "nan") and d["mutB"] in ("", "NA", "nan"):
                wt[(d["puzzle"], d["method"])] = d
    resc = {}
    for d in rows:
        if d["rescue_factor"] in ("", "NA", "nan") or d["SN_filter"] not in ("1", "1.0"):
            continue
        k = (d["puzzle"], d["method"]); p = wt.get(k)
        if p is None:
            continue
        ss = int(float(d["sub_start"])); P = M16.parse_pairs(d["target_structure"])
        try:
            ma, mb = int(float(d["mutA"])), int(float(d["mutB"]))
        except ValueError:
            continue
        a, b = ma + ss - 2, mb + ss - 2
        if P.get(a) != b:
            continue
        s_ = p["sequence"]; pt = "".join(sorted({s_[a], s_[b]}))
        if pt not in ("CG", "AU", "GU"):
            continue
        dep = 0
        while P.get(a - dep - 1) == b + dep + 1:
            dep += 1
        up = 0
        while P.get(a + up + 1) == b - up - 1:
            up += 1
        crossed = any(m < a < n < b or a < m < b < n for m, n in P.items() if m < n)
        resc[(k, a, b)] = {"design": k, "pt": pt, "depth": min(dep, up), "crossed": bool(crossed),
                           "y": float(d["rescue_factor"])}
    called = {}
    for k, p in wt.items():
        ss = int(float(p["sub_start"])); se = int(float(p["sub_end"]))
        ds = p["sequence"][ss - 1:se]
        if ds not in rnet:
            continue
        cnt = collections.Counter(); nrep = len(rnet[ds])
        for rs in rnet[ds]:
            Q = M16.parse_pairs(rs)
            for a, b in Q.items():
                if a < b:
                    cnt[(a + ss - 1, b + ss - 1)] += 1
        P = M16.parse_pairs(p["target_structure"])
        for a, b in P.items():
            if a < b:
                called[(k, a, b)] = cnt[(a, b)] / nrep
    keys = sorted(set(resc) & set(called))
    R1 = [{**resc[k]} for k in keys]
    R2 = [{**resc[k], "y": called[k]} for k in keys]
    return R1, R2, keys


def main() -> int:
    OUT = {}
    log("=== M36:常规读数究竟是什么,以及两个读数能不能相减 ===\n")

    # ---- S1: name the column, verbatim from the header of the file that is read
    hdr = next(csv.reader(open(M25.BENCH)))
    OUT["S1_source"] = {
        "file": os.path.basename(M25.BENCH),
        "column": RNET_COLUMN,
        "column_present": bool(RNET_COLUMN in hdr),
        "structure_columns_in_file": [c for c in hdr if "structure" in c.lower()],
        "profile_derived_structure_column_present": False,
    }
    log("--- S1 常规读数的来源 ---")
    log(f"  文件 {OUT['S1_source']['file']}")
    log(f"  这一列 {RNET_COLUMN}:在表头里 {OUT['S1_source']['column_present']}")
    log(f"  全表所有含 structure 的列:{OUT['S1_source']['structure_columns_in_file']}")
    log("  => 常规读数是 RibonanzaNet 自己的序列到结构预测,不是任何由 profile 推出的结构;"
        "本文件里根本没有那样一列")
    assert OUT["S1_source"]["column_present"], "RNet_structure not in the benchmark header"

    R1, R2, keys = build()
    log(f"\n两个读数都定义的配对 {len(keys)}")

    # ---- S2 / P1: reproduce m25's own numbers before touching anything
    def wts(rs):
        by = collections.defaultdict(lambda: {"CG": [], "AU": []})
        for r in rs:
            if r["pt"] in ("CG", "AU"):
                by[M25.strat(r)][r["pt"]].append(r)
        u = [(s, min(len(d["CG"]), len(d["AU"]))) for s, d in by.items() if d["CG"] and d["AU"]]
        return sum(w for _s, w in u), len(u)
    w1, s1 = wts(R1); w2, s2 = wts(R2)
    r1 = M25.E4_matched(R1, "y"); r2 = M25.E4_matched(R2, "y")
    OUT["S2_identical_design"] = {"weights": [w1, w2], "strata": [s1, s2],
                                  "identical": bool((w1, s1) == (w2, s2))}
    OUT["P1_raw"] = {"R1_rescue": round(r1, 5), "R2_rnet": round(r2, 5),
                     "matches_published": bool(abs(r1 - 0.0527) < 1e-3 and abs(r2 - 0.0198) < 1e-3)}
    log(f"--- S2/P1 ---\n  权重/层数 R1 {w1}/{s1}  R2 {w2}/{s2}  逐字相同 "
        f"{OUT['S2_identical_design']['identical']}")
    log(f"  R1 rescue {r1:+.5f}(已发表 +0.0527)  R2 RNet {r2:+.5f}(已发表 +0.0198)  "
        f"对得上 {OUT['P1_raw']['matches_published']}")
    if not (OUT["S2_identical_design"]["identical"] and OUT["P1_raw"]["matches_published"]):
        OUT["VOID"] = "S2_or_P1"
        log("  !! 复现不了已发表的数,本次运行作废")
        json.dump(OUT, open(os.path.join(HERE, "m36_readout_scale.json"), "w"),
                  indent=2, ensure_ascii=False)
        return 1

    # ---- P2/P3: the two columns' own scales
    sd1w, sd2w = within_sd(R1, "y"), within_sd(R2, "y")
    sd1m = float(np.std([r["y"] for r in R1], ddof=1))
    sd2m = float(np.std([r["y"] for r in R2], ddof=1))
    OUT["scales"] = {"rescue_within_sd": round(sd1w, 4), "rnet_within_sd": round(sd2w, 4),
                     "rescue_marginal_sd": round(sd1m, 4), "rnet_marginal_sd": round(sd2m, 4),
                     "sd_ratio_within": round(sd1w / sd2w, 4)}
    log(f"\n--- P2/P3 两列各自的尺度 ---")
    log(f"  rescue      层内 SD {sd1w:.4f}   边际 SD {sd1m:.4f}")
    log(f"  RNet 频率   层内 SD {sd2w:.4f}   边际 SD {sd2m:.4f}")
    log(f"  层内 SD 之比 {sd1w / sd2w:.4f} —— 相减之前它们不在一个尺度上")

    # ---- P4: the standardised paired difference, on the same design-clustered resample
    des = sorted({r["design"] for r in R1})
    byd = collections.defaultdict(lambda: ([], []))
    for a_, b_ in zip(R1, R2):
        byd[a_["design"]][0].append(a_); byd[a_["design"]][1].append(b_)
    rng = np.random.default_rng(SEED)
    raw, std = [], []
    for _ in range(NBOOT):
        ix = rng.integers(0, len(des), len(des))
        s1r = [x for j in ix for x in byd[des[j]][0]]
        s2r = [x for j in ix for x in byd[des[j]][1]]
        v1 = M25.E4_matched(s1r, "y"); v2 = M25.E4_matched(s2r, "y")
        if v1 is None or v2 is None:
            continue
        a1, a2 = within_sd(s1r, "y"), within_sd(s2r, "y")
        if not (a1 > 0 and a2 > 0):
            continue
        raw.append(v1 - v2); std.append(v1 / a1 - v2 / a2)

    def ci(v):
        a = np.sort(np.asarray(v))
        return [round(float(a[int(.025 * len(a))]), 5), round(float(a[int(.975 * len(a))]), 5)]

    z1, z2 = r1 / sd1w, r2 / sd2w
    OUT["standardised"] = {
        "rescue_contrast_in_sd": round(z1, 4), "rnet_contrast_in_sd": round(z2, 4),
        "paired_diff_in_sd": round(z1 - z2, 4), "ci": ci(std),
        "covers_zero": bool(ci(std)[0] <= 0.0 <= ci(std)[1]),
        "raw_paired_diff": round(r1 - r2, 5), "raw_ci": ci(raw), "n_boot": len(std)}
    S = OUT["standardised"]
    log(f"\n--- P4 标准化之后 ---")
    log(f"  rescue 对比 {z1:+.4f} 个层内 SD;RNet 对比 {z2:+.4f} 个层内 SD")
    log(f"  配对差 {z1 - z2:+.4f} SD  CI {S['ci']}  "
        f"{'含 0' if S['covers_zero'] else '不含 0'}({len(std)} 次抽样)")
    log(f"  原口径(未标准化,论文现在报的那个){r1 - r2:+.5f}  CI {S['raw_ci']}")

    OUT["staked"] = {
        "P1": OUT["P1_raw"]["matches_published"],
        "P2": bool(0.20 <= sd1w <= 0.35),
        "P3": bool(0.15 <= sd2w <= 0.35),
        "P4": bool(not S["covers_zero"] and (z1 - z2) > 0)}
    log("\n=== 押注 ===")
    for k, v in OUT["staked"].items():
        log(f"  {k}: {'成立' if v else '**未成立**'}")
    OUT["K1_claim1_comparison_withdrawn"] = bool(not OUT["staked"]["P4"])
    OUT["K2_sd_outside_staked_range"] = bool(not (OUT["staked"]["P2"] and OUT["staked"]["P3"]))
    OUT["verdict"] = ("comparison_survives_standardisation_but_must_be_reported_in_sd_units"
                      if OUT["staked"]["P4"] else "CLAIM1_CROSS_READOUT_COMPARISON_WITHDRAWN")
    log(f"\n=== 杀条件 ===\n  K1 撤回 Claim 1 的跨读数比较:{OUT['K1_claim1_comparison_withdrawn']}")
    log(f"  K2 SD 落在押注区间外:{OUT['K2_sd_outside_staked_range']}")
    log(f"  {OUT['verdict']}")

    OUT["log"] = LOG
    json.dump(OUT, open(os.path.join(HERE, "m36_readout_scale.json"), "w"),
              indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
