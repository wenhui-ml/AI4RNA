#!/usr/bin/env python3
"""R1b: give the 0.811/0.820 ceiling an interval (rule 49), from ALL repeats in the raw release.

`ceiling_denominator_review.md` section 4 recorded the debt in its own words: the ceiling that the whole
programme uses to say "how much room is left" rests on 31 pairs' median spread (4.097), a point estimate
with no interval, and a +-20% move in the assumed noise SD swings the ceiling 0.738 ~ 0.887. Section 5
said fixing this "比任何新列都优先". It was never run.

The six population files carry zero duplicate (target, sequence) rows -- sampling de-duplicated by
sequence -- so the repeats can only come from the raw release. This goes there.

  ceiling = 1 - (sigma_noise / sigma_within)^2
  sigma_noise estimated from repeated measurements of the SAME (puzzle, design_sequence)
  sigma_within = within-cell SD of wet, cells = (puzzle, method), unbiased (SSE/(N - n_cells))

Two estimators for sigma_noise, both reported:
  (a) spread-median route, the one the declaration used: SD ~ median(|diff|)/sqrt(2) for duplicate pairs
  (b) pooled route, which uses groups of any size: sigma^2 = sum_g SSE_g / sum_g (n_g - 1)
Bootstrap resamples the REPEAT GROUP (the cluster), not the row.

COST: $0, offline.
"""
from __future__ import annotations

import argparse, collections, csv, json, math, os, random, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
DATA = "/data/wenhui/openknot_data/Data/OpenKnotBench_data.v4.5.1.csv"
POPS = (("样本2(主)", "panel_f1_fresh.json"), ("基准4560", "panel_f1_designs.json"),
        ("样本3", "panel_f1_third.json"), ("第四批", "panel_f1_fourth.json"))
NBOOT = 4000


def repeats():
    """every (puzzle, design_sequence) measured more than once, under the release's own SN_filter."""
    csv.field_size_limit(10 ** 9)
    g = collections.defaultdict(list)
    n_rows = 0
    with open(DATA, newline="") as fh:
        r = csv.reader(fh)
        idx = {k: i for i, k in enumerate(next(r))}
        for x in r:
            if x[idx["SN_filter"]] != "1":
                continue
            s = x[idx["design_sequence"]]
            if not s:
                continue
            try:
                wet = float(x[idx["target_openknot_score"]])
            except (TypeError, ValueError):
                continue
            n_rows += 1
            g[(x[idx["puzzle"]], s)].append(wet)
    return {k: v for k, v in g.items() if len(v) >= 2}, n_rows, len(g)


def within_sd(fn):
    from e1_ensemble_marginals import load_pop
    rec, _s, _m = load_pop(fn)
    cells = collections.defaultdict(list)
    for r in rec:
        cells[(r["puzzle"], r["method"])].append(r["wet"])
    sse = n = k = 0.0
    for v in cells.values():
        if len(v) < 2:
            continue
        m = st.fmean(v)
        sse += sum((x - m) ** 2 for x in v); n += len(v); k += 1
    return math.sqrt(sse / (n - k)), int(n), int(k)


def sd_spread(groups):
    """(a) the declaration's route: pairs only, SD ~ median(|diff|)/sqrt(2)."""
    d = [abs(v[0] - v[1]) for v in groups if len(v) == 2]
    return (st.median(d) / math.sqrt(2) if d else None), len(d)


def sd_pooled(groups):
    """(b) pooled within-group SD, uses groups of any size."""
    sse = df = 0.0
    for v in groups:
        m = st.fmean(v)
        sse += sum((x - m) ** 2 for x in v); df += len(v) - 1
    return (math.sqrt(sse / df) if df > 0 else None), int(df)


def boot(groups, fn_, seed):
    rng = random.Random(seed); n = len(groups); out = []
    for _ in range(NBOOT):
        s = [groups[rng.randrange(n)] for _ in range(n)]
        v, _k = fn_(s)
        if v is not None:
            out.append(v)
    out.sort()
    return [out[int(0.025 * len(out))], out[int(0.975 * len(out))]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "r1b_ceiling_interval.json"))
    a = ap.parse_args()
    print("=== R1b:给 0.811/0.820 那根上限装区间(第 49 条)===", flush=True)
    G, n_rows, n_keys = repeats()
    gl = list(G.values())
    sizes = collections.Counter(len(v) for v in gl)
    print(f"  原始 release 通过 SN_filter 的行 {n_rows},不同 (靶标,序列) {n_keys}")
    print(f"  **重复测量组 {len(gl)}**(组大小分布 {dict(sorted(sizes.items()))}),"
          f"覆盖 {sum(len(v) for v in gl)} 行")
    if len(gl) < 20:
        json.dump({"verdict": "void_too_few_repeat_groups", "n": len(gl)},
                  open(a.out, "w"), indent=1, ensure_ascii=False)
        print("  **重复组不足 20 ⇒ 作废**"); return 0

    sa, npair = sd_spread(gl)
    sb, df = sd_pooled(gl)
    ca = boot(gl, sd_spread, a.seed + 11)
    cb = boot(gl, sd_pooled, a.seed + 23)
    print(f"\n  (a) 声明用的口径 中位spread/√2  sigma = **{sa:.4f}**  95% [{ca[0]:.4f},{ca[1]:.4f}]  (配对组 {npair})")
    print(f"  (b) 合并组内口径             sigma = **{sb:.4f}**  95% [{cb[0]:.4f},{cb[1]:.4f}]  (自由度 {df})")
    print(f"  声明里写的是 sigma ≈ 2.897(由 31 对推得)")

    print(f"\n=== 格内上限 = 1 − (sigma_noise / sigma_格内)^2 ===")
    print(f"  {'人群':<12}{'格内 SD':>9}{'分子':>7}{'格':>6}{'上限(a)':>22}{'上限(b)':>22}")
    R = {}
    for nm, fn in POPS:
        sw, N, K = within_sd(fn)
        def ceil(s):
            return max(0.0, 1 - (s / sw) ** 2)
        # the ceiling is monotone decreasing in sigma, so the interval maps端点对调
        ra = [ceil(ca[1]), ceil(ca[0])]
        rb = [ceil(cb[1]), ceil(cb[0])]
        R[fn] = {"name": nm, "within_sd": round(sw, 4), "n": N, "cells": K,
                 "ceiling_a": round(ceil(sa), 4), "ci_a": [round(x, 4) for x in ra],
                 "ceiling_b": round(ceil(sb), 4), "ci_b": [round(x, 4) for x in rb]}
        print(f"  {nm:<12}{sw:9.4f}{N:7d}{K:6d}"
              f"   {ceil(sa):.4f} [{ra[0]:.4f},{ra[1]:.4f}]"
              f"   {ceil(sb):.4f} [{rb[0]:.4f},{rb[1]:.4f}]")

    P = R["panel_f1_fresh.json"]
    print(f"\n=== 这对『我们占多少』那张表意味着什么(主人群)===")
    for tag, r2 in (("出厂判据", 0.047), ("两根 GC 列", 0.1001), ("整套 16 列面板", 0.1651)):
        lo = r2 / P["ci_a"][1] * 100; hi = r2 / P["ci_a"][0] * 100
        print(f"  {tag:<16}净 R² {r2:.4f} ⇒ 占上限 **{r2/P['ceiling_a']*100:.1f}%**  区间 [{lo:.1f}%, {hi:.1f}%]")
    json.dump({"n_repeat_groups": len(gl), "group_sizes": {str(k): v for k, v in sorted(sizes.items())},
               "sigma_a": round(sa, 4), "ci_a": [round(x, 4) for x in ca], "n_pairs": npair,
               "sigma_b": round(sb, 4), "ci_b": [round(x, 4) for x in cb], "df": df,
               "declared_sigma": 2.897, "per_pop": R},
              open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
