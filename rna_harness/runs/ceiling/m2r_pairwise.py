#!/usr/bin/env python3
"""Do the base pairs our panel claims survive a CAUSAL INTERVENTION -- not just fail to be contradicted?

THE DISTINCTION THIS SCRIPT EXISTS FOR. Everything this project has shipped is a NON-REFUTATION: three
thermodynamic predictors were given a sequence and did not contradict the target's crossing pairs. Even the
wet-lab score we test against in `openknot_grade_vs_shape.py` is an OBSERVATION -- SHAPE reactivity says a
nucleotide is constrained, it does not say WHO it is holding. A low-reactivity position could be paired with
the intended partner, paired with something else, stacked, or pinned by a magnesium ion.

Mutate-map-rescue is different in kind. For a claimed pair (i, j):

    mutate i          -> if the pair is real, j's protection should VANISH
    mutate j too, compensatorily -> if the pair is real, protection should RETURN

`rescue_factor` is the fraction of the SHAPE perturbation that the double mutant restores: 0 = no evidence
the pair exists, 1 = fully restored. That is a predicted counterfactual that either happens or does not. It
is the first outcome in this project that can say "i pairs with j SPECIFICALLY" rather than "nobody
objected".

WHAT IS BEING TESTED. Not a design's grade -- a single PAIR's fate. For every design, we fold the wild-type
sequence with pKiss / ProbKnot / ShapeKnots and ask, pair by pair, whether each oracle recovered it. The tube
answers the same question independently.

                                     rescue_factor high      rescue_factor low
    our panel recovered the pair          true positive         FALSE POSITIVE
    our panel did not                     false negative        true negative

THE FALSE-POSITIVE COLUMN IS THE POINT. Our shipped false-accept rate, 0.4%, was measured against RANDOM
SEQUENCES -- a null so weak that surviving it means very little. This measures the rate against pairs that
the molecule itself declines to form. It is the first honest denominator we have ever had.

POPULATION. OK7a_M2R_data.v4.5.1.csv, 4,752 double mutants over 20 Round-3 targets x 9 design methods (160
designs), each double mutant naming exactly one target base pair. Verified before writing this script:
4752/4752 (mutA, mutB) land exactly on a target pair once `target_structure` is sliced by
[sub_start, sub_end] -- the file stores it at full padded length while mutA/mutB are 1-based in the design.

    PRIMARY population: the 1,412 CROSSING pairs. Those are what makes a target a pseudoknot and the only
        pairs `_knot_recall` adjudicates. Nested pairs are reported as a declared secondary; they are not
        the claim goal 4 makes.
    Rows with unparseable rescue_factor are dropped and counted.
    Clustering is by DESIGN (puzzle, method): pairs within one molecule share its folding, its SHAPE
        normalisation and its experimental batch, so they are not independent.

BUILT-IN NEGATIVE CONTROL, found in the data before any hypothesis was written: 150 of the 4,752 tested
pairs are NOT complementary in the wild-type design -- the designer's own sequence cannot form them. Those
pairs should show low rescue. If they do not, the assay is too noisy to carry any conclusion and this study
is void regardless of what the main test says. This is the `assert_control_can_fail` habit applied to an
experiment we did not run ourselves.

PRE-REGISTERED, before any fold was computed. Three tests, Bonferroni threshold 0.05/3 = 0.0167.

    C0  CONTROL. Median rescue_factor of the 150 non-complementary pairs is BELOW the median of
        complementary pairs, Mann-Whitney one-sided p < 0.05. FAILS -> the whole study is void and reported
        as void; no other number from it may be quoted.

    H1  Crossing pairs recovered by at least one oracle have HIGHER rescue_factor than crossing pairs
        recovered by none. Per design, difference of medians; Wilcoxon signed-rank over designs, one-sided.
        PASS: median difference > 0 and p < 0.0167.

    H2  MORE ORACLES IS BETTER, not just non-zero. Within each design, Spearman rho between the number of
        oracles recovering a crossing pair (0..3) and its rescue_factor; Wilcoxon over designs.
        PASS: median rho > 0 and p < 0.0167. H1 without H2 means the panel is a binary smoke detector and
        the k=1/k=2/k=3 ladder carries no extra information -- which the product currently claims it does.

    H3  FALSE-POSITIVE RATE AGAINST REALITY. Among crossing pairs our panel recovered UNANIMOUSLY (k=3),
        the fraction with rescue_factor < 0.25 (the OpenKnot scoring threshold for "paired"), with a
        Clopper-Pearson interval. Descriptive, no threshold to pass -- but it replaces the 0.4% random-
        sequence figure in every place the product quotes an error rate, whatever it turns out to be.

    KILL: C0 fails -> void. H1 fails -> our pair-level claims are not supported by intervention, the
    certificate must stop implying they are, and this axis closes.

COST: $0 in API. Local binaries only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics as st
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)

DATA = "/data/wenhui/openknot_data/Data/OK7a_M2R_data.v4.5.1.csv"
ALPHA = 0.05 / 3
PAIRED_THRESHOLD = 0.25     # OpenKnot's own reactivity cut for "this residue is paired"
COMP = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")}


def load(path):
    """Return (designs, pairs). One design per (puzzle, method); one row per tested base pair."""
    csv.field_size_limit(10 ** 9)
    from rna_harness.structure_utils import pair_table
    want = ("id", "SN_filter", "puzzle", "method", "sub_start", "sub_end",
            "target_structure", "mutA", "mutB", "rescue_factor", "sequence")
    with open(path, newline="") as fh:
        r = csv.reader(fh)
        idx = {c: i for i, c in enumerate(next(r)) if c in want}
        rows = [{c: x[i] for c, i in idx.items()} for x in r]

    wt = {}
    for d in rows:
        if not d["mutA"]:
            wt.setdefault((d["puzzle"], d["method"]), d)

    designs = {}
    for k, w in wt.items():
        ss, se = int(w["sub_start"]), int(w["sub_end"])
        designs[k] = {"puzzle": k[0], "method": k[1], "seq": w["sequence"][ss - 1:se],
                      "target": w["target_structure"][ss - 1:se]}

    pairs, dropped = [], {"no_rescue_factor": 0, "no_wt": 0}
    for d in rows:
        if not (d["mutA"] and d["mutB"]):
            continue
        k = (d["puzzle"], d["method"])
        if k not in designs:
            dropped["no_wt"] += 1
            continue
        try:
            rf = float(d["rescue_factor"])
        except (TypeError, ValueError):
            dropped["no_rescue_factor"] += 1
            continue
        des = designs[k]
        a, b = int(d["mutA"]) - 1, int(d["mutB"]) - 1
        pt = pair_table(des["target"])
        if pt[a] != b:
            continue
        i, j = (a, b) if a < b else (b, a)
        pairs.append({"design": k, "i": i, "j": j, "rf": rf,
                      "crossed": des["target"][a] in "[]{}<>",
                      "wt_complementary": (des["seq"][a], des["seq"][b]) in COMP,
                      "sn": d["SN_filter"], "id": d["id"]})
    return designs, pairs, dropped


def fold_design(item):
    """Which of the three oracles recovers each pair of this design's wild type."""
    key, seq = item
    from rna_harness.oracles_pk import oracle_pairs
    try:
        ps = oracle_pairs(seq)
    except Exception as e:                                                     # noqa: BLE001
        return {"key": key, "error": repr(e)[:160]}
    out = {}
    for name, s in ps.items():
        out[name] = None if s is None else sorted(tuple(x) for x in s)
    return {"key": key, "oracle_pairs": out}


# ------------------------------------------------------------------------------------------ statistics
def _rankdata(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    return ranks


def _pearson(a, b):
    n = len(a)
    if n < 3:
        return None
    ma, mb = st.mean(a), st.mean(b)
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va <= 0 or vb <= 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va ** 0.5 * vb ** 0.5)


def spearman(a, b):
    return _pearson(_rankdata(a), _rankdata(b))


def mannwhitney(a, b):
    """One-sided: is `a` stochastically LARGER than `b`? Normal approximation with tie correction."""
    na, nb = len(a), len(b)
    if na < 3 or nb < 3:
        return None, None
    r = _rankdata(list(a) + list(b))
    ua = sum(r[:na]) - na * (na + 1) / 2.0
    mu = na * nb / 2.0
    counts = {}
    for v in r:
        counts[v] = counts.get(v, 0) + 1
    tie = sum(c ** 3 - c for c in counts.values())
    n = na + nb
    sd = math.sqrt(na * nb / 12.0 * ((n + 1) - tie / (n * (n - 1))))
    if sd <= 0:
        return ua, None
    z = (ua - mu - 0.5) / sd
    return ua, 0.5 * math.erfc(z / math.sqrt(2))          # P(U_a >= observed), one-sided "a larger"


def wilcoxon(vals, alt="greater"):
    nz = [v for v in vals if v != 0]
    n = len(nz)
    if n < 6:
        return None, None, n
    ranks = _rankdata([abs(v) for v in nz])
    wp = sum(r for r, v in zip(ranks, nz) if v > 0)
    wm = sum(r for r, v in zip(ranks, nz) if v < 0)
    w = min(wp, wm)
    mu = n * (n + 1) / 4.0
    counts = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    tie = sum(c ** 3 - c for c in counts.values())
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0 - tie / 48.0)
    if sd <= 0:
        return w, None, n
    z = (w - mu + 0.5) / sd
    p2 = math.erfc(abs(z) / math.sqrt(2))
    return w, (p2 if alt == "two-sided" else (p2 / 2 if wp > wm else 1 - p2 / 2)), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--min-pairs", type=int, default=4, help="pairs a design needs before it is a cluster")
    ap.add_argument("--out", default=os.path.join(HERE, "m2r_pairwise.json"))
    a = ap.parse_args()
    from external_judge import clopper_pearson

    t0 = time.time()
    designs, pairs, dropped = load(a.data)
    print(f"M2R:{len(designs)} 条设计(20 靶标 x 9 方法),{len(pairs)} 个被因果测试过的碱基对,"
          f"{time.time()-t0:.1f}s")
    print(f"  丢弃 {dropped}")
    cross = [p for p in pairs if p["crossed"]]
    print(f"  交叉配对(主分析){len(cross)};嵌套配对(次要){len(pairs)-len(cross)}", flush=True)

    print(f"\n=== C0 阴性对照:野生型序列根本不互补的碱基对 ===")
    nc = [p["rf"] for p in pairs if not p["wt_complementary"]]
    cc = [p["rf"] for p in pairs if p["wt_complementary"]]
    u, p_c0 = mannwhitney(cc, nc)
    print(f"  不互补 n={len(nc)} 中位 {st.median(nc):+.3f}   互补 n={len(cc)} 中位 {st.median(cc):+.3f}")
    print(f"  Mann-Whitney(互补 > 不互补)p = {p_c0:.3g}"
          f"   {'通过 —— 测定能分辨真假配对' if (p_c0 is not None and p_c0 < 0.05) else 'C0 未通过 —— 整个研究作废'}")
    c0_ok = bool(p_c0 is not None and p_c0 < 0.05 and st.median(nc) < st.median(cc))
    if not c0_ok:
        print("\n  !! C0 失败。按预注册,本研究作废,以下任何数字都不得引用。")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    print(f"\n=== 折叠 {len(designs)} 条野生型设计(三个神谕)===", flush=True)
    got, errs, t1 = 0, 0, time.time()
    folded = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(fold_design, (k, v["seq"])) for k, v in designs.items()]
        for f in as_completed(futs):
            r = f.result()
            got += 1
            if "error" in r:
                errs += 1
            else:
                folded[tuple(r["key"])] = {n: (None if v is None else {tuple(x) for x in v})
                                           for n, v in r["oracle_pairs"].items()}
            if got % 40 == 0 or got == len(futs):
                print(f"    {got}/{len(futs)}  {(time.time()-t1)/60:.1f} min  失败 {errs}", flush=True)

    for p in pairs:
        f = folded.get(p["design"])
        if not f:
            p["n_oracles"] = None
            continue
        hit = [n for n, s in f.items() if s is not None and (p["i"], p["j"]) in s]
        p["by"] = hit
        p["n_oracles"] = len(hit)

    usable = [p for p in cross if p.get("n_oracles") is not None]
    print(f"\n交叉配对可判 {len(usable)}/{len(cross)}")
    print("\n=== 描述:面板确认数 vs 湿实验补偿恢复(交叉配对)===")
    for k in range(4):
        s = [p["rf"] for p in usable if p["n_oracles"] == k]
        if not s:
            continue
        print(f"  {k} 个神谕确认  n={len(s):5d}   rescue 中位 {st.median(s):+.3f}"
              f"   rf>=0.5 占 {sum(x >= .5 for x in s)/len(s):5.1%}"
              f"   rf<{PAIRED_THRESHOLD} 占 {sum(x < PAIRED_THRESHOLD for x in s)/len(s):5.1%}")

    by_design = {}
    for p in usable:
        by_design.setdefault(p["design"], []).append(p)
    clusters = {k: v for k, v in by_design.items() if len(v) >= a.min_pairs}
    print(f"\n可用簇(每条设计 >= {a.min_pairs} 个交叉对):{len(clusters)}/{len(by_design)}")

    print(f"\n=== H1 被至少一个神谕确认的交叉对,补偿恢复更高吗(设计内配对比较)===")
    diffs = {}
    for k, ps in clusters.items():
        hi = [p["rf"] for p in ps if p["n_oracles"] >= 1]
        lo = [p["rf"] for p in ps if p["n_oracles"] == 0]
        if len(hi) >= 2 and len(lo) >= 2:
            diffs[k] = st.median(hi) - st.median(lo)
    if diffs:
        v = list(diffs.values())
        w, p1, n = wilcoxon(v, "greater")
        pos = sum(1 for x in v if x > 0)
        print(f"  两侧都有的设计 {len(v)} 条,中位差 {st.median(v):+.3f},正向 {pos}/{len(v)}")
        print(f"  Wilcoxon p = {p1:.3g}   {'通过' if (p1 is not None and p1 < ALPHA) else '未通过'}"
              f"(阈值 {ALPHA:.4f})")
    else:
        p1 = None
        print("  没有一条设计同时含有被确认与未被确认的交叉对 —— H1 无法检验")

    print(f"\n=== H2 确认它的神谕【越多】,恢复得越好吗(k=1/2/3 阶梯是否有信息)===")
    rhos = {}
    for k, ps in clusters.items():
        if len({p["n_oracles"] for p in ps}) < 2:
            continue
        r = spearman([p["n_oracles"] for p in ps], [p["rf"] for p in ps])
        if r is not None:
            rhos[k] = r
    if rhos:
        v = list(rhos.values())
        w, p2, n = wilcoxon(v, "greater")
        print(f"  有变异的设计 {len(v)} 条,中位 rho {st.median(v):+.3f},正向 "
              f"{sum(1 for x in v if x > 0)}/{len(v)}")
        print(f"  Wilcoxon p = {p2:.3g}   {'通过' if (p2 is not None and p2 < ALPHA) else '未通过'}"
              f"(阈值 {ALPHA:.4f})")
    else:
        p2 = None
        print("  无可用簇")

    print(f"\n=== H3 对着【现实】的假阳性率(取代 0.4% 那个随机序列数字)===")
    k3 = [p["rf"] for p in usable if p["n_oracles"] == 3]
    fp = sum(1 for x in k3 if x < PAIRED_THRESHOLD)
    if k3:
        lo, hi = clopper_pearson(fp, len(k3))
        print(f"  面板三家一致确认的交叉对 n={len(k3)},其中补偿恢复 < {PAIRED_THRESHOLD} 的占 "
              f"{fp/len(k3):.1%}  95% [{lo:.1%}, {hi:.1%}]")
        print(f"  对照:我们出厂引用的误收率 0.4% 是对【随机序列】测的 —— 那是一个弱得多的零假设")
    else:
        lo = hi = None
        print("  没有三家一致的交叉对")

    print(f"\n=== 次要:嵌套配对(不是目标 4 的主张)===")
    nest = [p for p in pairs if not p["crossed"] and p.get("n_oracles") is not None]
    for k in range(4):
        s = [p["rf"] for p in nest if p["n_oracles"] == k]
        if s:
            print(f"  {k} 个神谕确认  n={len(s):5d}   rescue 中位 {st.median(s):+.3f}")

    print(f"\n=== 预注册判决(阈值 {ALPHA:.4f})===")
    print(f"  C0 {'通过' if c0_ok else '失败 —— 研究作废'}")
    print(f"  H1 {'通过 —— 面板确认的交叉对确实经得起因果干预' if (c0_ok and p1 is not None and p1 < ALPHA) else 'KILL —— 逐对主张未获干预支持,证书不得再暗示它'}")
    print(f"  H2 {'通过 —— k=1/2/3 阶梯携带真实信息' if (p2 is not None and p2 < ALPHA) else '未通过 —— 面板只是个二值烟感,阶梯本身无增量信息'}")

    json.dump({"n_designs": len(designs), "n_pairs": len(pairs), "n_crossed": len(cross),
               "n_usable": len(usable), "fold_errors": errs, "dropped": dropped, "alpha": ALPHA,
               "C0": {"p": p_c0, "n_noncomp": len(nc), "median_noncomp": st.median(nc),
                      "median_comp": st.median(cc), "ok": c0_ok},
               "H1": {"n_designs": len(diffs), "median_diff": (st.median(list(diffs.values())) if diffs else None),
                      "p": p1},
               "H2": {"n_designs": len(rhos), "median_rho": (st.median(list(rhos.values())) if rhos else None),
                      "p": p2},
               "H3": {"n_k3": len(k3), "n_fp": fp, "rate": (fp/len(k3) if k3 else None), "ci": [lo, hi]},
               "pairs": [{k: p[k] for k in ("design", "i", "j", "rf", "crossed", "wt_complementary",
                                            "n_oracles", "id") if k in p} for p in pairs]},
              open(a.out, "w"), ensure_ascii=False, indent=1, default=list)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
