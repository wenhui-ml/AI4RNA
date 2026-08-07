#!/usr/bin/env python3
"""Does our certificate grade predict what happens in a tube -- and does it beat the learned criterion?

THE QUESTION. Every judge this project uses is a thermodynamic predictor. Six models cross-checking each
other tells us how much they agree (little: pairwise kappa 0.11-0.34) and how much of nature they see at all
(46.2%-73.7% of natural pseudoknots). It does not tell us whether our grade tracks REALITY. OpenKnotBench
gives 34,747 designs that were actually synthesised and probed with 2A3 SHAPE, each carrying
`target_openknot_score` -- the fraction of residues whose measured reactivity agrees with the target,
averaged with the same score computed over CROSSED pairs only. That crossed-pair half is measuring, by a
completely different route, the exact thing `_knot_recall` adjudicates.

WHAT MAKES THIS A HEAD-TO-HEAD AND NOT A SELF-ASSESSMENT. The same file carries `RNet_F1_crossed_pair`:
RibonanzaNet's predicted crossed pairs scored against the target. RibonanzaNet is the field's answer to the
same problem we answer differently -- it LEARNS the criterion from chemical-mapping data, we ADMIT the
criterion is model-relative and quantify the dependence. Both predictions sit in the same table against the
same wet-lab outcome. This is the comparison, and it is available for free.

We expect to lose H3. RNet was trained on chemical mapping of exactly this kind; a thermodynamic panel
should not out-predict it. Losing H3 is not a failure of the study -- H4 is the one that matters, and H3
exists so that H4 cannot be read as a win by omission.

POPULATION, fixed before any grade was computed.

    gradeable (openknot_preflight.py): SN_filter == 1, score parses, design_sequence over ACGU,
        len(design) == len(target), target is a pseudoknot, target brackets balance, length <= 250 nt
    MINUS method 'M2'              -- ids are W01_13C / W01_1C: single point mutants of a starting
                                      sequence, not independent designs
    MINUS method 'Starting sequence' -- the natural wild type, not a design
    deduplicated by (puzzle, design_sequence), score averaged over replicate measurements
    PRIMARY restricted to topologies where our calibrated rule applies (kissing / chain, higher-order,
        crossing non-H). H-type is adjudicated in the product by a DIFFERENT rule (panel-unanimous), and
        pooling two rules under one label is the error `_topology_point` exists to prevent. H-type is
        reported separately, with its own n, whatever it shows.

GRADE, the ordinal scale already shipped:  0 not_confirmed < 1 k=1 < 2 k=2 < 3 k=3 < 4 certified
(certified = all three oracles reproduce the WHOLE target, not just the crossing stems).

PRE-REGISTERED HYPOTHESES. Four tests, Bonferroni threshold 0.05/4 = 0.0125. Every one is reported
whichever way it lands.

    H1  THE INSTRUMENT PREDICTS REALITY. Within each puzzle, Spearman rho between grade and OpenKnot Score.
        Over the puzzles, Wilcoxon signed-rank of those rho against 0. Within-puzzle because targets differ
        enormously in difficulty and a pooled correlation would mostly measure that.
        PASS: median rho > 0 and p < 0.0125.

    H2  THE GRADE IS NOT JUST THE GENERATOR. Same test inside each (puzzle, method) cell holding >= 20
        designs. If H1 passes and H2 fails, the grade was reading generator quality, not design quality,
        and the product must not claim otherwise.

    H3  HEAD-TO-HEAD vs THE LEARNED CRITERION. Per puzzle, rho(grade, score) against
        rho(RNet_F1_crossed_pair, score); paired Wilcoxon over puzzles, two-sided.

    H4  DOES OUR PANEL SEE ANYTHING RNet DOES NOT. Within-puzzle Spearman PARTIAL correlation of grade with
        score, controlling for RNet_F1_crossed_pair, via rank residuals. Wilcoxon over puzzles.
        PASS: median partial rho > 0 and p < 0.0125. This is the only outcome that would make a
        thermodynamic panel worth keeping alongside a learned model.

    KILL: H1 fails -> our grade does not predict the wet lab. Report it plainly, stop describing the panel
    as anything more than an internal-consistency measure, and close the axis.

COST: $0 in API. Everything is local binaries (pKiss / ProbKnot / ShapeKnots) plus ViennaRNA.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as st
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)

DATA = "/data/wenhui/openknot_data/Data/OpenKnotBench_data.v4.5.1.csv"
MAX_LEN = 250
CALIBRATED = ("kissing / chain", "higher-order", "crossing (non-H)")
DROP_METHODS = ("M2", "Starting sequence")
MIN_CELL = 20                       # designs needed before a (puzzle, method) cell is testable in H2
ALPHA = 0.05 / 4


def build_population(path):
    csv.field_size_limit(10 ** 9)
    from rna_harness.structure_utils import has_pseudoknot, pair_table, classify_topology
    want = ("id", "SN_filter", "round", "puzzle", "method", "target_openknot_score",
            "design_sequence", "target_structure", "RNet_F1", "RNet_F1_crossed_pair")
    agg, topo_cache = {}, {}
    with open(path, newline="") as fh:
        r = csv.reader(fh)
        idx = {c: i for i, c in enumerate(next(r)) if c in want}
        for raw in r:
            d = {c: raw[i] for c, i in idx.items()}
            if d["SN_filter"] != "1" or d["method"] in DROP_METHODS:
                continue
            seq, tgt = d["design_sequence"], d["target_structure"]
            try:
                score = float(d["target_openknot_score"])
                rnet = float(d["RNet_F1_crossed_pair"])
            except (TypeError, ValueError):
                continue
            if (not seq or set(seq) - set("ACGU") or len(seq) != len(tgt)
                    or not has_pseudoknot(tgt) or len(seq) > MAX_LEN):
                continue
            try:
                pair_table(tgt)
            except ValueError:
                continue
            t = topo_cache.get(tgt)
            if t is None:
                t = topo_cache[tgt] = classify_topology(tgt)["topology"]
            k = (d["puzzle"], seq)
            if k in agg:                       # replicate measurement of the same design: average it
                agg[k]["scores"].append(score)
            else:
                agg[k] = {"puzzle": d["puzzle"], "method": d["method"], "seq": seq, "target": tgt,
                          "topology": t, "rnet": rnet, "scores": [score], "id": d["id"]}
    out = []
    for v in agg.values():
        v["score"] = st.mean(v["scores"])
        v["n_rep"] = len(v.pop("scores"))
        out.append(v)
    return out


def work(item):
    """Grade one design. Returns the ordinal plus the raw per-oracle numbers, so re-analysis never refolds."""
    from rna_harness.cognitive.certify import _knot_recall, _panel_verdict
    i, seq, tgt = item
    try:
        kr = _knot_recall(seq, tgt)
        pv = _panel_verdict(seq, tgt)
    except Exception as e:                                                     # noqa: BLE001
        return {"i": i, "error": repr(e)[:160]}
    k = len(kr["confirmed_by"])
    return {"i": i, "grade": 4 if pv["unanimous"] else k, "k": k,
            "confirmed_by": kr["confirmed_by"], "unanimous": pv["unanimous"],
            "n_cross": kr["n_cross"], "per_oracle": kr["per_oracle"]}


# ------------------------------------------------------------------ statistics on within-puzzle clusters
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


def partial_spearman(x, y, z):
    """rho(x, y | z) on ranks: correlate the residuals of x and y after linear regression on rank(z)."""
    rx, ry, rz = _rankdata(x), _rankdata(y), _rankdata(z)
    mz = st.mean(rz)
    vz = sum((v - mz) ** 2 for v in rz)
    if vz <= 0 or len(x) < 4:
        return None

    def resid(r):
        mr = st.mean(r)
        b = sum((v - mz) * (w - mr) for v, w in zip(rz, r)) / vz
        return [w - mr - b * (v - mz) for v, w in zip(rz, r)]
    return _pearson(resid(rx), resid(ry))


def wilcoxon(vals, alt="two-sided"):
    """Signed-rank against 0, normal approximation with tie correction. Returns (statistic, p, n_used)."""
    import math
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
    if alt == "two-sided":
        return w, p2, n
    return w, (p2 / 2 if wp > wm else 1 - p2 / 2), n


def clusterwise(rows, key, fn, min_n):
    """Apply fn to each cluster with >= min_n rows; return {cluster: value} dropping unevaluable ones."""
    groups = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    out = {}
    for g, rs in groups.items():
        if len(rs) < min_n:
            continue
        v = fn(rs)
        if v is not None:
            out[g] = v
    return out


def report(name, vals, alt="greater"):
    if not vals:
        print(f"  {name}: 无可用簇")
        return {"n": 0}
    v = list(vals.values())
    w, p, n = wilcoxon(v, alt)
    pos = sum(1 for x in v if x > 0)
    # A two-sided test that clears the threshold says the two are DIFFERENT; printing "通过" next to it
    # invites reading a loss as a win. The verdict word is reserved for the one-sided tests, where a small
    # p really does mean the pre-registered direction held.
    if p is None:
        tail = "  (簇太少,无法检验)"
    elif alt == "two-sided":
        tail = (f"   显著{'(方向:我们更高)' if st.median(v) > 0 else '(方向:我们更低)'}"
                if p < ALPHA else "   无显著差异") + f"(阈值 {ALPHA:.4f})"
    else:
        tail = f"   {'通过' if p < ALPHA else '未通过'}(阈值 {ALPHA:.4f})"
    print(f"  {name}\n"
          f"    簇 {len(v)} 个,中位 rho {st.median(v):+.3f},均值 {st.mean(v):+.3f},"
          f"正向 {pos}/{len(v)}\n"
          f"    Wilcoxon p = {p:.3g}" + tail)
    return {"n": len(v), "median": st.median(v), "mean": st.mean(v), "n_pos": pos, "p": p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="smoke test only; 0 = full population")
    ap.add_argument("--out", default=os.path.join(HERE, "openknot_grade_vs_shape.json"))
    a = ap.parse_args()

    t0 = time.time()
    pop = build_population(a.data)
    if a.limit:
        pop = pop[::max(1, len(pop) // a.limit)][:a.limit]
        print(f"!! 冒烟模式:只跑 {len(pop)} 条,结果不作数")
    print(f"总体:{len(pop)} 条去重设计,{len(set(p['puzzle'] for p in pop))} 个靶标,"
          f"{len(set(p['method'] for p in pop))} 个生成器,{time.time()-t0:.1f}s 构建\n"
          f"  已剔除 {DROP_METHODS}(突变体与野生型,不是独立设计);重复测量按设计取均值", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    jobs = [(i, p["seq"], p["target"]) for i, p in enumerate(pop)]
    got, errs, t1 = 0, 0, time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(work, j) for j in jobs]):
            r = f.result()
            if "error" in r:
                errs += 1
            else:
                pop[r["i"]].update(r)
            got += 1
            if got % 2000 == 0 or got == len(jobs):
                el = (time.time() - t1) / 60
                print(f"    {got}/{len(jobs)}  {el:.1f} min  (预计总 {el/got*len(jobs):.0f} min)"
                      f"  神谕失败 {errs}", flush=True)

    rows = [p for p in pop if "grade" in p]
    print(f"\n判定完成:{len(rows)} 条有等级,{errs} 条神谕失败")

    cal = [r for r in rows if r["topology"] in CALIBRATED]
    hty = [r for r in rows if r["topology"] not in CALIBRATED]
    print(f"  主分析(已标定拓扑){len(cal)} 条;次要(H-type 等){len(hty)} 条\n")

    dist = {}
    for r in cal:
        dist[r["grade"]] = dist.get(r["grade"], 0) + 1
    names = {0: "not_confirmed", 1: "k=1", 2: "k=2", 3: "k=3", 4: "certified"}
    print("=== 等级分布与实测分(主分析总体;此处是描述,不是检验)===")
    for g in sorted(dist):
        s = [r["score"] for r in cal if r["grade"] == g]
        print(f"  {names[g]:14s} {dist[g]:6d}   实测分中位 {st.median(s):5.1f}   >80 占 {sum(x>80 for x in s)/len(s):5.1%}")

    res = {}
    print(f"\n=== H1 等级是否预测湿实验(靶标内)===")
    h1 = clusterwise(cal, lambda r: r["puzzle"],
                     lambda rs: spearman([r["grade"] for r in rs], [r["score"] for r in rs]), MIN_CELL)
    res["H1"] = report("rho(grade, OpenKnot Score)", h1, "greater")

    print(f"\n=== H2 等级是否只是在读生成器质量(靶标 x 生成器 内)===")
    h2 = clusterwise(cal, lambda r: (r["puzzle"], r["method"]),
                     lambda rs: spearman([r["grade"] for r in rs], [r["score"] for r in rs]), MIN_CELL)
    res["H2"] = report("rho(grade, Score | 同靶标同生成器)", h2, "greater")

    print(f"\n=== H3 与 RNet(从湿实验学出来的判据)正面对比 ===")
    rn = clusterwise(cal, lambda r: r["puzzle"],
                     lambda rs: spearman([r["rnet"] for r in rs], [r["score"] for r in rs]), MIN_CELL)
    print(f"  RNet_F1_crossed_pair 自己:簇 {len(rn)}," +
          (f"中位 rho {st.median(list(rn.values())):+.3f}" if rn else "无可用簇"))
    diff = {k: h1[k] - rn[k] for k in h1 if k in rn}
    res["H3"] = report("rho(我们) - rho(RNet),两侧检验", diff, "two-sided")
    res["rnet_median"] = st.median(list(rn.values())) if rn else None

    print(f"\n=== H4 控制住 RNet 之后,我们还剩多少独立信号 ===")
    h4 = clusterwise(cal, lambda r: r["puzzle"],
                     lambda rs: partial_spearman([r["grade"] for r in rs], [r["score"] for r in rs],
                                                 [r["rnet"] for r in rs]), MIN_CELL)
    res["H4"] = report("rho(grade, Score | RNet)", h4, "greater")

    print(f"\n=== 次要:H-type(产品在这里用的是另一条规则)===")
    hh = clusterwise(hty, lambda r: r["puzzle"],
                     lambda rs: spearman([r["grade"] for r in rs], [r["score"] for r in rs]), MIN_CELL)
    res["Htype"] = report("rho(grade, Score) on H-type", hh, "greater")

    p1 = res["H1"].get("p")
    ok1 = bool(p1 is not None and p1 < ALPHA and res["H1"]["median"] > 0)
    print(f"\n=== 预注册判决(阈值 {ALPHA:.4f})===")
    print(f"  H1 {'通过 —— 我们的等级确实预测湿实验结果' if ok1 else 'KILL —— 等级不预测湿实验,该轴关闭'}")
    for h in ("H2", "H3", "H4"):
        pv = res[h].get("p")
        print(f"  {h} 中位 {res[h].get('median', float('nan')):+.3f}  p = "
              f"{'n/a' if pv is None else f'{pv:.3g}'}")

    json.dump({"n_pop": len(pop), "n_graded": len(rows), "n_calibrated": len(cal), "n_htype": len(hty),
               "errors": errs, "grade_dist": dist, "alpha": ALPHA, "results": res,
               "per_puzzle_rho": {"ours": h1, "rnet": rn, "partial": h4},
               "rows": [{k: r[k] for k in ("id", "puzzle", "method", "topology", "grade", "k",
                                           "confirmed_by", "unanimous", "score", "rnet", "n_rep")}
                        for r in rows]},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}   (每条设计的等级已存盘,重分析不必重折叠)")


if __name__ == "__main__":
    main()
