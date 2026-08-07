#!/usr/bin/env python3
"""Can our certificate grade be tested against a WET-LAB outcome at all? Feasibility only -- no inference.

WHY THIS EXISTS. Every judge this project has ever used is a thermodynamic predictor: pKiss, ProbKnot,
ShapeKnots inside the rule; RNAPKplex, Knotty, IPknot outside it. Six models cross-checking each other. The
honest description of what that measures is MODEL AGREEMENT, and we have quantified how little of it there
is (pairwise kappa 0.11-0.34; 97.1% of shipped pseudoknot certificates could fail under a different judge;
each outside judge recognises only 46.2%-73.7% of NATURAL pseudoknots). What we have never done is ask
whether our grade predicts anything that HAPPENS IN A TUBE.

OpenKnotBench (eternagame/OpenKnotAIDesignData, v4.5.1, Apache-2.0) is 36,761 designs across four rounds,
each probed with 2A3 SHAPE and scored 0-100 by `target_openknot_score`: the fraction of residues whose
measured reactivity agrees with the target structure, averaged with the same score restricted to CROSSED
pairs. That last clause matters -- the crossed-pair subscore is measuring exactly the thing `_knot_recall`
adjudicates, by a completely different route.

WHAT THIS SCRIPT DOES AND DOES NOT DO. It counts how many rows we could grade, how they spread over targets
and generators, and how long the panel would take. It DOES NOT grade them and DOES NOT touch the outcome
column beyond its dispersion. The experiment itself is a separate script written after this one passes.

WHAT WE ALREADY KNOW WE CANNOT DO. The 2x2 as originally drawn --

        our panel   wet lab
   DesiRNA     A         B
   Struct2SeQ  C         D

-- has an EMPTY CELL B. DesiRNA is not among OpenKnot's design methods, and we have no wet lab, so our own
designs can never be measured here. Chasing B is chasing a laboratory we do not have.

The executable form keeps the same payload and drops the unbuildable cell: the designs with measured scores
become a TEST SET FOR OUR INSTRUMENT, with the generator as a covariate instead of an arm.

    outcome   `target_openknot_score` (wet-lab)
    predictor our grade on (design_sequence, target_structure): not_confirmed < k=1 < k=2 < k=3 < certified
    cluster   puzzle (one target), because designs for the same target are not independent
    covariate method, because generators differ in quality and the grade must beat that explanation

    If grade predicts score  -> the instrument predicts reality, and "calibrated judge-dependence" stops
                                being a statistic about models and becomes a free proxy for an experiment.
    If it does not           -> the panel measures internal consistency only. Say so, stop selling it as
                                more, and the axis closes.

PRE-REGISTERED GATES, written before any count was computed. Row is gradeable iff: SN_filter == 1;
target_openknot_score parses; design_sequence is non-empty over ACGU only; len(design_sequence) ==
len(target_structure); has_pseudoknot(target_structure); design length <= 250 nt.

    AMENDED AFTER THE FIRST RUN CRASHED, before any gate was evaluated: the target structure must also
    PARSE. Some rows carry unbalanced brackets (five '[' closed by four ']'), which is a defect in the
    released data, not in our reader. The filter was added and its count is reported in the exclusion
    table; no gate threshold was touched. Recorded here because a filter added after seeing data is a
    filter that must be visible.

    G1  >= 300 gradeable designs in total.
    G2  >= 8 distinct puzzles carrying >= 20 gradeable designs each. Fewer clusters than that and the
        within-target analysis has no power, and between-target comparison is confounded by target.
    G3  >= 3 distinct methods carrying >= 50 gradeable designs each. Fails -> the generator dimension is
        degenerate; the study degrades to single-generator calibration and must not be called a 2x2.
    G4  interquartile range of target_openknot_score >= 10 points. A flat outcome cannot be predicted by
        anything, ours included.
    G5  >= 50% of gradeable designs sit in a topology where our calibrated rule applies (kissing / chain,
        higher-order, crossing non-H). Fails -> the main analysis is restricted to the calibrated subset
        and its n is declared separately; H-type is adjudicated by the panel, a different rule, and mixing
        two rules under one label is the mistake `_topology_point` exists to prevent.

    KILL: G1, G2 or G4 fails -> the axis closes and nothing further is spent on it.

TIMING SAMPLE. 12 designs are folded by the panel purely to measure seconds-per-design, sampled by stride so
they span the length range. Their grades are printed as an oracle-health check and are NOT used for any
inference; the experiment re-folds everything from scratch.
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


def load(path):
    csv.field_size_limit(10 ** 9)
    from rna_harness.structure_utils import has_pseudoknot, pair_table
    want = ("id", "SN_filter", "round", "puzzle", "method", "target_openknot_score",
            "design_sequence", "target_structure")
    rows, reject = [], {}
    with open(path, newline="") as fh:
        r = csv.reader(fh)
        idx = {c: i for i, c in enumerate(next(r)) if c in want}
        n_total = 0
        for raw in r:
            n_total += 1
            d = {c: raw[i] for c, i in idx.items()}
            if d["SN_filter"] != "1":
                reject["signal/noise 未过滤"] = reject.get("signal/noise 未过滤", 0) + 1
                continue
            try:
                d["score"] = float(d["target_openknot_score"])
            except (TypeError, ValueError):
                reject["无实测分"] = reject.get("无实测分", 0) + 1
                continue
            seq, tgt = d["design_sequence"], d["target_structure"]
            if not seq or set(seq) - set("ACGU"):
                reject["序列含非 ACGU"] = reject.get("序列含非 ACGU", 0) + 1
                continue
            if len(seq) != len(tgt):
                reject["序列与结构不等长"] = reject.get("序列与结构不等长", 0) + 1
                continue
            if not has_pseudoknot(tgt):
                reject["靶标不是假结"] = reject.get("靶标不是假结", 0) + 1
                continue
            # Some OpenKnot target structures do not balance -- e.g. five '[' closed by four ']'. The
            # submitted design is shorter than the target it was submitted for (the v1.1.0 notes record
            # exactly this for a batch of MPNN sequences), so slicing the structure to the design cut one
            # partner of a crossing pair off the end. A half-open crossing pair has no crossing set to
            # recall, so `_knot_recall` would silently score it against an empty reference. Dropped, and
            # counted, rather than repaired: guessing which bracket to delete would be inventing a target.
            try:
                pair_table(tgt)
            except ValueError:
                reject["靶标括号不配平(数据缺陷)"] = reject.get("靶标括号不配平(数据缺陷)", 0) + 1
                continue
            if len(seq) > MAX_LEN:
                reject[f"长于 {MAX_LEN} nt"] = reject.get(f"长于 {MAX_LEN} nt", 0) + 1
                continue
            rows.append(d)
    return rows, reject, n_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--timing-n", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(HERE, "openknot_preflight.json"))
    a = ap.parse_args()

    from rna_harness.structure_utils import classify_topology

    t0 = time.time()
    rows, reject, n_total = load(a.data)
    print(f"OpenKnotBench v4.5.1:{n_total} 行,{time.time()-t0:.1f}s 读完\n")
    print("=== 被排除的行 ===")
    for k, v in sorted(reject.items(), key=lambda kv: -kv[1]):
        print(f"  {k:24s} {v:6d}")
    print(f"  {'可判设计':24s} {len(rows):6d}")

    by_m, by_p, by_r = {}, {}, {}
    for d in rows:
        by_m[d["method"]] = by_m.get(d["method"], 0) + 1
        by_p[d["puzzle"]] = by_p.get(d["puzzle"], 0) + 1
        by_r[d["round"]] = by_r.get(d["round"], 0) + 1

    print(f"\n=== 生成器(method)===")
    for k, v in sorted(by_m.items(), key=lambda kv: -kv[1]):
        s = [d["score"] for d in rows if d["method"] == k]
        print(f"  {k:24s} {v:6d}   实测分中位 {st.median(s):5.1f}   >80 占 {sum(x > 80 for x in s)/len(s):5.1%}")

    print(f"\n=== 靶标(puzzle)===  {len(by_p)} 个;每轮:" +
          ", ".join(f"round {k}={v}" for k, v in sorted(by_r.items())))
    big = {k: v for k, v in by_p.items() if v >= 20}
    print(f"  >= 20 条的靶标 {len(big)} 个;最小 {min(by_p.values())},最大 {max(by_p.values())}")

    topo = {}
    seen_t = {}
    for d in rows:
        t = seen_t.get(d["target_structure"])
        if t is None:
            t = seen_t[d["target_structure"]] = classify_topology(d["target_structure"])["topology"]
        d["topology"] = t
        topo[t] = topo.get(t, 0) + 1
    print(f"\n=== 拓扑(决定用哪条规则)===")
    for k, v in sorted(topo.items(), key=lambda kv: -kv[1]):
        mark = "已标定规则" if k in CALIBRATED else "面板一致(不同的规则)"
        print(f"  {k:24s} {v:6d}   {mark}")
    n_cal = sum(v for k, v in topo.items() if k in CALIBRATED)

    scores = sorted(d["score"] for d in rows)
    q1, q3 = scores[len(scores) // 4], scores[3 * len(scores) // 4]
    print(f"\n=== 结局的离散度 ===")
    print(f"  OpenKnot Score  中位 {st.median(scores):.1f}   四分位 [{q1:.1f}, {q3:.1f}]   IQR {q3-q1:.1f}")

    lens = sorted(len(d["design_sequence"]) for d in rows)
    print(f"  设计长度        中位 {st.median(lens)}   范围 [{lens[0]}, {lens[-1]}]")

    print(f"\n=== 面板计时样本({a.timing_n} 条,仅测速,不进任何推断)===")
    from rna_harness.cognitive.certify import _knot_recall
    step = max(1, len(rows) // a.timing_n)
    sample = rows[::step][:a.timing_n]
    times, grades, failed = [], {}, 0
    for d in sample:
        s = time.time()
        try:
            kr = _knot_recall(d["design_sequence"], d["target_structure"])
        except Exception as e:                                                 # noqa: BLE001
            failed += 1
            print(f"    !! {d['id'][:30]} {repr(e)[:80]}")
            continue
        times.append(time.time() - s)
        g = f"k={len(kr['confirmed_by'])}" if kr["knot_confirmed"] else "not_confirmed"
        grades[g] = grades.get(g, 0) + 1
    if times:
        per = st.mean(times)
        print(f"    每条 {per:.2f}s(中位 {st.median(times):.2f}s);等级分布 {grades};神谕失败 {failed}")
        print(f"    全部 {len(rows)} 条:单核 {per*len(rows)/3600:.1f} 小时,16 并行约 "
              f"{per*len(rows)/3600/16:.2f} 小时。API 成本 $0 —— 全部本地二进制")
    else:
        print("    !! 计时样本全部失败,神谕不可用,后续实验无法进行")

    g1 = len(rows) >= 300
    g2 = len(big) >= 8
    g3 = sum(1 for v in by_m.values() if v >= 50) >= 3
    g4 = (q3 - q1) >= 10
    g5 = n_cal / max(1, len(rows)) >= 0.5
    print(f"\n=== 预注册门槛 ===")
    print(f"  G1 >= 300 条可判                 {'通过' if g1 else '未通过'}({len(rows)})")
    print(f"  G2 >= 8 个靶标各 >= 20 条         {'通过' if g2 else '未通过'}({len(big)})")
    print(f"  G3 >= 3 个生成器各 >= 50 条       "
          f"{'通过' if g3 else '未通过 —— 降级为单生成器校准,不得称作 2x2'}"
          f"({sum(1 for v in by_m.values() if v >= 50)})")
    print(f"  G4 实测分 IQR >= 10              {'通过' if g4 else '未通过'}({q3-q1:.1f})")
    print(f"  G5 已标定拓扑占比 >= 50%          "
          f"{'通过' if g5 else '未通过 —— 主分析限定在已标定子集,单独声明 n'}"
          f"({n_cal/max(1,len(rows)):.1%},n={n_cal})")
    kill = not (g1 and g2 and g4)
    print(f"\n  {'KILL —— 该轴关闭,不再投入' if kill else '可以立项。下一步写实验脚本,预注册主假设与检验。'}")

    json.dump({"n_total": n_total, "n_gradeable": len(rows), "reject": reject,
               "by_method": by_m, "by_puzzle": by_p, "by_round": by_r, "topology": topo,
               "n_calibrated_topology": n_cal, "score_median": st.median(scores),
               "score_iqr": [q1, q3], "len_median": st.median(lens), "len_range": [lens[0], lens[-1]],
               "panel_sec_per_design": (st.mean(times) if times else None),
               "timing_grades": grades, "timing_failed": failed,
               "G1": g1, "G2": g2, "G3": g3, "G4": g4, "G5": g5, "KILL": kill},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
