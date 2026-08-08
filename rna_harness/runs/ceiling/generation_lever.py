#!/usr/bin/env python3
"""Where does the generation gap live, and is there a handle on it we are not touching?

WHAT IS ESTABLISHED. At matched novelty -- our designs sit 28.5% identical to the wild type, the field's
learned designers 28-32% -- they score 0.93-0.98 crossed-pair F1 and we score 0.826. The gap is in the
whole distribution, not the tail we ship: our pool median loses to the novelty-matched peer median on 57 of
57 targets, only 7.5% of our raw candidates beat a typical peer design, and on 10 targets none of ~300 does.
Selection is not the lever (worth at most +0.55 wet-lab points) and neither is more search time. So the
question is what our generator is doing differently, and this looks for a handle before anything is built.

THE LEAD WORTH CHECKING FIRST. DesiRNA exposes an ACGU-composition constraint and our adapter has never
passed it -- every run this project has done carries `ACGUoff` in its own output directory name. A
thermodynamic optimiser with unconstrained composition is free to reach extreme GC content, because GC pairs
are simply stronger; a model trained on real molecules is not, because real molecules are not like that.
If that is what separates us, the fix is a command-line flag rather than a new generator.

    B1  ARE WE COMPOSITIONALLY DIFFERENT? GC fraction and longest homopolymer run, ours against the
        novelty-matched peers, paired within target. Descriptive -- it establishes whether there is
        anything to explain, not whether it matters.

    B2  DOES COMPOSITION PREDICT QUALITY IN THE FIELD'S OWN DATA? Within target, across real designs,
        Spearman between GC fraction and RNet crossed-pair F1, and between GC fraction and the measured
        wet-lab score. Clustered by target. If composition is unrelated to quality on molecules that were
        actually synthesised, it cannot be our lever no matter how different we are.

    B3  THE ONE THAT DECIDES IT. Inside OUR OWN pool of 15,226 scored candidates, does composition predict
        crossed-pair F1 within target? This is the counterfactual that matters: it asks whether steering
        our generator's composition would have produced better candidates from the search we already run.
        PASS: within-target Spearman is significantly negative (less GC -> better) or positive, with
        |median rho| >= 0.15 and p < 0.025 over targets.
        KILL: no relationship inside our own pool -> composition is not the handle, whatever B1 and B2 say,
        and the next candidate handle gets tested instead.

    B4  WHERE ELSE COULD IT LIVE? The gap against peers as a function of target length and of how many
        crossing pairs the target has. Descriptive, and it says whether a fix should be general or aimed
        at a regime.

Nothing here changes the product. It is a search for the handle, and it can come back saying there is none
in this direction, which is a useful answer at a cost of a few minutes.

COST: $0. Local CPU, already-computed scores plus published columns.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DATA = "/data/wenhui/openknot_data/Data/OpenKnotBench_data.v4.5.1.csv"
PEERS = ["Struct2SeQ", "MPNN-RFdiff", "gRNAde-no3d", "codesign-RFdiff"]
ALPHA = 0.025

_OPEN = {"(": ")", "[": "]", "{": "}", "<": ">"}
_CLOSE = {v: k for k, v in _OPEN.items()}


def pk_pairs(s):
    stacks, out = {o: [] for o in _OPEN}, set()
    for i, c in enumerate(s):
        if c in _OPEN:
            stacks[c].append(i)
        elif c in _CLOSE and stacks[_CLOSE[c]]:
            out.add((stacks[_CLOSE[c]].pop(), i))
    return out


def crossing(pairs):
    out = set()
    for (i, j) in pairs:
        for (m, n) in pairs:
            if (m < i < n < j) or (i < m < j < n):
                out.add((i, j))
                break
    return out


def gc(s):
    return (s.count("G") + s.count("C")) / len(s) if s else None


def maxrun(s):
    best = run = 1
    for i in range(1, len(s)):
        run = run + 1 if s[i] == s[i - 1] else 1
        best = max(best, run)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=os.path.join(HERE, "gap_score_dump.json"))
    ap.add_argument("--pools", default=os.path.join(HERE, "pool_rerank_pools.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "generation_lever.json"))
    a = ap.parse_args()

    from m2r_pairwise import spearman, wilcoxon

    T = json.load(open(a.pools))["targets"]
    P = {r["puzzle"]: r["pool"] for r in json.load(open(a.pools))["rows"] if "pool" in r}
    D = json.load(open(a.dump))["rows"]
    ours = collections.defaultdict(list)
    for r in D:
        seq = P[r["puzzle"]][r["i"]]["seq"] if r["i"] < len(P.get(r["puzzle"], [])) else None
        if seq:
            ours[r["puzzle"]].append((seq, r["f1c"]))
    print(f"我们的候选 {sum(len(v) for v in ours.values())} 条,{len(ours)} 个靶标")

    csv.field_size_limit(10 ** 9)
    theirs = collections.defaultdict(list)
    with open(DATA, newline="") as fh:
        r = csv.reader(fh)
        idx = {k: i for i, k in enumerate(next(r))}
        for x in r:
            if x[idx["SN_filter"]] != "1":
                continue
            p, s, m = x[idx["puzzle"]], x[idx["design_sequence"]], x[idx["method"]]
            if p not in T or not s or x[idx["target_structure"]] != T[p]["target"]:
                continue
            try:
                f1c = float(x[idx["RNet_F1_crossed_pair"]])
                wet = float(x[idx["target_openknot_score"]])
            except (TypeError, ValueError):
                continue
            theirs[p].append((m, s, f1c, wet))

    # ---- B1 ------------------------------------------------------------------------------------------
    print(f"\n=== B1 我们的候选与同新颖度同行,组成上有区别吗 ===")
    dg, dr = [], []
    for p in sorted(ours):
        pe = [s for m, s, _, _ in theirs.get(p, []) if m in PEERS]
        if not pe or not ours[p]:
            continue
        og, pg = st.median([gc(s) for s, _ in ours[p]]), st.median([gc(s) for s in pe])
        orr, pr = st.median([maxrun(s) for s, _ in ours[p]]), st.median([maxrun(s) for s in pe])
        dg.append(og - pg)
        dr.append(orr - pr)
    w, pg1, _ = wilcoxon(dg, "two-sided")
    w, pr1, _ = wilcoxon(dr, "two-sided")
    print(f"  GC 比例   我们 − 同行 中位 {st.median(dg):+.3f}({len(dg)} 个靶标),"
          f"p = {'n/a' if pg1 is None else f'{pg1:.3g}'}")
    print(f"  最长同碱基串 我们 − 同行 中位 {st.median(dr):+.2f},"
          f"p = {'n/a' if pr1 is None else f'{pr1:.3g}'}")

    # ---- B2 ------------------------------------------------------------------------------------------
    rf, rw = [], []
    for p in sorted(theirs):
        rs = theirs[p]
        if len(rs) < 30:
            continue
        g = [gc(s) for _, s, _, _ in rs]
        if len(set(g)) < 3:
            continue
        s1 = spearman(g, [f for _, _, f, _ in rs])
        s2 = spearman(g, [wv for _, _, _, wv in rs])
        if s1 is not None:
            rf.append(s1)
        if s2 is not None:
            rw.append(s2)
    print(f"\n=== B2 在【真实分子】上,GC 比例与质量有关系吗(靶标内)===")
    print(f"  rho(GC, RNet F1)      中位 {st.median(rf):+.3f}({len(rf)} 个靶标)")
    print(f"  rho(GC, 实测湿实验分) 中位 {st.median(rw):+.3f}({len(rw)} 个靶标)")

    # ---- B3 ------------------------------------------------------------------------------------------
    ro = []
    for p in sorted(ours):
        rs = ours[p]
        if len(rs) < 30:
            continue
        g = [gc(s) for s, _ in rs]
        if len(set(g)) < 3:
            continue
        s1 = spearman(g, [f for _, f in rs])
        if s1 is not None:
            ro.append(s1)
    w, p3, _ = wilcoxon(ro, "two-sided")
    print(f"\n=== B3 在【我们自己的池子】里,组成能预测质量吗(这条才决定是不是抓手)===")
    print(f"  rho(GC, 交叉配对 F1)中位 {st.median(ro):+.3f}({len(ro)} 个靶标),"
          f"为负的 {sum(1 for x in ro if x<0)}/{len(ro)},p = {'n/a' if p3 is None else f'{p3:.3g}'}")
    ok = bool(p3 is not None and p3 < ALPHA and abs(st.median(ro)) >= 0.15)
    print(f"  {'通过 —— 组成是一个抓手,而且我们从没设过这个参数' if ok else 'KILL —— 组成预测不了我们自己池子里的好坏,不是这个方向'}")

    # ---- B4 ------------------------------------------------------------------------------------------
    print(f"\n=== B4 缺口在哪些靶标上最大 ===")
    rows = []
    for p in sorted(ours):
        pe = [f for m, _, f, _ in theirs.get(p, []) if m in PEERS]
        if not pe:
            continue
        rows.append((p, len(T[p]["target"]), len(crossing(pk_pairs(T[p]["target"]))),
                     st.median([f for _, f in ours[p]]) - st.median(pe)))
    for lab, key in (("按长度", 1), ("按交叉配对数", 2)):
        rs = sorted(rows, key=lambda x: x[key])
        k = max(1, len(rs) // 3)
        lo, hi = rs[:k], rs[-k:]
        print(f"  {lab}:最小三分之一 缺口中位 {st.median([x[3] for x in lo]):+.3f}"
              f"(值 {lo[0][key]}-{lo[-1][key]});"
              f"最大三分之一 {st.median([x[3] for x in hi]):+.3f}(值 {hi[0][key]}-{hi[-1][key]})")

    json.dump({"B1": {"gc_diff_median": st.median(dg), "gc_p": pg1,
                      "maxrun_diff_median": st.median(dr), "maxrun_p": pr1, "n_targets": len(dg)},
               "B2": {"rho_gc_f1_median": st.median(rf), "rho_gc_wet_median": st.median(rw),
                      "n_targets": len(rf)},
               "B3": {"rho_median": st.median(ro), "p": p3, "n_targets": len(ro), "pass": ok},
               "B4": {"per_target": [{"puzzle": p, "n": n, "n_cross": nc, "gap": g}
                                     for p, n, nc, g in rows]}},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
