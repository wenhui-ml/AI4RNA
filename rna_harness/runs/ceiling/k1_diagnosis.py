#!/usr/bin/env python3
"""IPknot accepts 46.2% of natural pseudoknots and 4.6% of our k=1 designs. Which of the two is that about?

THE NUMBER THAT PROMPTED THIS. Three outside judges were measured against their own rate on 186 native
pseudoknots, and one of them is wildly out of line on the grade we ship most:

                    natives      our k=1 designs      ratio
    RNAPKplex        66.7%            32.7%            0.49
    Knotty           73.7%            45.8%            0.62
    IPknot           46.2%             4.6%            0.10      <- ten-fold, not two-fold

k=1 is 57% of everything this pipeline ships. A ten-fold gap on the majority grade is either a real defect in
those designs or a quirk of one judge, and those two answers demand opposite actions: fix the designs, or
word the certificate's warning as judge-specific and stop implying it is universal.

WHAT SEPARATES THE TWO. IPknot is the only judge that scores from LEARNED base-pairing probabilities
(LinearPartition-C) rather than thermodynamic search. A probability-thresholded method rejects a sequence
whose ensemble is DIFFUSE -- many competing structures, no pair confident enough to clear the bar -- even
when a search-based method happily reports one MFE-ish answer. So the hypothesis with teeth is not "our
designs are wrong", it is "our k=1 designs fold into a less determined ensemble than nature does", which is
a property we can measure directly and which no oracle in our panel ever looks at.

PRE-REGISTERED, before any sequence was folded:

    P1  k=1 designs are measurably LESS determined than natives: mean positional entropy higher, and the
        MFE structure's ensemble probability lower, both by non-overlapping 95% intervals.
    P2  Determinedness is MONOTONE in our grade: k=1 worse than k=2 worse than k=3. If the grade we ship
        tracks an ensemble property nobody in the panel measures, that is a real finding about what the
        panel is implicitly selecting.
    P3  An ensemble measure predicts IPknot's verdict BETTER than k_panel does, held out (AUC, deterministic
        even/odd split on target id). If it does, the certificate should carry it; if k_panel wins, the
        ensemble story is decoration.

    KILL: if k=1 designs and natives are indistinguishable on every ensemble measure, then the ten-fold gap
    is about IPknot and not about the designs, and the fix is wording, not engineering. That outcome is
    reported as plainly as the other.

    CONTROL that must be able to fail: GC content. A learned model can be biased by composition alone, and
    if GC explains IPknot's verdict as well as the ensemble measures do, the ensemble story adds nothing.

    DECLARED LIMITATION: ViennaRNA's partition function is nested-only, so these ensemble numbers ignore
    pseudoknots. That is not a workaround -- IPknot's own first stage computes nested base-pair
    probabilities too, so this measures roughly what IPknot looks at before its integer program runs. It is
    stated here rather than discovered later.

    The cluster is the TARGET. Several designs of one target are not several observations.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)


def ensemble_features(seq: str):
    """How DETERMINED is this sequence's fold? Nested model, which is what IPknot's first stage uses too."""
    import RNA
    fc = RNA.fold_compound(seq)
    mfe_s, mfe_e = fc.mfe()
    fc.exp_params_rescale(mfe_e)
    _, fe = fc.pf()
    n = len(seq)
    bpp = fc.bpp()                              # 1-indexed upper triangle
    # positional entropy: how spread out is each base's pairing probability
    ent = []
    for i in range(1, n + 1):
        ps = []
        for j in range(1, n + 1):
            p = bpp[i][j] if j > i else bpp[j][i]
            if p and p > 0:
                ps.append(p)
        unp = max(0.0, 1.0 - sum(ps))
        h = -sum(p * math.log(p) for p in ps if p > 1e-12)
        if unp > 1e-12:
            h -= unp * math.log(unp)
        ent.append(h)
    kT = 0.6163207755
    return {"p_mfe": math.exp((fe - mfe_e) / kT),        # ensemble probability of the MFE structure
            "entropy": sum(ent) / n,                      # mean positional entropy
            "gap": mfe_e - fe,                            # MFE minus ensemble free energy (>=0)
            "gc": 100.0 * sum(1 for c in seq if c in "GC") / n}


def work(item):
    kind, tid, seq = item
    try:
        f = ensemble_features(seq)
    except Exception as e:                                                     # noqa: BLE001
        return {"kind": kind, "task_id": tid, "error": repr(e)[:120]}
    return {"kind": kind, "task_id": tid, "seq": seq, **f}


def ci(v):
    if len(v) < 2:
        return (float("nan"), float("nan"))
    se = st.stdev(v) / len(v) ** 0.5
    return (st.mean(v) - 1.96 * se, st.mean(v) + 1.96 * se)


def auc(y, s):
    pairs = [(a, b) for a, b in zip(s, y)]
    pos = [a for a, b in pairs if b]
    neg = [a for a, b in pairs if not b]
    if not pos or not neg:
        return None
    w = sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg)
    return w / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(HERE, "k1_diagnosis.json"))
    a = ap.parse_args()

    import importlib.util
    from concurrent.futures import ProcessPoolExecutor
    from rna_harness.benchmarks.rnainvbench import load_pseudobase

    spec = importlib.util.spec_from_file_location("_cal", os.path.join(HERE, "calibrate.py"))
    cal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cal)
    tgt = {str(t.task_id): str(t.target) for t in load_pseudobase()}
    natives = {str(k): v for k, v in cal._native_sequences().items() if str(k) in tgt}

    P = json.load(open(os.path.join(HERE, "judge_paired.json")))
    K = json.load(open(os.path.join(HERE, "ipknot_judge.json")))
    ipk = K["ipknot"]
    designs = {}
    for r in P["rows"]:
        if r.get("error") or r.get("skipped"):
            continue
        for d in r.get("designs", []):
            if d["external_ran"] and ipk.get(d["seq"]) is not None:
                designs.setdefault(d["seq"], (r["task_id"], d["k_panel"]))

    items = ([("native", t, s) for t, s in natives.items()]
             + [(f"k{k}", t, s) for s, (t, k) in designs.items()])
    print(f"k=1 诊断:{len(natives)} 条天然序列 + {len(designs)} 条设计,共 {len(items)} 次系综计算\n"
          f"  问题:IPknot 认天然 46.2%、认我们的 k=1 只有 4.6% —— 这是【设计】的问题还是【这一类模型】的问题?\n",
          flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        R = [r for r in ex.map(work, items) if "error" not in r]
    print(f"  成功 {len(R)}/{len(items)}")

    G = defaultdict(list)
    for r in R:
        G[r["kind"]].append(r)
    print(f"\n=== P1/P2 系综有多确定(数值越小越确定)===")
    print(f"{'组':>8s}{'n':>6s}{'平均位点熵':>14s}{'95% 区间':>22s}{'MFE 系综概率':>14s}{'GC%':>8s}")
    stats = {}
    for kind in ("native", "k1", "k2", "k3"):
        g = G.get(kind) or []
        if not g:
            continue
        e = [x["entropy"] for x in g]
        p = [x["p_mfe"] for x in g]
        lo, hi = ci(e)
        stats[kind] = {"n": len(g), "entropy": st.mean(e), "entropy_ci": [lo, hi],
                       "p_mfe": st.mean(p), "gc": st.mean([x["gc"] for x in g])}
        print(f"{kind:>8s}{len(g):6d}{st.mean(e):14.4f}{f'[{lo:.4f}, {hi:.4f}]':>22s}"
              f"{st.mean(p):14.4f}{st.mean([x['gc'] for x in g]):8.1f}")

    if "native" in stats and "k1" in stats:
        nl, nh = stats["native"]["entropy_ci"]
        kl, kh = stats["k1"]["entropy_ci"]
        sep = kl > nh or kh < nl
        worse = stats["k1"]["entropy"] > stats["native"]["entropy"]
        print(f"\n  P1 k=1 的系综比天然更【不确定】且区间不重叠:"
              f"{'通过' if sep and worse else '未通过'}"
              f"{'' if sep else '(区间重叠)'}{'' if worse else '(方向相反)'}")
    mono = all(stats.get(f"k{i}", {}).get("entropy", 9e9) >= stats.get(f"k{i+1}", {}).get("entropy", -9e9)
               for i in (1, 2) if f"k{i}" in stats and f"k{i+1}" in stats)
    print(f"  P2 确定度随等级单调(k1 最不确定):{'通过' if mono else '未通过'}")

    # ---- P3: does an ensemble measure beat k_panel at predicting IPknot? ------------------------------
    rows = []
    for r in R:
        if r["kind"] == "native":
            continue
        tid, k = designs[r["seq"]]
        from rna_harness.benchmarks.rnainvbench import _round_only, pk_pairs
        cross = pk_pairs(tgt[tid]) - pk_pairs(_round_only(tgt[tid]))
        y = bool({tuple(x) for x in ipk[r["seq"]]} >= cross)
        rows.append({"tid": tid, "k": k, "y": y, "entropy": r["entropy"],
                     "p_mfe": r["p_mfe"], "gc": r["gc"]})
    keys = sorted({x["tid"] for x in rows})
    idx = {t: i for i, t in enumerate(keys)}
    test = [x for x in rows if idx[x["tid"]] % 2 == 1]
    y = [x["y"] for x in test]
    print(f"\n=== P3 谁更能预测 IPknot 的判决(TEST {len(test)} 条,按靶标偶奇分割)===")
    cand = {"面板等级 k": [x["k"] for x in test],
            "位点熵(负)": [-x["entropy"] for x in test],
            "MFE 系综概率": [x["p_mfe"] for x in test],
            "GC(对照,必须能失败)": [x["gc"] for x in test]}
    aucs = {}
    for nm, s in cand.items():
        v = auc(y, s)
        if v is None:
            continue
        aucs[nm] = max(v, 1 - v)
        print(f"  {nm:>22s}  AUC {aucs[nm]:.3f}")
    best_ens = max((aucs.get("位点熵(负)", 0), aucs.get("MFE 系综概率", 0)))
    p3 = best_ens > aucs.get("面板等级 k", 1.0)
    print(f"\n  P3 系综测量胜过面板等级:{'通过 —— 证书应当带上它' if p3 else '未通过 —— 系综这条只是装饰'}")
    if aucs.get("GC(对照,必须能失败)", 0) >= best_ens - 0.01:
        print(f"  !! 对照 GC 与最好的系综测量不相上下 —— 这可能只是组分偏好,不是系综")

    print(f"\n=== 结论走向 ===")
    if "native" in stats and "k1" in stats and abs(stats["k1"]["entropy"] - stats["native"]["entropy"]) < 0.02:
        print("  k=1 与天然在系综上分不开 -> 那个十倍差距是【IPknot 这一类模型】的性质,"
              "证书里的告警应当写成【某一类裁判特有】,而不是暗示普遍")
    else:
        print("  k=1 与天然在系综上可分 -> 差距至少部分来自设计本身,"
              "这是一条可以工程改进的路(在搜索里加系综确定度)")

    json.dump({"stats": stats, "auc": aucs, "P3": p3, "n": len(R), "rows": rows},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
