#!/usr/bin/env python3
"""Audit of everything claimed on 2026-08-07, run against the artifacts rather than against my memory of them.

Five things were asserted today that I could not justify from the runs themselves. Each is checked here and
reported whichever way it lands. Nothing is repaired silently; where a claim has to change, the change is
stated as a correction.

A  THE LADDER CONFLATION -- the one I think is a real error. `m2r_pairwise.py` counted, for each PAIR, how
   many oracles recovered THAT pair, and I reported the result as "the k=1/k=2/k=3 ladder is ordered in the
   tube" and wrote it into the certificate as `ladder_is_ordered_in_vitro`. But the certificate's k is a
   PER-DESIGN quantity: how many oracles recover ALL of the target's crossing pairs at tau=1.0. Per-pair
   agreement being ordered supports the MECHANISM behind the ladder; it is not the ladder. The design-level
   claim has to be tested at the design level or withdrawn.

B  DESIGN LENGTH IN THE M2R SET. `_INTERVENTION["scope"]` says "20 Round-3 targets, 52 nt designs". 52 came
   from ONE example row; the README calls Round 3 the 100-mer round and the row ids say `ok7b`, which is the
   240-mer lab. If the lengths are not 52 the shipped scope string is wrong.

C  THE THRESHOLD BEHIND 14.1%. The pre-registration justified `rescue_factor < 0.25` as "OpenKnot's own
   reactivity cut for 'paired'". That is a category error: OpenKnot's 0.25 is a cut on normalised SHAPE
   REACTIVITY; rescue_factor is a fraction of restored RMSD. Borrowing the number across quantities is not a
   justification, it is a coincidence of digits. The rate is re-reported across the whole threshold range so
   a reader can see how much of 14.1% is the threshold's doing.

D  POOL SIZE. `selector_vs_reality.py` pre-registered N=20 as "the order of magnitude of the candidate pool
   the solver actually produces per target inside one budget tier". The product's own stored
   `n_candidates_screened` has median 131 (range 71-200). The pre-registered primary was therefore set at
   the wrong operating point, and the product-relevant number is re-run here.

E  WILD-TYPE REPLICATES. `m2r_pairwise.load` takes the FIRST wild-type row per (puzzle, method) as the
   design. There are ~30 wild-type rows per design. If they do not all carry the same sequence, the design
   that got folded was chosen by file order.
"""
from __future__ import annotations

import collections
import csv
import json
import os
import random
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from m2r_pairwise import spearman, wilcoxon, mannwhitney                       # noqa: E402

M2R = "/data/wenhui/openknot_data/Data/OK7a_M2R_data.v4.5.1.csv"
CALIBRATED = ("kissing / chain", "higher-order", "crossing (non-H)")
ALPHA_A = 0.05


def load_m2r():
    csv.field_size_limit(10 ** 9)
    with open(M2R, newline="") as fh:
        r = csv.reader(fh)
        idx = {c: i for i, c in enumerate(next(r))
               if c in ("id", "SN_filter", "puzzle", "method", "sub_start", "sub_end",
                        "target_structure", "mutA", "mutB", "rescue_factor", "sequence")}
        return [{c: x[i] for c, i in idx.items()} for x in r]


def main():
    rows = load_m2r()
    run = json.load(open(os.path.join(HERE, "m2r_pairwise.json")))

    # ---------------------------------------------------------------- E first: it decides what A folds
    print("=" * 78)
    print("E  野生型的 ~30 个重复行,序列是不是同一条")
    print("=" * 78)
    wt_all = collections.defaultdict(set)
    for d in rows:
        if not d["mutA"]:
            ss, se = int(d["sub_start"]), int(d["sub_end"])
            wt_all[(d["puzzle"], d["method"])].add(d["sequence"][ss - 1:se])
    multi = {k: v for k, v in wt_all.items() if len(v) > 1}
    print(f"  设计数 {len(wt_all)};野生型序列不唯一的设计 {len(multi)}")
    print(f"  -> {'通过:取第一行是安全的' if not multi else '问题:折叠哪一条取决于文件顺序'}")

    print("\n" + "=" * 78)
    print("B  M2R 设计的实际长度(证书里写的是 52 nt)")
    print("=" * 78)
    lens = collections.Counter(int(d["sub_end"]) - int(d["sub_start"]) + 1
                               for d in rows if not d["mutA"])
    print(f"  长度分布 {dict(sorted(lens.items()))}")
    pz = collections.Counter(d["puzzle"][0] for d in rows if not d["mutA"])
    print(f"  靶标编号前缀 {dict(pz)}   (P = Round 3 的 100-mer 组;Q = Round 4 的 240-mer 组)")
    ok_b = set(lens) == {52}
    print(f"  -> {'证书里的 52 nt 正确' if ok_b else '证书里的 scope 字符串写错了,必须改'}")

    print("\n" + "=" * 78)
    print("C  14.1% 有多少是【阈值】做出来的")
    print("=" * 78)
    pairs = run["pairs"]
    k3 = [p["rf"] for p in pairs if p["crossed"] and p.get("n_oracles") == 3]
    k0 = [p["rf"] for p in pairs if p["crossed"] and p.get("n_oracles") == 0]
    print(f"  三家一致的交叉对 n={len(k3)};无人确认的 n={len(k0)}")
    print(f"  {'阈值':>8s} {'三家一致的假阳性':>18s} {'无人确认的':>12s} {'比值':>8s}")
    for t in (0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5):
        a = sum(1 for x in k3 if x < t) / len(k3)
        b = sum(1 for x in k0 if x < t) / len(k0)
        print(f"  {t:8.2f} {a:17.1%} {b:11.1%} {(b/a if a else float('nan')):8.2f}x")
    print("  -> 点估计随阈值从 6% 走到 51%。【阈值必须和数字一起报,单报 14.1% 是误导】")
    print("     阈值无关的说法:三家一致的配对,其 rescue 分布整体高于无人确认的 —— 见下检验")
    u, p_tf = mannwhitney(k3, k0)
    print(f"     Mann-Whitney(三家一致 > 无人确认)p = {p_tf:.3g}"
          f"   中位 {st.median(k3):+.3f} 对 {st.median(k0):+.3f}")

    print("\n" + "=" * 78)
    print("A  证书的 k 是【每条设计】的量,而我测的是【每个配对】的量 —— 补做设计层检验")
    print("=" * 78)
    # design-level k: how many oracles recover ALL crossing pairs of this design's target, tau = 1.0
    from rna_harness.cognitive.certify import _knot_recall
    wt = {}
    for d in rows:
        if not d["mutA"]:
            wt.setdefault((d["puzzle"], d["method"]), d)
    designs = {}
    for k, w in wt.items():
        ss, se = int(w["sub_start"]), int(w["sub_end"])
        designs[k] = {"seq": w["sequence"][ss - 1:se], "target": w["target_structure"][ss - 1:se],
                      "puzzle": k[0], "method": k[1]}
    rf_by_design = collections.defaultdict(list)
    for p in pairs:
        if p["crossed"]:
            rf_by_design[tuple(p["design"])].append(p["rf"])

    from concurrent.futures import ProcessPoolExecutor
    def job(item):
        k, seq, tgt = item
        kr = _knot_recall(seq, tgt)
        return k, len(kr["confirmed_by"]), kr["per_oracle"]
    items = [(k, v["seq"], v["target"]) for k, v in designs.items()]
    out = []
    with ProcessPoolExecutor(max_workers=16) as ex:
        for r in ex.map(_job_wrapper, items):
            out.append(r)
    kdes = {tuple(k): kk for k, kk, _ in out}

    recs = []
    for k, v in designs.items():
        rfs = rf_by_design.get(k)
        if not rfs:
            continue
        recs.append({"design": k, "puzzle": k[0], "k": kdes[k], "mean_rf": st.mean(rfs),
                     "median_rf": st.median(rfs), "n_pairs": len(rfs)})
    dist = collections.Counter(r["k"] for r in recs)
    print(f"  160 条设计的【证书等级】分布 {dict(sorted(dist.items()))}")
    for kk in sorted(dist):
        s = [r["mean_rf"] for r in recs if r["k"] == kk]
        print(f"    k={kk}  n={len(s):3d}   该设计交叉对的平均 rescue 中位 {st.median(s):+.3f}")
    # within-target, because targets differ in difficulty
    by_t = collections.defaultdict(list)
    for r in recs:
        by_t[r["puzzle"]].append(r)
    rhos = {}
    for t, v in by_t.items():
        if len(v) >= 4 and len({r["k"] for r in v}) >= 2:
            rr = spearman([r["k"] for r in v], [r["mean_rf"] for r in v])
            if rr is not None:
                rhos[t] = rr
    if rhos:
        vals = list(rhos.values())
        w, p, n = wilcoxon(vals, "greater")
        print(f"  靶标内 Spearman(证书 k, 该设计的平均 rescue):簇 {len(vals)},"
              f"中位 rho {st.median(vals):+.3f},正向 {sum(1 for x in vals if x>0)}/{len(vals)}")
        print(f"    Wilcoxon p = {p:.3g}   "
              f"{'设计层的阶梯主张成立' if p < ALPHA_A else '设计层的阶梯主张【不成立】—— 证书里的措辞必须改'}")
    else:
        p = None
        print("  无可用簇 —— 设计层无法检验")

    print("\n" + "=" * 78)
    print("D  用产品真实池子大小(中位 131)重跑选择实验")
    print("=" * 78)
    src = json.load(open(os.path.join(HERE, "openknot_grade_vs_shape.json")))
    grows = [r for r in src["rows"] if r["topology"] in CALIBRATED]
    by_p = collections.defaultdict(list)
    for r in grows:
        by_p[r["puzzle"]].append(r)
    ARMS = ("random", "grade", "rnet", "grade+rnet")
    def pick(arm, pool, rng):
        q = list(pool); rng.shuffle(q)
        if arm == "random":
            return q[0]
        key = {"grade": lambda r: r["grade"], "rnet": lambda r: r["rnet"],
               "grade+rnet": lambda r: (r["grade"], r["rnet"])}[arm]
        return max(q, key=key)
    for n_pool in (131, 71, 200):
        rng = random.Random(777 + n_pool)
        per = {a: {} for a in ARMS}
        for t, cand in sorted(by_p.items()):
            if len(cand) < n_pool:
                continue
            acc = {a: [] for a in ARMS}
            for _ in range(300):
                pool = rng.sample(cand, n_pool)
                for a in ARMS:
                    acc[a].append(pick(a, pool, rng)["score"])
            for a in ARMS:
                per[a][t] = st.mean(acc[a])
        ts = sorted(per["grade"])
        d1 = [per["grade"][t] - per["random"][t] for t in ts]
        d2 = [per["grade+rnet"][t] - per["rnet"][t] for t in ts]
        _, p1, _ = wilcoxon(d1, "greater")
        _, p2, _ = wilcoxon(d2, "greater")
        print(f"  N={n_pool:3d}  靶标 {len(ts):2d}   "
              f"等级>随机 {st.median(d1):+.3f} 分 p={p1:.3g}   "
              f"等级+RNet>纯RNet {st.median(d2):+.3f} 分 p={p2:.3g}")
    print("  -> 产品真实工作点上,现行选择器的增益是多少,就照这个报")

    json.dump({"E_wt_unique": len(multi) == 0, "B_lengths": dict(lens), "B_ok": ok_b,
               "C_threshold_sweep": {str(t): sum(1 for x in k3 if x < t)/len(k3)
                                     for t in (0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5)},
               "C_threshold_free_p": p_tf,
               "A_design_level": {"n_designs": len(recs), "k_dist": dict(dist),
                                  "median_rho": (st.median(list(rhos.values())) if rhos else None),
                                  "p": p, "n_clusters": len(rhos)}},
              open(os.path.join(HERE, "audit_0807.json"), "w"), ensure_ascii=False, indent=1)
    print("\n  wrote audit_0807.json")


def _job_wrapper(item):
    from rna_harness.cognitive.certify import _knot_recall
    k, seq, tgt = item
    kr = _knot_recall(seq, tgt)
    return k, len(kr["confirmed_by"]), kr["per_oracle"]


if __name__ == "__main__":
    main()
