#!/usr/bin/env python3
"""Adversarial review of m2r_pairwise.py: three objections I could not answer from the run itself.

OBJECTION 1 -- A FILTER I APPLIED EVERYWHERE ELSE AND FORGOT HERE. `openknot_preflight.py` and
`openknot_grade_vs_shape.py` both require SN_filter == 1. `m2r_pairwise.py` does not, so 722 of the 14,416
M2R rows with signal/noise <= 1 or reads <= 100 were allowed in. A rescue_factor computed from a noisy
profile is a noisy number. Not silently repaired -- the pre-registered primary stands as run, and this is
reported as a robustness check. If the conclusion depends on the filter, the conclusion is fragile and must
be quoted with the filter.

OBJECTION 2 -- THE STABILITY CONFOUND, and it is the one that could hollow out the whole result. Our oracles
confirm thermodynamically strong pairs. Strong pairs may also rescue better for reasons having nothing to do
with whether the pair belongs to the intended structure: a GC pair perturbs SHAPE more when broken and
restores more when repaired. If that is all H2 is measuring, the panel is reading BASE COMPOSITION, not
STRUCTURE, and "the k ladder carries real information" is an illusion.

    Test: within-design Spearman of n_oracles with rescue_factor, PARTIAL on pair type ranked by stability
    (GU < AU < GC). If the partial correlation collapses, say so.

OBJECTION 3 -- H3's INTERVAL ASSUMED INDEPENDENCE. 453 crossing pairs come from 60 designs; pairs inside one
molecule share its fold, its normalisation and its batch. Clopper-Pearson on 453 is too narrow. Replaced by
a cluster bootstrap over designs.

Nothing here changes a pre-registered threshold. Each objection is reported whichever way it lands.
"""
from __future__ import annotations

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

DATA = "/data/wenhui/openknot_data/Data/OK7a_M2R_data.v4.5.1.csv"
RUN = os.path.join(HERE, "m2r_pairwise.json")
ALPHA = 0.05 / 3
STABILITY = {("G", "C"): 3, ("C", "G"): 3, ("A", "U"): 2, ("U", "A"): 2, ("G", "U"): 1, ("U", "G"): 1}

from m2r_pairwise import spearman, wilcoxon, mannwhitney, _rankdata, _pearson   # noqa: E402


def partial_spearman(x, y, z):
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


def main():
    run = json.load(open(RUN))
    pairs = run["pairs"]
    by_id = {p["id"]: p for p in pairs}

    csv.field_size_limit(10 ** 9)
    with open(DATA, newline="") as fh:
        r = csv.reader(fh)
        hdr = next(r)
        idx = {c: i for i, c in enumerate(hdr)
               if c in ("id", "SN_filter", "puzzle", "method", "sub_start", "sub_end", "sequence", "mutA")}
        rows = [{c: x[i] for c, i in idx.items()} for x in r]
    sn = {d["id"]: d["SN_filter"] for d in rows}
    wt = {}
    for d in rows:
        if not d["mutA"]:
            wt.setdefault((d["puzzle"], d["method"]), d)
    seqs = {k: w["sequence"][int(w["sub_start"]) - 1:int(w["sub_end"])] for k, w in wt.items()}

    for p in pairs:
        p["sn"] = sn.get(p["id"], "1")
        s = seqs.get(tuple(p["design"]))
        p["stab"] = STABILITY.get((s[p["i"]], s[p["j"]]), 0) if s else 0
    cross = [p for p in pairs if p["crossed"] and p.get("n_oracles") is not None]

    print("=" * 78)
    print("异议 1:我在别处都加了、这里漏掉的 SN_filter")
    print("=" * 78)
    bad = [p for p in pairs if p["sn"] != "1"]
    print(f"  信噪比未过滤的配对行 {len(bad)}/{len(pairs)} = {len(bad)/len(pairs):.1%}")
    for label, sub in (("原样(预注册主分析)", cross),
                       ("仅 SN_filter=1", [p for p in cross if p["sn"] == "1"])):
        print(f"\n  --- {label}:交叉对 n={len(sub)} ---")
        for k in range(4):
            s = [p["rf"] for p in sub if p["n_oracles"] == k]
            if s:
                print(f"      {k} 家确认 n={len(s):4d}  中位 {st.median(s):+.3f}"
                      f"  rf<0.25 占 {sum(x < 0.25 for x in s)/len(s):5.1%}")
        g = {}
        for p in sub:
            g.setdefault(tuple(p["design"]), []).append(p)
        rho = {k: spearman([p["n_oracles"] for p in v], [p["rf"] for p in v])
               for k, v in g.items() if len(v) >= 4 and len({p["n_oracles"] for p in v}) >= 2}
        rho = {k: v for k, v in rho.items() if v is not None}
        w, pv, n = wilcoxon(list(rho.values()), "greater")
        print(f"      H2 重算:簇 {len(rho)},中位 rho {st.median(list(rho.values())):+.3f},"
              f"p = {pv:.3g}  {'通过' if pv < ALPHA else '未通过'}")

    print("\n" + "=" * 78)
    print("异议 2:面板是不是只在读【碱基对强度】,而不是读结构")
    print("=" * 78)
    for k in range(4):
        s = [p["stab"] for p in cross if p["n_oracles"] == k]
        if s:
            comp = {3: "GC", 2: "AU", 1: "GU", 0: "非配对"}
            c = {comp[x]: sum(1 for y in s if y == x) / len(s) for x in set(s)}
            print(f"  {k} 家确认  平均强度 {st.mean(s):.2f}   " +
                  "  ".join(f"{a} {b:.0%}" for a, b in sorted(c.items())))
    stab_rf = spearman([p["stab"] for p in cross], [p["rf"] for p in cross])
    print(f"\n  强度本身与 rescue 的相关(混合所有设计):rho = {stab_rf:+.3f}")

    g = {}
    for p in cross:
        g.setdefault(tuple(p["design"]), []).append(p)
    raw, par = {}, {}
    for k, v in g.items():
        if len(v) < 4 or len({p["n_oracles"] for p in v}) < 2:
            continue
        a = spearman([p["n_oracles"] for p in v], [p["rf"] for p in v])
        b = partial_spearman([p["n_oracles"] for p in v], [p["rf"] for p in v], [p["stab"] for p in v])
        if a is not None and b is not None:
            raw[k], par[k] = a, b
    wa, pa, _ = wilcoxon(list(raw.values()), "greater")
    wb, pb, _ = wilcoxon(list(par.values()), "greater")
    print(f"  同一批 {len(raw)} 条设计上:")
    print(f"    未控制强度  中位 rho {st.median(list(raw.values())):+.3f}   p = {pa:.3g}")
    print(f"    控制强度后  中位 rho {st.median(list(par.values())):+.3f}   p = {pb:.3g}"
          f"   {'通过 —— 不是在读碱基组成' if pb < ALPHA else '未通过 —— H2 可能只是强度的影子'}")

    print("\n" + "=" * 78)
    print("异议 3:H3 的区间假设了独立,而配对在同一个分子里")
    print("=" * 78)
    k3 = {}
    for p in cross:
        if p["n_oracles"] == 3:
            k3.setdefault(tuple(p["design"]), []).append(p["rf"])
    flat = [x for v in k3.values() for x in v]
    pt = sum(1 for x in flat if x < 0.25) / len(flat)
    random.seed(0)
    ks = list(k3.values())
    B = []
    for _ in range(8000):
        s = [random.choice(ks) for _ in ks]
        f = [x for v in s for x in v]
        B.append(sum(1 for x in f if x < 0.25) / len(f))
    B.sort()
    print(f"  三家一致的交叉对 n={len(flat)},来自 {len(ks)} 条设计")
    print(f"  假阳性率 {pt:.1%}   簇自助法 95% [{B[200]:.1%}, {B[7800]:.1%}]")
    print(f"  (原文用的 Clopper-Pearson 是 {run['H3']['ci'][0]:.1%}-{run['H3']['ci'][1]:.1%},"
          f"它把 {len(flat)} 个配对当成了独立观测)")
    print(f"  -> 产品引用簇自助法的区间。点估计不变。")

    json.dump({"objection1": {"n_unfiltered": len(bad), "share": len(bad)/len(pairs)},
               "objection2": {"stability_rho": stab_rf, "raw_median": st.median(list(raw.values())),
                              "partial_median": st.median(list(par.values())), "p_raw": pa, "p_partial": pb,
                              "n_designs": len(raw)},
               "objection3": {"point": pt, "ci_cluster": [B[200], B[7800]],
                              "ci_naive": run["H3"]["ci"], "n_pairs": len(flat), "n_designs": len(ks)}},
              open(os.path.join(HERE, "m2r_review.json"), "w"), ensure_ascii=False, indent=1)
    print("\n  wrote m2r_review.json")


if __name__ == "__main__":
    main()
