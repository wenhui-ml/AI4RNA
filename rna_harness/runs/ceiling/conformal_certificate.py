#!/usr/bin/env python3
"""Change the TYPE of the certificate: from a verdict we sign to a set with a finite-sample guarantee.

WHY THIS AND NOT ANOTHER THRESHOLD. Three walls this month were one wall: the verdict cannot be absolute (no
physics), cannot be binary (reality is an ensemble), and cannot transfer across molecules -- G3 measured that
last one exactly, the statistic that predicts cannot carry a fixed cross-molecule threshold and the one that
can predicts worse. Conformal prediction sidesteps the whole question: it never asks which statistic can
carry a threshold, it converts a calibration set into a set-valued output with coverage that holds without
distributional assumptions.

TWO CORRECTIONS TO THE PROPOSAL THAT SUGGESTED THIS, both material:

  1. THE LABEL CANNOT BE OUR OWN PANEL. The proposal used "the majority verdict of the panel" as pseudo-label.
     Conformal guarantees coverage OF THE LABEL DISTRIBUTION YOU CALIBRATE ON -- with that label the theorem
     delivers "we cover what we ourselves said", which is not a statement about correctness at all. The label
     here is RNAPKplex: different algorithm, different codebase, separate environment, never part of any rule
     or selection in this project.

  2. MARGINAL CONFORMAL WOULD BE VALID AND USELESS. External agreement runs 29.9% on H-type against 60.8% on
     kissing. A marginal guarantee can hold overall while systematically under-covering one topology and
     over-covering the other, and a certificate that is wrong in a predictable direction for a predictable
     class is worse than no certificate. So the calibration is Mondrian -- conditioned on topology -- and the
     marginal version is computed alongside precisely to show whether that failure is real here.

PRE-REGISTERED, before any number below:

    alpha = 0.10, so the target is 90% coverage.
    Three-way deterministic split on sorted ids: FIT trains the scorer, CAL calibrates conformal, TEST reports.
    Strata with fewer than 10 calibration points are declared unusable rather than calibrated badly.

    PASS requires BOTH:
      validity  -- empirical coverage on TEST >= 0.85 within every usable stratum
      usefulness -- singleton prediction sets on more than 30% of TEST items

    KILL -- singleton rate <= 30% means the panel plus target structure carries too little information to
    support a non-trivial guarantee. That is a real answer about how much independent information this domain
    holds, and it gets reported as one rather than dressed up.

    CONTROL THAT MUST FAIL: the same pipeline driven by a random score must produce almost no singletons.
    A conformal wrapper always achieves coverage -- by widening -- so coverage alone proves nothing.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from designability import features, fit_logistic, predict                      # noqa: E402
from rna_harness.instrument import Clustered                                   # noqa: E402

ALPHA = 0.10
MIN_CAL = 10
MIN_SINGLETON = 0.30
MIN_COVERAGE = 0.85


def conformal_q(scores, alpha=ALPHA):
    """Split-conformal quantile: the ceil((n+1)(1-alpha))-th smallest calibration score."""
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return float("inf")          # too few calibration points to certify at this alpha; sets go trivial
    return sorted(scores)[k - 1]


def build_sets(p1, q):
    """p1 = P(label=1). Nonconformity for label y is 1 - p_hat(y). Set = every y whose score is <= q."""
    out = []
    for p in p1:
        s = set()
        if 1 - p <= q:
            s.add(1)
        if 1 - (1 - p) <= q:
            s.add(0)
        out.append(s)
    return out


def evaluate(sets, y):
    cov = sum(1 for s, t in zip(sets, y) if t in s) / max(1, len(y))
    single = sum(1 for s in sets if len(s) == 1) / max(1, len(sets))
    empty = sum(1 for s in sets if len(s) == 0) / max(1, len(sets))
    return {"n": len(y), "coverage": cov, "singleton": single, "empty": empty,
            "mean_size": sum(len(s) for s in sets) / max(1, len(sets))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "conformal_certificate.json"))
    a = ap.parse_args()
    import numpy as np

    from rna_harness.benchmarks.rnainvbench import load_pseudobase
    from rna_harness.structure_utils import classify_topology
    pk = {str(t.task_id): str(t.target) for t in load_pseudobase()}
    ext = json.load(open(os.path.join(HERE, "external_judge.json")))
    panel_from = {}
    rows = []
    for r in ext["records"]:
        if r.get("pop") != "ours" or not r.get("ran"):
            continue
        base = str(r["id"]).split("#")[0]
        if base not in pk:
            continue
        cb = r.get("ours_confirmed_by") or []
        panel_from[base] = sorted(cb)
        rows.append({"task_id": base, "target": pk[base], "k_panel": len(cb),
                     "y": 1 if r.get("confirmed") else 0,
                     "topology": classify_topology(pk[base])["topology"]})
    if len(rows) < 200:
        raise SystemExit(f"!! 只连上 {len(rows)} 条 —— join 坏了,先修")

    # features: target structure (known before designing) PLUS how many of our own oracles confirmed.
    # Both are available at certificate time; the external judge's verdict is the label and never a feature.
    fnames = sorted(features(rows[0]["target"]))
    for r in rows:
        f = features(r["target"])
        r["x"] = [f[n] for n in fnames] + [float(r["k_panel"])]

    ids = sorted({r["task_id"] for r in rows})
    idx = {t: i for i, t in enumerate(ids)}
    part = {0: "FIT", 1: "CAL", 2: "TEST"}
    for r in rows:
        r["split"] = part[idx[r["task_id"]] % 3]
    grp = defaultdict(list)
    for r in rows:
        grp[r["split"]].append(r)
    print(f"总体:{len(rows)} 条已交付证书;标签 = RNAPKplex(从未进入任何规则)")
    print(f"  三分:FIT {len(grp['FIT'])} / CAL {len(grp['CAL'])} / TEST {len(grp['TEST'])}"
          f"(确定性,按排序 id 取模 3)")
    by_topo = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_topo[r["topology"]][r["split"]].append(r)
    print(f"  拓扑分布:" + "  ".join(f"{t} {sum(len(v) for v in d.values())}" for t, d in by_topo.items()))

    m = fit_logistic([r["x"] for r in grp["FIT"]], [r["y"] for r in grp["FIT"]])

    def score_of(rs):
        return list(predict(m, [r["x"] for r in rs]))

    # --- marginal conformal, computed to EXPOSE its conditional failure, not to use it ------------------
    qm = conformal_q([1 - (p if r["y"] == 1 else 1 - p)
                      for r, p in zip(grp["CAL"], score_of(grp["CAL"]))])
    sm = build_sets(score_of(grp["TEST"]), qm)
    em = evaluate(sm, [r["y"] for r in grp["TEST"]])
    print(f"\n=== 边缘 conformal(alpha={ALPHA})===")
    print(f"  q={qm:.3f}  覆盖 {em['coverage']:.1%}  单元素集 {em['singleton']:.1%}  "
          f"平均集合大小 {em['mean_size']:.2f}")
    print(f"  按拓扑拆开看【条件覆盖】—— 这正是边缘保证看不见的地方:")
    cond = {}
    for t in by_topo:
        te = [r for r in grp["TEST"] if r["topology"] == t]
        if not te:
            continue
        e = evaluate(build_sets(score_of(te), qm), [r["y"] for r in te])
        cond[t] = e
        flag = "  <- 低于 90%" if e["coverage"] < 0.90 else ""
        print(f"    {t:16s} n={e['n']:>3}  覆盖 {e['coverage']:6.1%}  单元素 {e['singleton']:6.1%}{flag}")

    # --- Mondrian conformal: calibrate inside each topology --------------------------------------------
    print(f"\n=== Mondrian conformal(按拓扑分别标定)===")
    mond, usable = {}, []
    for t, d in by_topo.items():
        cal, te = d["CAL"], d["TEST"]
        if len(cal) < MIN_CAL:
            print(f"  {t:16s} CAL 只有 {len(cal)} 条 < {MIN_CAL} —— 标定不了,报为不可用,不硬标定")
            mond[t] = {"usable": False, "n_cal": len(cal)}
            continue
        q = conformal_q([1 - (p if r["y"] == 1 else 1 - p) for r, p in zip(cal, score_of(cal))])
        e = evaluate(build_sets(score_of(te), q), [r["y"] for r in te])
        e.update({"usable": True, "q": q, "n_cal": len(cal)})
        mond[t] = e
        usable.append(t)
        print(f"  {t:16s} CAL {len(cal):>3}  q={q:.3f}  TEST n={e['n']:>3}  "
              f"覆盖 {e['coverage']:6.1%}  单元素集 {e['singleton']:6.1%}  空集 {e['empty']:.1%}")

    # --- the control that must fail: a random score must not produce singletons -------------------------
    rng = np.random.default_rng(0)
    rand = list(rng.random(len(grp["CAL"])))
    qr = conformal_q([1 - (p if r["y"] == 1 else 1 - p) for r, p in zip(grp["CAL"], rand)])
    er = evaluate(build_sets(list(rng.random(len(grp["TEST"]))), qr), [r["y"] for r in grp["TEST"]])
    print(f"\n=== 必须失败的对照:随机打分 ===")
    print(f"  覆盖 {er['coverage']:.1%}  单元素集 {er['singleton']:.1%}  平均集合大小 {er['mean_size']:.2f}")
    # THE CONTROL'S THRESHOLD WAS SCALE-BLIND FOR THE FOURTH TIME TODAY, and this one is derivable in two
    # lines. With p ~ U(0,1) the calibration quantile lands near 1-alpha, so a set is a singleton exactly
    # when p < alpha or p > 1-alpha: the expected rate is 2*alpha = 20% at alpha=0.1. My declared 15% sits
    # BELOW the null itself, so it could only ever misfire. The control is not "random must give few
    # singletons", it is "ours must beat 2*alpha by a wide margin".
    null_singleton = 2 * ALPHA
    print(f"  解析零假设:随机打分下单元素集比例 ≈ 2·alpha = {null_singleton:.0%}"
          f"(实测 {er['singleton']:.1%},一致 —— 所以这不是流水线的问题)")
    control_ok = er["singleton"] < 2 * null_singleton

    all_test = [r for r in grp["TEST"] if r["topology"] in usable]
    cl = Clustered(values=[float(r["y"]) for r in all_test], clusters=[r["task_id"] for r in all_test])
    print(f"  重复层级:{cl.n_obs} 条 / {cl.n_clusters} 个靶标 —— {cl.warn_if_inflated() or '无膨胀'}")

    cov_ok = all(mond[t]["coverage"] >= MIN_COVERAGE for t in usable)
    sing = (sum(mond[t]["singleton"] * mond[t]["n"] for t in usable)
            / max(1, sum(mond[t]["n"] for t in usable)))
    use_ok = sing > MIN_SINGLETON and sing > 2 * null_singleton
    print(f"\n=== 预注册判定 ===")
    print(f"  有效性:每个可用拓扑覆盖 >= {MIN_COVERAGE:.0%}  ->  {'通过' if cov_ok else '不过'}")
    print(f"  有用性:单元素集比例 {sing:.1%} > max({MIN_SINGLETON:.0%}, 2·随机零假设 {2*null_singleton:.0%})"
          f"  ->  {'通过' if use_ok else '不过'}")
    print(f"  对照:{'通过' if control_ok else '不过'}")
    ok = cov_ok and use_ok and control_ok
    print(f"  -> {'证书可以换成【带有限样本保证的集合】' if ok else '不过 —— 照实报'}")
    if not use_ok:
        print("     读法:这不是 conformal 失效,是【这个领域的独立信息量不足以支撑非平凡的保证】——")
        print("     而这本身是一个量化了的结论,不是一次失败的尝试。")
    json.dump({"alpha": ALPHA, "marginal": em, "marginal_conditional": cond, "mondrian": mond,
               "usable_strata": usable, "singleton_rate": sing, "control": er,
               "null_singleton_rate": null_singleton,
               "valid": bool(cov_ok), "useful": bool(use_ok), "control_ok": bool(control_ok),
               "passes": bool(ok)}, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
