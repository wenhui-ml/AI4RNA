#!/usr/bin/env python3
"""M26v: recompute the new headline with a second, independently written estimator.

The claim that now carries the paper is a sign: in matched strata the thermodynamic model assigns
LOWER pairing probability to C-G targets than to A-U targets, while the measurement puts C-G
higher. It replaces a retention ratio that our own calibration showed to be uninformative, so it
had better not rest on one implementation of one function.

M26.matched accumulates per stratum with dicts. designaudit.core._weighted does the same
arithmetic with bincount over a hashed stratum id, written months apart and never run against the
other. This puts both on the same records and requires them to agree.

It also asks where the sign comes from: per depth, per crossing, per design-cell, and with the
GU arm included -- a sign that lives in one stratum is a different claim from one that is general.
"""
from __future__ import annotations

import collections, json, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, "/data/wenhui/openclaude_rna/rna_harness/tools")
import m26_model_scored as M26
from designaudit import core

LOG = []


def log(s=""):
    print(s, flush=True); LOG.append(s)


def main():
    log("=== M26v:用第二套独立实现复算新头条 ===\n")
    recs, _ = M26.build()
    z = np.load(os.path.join(HERE, "m26c_pneural_cache.npz"), allow_pickle=True)
    pn = {k: v for k, v in zip(z["keys"], z["mats"])}
    for r in recs:
        i, j = min(r["a"], r["b"]), max(r["a"], r["b"])
        r["nmut"] = float(pn[r["mut_seq"]][i, j])
    log(f"纳入 {len(recs)}")

    sb = lambda r: (r["design"], min(r["depth"], 3), r["crossed"])
    dr = [{"design": r["design"], "pt": r["pt"], "depth3": min(r["depth"], 3),
           "crossed": r["crossed"], "puzzle": r["puzzle"], "rf": r["rf"],
           "pmut": r["pmut"], "nmut": r["nmut"]} for r in recs]

    log("\n=== 两套实现对拍(同一批记录、同一套分层)===")
    log(f"  {'量':<12}{'M26.matched':>14}{'designaudit':>14}{'差':>12}{'权重':>9}")
    OUT = {}
    for key, nm in (("rf", "rescue(结果)"), ("pmut", "P_vienna"), ("nmut", "P_neural")):
        a1, w1, _ = M26.matched(recs, "CG", "AU", sb, key=key)
        c = core.Corpus.from_records(dr, design="design", treatment="pt", outcome=key,
                                     covariates=("depth3", "crossed"), cluster="puzzle")
        est = core.contrast(c, ("CG", "AU"), strata=("depth3", "crossed"), nperm=1000, nboot=300)
        log(f"  {nm:<12}{a1:>+14.5f}{est.effect:>+14.5f}{abs(a1-est.effect):>12.2e}{w1:>9.0f}")
        OUT[key] = {"m26": round(a1, 5), "designaudit": round(est.effect, 5),
                    "abs_diff": abs(a1 - est.effect), "p": est.p,
                    "ci": [round(x, 4) for x in est.ci], "weight": est.weight}
        log(f"     designaudit 置换 p {est.p:.4f}  cluster CI [{est.ci[0]:+.4f}, {est.ci[1]:+.4f}]")
    agree = all(v["abs_diff"] < 1e-9 for v in OUT.values())
    log(f"  **两套实现{'一致(<1e-9)' if agree else '**不一致 ⇒ 先查再判**'}**")
    OUT["implementations_agree"] = bool(agree)

    log("\n=== 符号来自哪里:逐 depth / crossed 拆开 ===")
    log(f"  {'层':<22}{'n':>6}{'rescue':>10}{'P_vienna':>11}{'P_neural':>11}")
    br = {}
    for d3 in (0, 1, 2, 3):
        for cr in (False, True):
            sub = [r for r in recs if min(r["depth"], 3) == d3 and r["crossed"] == cr]
            if len(sub) < 60:
                continue
            f = lambda k: M26.matched(sub, "CG", "AU", sb, key=k)[0]
            vs = [f("rf"), f("pmut"), f("nmut")]
            if any(v is None for v in vs):
                continue
            tag = f"depth={d3} crossed={cr}"
            br[tag] = {"n": len(sub), "rf": round(vs[0], 5), "pmut": round(vs[1], 5),
                       "nmut": round(vs[2], 5)}
            log(f"  {tag:<22}{len(sub):>6}{vs[0]:>+10.4f}{vs[1]:>+11.4f}{vs[2]:>+11.4f}")
    OUT["breakdown"] = br
    neg = sum(1 for v in br.values() if v["pmut"] < 0)
    log(f"  **P_vienna 为负的层: {neg}/{len(br)}**"
        f"{'  ⇒ 符号是普遍的' if neg >= 0.75 * len(br) else '  ⇒ 符号集中在少数层,必须照这么写'}")

    log("\n=== 每个设计单元一票(避免大单元主导)===")
    per = collections.defaultdict(list)
    for r in recs:
        per[r["design"]].append(r)
    cells = []
    for d, rs in per.items():
        cg = [x["pmut"] for x in rs if x["pt"] == "CG"]
        au = [x["pmut"] for x in rs if x["pt"] == "AU"]
        if len(cg) >= 3 and len(au) >= 3:
            cells.append(float(np.mean(cg) - np.mean(au)))
    cells = np.array(cells)
    log(f"  可用设计单元 {len(cells)}   均值 {cells.mean():+.5f}   中位 {np.median(cells):+.5f}")
    log(f"  **为负的单元 {int((cells<0).sum())}/{len(cells)} = {100*(cells<0).mean():.1f}%**")
    from scipy import stats as st
    try:
        sgn = st.binomtest(int((cells < 0).sum()), len(cells), 0.5).pvalue
    except AttributeError:
        sgn = st.binom_test(int((cells < 0).sum()), len(cells), 0.5)
    log(f"  符号检验 p = {sgn:.5f}")
    OUT["per_cell"] = {"n": len(cells), "mean": round(float(cells.mean()), 5),
                       "median": round(float(np.median(cells)), 5),
                       "frac_negative": round(float((cells < 0).mean()), 4),
                       "sign_test_p": float(sgn)}

    json.dump(OUT, open(os.path.join(HERE, "m26v_alpha_verify.json"), "w"),
              ensure_ascii=False, indent=1)
    open(os.path.join(HERE, "m26v_alpha_verify.log"), "w").write("\n".join(LOG) + "\n")
    log("\n写出 m26v_alpha_verify.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
