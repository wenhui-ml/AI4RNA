#!/usr/bin/env python3
"""M21b: replace M21's retention figure with the direct quantity, for the same reason as M26.

m26r3 calibrated the retention estimator against a known mediated fraction and it does not track
truth at this corpus's signal-to-noise. M21 reported a retention figure computed with the same
machinery on the same records, so it falls with the others. This computes what replaces it: the
matched G-C minus A-U contrast IN the mediator itself, with a puzzle-clustered interval, alongside
the same contrast in the outcome.

That is a difference of two matched contrasts rather than a ratio of them, and it says directly
what the retention figure was being read as saying: whether the quantity the designers were
optimising differs between the two pair types at all, in matched contexts.
"""
from __future__ import annotations
import collections, json, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, "/data/wenhui/openclaude_rna/rna_harness/tools")
import m26_model_scored as M26
import m21_thermo_mediator as M21
from designaudit import core

LOG = []


def log(s=""):
    print(s, flush=True); LOG.append(s)


def main():
    log("=== M21b:把 M21 的留存数换成直接量 ===")
    log("  理由与 M26 相同:同一个估计量,已被 m26r3 的标定推翻。\n")
    # M21's own builder, not a reimplementation -- it self-checks against M16 on every shared field
    recs = M21.build_with_dg()
    log(f"M21 记录 {len(recs)}")
    # P_model from M26's builder, joined on the design+pair key so nothing is recomputed here
    m26recs = M26.build()[0]
    pm = {}
    for r in m26recs:
        pm.setdefault((r["design"], r["pt"], round(r["rf"], 6)), r["pmut"])
    n_join = 0
    for r in recs:
        k = (r["design"], r["pt"], round(r["rf"], 6))
        r["pmut"] = pm.get(k)
        n_join += r["pmut"] is not None
    log(f"  与 M26 的 P_model 连上 {n_join}/{len(recs)}")

    dr = [{"design": r["design"], "pt": r["pt"], "depth3": min(r["depth"], 3),
           "crossed": r["crossed"], "puzzle": r["puzzle"], "rf": r["rf"],
           "dg": r["dG"], "pmut": (r["pmut"] if r["pmut"] is not None else float("nan"))}
          for r in recs]
    dr = [d for d in dr if d["pmut"] == d["pmut"]]
    recs = [r for r in recs if r["pmut"] is not None]
    log(f"  两侧都有的记录 {len(dr)}")
    OUT = {"n": len(recs)}
    log(f"\n  {'配平后 G-C 减 A-U 的对比':<34}{'估计':>10}{'95% CI':>24}{'p':>9}")
    for key, nm in (("rf", "结果 rescue"), ("dg", "中介物 ΔG_loc"), ("pmut", "中介物 P_model")):
        c = core.Corpus.from_records(dr, design="design", treatment="pt", outcome=key,
                                     covariates=("depth3", "crossed"), cluster="puzzle")
        e = core.contrast(c, ("CG", "AU"), strata=("depth3", "crossed"), nperm=2000, nboot=400)
        log(f"  {nm:<34}{e.effect:>+10.4f}   [{e.ci[0]:+.4f}, {e.ci[1]:+.4f}]{e.p:>9.4f}")
        OUT[key] = {"effect": round(e.effect, 5), "ci": [round(x, 4) for x in e.ci],
                    "p": round(e.p, 5), "weight": e.weight, "strata": e.strata}
    sb = lambda r: (r["design"], min(r["depth"], 3), r["crossed"])
    for key, nm in (("dG", "ΔG_loc"), ("pmut", "P_model")):
        s, n = M26.slope_in(recs, sb, key)
        jk = "dg" if key == "dG" else key
        OUT[jk]["within_slope"] = round(s, 5)
        bound = abs(OUT[jk]["effect"] * s / OUT["rf"]["effect"])
        OUT[jk]["structural_bound"] = round(bound, 4)
        log(f"  {nm} 的层内斜率 {s:+.5f}  ⇒ 乘积路线上限 {100*bound:.1f}%")
    json.dump(OUT, open(os.path.join(HERE, "m21b_direct_contrast.json"), "w"),
              ensure_ascii=False, indent=1)
    open(os.path.join(HERE, "m21b_direct_contrast.log"), "w").write("\n".join(LOG) + "\n")
    log("\n写出 m21b_direct_contrast.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
