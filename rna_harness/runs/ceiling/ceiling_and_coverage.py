#!/usr/bin/env python3
"""What the CURRENT shipping rule actually delivers, against each outside judge's OWN ceiling.

WHY THIS EXISTS, and it starts with an error of mine. Weekly-report 4.15/4.16 compared candidate rules
against a row labelled "panel unanimity (current)". That label was wrong. The pseudoknot path ships on a
TWO-RUNG ladder: `certified` when all three predictors reproduce the WHOLE structure, and `knot_confirmed`
when AT LEAST ONE fully reproduces the crossing stems. Every one of the 4388 designs in judge_paired.json
already passes the second rung -- so the comparison was never "unanimity vs ShapeKnots", it was "the rule we
actually ship vs several strictly harder ones".

TWO CONSEQUENCES THAT THE OLD FRAMING HID:

    Recall was counted over DESIGNS. Counted over TARGETS, `ShapeKnots and k>=2` covers 131 of 250 -- it
    refuses to answer on 119 targets that get a certificate today. For a product whose north star is that a
    non-expert can cheaply GET an answer, that is the dominant cost and it was invisible.

    "25.2% outside agreement" cannot be read as "74.8% wrong". Every predictor in this field misses most of
    nature; the fourth-judge protocol (external_judge.py) fixed this by measuring the judge's OWN rate on
    NATIVE sequences in the same run. That reference was missing here, so this run measures it.

PRE-REGISTERED, before either ceiling was computed:

    P1  The current rule's agreement with an outside judge is BELOW that judge's native ceiling S_ext.
        If it is at or above S_ext, the rule is already at the instrument's limit and no stricter rule can
        be justified by outside agreement at all.
    P2  A stricter rule buys outside agreement ABOVE S_ext -- i.e. it is not merely recovering the ceiling
        but selecting sequences the outside judge likes better than it likes nature.
        If a rule exceeds S_ext it is worth having; if it merely approaches S_ext, the honest statement is
        that it reaches the instrument's limit, and the coverage it costs is the whole price.

    Reported either way, including the outcome where the ceilings are so low that no rule separates.
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from external_judge import clopper_pearson, pkplex_pairs                       # noqa: E402
from knotty_judge import knotty_pairs                                          # noqa: E402


def _cal():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_cal", os.path.join(HERE, "calibrate.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def native_job(job):
    tid, target, seq = job
    from rna_harness.benchmarks.rnainvbench import _round_only, pk_pairs
    cross = pk_pairs(target) - pk_pairs(_round_only(target))
    if not cross or len(seq) != len(target):
        return None
    pp, kk = pkplex_pairs(seq), knotty_pairs(seq)
    return (tid, None if pp is None else bool(pp >= cross), None if kk is None else bool(kk >= cross))


def main():
    from concurrent.futures import ProcessPoolExecutor
    from rna_harness.benchmarks.rnainvbench import _round_only, load_pseudobase, pk_pairs

    tgt = {str(t.task_id): str(t.target) for t in load_pseudobase()}
    natives = _cal()._native_sequences()
    jobs = [(str(k), tgt[str(k)], v) for k, v in natives.items() if str(k) in tgt]
    print(f"=== S_ext:两个外部裁判在【天然序列】上的自身天花板(n={len(jobs)} 条天然序列)===", flush=True)
    with ProcessPoolExecutor(max_workers=32) as ex:
        NAT = [r for r in ex.map(native_job, jobs) if r]

    S = {}
    for i, name in ((1, "RNAPKplex"), (2, "Knotty")):
        v = [r[i] for r in NAT if r[i] is not None]
        lo, hi = clopper_pearson(sum(v), len(v))
        S[name] = (sum(v) / len(v), lo, hi, len(v))
        print(f"  {name:10s} {sum(v):3d}/{len(v):3d} = {sum(v)/len(v):6.1%}  95% [{lo:.1%}, {hi:.1%}]")
    bo = [(a, b) for _t, a, b in NAT if a is not None and b is not None]
    v = [a and b for a, b in bo]
    lo, hi = clopper_pearson(sum(v), len(v))
    S["both"] = (sum(v) / len(v), lo, hi, len(v))
    print(f"  {'两家都认':8s} {sum(v):3d}/{len(v):3d} = {sum(v)/len(v):6.1%}  95% [{lo:.1%}, {hi:.1%}]"
          f"   <- 严标签的天花板")
    print("  ↑ 这不是正确率,是【这两个裁判连天然的真结构都只认出这么多】。"
          "任何设计集的同意率都必须对着它读。", flush=True)

    # ---- the delivered designs, with both outside verdicts ------------------------------------------
    P = json.load(open(os.path.join(HERE, "judge_paired.json")))
    K = json.load(open(os.path.join(HERE, "knotty_judge.json")))
    kn = {s: (None if v is None else {tuple(x) for x in v}) for s, v in K["knotty"].items()}
    by = defaultdict(list)
    for r in P["rows"]:
        if r.get("error") or r.get("skipped"):
            continue
        tg = tgt.get(r["task_id"])
        if not tg:
            continue
        cross = pk_pairs(tg) - pk_pairs(_round_only(tg))
        for d in r.get("designs", []):
            if not d["external_ran"] or kn.get(d["seq"]) is None:
                continue
            by[r["task_id"]].append({**d, "pk": bool(d["external"]),
                                     "kn": bool(kn[d["seq"]] >= cross)})
    allv = [d for v in by.values() for d in v]
    NT = len(by)
    sk = lambda d: "shapeknots" in d["confirmed_by"]                            # noqa: E731
    RULES = [("现行 k>=1(全部发证)", lambda d: True),
             ("k>=2", lambda d: d["k_panel"] >= 2),
             ("k==3", lambda d: d["k_panel"] == 3),
             ("ShapeKnots 确认", sk),
             ("ShapeKnots 且 k>=2", lambda d: sk(d) and d["k_panel"] >= 2)]

    print(f"\n=== 出厂规则:同意率 vs 覆盖(共 {len(allv)} 条设计 / {NT} 个靶标,【全部】通过现行规则)===")
    print(f"{'规则':>22s}{'发证':>6s}{'覆盖靶标':>9s}{'丢靶标':>7s}"
          f"{'PKplex':>9s}{'Knotty':>9s}{'两家都认':>10s}")
    out = {}
    for lab, f in RULES:
        hi_ = [d for d in allv if f(d)]
        cov = len({t for t, v in by.items() if any(f(d) for d in v)})
        rates = {k: st.mean([d[k2] if k2 != "both" else (d["pk"] and d["kn"]) for d in hi_])
                 for k, k2 in (("RNAPKplex", "pk"), ("Knotty", "kn"), ("both", "both"))}
        out[lab] = {"n": len(hi_), "coverage": cov, "lost": NT - cov, **rates}
        print(f"{lab:>22s}{len(hi_):6d}{cov:9d}{NT-cov:7d}"
              f"{rates['RNAPKplex']:9.1%}{rates['Knotty']:9.1%}{rates['both']:10.1%}")

    print(f"\n=== 对着天花板读(同意率 / S_ext,>100% 表示比天然还更受该裁判认可)===")
    print(f"{'规则':>22s}{'PKplex':>10s}{'Knotty':>10s}{'两家都认':>11s}   判定")
    for lab, _f in RULES:
        r = out[lab]
        ratios = {k: r[k] / S[k][0] if S[k][0] else float("nan") for k in ("RNAPKplex", "Knotty", "both")}
        above = sum(1 for k in ("RNAPKplex", "Knotty", "both") if r[k] > S[k][2])   # above the CI upper end
        verdict = ("超过全部三个天花板" if above == 3 else
                   f"超过 {above}/3 个天花板" if above else "全部低于天花板")
        out[lab]["ratios"] = ratios
        out[lab]["n_above_ceiling"] = above
        print(f"{lab:>22s}{ratios['RNAPKplex']:9.0%}{ratios['Knotty']:10.0%}{ratios['both']:11.0%}   {verdict}")

    cur = out["现行 k>=1(全部发证)"]
    p1 = all(cur[k] < S[k][0] for k in ("RNAPKplex", "Knotty", "both"))
    best = max((l for l, _ in RULES if l != "现行 k>=1(全部发证)"), key=lambda l: out[l]["n_above_ceiling"])
    p2 = out[best]["n_above_ceiling"] > 0
    print(f"\n=== 预注册判决 ===")
    print(f"  P1 现行规则的同意率{'低于' if p1 else '不低于'}裁判自身天花板 —— "
          f"{'成立,存在改进空间' if p1 else '不成立:现行规则已到仪器上限,任何更严的规则都不能用外部同意率来论证'}")
    print(f"  P2 更严的规则{'能' if p2 else '不能'}把同意率买到天花板【之上】 —— 最好的是「{best}」"
          f"({out[best]['n_above_ceiling']}/3);"
          f"{'值得,但要付覆盖的价' if p2 else '它只是逼近仪器上限,而覆盖的代价是全部的价钱'}")
    print(f"  代价:「{best}」丢掉 {out[best]['lost']}/{NT} 个靶标 = "
          f"{out[best]['lost']/NT:.1%} 的靶标今天有证书、换规则后没有")

    json.dump({"S_ext": {k: {"rate": v[0], "lo": v[1], "hi": v[2], "n": v[3]} for k, v in S.items()},
               "rules": out, "n_targets": NT, "n_designs": len(allv),
               "P1_current_below_ceiling": p1, "P2_best": best, "P2_passes": p2},
              open(os.path.join(HERE, "ceiling_and_coverage.json"), "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote ceiling_and_coverage.json")


if __name__ == "__main__":
    main()
