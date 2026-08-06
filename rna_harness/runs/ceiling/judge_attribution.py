#!/usr/bin/env python3
"""How much of a delivery verdict is EVIDENCE, and how much is WHICH JUDGE WE ASKED?

THE BIGGER CIRCLE, and why it is forced by the evidence rather than invented. Three separate walls this month
turned out to be one wall:

    the verdict cannot be ABSOLUTE          -- nothing here is grounded in physics, only in other models
    the verdict cannot be BINARY            -- reality is an ensemble and exactness is what stalled everything
    the verdict cannot TRANSFER ACROSS MOLECULES -- G3: the quantity that predicts cannot carry a fixed
                                               cross-molecule threshold, and the one that can predicts worse

All three say the same thing: we keep trying to produce `certify(sequence) -> verdict`, a total, absolute,
binary, transferable function, and the domain does not support one. Every attempt to route around it changed
the CONTENTS -- swap the judge, swap the criterion, swap the domain, reach for SHAPE data -- and kept the
TYPE. A type error cannot be fixed by changing values, which is the exact content of "we cannot get around it".

So change the type of the deliverable:

    certify(sequence, question) -> (verdict, how much of this verdict is judge-choice)

That claim needs no physics, is not bounded by the 70 nt ceiling, and turns the model-relativity that has been
a caveat in every report into the measured object. But it is only worth anything if it can be FALSE.

PRE-REGISTERED, written before any number below was computed:

    CLAIM: the share of a delivery verdict attributable to judge-choice is measurable and STABLE.

    Substrate: kissing_deliver.json -- 251 DESIGNED sequences, the deployment population, each carrying the
    set of oracles that confirmed it. Not the natives, because the entry-check lesson from the third domain
    was that the population must be the one the criterion signs off on.

    judge_share = 1 - (unanimously confirmed) / (confirmed by at least one)

    KILL 1 -- INSTABILITY. Computed on the project's existing deterministic split (sorted ids, even/odd), the
      two halves must agree within 0.15. Beyond that the quantity is not a property of the domain and the
      whole frame is empty.
    KILL 2 -- A PANEL THAT IS REALLY ONE ORACLE. If the dominant oracle alone delivers essentially what the
      union delivers (within 5 points), there is nothing to decompose and the frame is empty for a second,
      independent reason.

    Reported whatever it says. A frame that survives only by being untestable is the failure mode here, and
    the honest risk is that this is a negative result about model-relativity wearing a positive result's
    clothes -- these two kill conditions are the only thing separating the two.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)

from rna_harness.instrument import Clustered, gate_margin                      # noqa: E402

ORACLES = ("pkiss", "probknot", "shapeknots")
MAX_HALF_GAP = 0.15          # KILL 1
MAX_DOMINANT_GAP = 0.05      # KILL 2


def load(path):
    rows = json.load(open(path))["rows"]
    out = []
    for r in rows:
        cb = r.get("confirmed_by") or []
        out.append({"task_id": r["task_id"], "confirmed_by": frozenset(cb),
                    "n_cross": r.get("n_cross"), "delivered": r.get("delivered")})
    return out


def share(rows):
    """judge_share plus everything needed to see where it comes from."""
    any_ = [r for r in rows if r["confirmed_by"]]
    unan = [r for r in any_ if len(r["confirmed_by"]) == len(ORACLES)]
    single = {o: sum(1 for r in rows if o in r["confirmed_by"]) for o in ORACLES}
    pairs = {"+".join(p): sum(1 for r in rows if set(p) <= r["confirmed_by"])
             for p in combinations(ORACLES, 2)}
    dom = max(single, key=single.get)
    return {"n": len(rows), "n_any": len(any_), "n_unanimous": len(unan),
            "judge_share": (1 - len(unan) / len(any_)) if any_ else None,
            "per_oracle": single, "per_pair": pairs, "dominant": dom,
            "dominant_vs_union": (len(any_) - single[dom]) / max(1, len(rows)),
            "patterns": Counter("+".join(sorted(r["confirmed_by"])) or "(none)"
                                for r in rows).most_common()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "kissing_deliver.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "judge_attribution.json"))
    a = ap.parse_args()
    rows = load(a.src)
    print(f"部署总体:{len(rows)} 条【已设计】序列(不是天然序列 —— 判据要签字的是这一批)\n")

    full = share(rows)
    print("=== 判决的构成 ===")
    for pat, k in full["patterns"]:
        print(f"  {pat:34s} {k:>4}")
    print(f"\n  至少一家确认 {full['n_any']}/{full['n']}   三家一致 {full['n_unanimous']}")
    print(f"  judge_share = 1 - {full['n_unanimous']}/{full['n_any']} = {full['judge_share']:.1%}")
    print(f"    读法:交付出去的证书里,有这么大比例【换一家裁判就可能不成立】")

    print(f"\n=== 单家 vs 组合的交付数(同一批序列)===")
    for o, k in sorted(full["per_oracle"].items(), key=lambda x: -x[1]):
        print(f"  {o:12s} {k:>4}")
    for p, k in sorted(full["per_pair"].items(), key=lambda x: -x[1]):
        print(f"  {p:24s} {k:>4}")
    print(f"  三家一致 {full['n_unanimous']:>4}   至少一家 {full['n_any']:>4}")
    lo = min(full["per_oracle"].values()); hi = full["n_any"]
    print(f"  -> 同一批序列,交付数在 {lo}..{hi} 之间,取决于你信谁({hi/max(1,lo):.1f}×)")

    # --- KILL 1: stability on the project's existing deterministic split ---------------------------------
    ids = sorted({r["task_id"] for r in rows})
    idx = {t: i for i, t in enumerate(ids)}
    halves = {"FIT(偶)": [r for r in rows if idx[r["task_id"]] % 2 == 0],
              "TEST(奇)": [r for r in rows if idx[r["task_id"]] % 2 == 1]}
    print(f"\n=== KILL 1:两个半集上稳不稳(确定性分割,与本项目其它地方同一口径)===")
    hs = {}
    for k, v in halves.items():
        s = share(v)
        hs[k] = s
        print(f"  {k:10s} n={s['n']:>3}  至少一家 {s['n_any']:>3}  一致 {s['n_unanimous']:>2}  "
              f"judge_share {s['judge_share']:.1%}")
    gap = abs(hs["FIT(偶)"]["judge_share"] - hs["TEST(奇)"]["judge_share"])
    k1 = gap <= MAX_HALF_GAP
    print(f"  两半差 {gap:.1%}   门槛 {MAX_HALF_GAP:.0%}  -> {'稳定' if k1 else '不稳定 —— 框架作废'}")

    # the split is one draw; the interval says whether the halves could differ this much by chance alone
    per_item = [1.0 if (r["confirmed_by"] and len(r["confirmed_by"]) < len(ORACLES)) else 0.0
                for r in rows if r["confirmed_by"]]
    gm = gate_margin(per_item, 0.5)
    print(f"  自助区间 [{gm['ci95'][0]:.1%}, {gm['ci95'][1]:.1%}]  (逐条,{len(per_item)} 条)")
    cl = Clustered(values=per_item, clusters=[r["task_id"] for r in rows if r["confirmed_by"]])
    print(f"  重复层级:{cl.n_obs} 条来自 {cl.n_clusters} 个靶标 —— {cl.warn_if_inflated() or '一条一靶标,无膨胀'}")

    # --- the refinement this project's OWN earlier measurement demands ------------------------------------
    # Pairwise kappa said ProbKnot~ShapeKnots = 0.74 and the panel is TWO effective sources, not three.
    # "Unanimous among 3" is therefore a stricter bar than "unanimous among the sources that are actually
    # independent". Both are reported; the pre-registered one is primary. Reporting only the second after
    # seeing the first would be switching the statistic to suit the result, which is the same act as
    # retuning a threshold after seeing it.
    def eff2(r):
        out = set()
        if "pkiss" in r["confirmed_by"]:
            out.add("A")
        if r["confirmed_by"] & {"probknot", "shapeknots"}:
            out.add("B")
        return out
    any2 = [r for r in rows if eff2(r)]
    un2 = [r for r in any2 if len(eff2(r)) == 2]
    js2 = 1 - len(un2) / len(any2) if any2 else None
    print(f"\n=== 按【两个有效独立源】重算(kappa 0.74 说 ProbKnot 与 ShapeKnots 高度相关)===")
    print(f"  至少一源 {len(any2)}   两源一致 {len(un2)}   judge_share = {js2:.1%}")
    print(f"  预注册的三家口径 {full['judge_share']:.1%};两者都报,主口径是预注册那个")

    # --- KILL 2: is the panel really one oracle? ---------------------------------------------------------
    print(f"\n=== KILL 2:这个面板是不是其实只有一家 ===")
    d = full["dominant"]
    print(f"  最强的一家 {d}:单独交付 {full['per_oracle'][d]},并集交付 {full['n_any']}")
    print(f"  差 {full['dominant_vs_union']:.1%}   门槛 {MAX_DOMINANT_GAP:.0%}")
    k2 = full["dominant_vs_union"] > MAX_DOMINANT_GAP
    print(f"  -> {'不是一家 —— 有东西可分解' if k2 else '等价于一家 —— 没有可分解的东西,框架作废'}")

    verdict = bool(k1 and k2)
    print(f"\n=== 预注册判定 ===")
    print(f"  KILL 1 稳定性  {'通过' if k1 else '不过'}")
    print(f"  KILL 2 可分解  {'通过' if k2 else '不过'}")
    print(f"  -> {'这个量可测且稳定,更大的圈站得住第一步' if verdict else '框架作废,照实报,不找第三种切法'}")
    json.dump({"population": "designed sequences (deployment)", "full": full,
               "effective_two_sources": {"n_any": len(any2), "n_unanimous": len(un2), "judge_share": js2},
               "halves": {k: v for k, v in hs.items()}, "half_gap": gap,
               "bootstrap_ci": gm["ci95"], "kill1_stable": bool(k1), "kill2_decomposable": bool(k2),
               "passes": verdict}, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
