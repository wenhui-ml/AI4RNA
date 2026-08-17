#!/usr/bin/env python3
"""Score the holdout: does the structure term survive targets that had no part in choosing it?

RUNS IN THE RibonanzaNet venv.

    H1  CONFIRMATION. `mcc` against `base` on the 19 untouched targets, F1=0 fraction, paired within target,
        Wilcoxon one-sided less. PASS: median reduction > 0 and p < 0.025.
    H2  THE MECHANISM'S QUANTITATIVE PREDICTION, separately failable. These targets sit between the two
        groups already measured, so the reduction should land between the control group's and the
        low-crossing group's -- both read from objective_arms_score.json rather than typed in here.
    H3  `mcc_only`, the single-term arm that parsed correctly before the DesiRNA patch, so the conclusion
        does not rest on that patch.

COST: $0. Local CPU.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics as st
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pool_rerank_score import _score_chunk                                      # noqa: E402

ALPHA = 0.025


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default=os.path.join(HERE, "objective_arms_holdout_pools.json"))
    ap.add_argument("--ref", default=os.path.join(HERE, "objective_arms_score.json"))
    ap.add_argument("--workers", type=int, default=56)
    ap.add_argument("--out", default=os.path.join(HERE, "objective_arms_holdout_score.json"))
    a = ap.parse_args()

    from m2r_pairwise import wilcoxon

    P = json.load(open(a.pools))
    T, arms, nc = P["targets"], P["arms"], P["n_cross"]
    rows = [r for r in P["rows"] if "error" not in r and r["pool"]]
    jobs = [((r["arm"], r["puzzle"], i), s, T[r["puzzle"]]["target"])
            for r in rows for i, s in enumerate(r["pool"])]
    print(f"{len(rows)} 个(臂,靶标)组合,{len(jobs)} 条候选,{a.workers} 进程\n", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    CH = max(1, len(jobs) // (a.workers * 8))
    chunks = [jobs[i:i + CH] for i in range(0, len(jobs), CH)]
    sc, t0, done = {}, time.time(), 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for res in ex.map(_score_chunk, chunks):
            for k, v in res:
                sc[k] = v
            done += 1
            if done % 40 == 0 or done == len(chunks):
                print(f"    {done}/{len(chunks)}  {(time.time()-t0)/60:.1f} min", flush=True)

    per = collections.defaultdict(dict)
    for r in rows:
        v = [sc[(r["arm"], r["puzzle"], i)] for i in range(len(r["pool"]))
             if (r["arm"], r["puzzle"], i) in sc]
        if v:
            per[r["puzzle"]][r["arm"]] = {"zero_frac": sum(1 for x in v if x == 0.0) / len(v),
                                          "median": st.median(v), "n": len(v)}

    ps = sorted(p for p in per if "base" in per[p])
    print(f"=== 留出集 {len(ps)} 个靶标(交叉数 {min(nc[p] for p in ps)}-{max(nc[p] for p in ps)})===")
    print(f"  {'臂':11s} {'F1=0 占比':>10s} {'对 base':>10s} {'更低':>8s} {'p':>10s} {'池中位':>8s}")
    out = {"n_targets": len(ps), "arms": {}}
    for arm in arms:
        q = [p for p in ps if arm in per[p]]
        if not q:
            continue
        z = st.median([per[p][arm]["zero_frac"] for p in q])
        m = st.median([per[p][arm]["median"] for p in q])
        if arm == "base":
            print(f"  {arm:11s} {z:9.1%} {'—':>10s} {'—':>8s} {'—':>10s} {m:8.3f}")
            out["arms"][arm] = {"zero_frac": z, "median": m}
            continue
        d = [per[p][arm]["zero_frac"] - per[p]["base"]["zero_frac"] for p in q]
        w, pv, _ = wilcoxon(d, "less")
        print(f"  {arm:11s} {z:9.1%} {st.median(d):+10.1%} {sum(1 for x in d if x<0):5d}/{len(d):<3d} "
              f"{'n/a' if pv is None else f'{pv:.3g}':>10s} {m:8.3f}")
        out["arms"][arm] = {"zero_frac": z, "median": m, "delta_vs_base": st.median(d), "p": pv,
                            "lower": sum(1 for x in d if x < 0), "n_targets": len(d)}

    mcc = out["arms"].get("mcc", {})
    h1 = bool(mcc.get("p") is not None and mcc["p"] < ALPHA and mcc["delta_vs_base"] < 0)
    print(f"\n=== H1 确认(它有没有在没参与选择的靶标上重现)===")
    print(f"  mcc 降幅 {mcc.get('delta_vs_base', float('nan')):+.1%},p = {mcc.get('p')}")
    print(f"  {'通过 —— 不是四选一挑出来的运气' if h1 else 'KILL —— 原结果没能迁移到未见靶标'}")

    R = json.load(open(a.ref))["by_group"]
    lo_d, hi_d = R["low"]["mcc"]["delta_vs_base"], R["high"]["mcc"]["delta_vs_base"]
    got = mcc.get("delta_vs_base")
    inside = got is not None and min(lo_d, hi_d) <= got <= max(lo_d, hi_d)
    print(f"\n=== H2 机制的定量预测(可单独失败)===")
    print(f"  已测两组:最少交叉 {lo_d:+.1%}(交叉数 8-18),最多交叉 {hi_d:+.1%}(30-88)")
    print(f"  留出集交叉数 {min(nc[p] for p in ps)}-{max(nc[p] for p in ps)},降幅 {got:+.1%}")
    print(f"  {'落在两者之间 —— 机制的定量预测命中' if inside else '落在区间之外 —— 旋钮仍然有用,但我给的机制解释在数量上不成立,照此上报'}")

    mo = out["arms"].get("mcc_only", {})
    print(f"\n=== H3 不依赖补丁的那一臂 ===")
    print(f"  mcc_only 降幅 {mo.get('delta_vs_base', float('nan')):+.1%},p = {mo.get('p')}"
          f"  -> {'同向,结论不依赖我的 DesiRNA 补丁' if mo.get('delta_vs_base', 0) < 0 else '不同向,需要重新审视补丁'}")

    out.update({"H1_pass": h1, "H2_inside": bool(inside),
                "H2_reference": {"low": lo_d, "high": hi_d, "holdout": got},
                "per_target": {p: per[p] for p in ps}})
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
