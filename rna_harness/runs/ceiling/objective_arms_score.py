#!/usr/bin/env python3
"""Stage B: score the objective arms, and test the mechanism as well as the intervention.

RUNS IN THE RibonanzaNet venv. Every arm came from the same reproducible solver machinery at the same step
budget, so arm is the only variable and there is no run-to-run noise between them.

    M1  on the fewest-crossing targets: fraction of candidates at exactly 0.000 crossed-pair F1, each arm
        against base, paired within target, Wilcoxon one-sided less. PASS: median drop >= 10 points, p<0.025.
    M2  the same on the most-crossing targets. The mechanism predicts M1 >> M2. Equal help in both regimes
        means the knob may work but the explanation is wrong.
    M3  pool median crossed-pair F1 per arm, to catch a zero-fraction bought by trading median quality away.

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
    ap.add_argument("--pools", default=os.path.join(HERE, "objective_arms_pools.json"))
    ap.add_argument("--workers", type=int, default=56)
    ap.add_argument("--out", default=os.path.join(HERE, "objective_arms_score.json"))
    a = ap.parse_args()

    from m2r_pairwise import wilcoxon

    P = json.load(open(a.pools))
    T, group, arms = P["targets"], P["group"], P["arms"]
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
            per[r["puzzle"]][r["arm"]] = {
                "zero_frac": sum(1 for x in v if x == 0.0) / len(v),
                "median": st.median(v), "max": max(v), "n": len(v)}

    out = {"arms": arms, "by_group": {}}
    for grp, label in (("low", "最少交叉(缺口最大的那一端)"), ("high", "最多交叉(对照)")):
        ps = [p for p in per if group.get(p) == grp and "base" in per[p]]
        print(f"\n=== {label}:{len(ps)} 个靶标 ===")
        print(f"  {'臂':11s} {'F1=0 占比':>10s} {'对 base 的变化':>14s} {'更低':>7s} {'p':>9s} {'池中位':>8s}")
        g = {}
        for arm in arms:
            q = [p for p in ps if arm in per[p]]
            if not q:
                continue
            z = st.median([per[p][arm]["zero_frac"] for p in q])
            m = st.median([per[p][arm]["median"] for p in q])
            if arm == "base":
                print(f"  {arm:11s} {z:9.1%} {'—':>14s} {'—':>7s} {'—':>9s} {m:8.3f}")
                g[arm] = {"zero_frac": z, "median": m, "n_targets": len(q)}
                continue
            d = [per[p][arm]["zero_frac"] - per[p]["base"]["zero_frac"] for p in q]
            w, pv, _ = wilcoxon(d, "less")
            print(f"  {arm:11s} {z:9.1%} {st.median(d):+13.1%} "
                  f"{sum(1 for x in d if x<0):5d}/{len(d):<3d} "
                  f"{'n/a' if pv is None else f'{pv:.3g}':>9s} {m:8.3f}")
            g[arm] = {"zero_frac": z, "median": m, "delta_vs_base": st.median(d), "p": pv,
                      "lower": sum(1 for x in d if x < 0), "n_targets": len(d)}
        out["by_group"][grp] = g

    lo, hi = out["by_group"].get("low", {}), out["by_group"].get("high", {})
    cands = [(k, v) for k, v in lo.items() if k != "base" and v.get("p") is not None]
    best = min(cands, key=lambda kv: kv[1]["delta_vs_base"]) if cands else None
    print(f"\n=== 判读 ===")
    if not best:
        out["verdict"] = "没有可比较的臂"
    else:
        arm, v = best
        m1 = v["p"] < ALPHA and v["delta_vs_base"] <= -0.10
        print(f"  M1 最佳臂 `{arm}`:F1=0 占比变化 {v['delta_vs_base']:+.1%},p = {v['p']:.3g} "
              f"-> {'通过' if m1 else 'KILL —— 目标函数不是小假结被忽略的原因'}")
        hv = hi.get(arm, {})
        if m1 and hv.get("delta_vs_base") is not None:
            # COMPARE ON THE SIGNED SCALE, NOT ON MAGNITUDES. The first version divided |low| by |high|,
            # which reads an OPPOSITE effect in the control as a large ratio: -7.3 against +1.5 came out as
            # "4.9x", when the honest statement is that the control moved the wrong way and there is no
            # ratio to take. Magnitudes also make a control that HELPS MORE (say -20 against -7) look like
            # 0.4x rather than like the refutation it would be.
            lo_d, hi_d = v["delta_vs_base"], hv["delta_vs_base"]
            if hi_d >= 0:
                mech, how = True, (f"对照组根本没有改善({hi_d:+.1%}),效应只出现在小假结端"
                                   f"({lo_d:+.1%})")
                ratio = None
            else:
                ratio = lo_d / hi_d          # both negative: >1 means the low end improved more
                mech = ratio >= 2.0
                how = (f"两端都改善,小假结端是对照端的 {ratio:.1f} 倍"
                       f"({lo_d:+.1%} 对 {hi_d:+.1%})")
            print(f"  M2 同一臂在对照组:{hi_d:+.1%} —— {how}")
            print(f"  {'机制成立 —— 效应集中在小假结上,与「小假结被忽略」一致' if mech else '机制不成立 —— 两端帮助相当,这个旋钮也许有用,但我给的解释是错的'}")
            out["mechanism_supported"] = bool(mech)
            out["low_vs_high"] = {"low_delta": lo_d, "high_delta": hi_d, "ratio": ratio}
        base_med = lo["base"]["median"]
        print(f"  M3 池中位 base {base_med:.3f} -> `{arm}` {v['median']:.3f} "
              f"({'未以中位质量换取' if v['median'] >= base_med - 0.02 else '注意:中位质量被换掉了一部分'})")
        out["M1_pass"], out["best_arm"] = bool(m1), arm

    out["per_target"] = {p: per[p] for p in per}
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
