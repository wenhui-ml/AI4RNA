#!/usr/bin/env python3
"""A capability nobody can reach is not a capability: does the agent CHOOSE the new primitives?

WHY THIS IS THE NEXT THING RATHER THAN MORE SCIENCE. Two requirements were confirmed as gaps, shipped as
primitives, and verified to work when called directly -- `design` screens on them and refuses to finish
without them. What was never measured is whether the agent REACHES for them when a person describes the
requirement in their own words. Shipping a primitive is step one; being selected is step two, and only the
second one reaches a user.

The literature count makes this the highest-value gap of the three: the two-state switch is the most-named
requirement of nine across 1676 RNA-design papers (36 hits, 2.15%) and not one of five external tools exposes
it. Demand with no supply, and our supply currently sits behind a name only we know.

THE REQUESTS DELIBERATELY DO NOT REUSE THE SYSTEM PROMPT'S WORDING. That prompt says "a riboswitch, a toehold
switch, a sensor, or anything described as changing shape when something binds". Testing with those words
would measure string matching and would pass trivially. So each request here describes the FUNCTION in a way
a biologist would say it, drawn from how the literature names the concept rather than from how our
documentation does.

BOTH DIRECTIONS ARE SCORED, because a primitive that fires on everything is as broken as one that never
fires. Distractors are plain fold requests with no switching or kinetic content at all; selecting a
requirement there is a FALSE POSITIVE and costs the user money and time for a constraint they never asked
for.

    hit rate         of requests that describe the requirement, how many set it
    false-positive   of plain requests, how many set it anyway
    delivered        of requests where it was set, how many ended with a certificate that HOLDS it

The third matters separately: choosing the primitive and then failing to satisfy it is a different failure
from never choosing it, and averaging them would hide which one we have.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)

# A target with the toehold motif the switch requirement needs, and a plain one for the distractors.
SWITCH_TARGET = "....((((((....))))))......"
PLAIN_TARGET = "((((((....))))))"

REQUESTS = [
    # --- should set `switch` -------------------------------------------------------------------------
    ("switch", f"我有一个结构 {SWITCH_TARGET} ,平时它要保持关闭,但只要环境里出现一段跟它互补的 RNA,"
               f"它就得打开。请给我一条这样的序列"),
    ("switch", f"结构 {SWITCH_TARGET} 。我要做一个检测用的元件:没有目标 RNA 的时候它折成这个形状,"
               f"目标 RNA 一来就把茎撬开。序列给我一条"),
    ("switch", f"我需要一条 RNA,结构是 {SWITCH_TARGET} ,它必须能被另一条链触发而改变构象 —— "
               f"像核糖开关那样。请设计"),
    # --- should set `cotranscriptional` --------------------------------------------------------------
    ("cotrans", f"结构 {PLAIN_TARGET} 。这条 RNA 会在细胞里边转录边折叠,我不希望它在合成到一半的时候"
                f"先折成别的形状再卡住。请给我一条序列"),
    ("cotrans", f"我要 {PLAIN_TARGET} 这个结构的序列,但它是体内表达的,"
                f"折叠路径上不能有会把它困住的中间态"),
    # --- distractors: must NOT set either -------------------------------------------------------------
    ("plain", f"我有一个结构 {PLAIN_TARGET} ,请给我一条能折叠成它的 RNA 序列"),
    ("plain", f"请给我一条折叠成 {PLAIN_TARGET} 的序列,GC 含量在 50% 左右"),
    ("plain", f"结构 {PLAIN_TARGET} ,序列越稳定越好"),
]


def run_one(job):
    kind, request, model, budget = job
    from rna_harness.cognitive.agent import run
    t0 = time.time()
    try:
        sess = run(request, model=model, max_cost_usd=budget)
    except Exception as e:                                                    # noqa: BLE001
        return {"kind": kind, "error": repr(e)[:200], "request": request[:60]}
    actions = [s.action for s in sess.steps]
    cert = sess.certificate or {}
    det = cert.get("detail") or {}
    return {"kind": kind, "request": request[:70], "actions": actions,
            "set_switch": "set_switch" in actions,
            "set_cotranscriptional": "set_cotranscriptional" in actions,
            "verdict": cert.get("verdict"), "outcome": sess.outcome,
            # did the DELIVERED certificate actually carry the requirement, not just the action get called
            "switch_held": bool((sess.switch or {}) and (det.get("switch") or {}).get("held")),
            "kin_held": bool((sess.kin or {}) and (det.get("kinetics") or {}).get("held")),
            "cost_usd": sess.total_cost_usd, "llm_cost_usd": sess.llm_cost_usd,
            "n_steps": len(sess.steps), "wall_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="z-ai/glm-5.2")
    ap.add_argument("--budget", type=float, default=0.10)
    ap.add_argument("--repeats", type=int, default=3, help="the loop is stochastic; one run is an anecdote")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(HERE, "primitive_choice.json"))
    a = ap.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("!! 没有 OPENROUTER_API_KEY —— 这个基准要跑 agent 循环。设好再跑,不要用桩数据顶替。")
        return

    jobs = [(k, r, a.model, a.budget) for _ in range(a.repeats) for k, r in REQUESTS]
    print(f"原语选择:{len(REQUESTS)} 条请求 × {a.repeats} 次复跑 = {len(jobs)} 次\n"
          f"  (请求措辞刻意不复用系统提示里的词,否则测的是字符串匹配)\n", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    R, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, f in enumerate(as_completed([ex.submit(run_one, j) for j in jobs]), 1):
            R.append(f.result())
            if n % 4 == 0 or n == len(jobs):
                print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)

    from scipy.stats import beta

    def cp(k, n):
        if n == 0:
            return (0.0, 1.0)
        lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
        hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
        return lo, hi

    ok = [r for r in R if not r.get("error")]
    print(f"\n=== {len(ok)}/{len(R)} 次完成 ===")
    out = {}
    for kind, flag, held in (("switch", "set_switch", "switch_held"),
                             ("cotrans", "set_cotranscriptional", "kin_held")):
        sub = [r for r in ok if r["kind"] == kind]
        hit = sum(1 for r in sub if r[flag])
        lo, hi = cp(hit, len(sub))
        dsub = [r for r in ok if r["kind"] == "plain"]
        fp = sum(1 for r in dsub if r[flag])
        flo, fhi = cp(fp, len(dsub))
        deliv = sum(1 for r in sub if r[flag] and r[held])
        out[kind] = {"hit": hit, "n": len(sub), "hit_lo": lo, "hit_hi": hi,
                     "false_pos": fp, "n_plain": len(dsub), "fp_lo": flo, "fp_hi": fhi,
                     "delivered_holding": deliv}
        print(f"  {kind:<10} 该选而选中 {hit}/{len(sub)} = {hit/max(1,len(sub)):5.1%} [{lo:.0%},{hi:.0%}]"
              f"   误选(普通请求) {fp}/{len(dsub)} = {fp/max(1,len(dsub)):5.1%} [{flo:.0%},{fhi:.0%}]"
              f"   选中且证书守住 {deliv}/{hit if hit else 0}")
    cost = [r["cost_usd"] for r in ok]
    if cost:
        import statistics as st
        print(f"\n  单次花费中位 ${st.median(cost):.4f}   思考占 "
              f"${st.median([r['llm_cost_usd'] for r in ok]):.4f}")
    json.dump({"requests": [{"kind": k, "text": r} for k, r in REQUESTS], "summary": out, "rows": R},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
