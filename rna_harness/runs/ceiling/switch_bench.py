#!/usr/bin/env python3
"""The partner coordinate: does a two-state switch open a gap, or is the third one a null too?

Same discipline as every gap claim here -- both arms pay for the identical DesiRNA invocation and differ only
in what they do with what comes back, both arms must satisfy BOTH states, and the GC-band control is paired
and runs in the same invocation.

Two traps this file is written to avoid, both of which bit earlier benchmarks in this directory:

  * NESTED ARMS. `ours` scans a pool whose first element is the tool's own answer, so it can never lose and a
    paired p-value against it is circular. The number reported as the result is therefore the RESCUE RATE --
    of the targets where the tool's own answer fails, how many does screening recover -- and the p-value is
    printed with that caveat attached rather than as the headline.
  * A STATISTIC THAT IS A DRAW. Every quantity here is exhaustive or exact: MFE on the monomer, MFE on the
    two-chain complex, and a pair table read for PARTNER identity rather than for brackets. Re-running moves
    only the pool, not the measurement.

The requirement, the toehold + invasion window, and the three ways this dies are declared in
benchmarks/switch_suite.py.
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
GC_BAND = (45.0, 58.0)


def run_one(job):
    tid, target, at, timelimit, seed = job
    from rna_harness.benchmarks.switch_suite import gc_percent, screen, switches
    from rna_harness.generators.desirna import DesiRNAGenerator
    from rna_harness.task import DesignTask, OracleConfig
    t0 = time.time()
    oc = OracleConfig(param_set="turner2004", temperature=37.0, dangles=2)
    try:
        res = DesiRNAGenerator(replicas=10, timelimit_s=timelimit, results_number=50).design(
            DesignTask(task_id="sw", target=target, oracle=oc), seed=seed)
    except Exception as e:                                                    # noqa: BLE001
        return {"task_id": tid, "n": len(target), "error": repr(e)[:160]}

    pool, seen = [], set()
    for s in ([res.best_sequence] if res.best_sequence else []) + list(res.candidates or []):
        if s and len(s) == len(target) and s not in seen:
            seen.add(s)
            pool.append(s)

    tool = None
    if res.best_sequence:
        r = switches(res.best_sequence, target, at)
        tool = {"sequence": res.best_sequence, "gc": round(gc_percent(res.best_sequence), 1),
                "free_ok": r["free_ok"], "invasion": r["invasion"], "held": r["held"],
                "complex_dg": r["complex_dg"]}

    free = screen(pool, target, at)
    band = screen(pool, target, at, gc_band=GC_BAND)
    return {"task_id": tid, "n": len(target), "window_at": at, "pool": len(pool),
            "wall_s": round(time.time() - t0, 1), "tool": tool,
            "tool_in_band": bool(tool and GC_BAND[0] <= tool["gc"] <= GC_BAND[1]),
            "free": {k: v for k, v in free.items() if k != "rows"},
            "band": {k: v for k, v in band.items() if k != "rows"},
            "free_rows": free["rows"], "band_rows": band["rows"]}


def sign_test(a: int, b: int) -> float:
    from math import comb
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--timelimit", type=int, default=15)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(HERE, "switch_bench.json"))
    a = ap.parse_args()

    from rna_harness.benchmarks.bprna_nested import load_bprna_nested
    from rna_harness.benchmarks.switch_suite import INVADE, TOEHOLD, switch_window
    rows = [r for r in load_bprna_nested(400) if 25 <= len(str(r["target"])) <= 70]
    jobs, dropped = [], 0
    for r in rows:
        at = switch_window(str(r["target"]))
        if at is None:
            dropped += 1                  # no stacked stem of WINDOW pairs -- reported, never silently cut
            continue
        jobs.append((str(r["id"]), str(r["target"]), at, a.timelimit, 0))
        if len(jobs) >= a.n:
            break
    print(f"两态开关:{len(jobs)} 条自然结构,toehold {TOEHOLD} nt + 入侵 {INVADE} 对,"
          f"因无此基序而丢弃 {dropped} 条\n", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    R, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, f in enumerate(as_completed([ex.submit(run_one, j) for j in jobs]), 1):
            R.append(f.result())
            print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)

    ok = [r for r in R if not r.get("error") and r.get("tool")]
    t_ok = sum(1 for r in ok if r["tool"]["held"])
    o_ok = sum(1 for r in ok if r["free"]["chosen"])
    failed = sum(1 for r in ok if not r["tool"]["held"])
    rescued = sum(1 for r in ok if r["free"]["chosen"] and not r["tool"]["held"])
    lost = sum(1 for r in ok if r["tool"]["held"] and not r["free"]["chosen"])
    sub_b = [r for r in ok if r["tool_in_band"]]
    tb = sum(1 for r in sub_b if r["tool"]["held"])
    ob = sum(1 for r in sub_b if r["band"]["chosen"])
    import statistics as st
    rates = sorted(r["free"]["pool_pass_rate"] for r in ok)
    # WHICH STATE FAILED, kept apart: "the fold is wrong" and "the trigger cannot open it" are different
    # diagnoses and averaging them would hide whichever one is doing the work.
    only_free = sum(1 for r in ok if r["tool"]["free_ok"] and not r["tool"]["held"])
    depths = sorted(r["tool"]["invasion"] for r in ok if r["tool"]["free_ok"])

    print(f"\n=== {len(ok)}/{len(R)} 条完成 ===")
    print(f"  自由:   工具自己的答案 {t_ok}/{len(ok)}     筛同一批池 {o_ok}/{len(ok)}"
          f"   (符号检验 p={sign_test(rescued, lost):.3g},两臂嵌套,见下)")
    print(f"  救回率:它失败 {failed} 条,筛同一批池救回 {rescued} 条;它成功而我们失败 {lost} 条")
    print(f"  控制:   GC {GC_BAND[0]}-{GC_BAND[1]}% 带内   工具 {tb}/{len(sub_b)}(它落在带内的答案)"
          f"     我们 {ob}/{len(ok)}")
    print(f"  池通过率中位 {rates[len(rates)//2] if rates else 0:.3f}   (接近 0 = M1 天花板)")
    print(f"  失败落在哪一态:折叠对但触发打不开 {only_free}/{len(ok)};"
          f"  它自己答案的入侵深度中位 {st.median(depths) if depths else 0:.0f}/{INVADE}")
    # NO SINGLE CUTOFF IS LOAD-BEARING: sweep the requirement over every achievable invasion depth.
    print(f"  入侵深度扫描(两臂同时重算):")
    sweep = []
    for k in range(1, INVADE + 1):
        t_k = sum(1 for r in ok if r["tool"]["free_ok"] and r["tool"]["invasion"] >= k)
        o_k = sum(1 for r in ok if any(x["invasion"] >= k for x in r["free_rows"]))
        tb_k = sum(1 for r in sub_b if r["tool"]["free_ok"] and r["tool"]["invasion"] >= k)
        ob_k = sum(1 for r in sub_b if any(x["invasion"] >= k for x in r["band_rows"]))
        sweep.append({"depth": k, "tool": t_k, "ours": o_k, "tool_in_band": tb_k,
                      "ours_in_band": ob_k, "n": len(ok), "n_in_band": len(sub_b)})
        mark = "  <-- 满入侵(声明的头条)" if k == INVADE else ""
        print(f"    >={k}   工具 {t_k:>2}/{len(ok)}   我们 {o_k:>2}/{len(ok)}"
              f"   |带内| 工具 {tb_k}/{len(sub_b)}  我们 {ob_k}/{len(sub_b)}{mark}")
    if sub_b:
        gt = [r["tool"]["gc"] for r in sub_b]
        go = [r["band"]["chosen"]["gc"] for r in sub_b if r["band"]["chosen"]]
        print(f"  配对带内 GC 中位:工具 {st.median(gt):.1f}%"
              + (f"  我们 {st.median(go):.1f}%" if go else "  我们无选中"))
    json.dump({"requirement": f"free fold == target AND all {INVADE} invaded stem pairs bound by the trigger in the complex",
               "gc_band": GC_BAND, "timelimit_s": a.timelimit, "dropped_no_stem": dropped,
               "free": {"tool": t_ok, "ours": o_ok, "n": len(ok)},
               "rescue": {"tool_failed": failed, "recovered": rescued, "lost": lost,
                          "note": "arms are nested; 'tool only' is structurally 0"},
               "band": {"tool": tb, "n_in_band": len(sub_b), "ours": ob, "n": len(ok)},
               "sweep": sweep,
               "median_pool_pass_rate": rates[len(rates) // 2] if rates else None,
               "rows": R}, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
