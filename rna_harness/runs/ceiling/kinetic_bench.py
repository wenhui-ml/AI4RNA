#!/usr/bin/env python3
"""Is there a SECOND gap -- one that does not collapse to GC content?

TWO ARMS OFF THE SAME SOLVER RUNS, which is the discipline every gap claim in this project is held to. Both
arms pay for the identical DesiRNA invocation; the only difference is what is done with what comes back:

    tool    take the answer it hands you (best_sequence) and ask whether it meets the requirement
    ours    screen the pool THOSE SAME RUNS returned, and hand back the first sequence that does

So a difference cannot be "we bought more compute" -- the compute is the same object.

AND THE CONTROL RUNS IN THE SAME INVOCATION, not as a follow-up nobody gets around to. Every target is scored
twice: free, and with both arms restricted to 45-58% GC. If the gap survives the band, this requirement is
orthogonal to composition and the project has a second capability. If it does not, that is a fourth
confirmation of the one-gap conclusion and gets written down as one.

The requirement, the 0.10 threshold and the three refutation modes are declared in benchmarks/kinetic_suite.py
before any of this runs.
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
    tid, target, timelimit, seed = job
    from rna_harness.benchmarks.kinetic_suite import cotranscriptional, gc_percent, screen
    from rna_harness.generators.desirna import DesiRNAGenerator
    from rna_harness.task import DesignTask, OracleConfig
    t0 = time.time()
    oc = OracleConfig(param_set="turner2004", temperature=37.0, dangles=2)
    try:
        res = DesiRNAGenerator(replicas=10, timelimit_s=timelimit, results_number=50).design(
            DesignTask(task_id="kin", target=target, oracle=oc), seed=seed)
    except Exception as e:                                                    # noqa: BLE001
        return {"task_id": tid, "n": len(target), "error": repr(e)[:160]}

    pool, seen = [], set()
    for s in ([res.best_sequence] if res.best_sequence else []) + list(res.candidates or []):
        if s and len(s) == len(target) and s not in seen:
            seen.add(s)
            pool.append(s)

    # ---- arm `tool`: its own answer, judged by the same requirement --------------------------------
    tool = None
    if res.best_sequence:
        k = cotranscriptional(res.best_sequence, target)
        tool = {"sequence": res.best_sequence, "gc": round(gc_percent(res.best_sequence), 1),
                "trap_fraction": k["trap_fraction"], "held": k["held"], "worst": k["worst"]}

    # ---- arm `ours`: the same pool, screened; free and inside the band -----------------------------
    free = screen(pool, target)
    band = screen(pool, target, gc_band=GC_BAND)
    tool_in_band = bool(tool and GC_BAND[0] <= tool["gc"] <= GC_BAND[1])

    return {"task_id": tid, "n": len(target), "pool": len(pool), "wall_s": round(time.time() - t0, 1),
            "tool": tool, "tool_in_band": tool_in_band,
            "free": {k: v for k, v in free.items() if k != "rows"},
            "band": {k: v for k, v in band.items() if k != "rows"},
            "free_rows": free["rows"], "band_rows": band["rows"]}


def mcnemar(a_only: int, b_only: int) -> float:
    """Exact two-sided binomial on the discordant pairs -- the same test used everywhere else here."""
    from math import comb
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--timelimit", type=int, default=15)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(HERE, "kinetic_bench.json"))
    a = ap.parse_args()

    from rna_harness.benchmarks.bprna_nested import load_bprna_nested
    rows = [r for r in load_bprna_nested(400) if 25 <= len(str(r["target"])) <= 70]
    sel = rows[:a.n]
    jobs = [(str(r["id"]), str(r["target"]), a.timelimit, 0) for r in sel]
    print(f"动力学/共转录:{len(jobs)} 条自然结构,同一批求解器运行两臂读法,"
          f"外加 GC {GC_BAND[0]}-{GC_BAND[1]}% 控制变量\n", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    R, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, f in enumerate(as_completed([ex.submit(run_one, j) for j in jobs]), 1):
            R.append(f.result())
            print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)

    ok = [r for r in R if not r.get("error")]
    # FREE: tool's own answer vs screening the same pool
    t_ok = sum(1 for r in ok if (r.get("tool") or {}).get("held"))
    o_ok = sum(1 for r in ok if r["free"]["chosen"])
    a_only = sum(1 for r in ok if r["free"]["chosen"] and not (r.get("tool") or {}).get("held"))
    b_only = sum(1 for r in ok if (r.get("tool") or {}).get("held") and not r["free"]["chosen"])
    # BANDED: the control
    tb_ok = sum(1 for r in ok if r["tool_in_band"] and (r.get("tool") or {}).get("held"))
    ob_ok = sum(1 for r in ok if r["band"]["chosen"])
    rates = sorted(r["free"]["pool_pass_rate"] for r in ok)
    gt = [r["tool"]["gc"] for r in ok if r.get("tool")]
    go = [r["free"]["chosen"]["gc"] for r in ok if r["free"]["chosen"]]

    print(f"\n=== {len(ok)}/{len(R)} 条完成 ===")
    print(f"  自由:   工具自己的答案 {t_ok}/{len(ok)}     筛同一批池 {o_ok}/{len(ok)}"
          f"   McNemar p={mcnemar(a_only, b_only):.3g}  (仅我们 {a_only} / 仅它 {b_only})")
    print(f"  控制:   GC {GC_BAND[0]}-{GC_BAND[1]}% 带内   工具 {tb_ok}/{len(ok)}     我们 {ob_ok}/{len(ok)}")
    print(f"  池通过率中位 {rates[len(rates)//2] if rates else 0:.3f}"
          f"   (M1 天花板判据:接近 0 就是池里根本没有)")
    if gt:
        print(f"  GC:  工具答案中位 {sorted(gt)[len(gt)//2]:.1f}%"
              + (f"   我们选中的中位 {sorted(go)[len(go)//2]:.1f}%" if go else "   我们无选中"))
    json.dump({"requirement": "co-transcriptional: trap_fraction <= 0.10 (declared pre-run)",
               "gc_band": GC_BAND, "timelimit_s": a.timelimit,
               "free": {"tool": t_ok, "ours": o_ok, "n": len(ok), "ours_only": a_only,
                        "tool_only": b_only, "p": mcnemar(a_only, b_only)},
               "band": {"tool": tb_ok, "ours": ob_ok, "n": len(ok)},
               "median_pool_pass_rate": rates[len(rates) // 2] if rates else None,
               "rows": R}, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
