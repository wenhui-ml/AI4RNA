#!/usr/bin/env python3
"""Is there a SECOND gap -- one that does not collapse to GC content?

TWO ARMS OFF THE SAME SOLVER RUNS, which is the discipline every gap claim in this project is held to. Both
arms pay for the identical DesiRNA invocation; the only difference is what is done with what comes back:

    tool    take the answer it hands you (best_sequence) and ask whether it meets the requirement
    ours    screen the pool THOSE SAME RUNS returned, and hand back the first sequence that does

So a difference cannot be "we bought more compute" -- the compute is the same object.

WHAT THE PAIRED TEST CAN AND CANNOT SAY, because the arms are not independent and reporting a p-value without
saying so would be circular. `ours` scans the pool in order and the pool's first element IS the tool's own
answer, so whenever the tool passes, ours passes: the "tool only" cell of the McNemar table is zero BY
CONSTRUCTION, not by measurement. The arms are nested, and the quantity that actually carries information is
the rescue rate -- of the targets where the tool's own answer fails, how many does screening the same pool
recover. The p-value is reported because the count is what it is, but the rescue rate is the claim.

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
from rna_harness.benchmarks.kinetic_suite import TRAP_FRACTION_MAX   # noqa: E402  (after sys.path)


def run_one(job):
    tid, target, timelimit, seed = job
    from rna_harness.benchmarks.kinetic_suite import (TRAP_FRACTION_MAX, cotranscriptional,
                                                  gc_percent, screen)
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

    # ---- arm `tool`: its own answer, judged by the SAME two conditions our arm has to meet ---------
    # BOTH ARMS MUST FOLD AND PASS. The first version of this file scored the tool on kinetics alone while
    # requiring our arm to fold as well, and that asymmetry handed the tool a free point: one sequence passed
    # the kinetic filter while not folding to the target at all. It inflated the BASELINE rather than us, so
    # the published direction was safe, but a benchmark whose two arms answer different questions is not a
    # benchmark. `folds` is recorded separately so a reader can see which of the two conditions failed.
    from rna_harness.cognitive.certify import certify_sequence
    tool = None
    if res.best_sequence:
        k = cotranscriptional(res.best_sequence, target)
        folds = certify_sequence(res.best_sequence, target).verdict in ("certified", "knot_confirmed")
        tool = {"sequence": res.best_sequence, "gc": round(gc_percent(res.best_sequence), 1),
                "trap_fraction": k["trap_fraction"], "kinetics_held": k["held"], "folds": folds,
                "held": bool(folds and k["held"]), "worst": k["worst"]}

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
    # RE-DERIVE THE SUMMARY FROM THE ARCHIVED PER-ITEM ROWS, without touching the solver. The threshold sweep
    # was added to this file AFTER the run it is quoted from, so the archived json did not contain the table
    # the weekly report cites -- a number in a report that cannot be regenerated from its own artifact is not
    # a result. Every quantity below is a function of `rows`, so this reproduces all of them exactly.
    ap.add_argument("--analyse-only", action="store_true")
    a = ap.parse_args()

    from rna_harness.benchmarks.bprna_nested import load_bprna_nested
    rows = [r for r in load_bprna_nested(400) if 25 <= len(str(r["target"])) <= 70]
    sel = rows[:a.n]
    jobs = [(str(r["id"]), str(r["target"]), a.timelimit, 0) for r in sel]
    print(f"动力学/共转录:{len(jobs)} 条自然结构,同一批求解器运行两臂读法,"
          f"外加 GC {GC_BAND[0]}-{GC_BAND[1]}% 控制变量\n", flush=True)

    if a.analyse_only:
        R = json.load(open(a.out))["rows"]
        print(f"  从归档的 {len(R)} 条逐条数据重算,不重跑求解器\n", flush=True)
    else:
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
    # BANDED: the control. The tool produces ONE answer, so it cannot be "restricted" to the band the way a
    # pool can -- the only honest denominator for its in-band rate is the answers that ARE in band. Quoting
    # it out of 24 would silently score "its answer sits at 66% GC" as a kinetic failure.
    n_in_band = sum(1 for r in ok if r["tool_in_band"])
    tb_ok = sum(1 for r in ok if r["tool_in_band"] and (r.get("tool") or {}).get("held"))
    ob_ok = sum(1 for r in ok if r["band"]["chosen"])
    rates = sorted(r["free"]["pool_pass_rate"] for r in ok)
    gt = [r["tool"]["gc"] for r in ok if r.get("tool")]
    go = [r["free"]["chosen"]["gc"] for r in ok if r["free"]["chosen"]]

    print(f"\n=== {len(ok)}/{len(R)} 条完成 ===")
    print(f"  自由:   工具自己的答案 {t_ok}/{len(ok)}     筛同一批池 {o_ok}/{len(ok)}"
          f"   McNemar p={mcnemar(a_only, b_only):.3g}  (仅我们 {a_only} / 仅它 {b_only})")
    print(f"  控制:   GC {GC_BAND[0]}-{GC_BAND[1]}% 带内   工具 {tb_ok}/{n_in_band}(它落在带内的答案)"
          f"     我们 {ob_ok}/{len(ok)}")
    print(f"  池通过率中位 {rates[len(rates)//2] if rates else 0:.3f}"
          f"   (M1 天花板判据:接近 0 就是池里根本没有)")
    if gt:
        print(f"  GC:  工具答案中位 {sorted(gt)[len(gt)//2]:.1f}%"
              + (f"   我们选中的中位 {sorted(go)[len(go)//2]:.1f}%" if go else "   我们无选中"))
    # THE THRESHOLD SWEEP, BOTH ARMS, printed rather than left for a reader to trust. A declared threshold
    # is only worth something if the conclusion does not hinge on it, and the way to show that is to move it.
    sub_b = [r for r in ok if r["tool_in_band"]]
    sweep = []
    for th in (0.05, 0.10, 0.15, 0.20, 0.25, 0.35):
        sweep.append({
            "threshold": th,
            "tool": sum(1 for r in ok if (r.get("tool") or {}).get("folds")
                        and r["tool"]["trap_fraction"] <= th),
            "ours": sum(1 for r in ok if any(x["trap_fraction"] <= th for x in r["free_rows"])),
            "tool_in_band": sum(1 for r in sub_b if r["tool"]["folds"]
                                and r["tool"]["trap_fraction"] <= th),
            "ours_in_band": sum(1 for r in sub_b if any(x["trap_fraction"] <= th for x in r["band_rows"])),
            "n": len(ok), "n_in_band": len(sub_b)})
    print(f"  阈值敏感性(两臂同时重算):")
    for w in sweep:
        print(f"    {w['threshold']:.2f}   工具 {w['tool']:>2}/{w['n']}   我们 {w['ours']:>2}/{w['n']}"
              f"   |带内| 工具 {w['tool_in_band']}/{w['n_in_band']}  我们 {w['ours_in_band']}/{w['n_in_band']}")
    tfs = sorted(r["tool"]["trap_fraction"] for r in ok if r.get("tool"))
    near = [x for x in tfs if 0.5 * TRAP_FRACTION_MAX < x < 2.0 * TRAP_FRACTION_MAX]
    rescued = sum(1 for r in ok if r["free"]["chosen"] and not (r.get("tool") or {}).get("held"))
    failed = sum(1 for r in ok if not (r.get("tool") or {}).get("held"))
    print(f"  救回率:它自己的答案失败 {failed} 条,筛同一批池救回 {rescued} 条"
          f"   (两臂是嵌套的,'仅它' 恒为 0 是构造决定的,不是测出来的)")
    print(f"  阈值稳健性:trap_fraction 落在 {0.5*TRAP_FRACTION_MAX:.3f}-{2*TRAP_FRACTION_MAX:.3f} 的只有 "
          f"{len(near)}/{len(tfs)} 条 -- 阈值不吃重则分布应是双峰的")
    json.dump({"requirement": "co-transcriptional: trap_fraction <= 0.10 (declared pre-run)",
               "gc_band": GC_BAND, "timelimit_s": a.timelimit,
               "free": {"tool": t_ok, "ours": o_ok, "n": len(ok), "ours_only": a_only,
                        "tool_only": b_only, "p": mcnemar(a_only, b_only)},
               "band": {"tool": tb_ok, "tool_n_in_band": n_in_band, "ours": ob_ok, "n": len(ok)},
               "median_pool_pass_rate": rates[len(rates) // 2] if rates else None,
               "threshold_sweep": sweep, "trap_fraction_sorted": tfs, "n_near_threshold": len(near),
               "rescue": {"tool_failed": failed, "recovered_by_screening": rescued,
                          "note": "arms are nested; the tool-only cell is structurally 0"},
               "rows": R}, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
