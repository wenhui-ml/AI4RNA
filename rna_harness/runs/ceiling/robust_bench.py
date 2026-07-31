#!/usr/bin/env python3
"""Does the point-versus-neighbourhood split predict a THIRD gap, or was co-transcriptional folding a one-off?

Same discipline as every gap claim here: both arms pay for the identical DesiRNA invocation, and the only
difference is what is done with what comes back. Both arms must fold AND meet the requirement -- the kinetic
benchmark's first version scored them on different conditions and that is the defect this one starts from.

The GC-band control is PAIRED and runs in the same invocation: restrict to the targets whose own answer is
already inside 45-58% GC, and screen our arm inside the same band. That form was arrived at because the
unpaired version turned out to be a draw -- `screen` returns a candidate from a freshly sampled pool, so the
chosen sequence's GC varies run to run and cannot carry an orthogonality claim.

The requirement, the 0.50 threshold, the sweep and the three ways this dies are declared in
benchmarks/robust_suite.py before anything here runs.
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
    from rna_harness.benchmarks.robust_suite import gc_percent, neutral_fraction, screen
    from rna_harness.cognitive.certify import certify_sequence
    from rna_harness.generators.desirna import DesiRNAGenerator
    from rna_harness.task import DesignTask, OracleConfig
    t0 = time.time()
    oc = OracleConfig(param_set="turner2004", temperature=37.0, dangles=2)
    try:
        res = DesiRNAGenerator(replicas=10, timelimit_s=timelimit, results_number=50).design(
            DesignTask(task_id="rob", target=target, oracle=oc), seed=seed)
    except Exception as e:                                                    # noqa: BLE001
        return {"task_id": tid, "n": len(target), "error": repr(e)[:160]}

    pool, seen = [], set()
    for s in ([res.best_sequence] if res.best_sequence else []) + list(res.candidates or []):
        if s and len(s) == len(target) and s not in seen:
            seen.add(s)
            pool.append(s)

    tool = None
    if res.best_sequence:
        nf = neutral_fraction(res.best_sequence, target)
        folds = certify_sequence(res.best_sequence, target).verdict in ("certified", "knot_confirmed")
        tool = {"sequence": res.best_sequence, "gc": round(gc_percent(res.best_sequence), 1),
                "folds": folds, **nf}

    free = screen(pool, target)
    band = screen(pool, target, gc_band=GC_BAND)
    return {"task_id": tid, "n": len(target), "pool": len(pool), "wall_s": round(time.time() - t0, 1),
            "tool": tool, "tool_in_band": bool(tool and GC_BAND[0] <= tool["gc"] <= GC_BAND[1]),
            "free": {k: v for k, v in free.items() if k != "rows"},
            "band": {k: v for k, v in band.items() if k != "rows"},
            "free_rows": free["rows"], "band_rows": band["rows"]}


def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided binomial on the discordant pairs."""
    from math import comb
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--timelimit", type=int, default=15)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(HERE, "robust_bench.json"))
    a = ap.parse_args()

    from rna_harness.benchmarks.bprna_nested import load_bprna_nested
    from rna_harness.benchmarks.robust_suite import MAX_SCREEN, NEUTRAL_MIN, SWEEP
    rows = [r for r in load_bprna_nested(400) if 25 <= len(str(r["target"])) <= 70]
    jobs = [(str(r["id"]), str(r["target"]), a.timelimit, 0) for r in rows[:a.n]]
    print(f"突变鲁棒性:{len(jobs)} 条自然结构(与动力学基准同一批),同一批求解器运行两臂读法;"
          f"声明阈值 {NEUTRAL_MIN},每条最多筛 {MAX_SCREEN} 个候选\n", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    R, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, f in enumerate(as_completed([ex.submit(run_one, j) for j in jobs]), 1):
            R.append(f.result())
            print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)

    ok = [r for r in R if not r.get("error") and r.get("tool")]
    # ---- primary, THRESHOLD-FREE: paired difference in neutral fraction -----------------------------
    pairs = [(r["tool"]["neutral_fraction"], (r["free"]["best"] or {}).get("neutral_fraction"))
             for r in ok if r["free"]["best"]]
    wins = sum(1 for t, o in pairs if o > t)
    losses = sum(1 for t, o in pairs if o < t)
    import statistics as st
    dt = [o - t for t, o in pairs]
    print(f"\n=== {len(ok)}/{len(R)} 条完成 ===")
    print(f"  无阈值主统计(配对,池内最好 vs 它自己的答案):")
    print(f"    我们更鲁棒 {wins} 条,它更鲁棒 {losses} 条,符号检验 p={sign_test(wins, losses):.3g}")
    print(f"    中位提升 {st.median(dt):+.3f}   (工具中位 {st.median([t for t, _ in pairs]):.3f} "
          f"-> 我们 {st.median([o for _, o in pairs]):.3f})")

    # ---- requirement-shaped, with the sweep ---------------------------------------------------------
    sub_b = [r for r in ok if r["tool_in_band"]]
    print(f"  阈值扫描(两臂同时重算;声明值 {NEUTRAL_MIN}):")
    print(f"    {'阈值':<8}{'工具':>10}{'我们':>10}    {'工具|带内':>12}{'我们|带内':>12}")
    sweep = []
    for th in SWEEP:
        t_ok = sum(1 for r in ok if r["tool"]["folds"] and r["tool"]["neutral_fraction"] >= th)
        o_ok = sum(1 for r in ok if any(x["neutral_fraction"] >= th for x in r["free_rows"]))
        tb = sum(1 for r in sub_b if r["tool"]["folds"] and r["tool"]["neutral_fraction"] >= th)
        ob = sum(1 for r in sub_b if any(x["neutral_fraction"] >= th for x in r["band_rows"]))
        sweep.append({"threshold": th, "tool": t_ok, "ours": o_ok, "tool_in_band": tb,
                      "ours_in_band": ob, "n": len(ok), "n_in_band": len(sub_b)})
        mark = "  <-- 声明值" if abs(th - NEUTRAL_MIN) < 1e-9 else ""
        print(f"    {th:<8.2f}{t_ok:>8}/{len(ok)}{o_ok:>8}/{len(ok)}"
              f"    {tb:>10}/{len(sub_b)}{ob:>10}/{len(sub_b)}{mark}")

    # ---- the two ways this could be a null, measured rather than argued ------------------------------
    rates = sorted((sum(1 for x in r["free_rows"] if x["neutral_fraction"] >= NEUTRAL_MIN)
                    / max(1, len(r["free_rows"]))) for r in ok if r["free_rows"])
    gt = [r["tool"]["gc"] for r in sub_b]
    go = [(r["band"]["best"] or {}).get("gc") for r in sub_b if r["band"]["best"]]
    print(f"  池通过率中位 {rates[len(rates)//2] if rates else 0:.3f}"
          f"   (接近 0 = M1 天花板,筛选变不出不存在的东西)")
    if gt and go:
        print(f"  配对带内 GC 中位:工具 {st.median(gt):.1f}%  我们 {st.median(go):.1f}%"
              f"   (相同才说明缺口不是组分造成的)")
    capped = sum(1 for r in ok if r["free"].get("capped"))
    print(f"  触到 {MAX_SCREEN} 候选上限的靶标:{capped}/{len(ok)}  (上限是声明的,不是静默截断)")

    json.dump({"requirement": f"neutral_fraction >= {NEUTRAL_MIN} (declared pre-run)",
               "gc_band": GC_BAND, "cap": MAX_SCREEN, "timelimit_s": a.timelimit,
               "paired": {"wins": wins, "losses": losses, "p": sign_test(wins, losses),
                          "median_delta": st.median(dt) if dt else None},
               "sweep": sweep, "median_pool_pass_rate": rates[len(rates) // 2] if rates else None,
               "n_capped": capped, "rows": R}, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
