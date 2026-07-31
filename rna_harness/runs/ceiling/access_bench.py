#!/usr/bin/env python3
"""The last coordinate. Predicted NULL -- and the control is the per-target GC match, not a band.

Both arms pay for the identical DesiRNA invocation and differ only in what they do with what comes back;
both must fold AND meet the requirement. That much is standard here. What is new since the switch benchmark
is the control:

    A BAND CONSTRAINS A RANGE. IT DOES NOT MATCH A PAIR. On 72 targets the two arms sat 3.2 GC points apart
    INSIDE the 45-58% band, which is the same drift that forced the kinetic benchmark's orthogonality claim
    to be withdrawn. So composition is pinned PER TARGET: our arm may use only candidates whose GC equals the
    tool's own answer, to a stated tolerance, and at tolerance 0 the two arms have literally the same base
    composition. The band is reported too, as the weaker control it is.

And the nested-arm caveat is printed inline rather than left for a reader to notice: `ours` scans a pool
whose first element IS the tool's own answer, so it cannot lose and any paired p-value against it has a
structurally empty cell. The rescue rate is the result.
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
    tid, target, at, length, timelimit, seed = job
    from rna_harness.benchmarks.access_suite import accessibility, gc_percent, screen
    from rna_harness.cognitive.certify import certify_sequence
    from rna_harness.generators.desirna import DesiRNAGenerator
    from rna_harness.task import DesignTask, OracleConfig
    t0 = time.time()
    oc = OracleConfig(param_set="turner2004", temperature=37.0, dangles=2)
    try:
        res = DesiRNAGenerator(replicas=10, timelimit_s=timelimit, results_number=50).design(
            DesignTask(task_id="acc", target=target, oracle=oc), seed=seed)
    except Exception as e:                                                    # noqa: BLE001
        return {"task_id": tid, "n": len(target), "error": repr(e)[:160]}

    pool, seen = [], set()
    for s in ([res.best_sequence] if res.best_sequence else []) + list(res.candidates or []):
        if s and len(s) == len(target) and s not in seen:
            seen.add(s)
            pool.append(s)

    from rna_harness.benchmarks.access_suite import ACCESS_MIN
    tool = None
    if res.best_sequence:
        acc = accessibility(res.best_sequence, at, length)
        folds = certify_sequence(res.best_sequence, target).verdict in ("certified", "knot_confirmed")
        tool = {"sequence": res.best_sequence, "gc": round(gc_percent(res.best_sequence), 1),
                "folds": folds, "accessibility": round(acc, 4),
                "held": bool(folds and acc >= ACCESS_MIN)}

    free = screen(pool, target, at, length)
    band = screen(pool, target, at, length, gc_band=GC_BAND)
    return {"task_id": tid, "n": len(target), "window_at": at, "window_len": length,
            "pool": len(pool), "wall_s": round(time.time() - t0, 1), "tool": tool,
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
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--pool", type=int, default=2000)
    ap.add_argument("--timelimit", type=int, default=15)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(HERE, "access_bench.json"))
    ap.add_argument("--analyse-only", action="store_true")
    a = ap.parse_args()

    from rna_harness.benchmarks.access_suite import ACCESS_MIN, MIN_WINDOW, SWEEP, longest_unpaired
    from rna_harness.benchmarks.bprna_nested import load_bprna_nested
    rows = [r for r in load_bprna_nested(a.pool) if 25 <= len(str(r["target"])) <= 70]
    jobs, dropped = [], 0
    for r in rows:
        at, L = longest_unpaired(str(r["target"]))
        if L < MIN_WINDOW:
            dropped += 1                 # a 3 nt window makes the mean noise -- counted, never silently cut
            continue
        jobs.append((str(r["id"]), str(r["target"]), at, L, a.timelimit, 0))
        if len(jobs) >= a.n:
            break
    print(f"可及性:{len(jobs)} 条自然结构,窗口 = 最长未配对段(>= {MIN_WINDOW} nt),"
          f"因窗口过短丢弃 {dropped} 条;声明阈值 {ACCESS_MIN}\n", flush=True)

    if a.analyse_only:
        R = json.load(open(a.out))["rows"]
        print(f"  从归档的 {len(R)} 条逐条数据重算,不重跑求解器\n", flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        R, t0 = [], time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for n, f in enumerate(as_completed([ex.submit(run_one, j) for j in jobs]), 1):
                R.append(f.result())
                if n % 8 == 0 or n == len(jobs):
                    print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)

    ok = [r for r in R if not r.get("error") and r.get("tool")]
    t_ok = sum(1 for r in ok if r["tool"]["held"])
    o_ok = sum(1 for r in ok if r["free"]["chosen"])
    failed = sum(1 for r in ok if not r["tool"]["held"])
    rescued = sum(1 for r in ok if r["free"]["chosen"] and not r["tool"]["held"])
    import statistics as st
    print(f"\n=== {len(ok)}/{len(R)} 条完成 ===")
    print(f"  自由:   工具自己的答案 {t_ok}/{len(ok)}     筛同一批池 {o_ok}/{len(ok)}")
    print(f"  救回率:它失败 {failed} 条,救回 {rescued} 条  (两臂嵌套,'仅它' 构造上恒为 0)")

    # THE DECISIVE CONTROL: composition pinned per target, not to a band.
    print(f"  GC 严格配对(逐条钉在它自己答案的 GC 上):")
    matched = []
    for tol in (2.0, 1.0, 0.0):
        w = l = both = nrows = 0
        for r in ok:
            if not r["tool"]["folds"]:
                continue
            cand = [x for x in r["free_rows"] if abs(x["gc"] - r["tool"]["gc"]) <= tol]
            if not cand:
                continue
            nrows += 1
            oo, tt = any(x["held"] for x in cand), r["tool"]["held"]
            w += oo and not tt
            l += tt and not oo
            both += oo and tt
        matched.append({"tolerance": tol, "n": nrows, "tool": both + l, "ours": both + w,
                        "ours_only": w, "tool_only": l, "p": sign_test(w, l)})
        print(f"    ±{tol:<4} (n={nrows})  工具 {both + l:>2}/{nrows}  我们 {both + w:>2}/{nrows}"
              f"   仅我们 {w} / 仅它 {l}   p={sign_test(w, l):.3g}")

    print(f"  阈值扫描(两臂同时重算;声明值 {ACCESS_MIN}):")
    sweep = []
    for th in SWEEP:
        t_k = sum(1 for r in ok if r["tool"]["folds"] and r["tool"]["accessibility"] >= th)
        o_k = sum(1 for r in ok if any(x["accessibility"] >= th for x in r["free_rows"]))
        sweep.append({"threshold": th, "tool": t_k, "ours": o_k, "n": len(ok)})
        mark = "  <-- 声明值" if abs(th - ACCESS_MIN) < 1e-9 else ""
        print(f"    {th:.2f}   工具 {t_k:>2}/{len(ok)}   我们 {o_k:>2}/{len(ok)}{mark}")

    rates = sorted(r["free"]["pool_pass_rate"] for r in ok)
    ta = [r["tool"]["accessibility"] for r in ok]
    print(f"  池通过率中位 {rates[len(rates)//2] if rates else 0:.3f}   (接近 1 = 免费,接近 0 = M1 天花板)")
    print(f"  工具答案可及性中位 {st.median(ta):.3f}")
    json.dump({"requirement": f"mean unpaired probability over the longest loop >= {ACCESS_MIN}",
               "prediction": "NULL -- accessibility is another readout of the same partition function",
               "gc_band": GC_BAND, "dropped_short_window": dropped,
               "free": {"tool": t_ok, "ours": o_ok, "n": len(ok)},
               "rescue": {"tool_failed": failed, "recovered": rescued,
                          "note": "arms are nested; 'tool only' is structurally 0"},
               "gc_matched": matched, "sweep": sweep,
               "median_pool_pass_rate": rates[len(rates) // 2] if rates else None,
               "rows": R}, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
