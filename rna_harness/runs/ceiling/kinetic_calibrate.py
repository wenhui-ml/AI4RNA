#!/usr/bin/env python3
"""Does the SAME method work twice? A calibrated criterion for a SECOND judgment-missing domain.

WHAT GOAL 4 ACTUALLY CLAIMS, and why one instance is not enough. The claim is not "we solved pseudoknots".
It is "in a domain where the field has no agreed verdict, we can build a criterion that is calibrated,
externally validated, and honest about its scope". We have shown that once. Shown twice it is a METHOD;
shown once it is a result, and the difference is the whole distance between a paper and an anecdote.

Co-transcriptional folding is the natural second instance because it is judgment-missing in the same specific
way pseudoknots were: Kinfold is a stochastic simulation, findpath is a heuristic, prefix-MFE is a
deterministic approximation, and they do not agree. Nobody can cheaply say whether a designed sequence folds
correctly as it is made.

THE VALIDATION IS NOT A COPY OF THE PSEUDOKNOT ONE, and the difference matters. There, natives gave a usable
positive set: the wild-type sequence really does form its published structure. Here that fails -- ViennaRNA
recovers a native's own structure rarely enough that a native-based sensitivity would measure the folding
model, not the kinetic criterion.

So the question is put directly instead:

    DOES trap_fraction PREDICT WHAT AN INDEPENDENT KINETIC SIMULATOR SEES?

Population: DesiRNA designs for natural targets, every one verified to fold to its target, so the folding
question is held fixed and only kinetics varies. Split by our declared criterion (trap_fraction <= 0.10,
fixed in kinetic_suite.py long before this file). Then ask Kinfold -- stochastic, mechanistically unrelated,
never part of any screening here -- to grow each chain co-transcriptionally and report how often it reaches
the target.

    accepted reach >> rejected reach   ->  the criterion predicts an independent simulator, and the second
                                           domain is calibrated the way the first one was
    accepted ~= rejected               ->  trap_fraction measures nothing that matters kinetically, the
                                           "second gap" needs heavy qualification, and this is the single
                                           most important thing we could learn

DECLARED BEFORE THE RUN: the criterion passes if accepted-minus-rejected reach rate exceeds 20 points with
non-overlapping 95% intervals. Anything less is reported as it comes.

Kinfold's parse is one line: a trajectory that reaches the stop structure ends its final line with the marker
X, and one that times out ends with O. Verified on a positive control that reaches 5 of 5 and a poly-A
control that reaches 0.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)

KINFOLD = os.environ.get("KINFOLD", "/home/ubuntu/miniconda3/envs/vpk/bin/Kinfold")
TRAJ = 20                 # trajectories per sequence; Kinfold is milliseconds, so this is cheap
SIM_TIME = 8000.0         # arbitrary units; declared and held constant across every sequence
GROW = 20.0               # chain grows every GROW time units -- this is what makes it CO-transcriptional
MIN_GAP = 0.20            # declared pass threshold: accepted minus rejected reach rate


def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    from scipy.stats import beta
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return (lo, hi)


def kinfold_reach(seq: str, target: str, traj: int = TRAJ, grow: float = GROW,
                  timeout_s: float = 120.0):
    """Fraction of CO-TRANSCRIPTIONAL trajectories that reach the target. None if the tool did not run.

    None is never 0. A tool that failed to launch and a sequence that never folds are opposite findings, and
    conflating them is the silent-failure class this project keeps catching in other people's results.
    """
    if not os.path.exists(KINFOLD):
        return None
    try:
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run(
                [KINFOLD, "--stop", "--fpt", "--grow", str(grow), "--glen", "10",
                 "--num", str(traj), "--time", str(SIM_TIME)],
                input=f"{seq}\n{target}\n", capture_output=True, text=True,
                timeout=timeout_s, cwd=d)
    except Exception:                                                          # noqa: BLE001
        return None
    lines = [ln.rstrip() for ln in (r.stdout or "").splitlines() if ln.strip()]
    reached = sum(1 for ln in lines if ln.endswith("X1") or ln.rstrip().split()[-1].startswith("X"))
    ended = sum(1 for ln in lines
                if ln.endswith("X1") or ln.rstrip().split()[-1] in ("O",)
                or ln.rstrip().split()[-1].startswith("X"))
    if ended == 0:
        return None
    return {"reach": reached / ended, "n_traj": ended, "n_reached": reached}


def work(job):
    tid, target, timelimit, seed = job
    from rna_harness.benchmarks.kinetic_suite import TRAP_FRACTION_MAX, cotranscriptional, gc_percent
    from rna_harness.cognitive.certify import certify_sequence
    from rna_harness.generators.desirna import DesiRNAGenerator
    from rna_harness.task import DesignTask, OracleConfig
    t0 = time.time()
    oc = OracleConfig(param_set="turner2004", temperature=37.0, dangles=2)
    try:
        res = DesiRNAGenerator(replicas=10, timelimit_s=timelimit, results_number=50).design(
            DesignTask(task_id="kcal", target=target, oracle=oc), seed=seed)
    except Exception as e:                                                    # noqa: BLE001
        return {"task_id": tid, "error": repr(e)[:160]}
    pool, seen = [], set()
    for s in ([res.best_sequence] if res.best_sequence else []) + list(res.candidates or []):
        if s and len(s) == len(target) and s not in seen:
            seen.add(s)
            pool.append(s)
    out = []
    for s in pool[:12]:
        # FOLDING IS HELD FIXED. Only candidates that answer the shape question enter, so any difference in
        # reach rate is about kinetics rather than about one arm folding wrong.
        if certify_sequence(s, target).verdict not in ("certified", "knot_confirmed"):
            continue
        k = cotranscriptional(s, target)
        kf = kinfold_reach(s, target)
        out.append({"seq": s, "gc": round(gc_percent(s), 1),
                    "trap_fraction": k["trap_fraction"],
                    "accepted": k["trap_fraction"] <= TRAP_FRACTION_MAX,
                    "kinfold": kf})
    return {"task_id": tid, "n": len(target), "n_folding": len(out),
            "wall_s": round(time.time() - t0, 1), "candidates": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--timelimit", type=int, default=15)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "kinetic_calibrate.json"))
    a = ap.parse_args()
    if not os.path.exists(KINFOLD):
        print(f"!! 找不到外部裁判 {KINFOLD} —— 不要用我们自己的前缀 MFE 顶替,那是自证。")
        return

    from rna_harness.benchmarks.bprna_nested import load_bprna_nested
    from rna_harness.benchmarks.kinetic_suite import TRAP_FRACTION_MAX
    rows = [r for r in load_bprna_nested(400) if 25 <= len(str(r["target"])) <= 70][:a.n]
    jobs = [(str(r["id"]), str(r["target"]), a.timelimit, a.seed) for r in rows]
    print(f"共转录判据的外部验证:{len(jobs)} 条自然结构\n"
          f"  判据 trap_fraction <= {TRAP_FRACTION_MAX}(早已声明);外部裁判 Kinfold --grow,"
          f"{TRAJ} 条轨迹/序列\n", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    R, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, f in enumerate(as_completed([ex.submit(work, j) for j in jobs]), 1):
            R.append(f.result())
            if n % 8 == 0 or n == len(jobs):
                print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)

    cands = [c for r in R if not r.get("error") for c in r["candidates"] if c["kinfold"]]
    acc = [c for c in cands if c["accepted"]]
    rej = [c for c in cands if not c["accepted"]]
    print(f"\n=== {len(cands)} 条【折叠正确】的候选,其中判据接受 {len(acc)},拒绝 {len(rej)} ===")

    def rate(group):
        tot = sum(c["kinfold"]["n_traj"] for c in group)
        hit = sum(c["kinfold"]["n_reached"] for c in group)
        lo, hi = clopper_pearson(hit, tot)
        return hit, tot, (hit / tot if tot else 0.0), lo, hi

    ah, at_, ar, alo, ahi = rate(acc)
    rh, rt, rr, rlo, rhi = rate(rej)
    print(f"  Kinfold 共转录到达率(按轨迹计):")
    print(f"    判据【接受】的  {ah}/{at_} = {ar:6.1%}  [{alo:.1%}, {ahi:.1%}]")
    print(f"    判据【拒绝】的  {rh}/{rt} = {rr:6.1%}  [{rlo:.1%}, {rhi:.1%}]")
    gap = ar - rr
    disjoint = alo > rhi or rlo > ahi
    print(f"\n  差距 {gap:+.1%}   区间不重叠 = {disjoint}")
    print(f"  预注册判据:差距 > {MIN_GAP:.0%} 且区间不重叠 -> "
          f"{'通过 —— 判据能预测一个独立模拟器' if (gap > MIN_GAP and disjoint) else '未通过 —— 照实报'}")

    # THE RELATIONSHIP, not just the two buckets: a threshold that happens to split well is weaker evidence
    # than a monotone trend across the whole range.
    import statistics as st
    print(f"\n  trap_fraction 分档 vs 到达率(检验是不是单调,而不只是这个阈值恰好切得好):")
    bands = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.40), (0.40, 1.01)]
    trend = []
    for lo_, hi_ in bands:
        g = [c for c in cands if lo_ <= c["trap_fraction"] < hi_]
        if not g:
            continue
        h, t_, r_, l_, u_ = rate(g)
        trend.append({"lo": lo_, "hi": hi_, "n_seq": len(g), "reach": r_})
        print(f"    [{lo_:.2f}, {hi_:.2f})  序列 {len(g):>3}  到达 {h}/{t_} = {r_:6.1%} [{l_:.0%},{u_:.0%}]")
    if len(cands) > 3:
        xs = [c["trap_fraction"] for c in cands]
        ys = [c["kinfold"]["reach"] for c in cands]
        try:
            from scipy.stats import spearmanr
            rho, p = spearmanr(xs, ys)
            print(f"\n  Spearman 秩相关 rho={rho:+.3f}  p={p:.3g}"
                  f"   (负相关 = 陷阱越多、到达越少,方向正确)")
        except Exception:                                                      # noqa: BLE001
            rho, p = None, None
    json.dump({"criterion": f"trap_fraction <= {TRAP_FRACTION_MAX}", "judge": "Kinfold --grow",
               "traj_per_seq": TRAJ, "sim_time": SIM_TIME, "grow": GROW,
               "accepted": {"reached": ah, "traj": at_, "rate": ar, "lo95": alo, "hi95": ahi},
               "rejected": {"reached": rh, "traj": rt, "rate": rr, "lo95": rlo, "hi95": rhi},
               "gap": gap, "intervals_disjoint": disjoint, "trend": trend,
               "spearman": {"rho": rho, "p": p} if len(cands) > 3 else None,
               "rows": R}, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
