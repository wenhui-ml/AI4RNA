#!/usr/bin/env python3
"""Stage A: how much of our candidate pool is the solver's ANSWER, and how much is its scratch work?

THE DEFECT. `DesiRNAGenerator._run` collects candidates by walking every file the solver wrote and
regex-scraping every length-n ACGU string out of all of them, then keeping the first 200. DesiRNA writes its
ranked answers to `results.csv`, `mid_results.csv` and `best_fasta.fas` -- 20 each, which is the `-r 20` we
asked for -- and separately writes `trajectory_files/traj.csv`, `replicas.csv` and `random.csv`, which are
the raw optimisation trace. At a 300 s limit those three trace files held 1,450 sequences each. So of the
4,422 strings on disk, about 98% are intermediate states the solver had already rejected, and our 200-slot
pool ends up as roughly 20 real answers diluted with 180 early, barely-optimised ones.

    This is the mechanism behind the tier decline measured in `overoptimization.py`: at 300 s the pool got
    WORSE on our own panel (-0.067, down on 34 of 57 targets, p = 1.6e-06) as well as on RibonanzaNet
    (-0.084, down on 44 of 57, p = 1.2e-07). A longer run does not write worse answers; it writes a longer
    trajectory, so the first 180 trace rows sit proportionally earlier in the optimisation. Nothing is
    wrong with the solver. The scrape is wrong.

WHY IT MATTERS BEYOND THE TIER QUESTION. Every pool measurement made today used this scrape, and the
comparison that produced the day's headline gap -- our pool median 0.209 below the novelty-matched peer
median, on 57 of 57 targets -- was therefore comparing THEIR submitted designs against OUR scratch work.
That comparison is unfair in our own disfavour and the gap is very likely overstated. It is also a live
product defect and not merely a measurement one, because the shipped answer is chosen from this same pool,
so a trajectory state can be certified and returned to a user.

    T1  SEPARATION. Per target, median RNet crossed-pair F1 of the reported answers versus the trajectory
        states. Paired within target, Wilcoxon, one-sided greater.
        PASS: reported beats trajectory, p < 0.025.
        KILL: no difference -> the trace states are as good as the answers, the scrape is harmless, and
        both the tier decline and the overstated-gap story need another explanation.

    T2  CONTAMINATION. What share of the 200-slot pool the product actually builds is trajectory.

    T3  COST. Per target, the pool median as built today versus the pool median if the trace were excluded.
        This is the size of the free win available from a change to one loop.

Provenance is recorded per candidate, so nothing here depends on rerunning anything: the same run yields
both arms. Stage B (`trajectory_contamination_score.py`) scores them in the RibonanzaNet venv.

COST: $0 API. Local solver.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from leaderboard_design import targets                                          # noqa: E402

# The files DesiRNA uses to report its ranked answers. Everything under trajectory_files/ is the raw trace,
# except best_fasta.fas and best_str, which DesiRNA writes there but which ARE reported answers.
REPORTED = ("_results.csv", "_mid_results.csv", "_best_fasta.fas", "_best_str")
TRACE = ("_traj.csv", "_replicas.csv", "_random.csv")


def work(item):
    pid, target, tl, seed = item
    from rna_harness.generators.desirna import _desirna_cmd, DesiRNAGenerator
    cmd = _desirna_cmd()
    g = DesiRNAGenerator(replicas=10, timelimit_s=tl, results_number=20)
    n = len(target)
    t0 = time.time()
    from rna_harness.task import DesignTask, OracleConfig
    task = DesignTask(task_id="traj", target=target,
                      oracle=OracleConfig(param_set="turner2004", temperature=37.0, dangles=2))
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "input.txt"), "w") as f:
            f.write(g._build_input(task))
        try:
            subprocess.run(g._argv(cmd, "input.txt", seed, "2004"), cwd=td,
                           capture_output=True, text=True, timeout=g.timeout_s)
        except Exception as e:                                                 # noqa: BLE001
            return {"puzzle": pid, "error": repr(e)[:160], "reported": [], "trace": []}
        # Walk in the SAME order the shipped adapter walks, so `order` reproduces which candidates would
        # have filled its 200 slots.
        rep, tr, seen, order = [], [], set(), []
        for root, _dirs, files in os.walk(td):
            for fn in sorted(files):
                if fn == "input.txt":
                    continue
                path = os.path.join(root, fn)
                try:
                    blob = open(path, errors="ignore").read()
                except Exception:                                              # noqa: BLE001
                    continue
                kind = ("reported" if any(fn.endswith(s) for s in REPORTED) else
                        "trace" if any(fn.endswith(s) for s in TRACE) else "other")
                for m in re.findall(r"[ACGUacgu]{%d}" % n, blob):
                    s = m.upper().replace("T", "U")
                    if s in seen:
                        continue
                    seen.add(s)
                    order.append(kind)
                    if kind == "reported":
                        rep.append(s)
                    elif kind == "trace":
                        tr.append(s)
    return {"puzzle": pid, "tier": tl, "reported": rep, "trace": tr,
            "shipped_slice_kinds": order[:200], "wall_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=60)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "trajectory_contamination_pools.json"))
    a = ap.parse_args()

    cmd = os.environ.get("DESIRNA_CMD")
    assert cmd and os.path.exists(cmd.split()[-1]), "DESIRNA_CMD unset"
    print(f"求解器:{cmd}")

    T = targets()
    jobs = [(p, v["target"], a.tier, a.seed) for p, v in sorted(T.items())]
    print(f"{len(jobs)} 个靶标,时限 {a.tier}s(产品 standard 阶梯的最高一档),按来源文件区分候选\n",
          flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    R, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for n, f in enumerate(as_completed(futs), 1):
            R.append(f.result())
            json.dump({"tier": a.tier, "seed": a.seed, "targets": T, "rows": R},
                      open(a.out, "w"), ensure_ascii=False)
            if n % 10 == 0 or n == len(jobs):
                print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)

    import statistics as st
    rep = [len(r["reported"]) for r in R if "error" not in r]
    tr = [len(r["trace"]) for r in R if "error" not in r]
    frac = [sum(1 for k in r["shipped_slice_kinds"] if k == "trace") / max(1, len(r["shipped_slice_kinds"]))
            for r in R if "error" not in r]
    print(f"\n=== T2 产品实际取用的那 200 个名额里,轨迹噪声占多少 ===")
    print(f"  上报答案 中位 {st.median(rep):.0f} 条,轨迹状态 中位 {st.median(tr):.0f} 条")
    print(f"  200 名额中轨迹占比 中位 {st.median(frac):.1%}(范围 {min(frac):.1%}-{max(frac):.1%})")
    print(f"\n  wrote {a.out}   (下一步:用 rnet venv 跑 trajectory_contamination_score.py)")


if __name__ == "__main__":
    main()
