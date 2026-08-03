#!/usr/bin/env python3
"""A judge that asks how CLOSE the molecule got, not whether it hit one exact structure.

WHY, and it is the same cause behind four separate dead ends this week. Every criterion in this project is
defined on EXACT structure equality: trap_fraction fires on any pair the target lacks, knot_confirmed demands
full reproduction at tau=1.0, the strict interaction rule demands four identical answers, and Kinfold --stop
demands landing on one exact microstate. Exactness is why the pseudoknot ruler works -- the register-shift
decoy is refused 0/386 precisely because one nucleotide of slip is not forgiven. It is also why everything
else stalled:

    Kinfold reaches nothing at >=100 nt and 100x the simulation budget does not help, because hitting one
    exact structure gets combinatorially unlikely as the molecule grows. 17 of 22 natural constructs came
    back with reach identically 0, so the judge -- not the criterion -- had no dynamic range, and the
    native-coherence question could not be asked at all.

The trajectory already contains the answer. Kinfold prints every accepted move; --stop only decides when to
quit. So the crossing is to read what was always there: over the full-length portion of each trajectory,
what is the SMALLEST base-pair distance to the target that the molecule actually visited?

    reach_0  == the old verdict, by construction
    reach_d  == fraction of trajectories that came within d base pairs

THIS IS NOT A SOFTER STANDARD UNLESS IT FAILS THE FOLLOWING, and that is checked before anything is claimed:

  1. AGREEMENT AT d=0. On the 25-70 nt population where the old judge worked, reach_0 from this parser must
     equal reach from Kinfold's own --stop accounting. Any disagreement means the parser is wrong, and a
     wrong parser that happens to produce a wider dynamic range would look exactly like success.

     THE CHECK FAILED AND THE PRE-REGISTRATION WAS MIS-SPECIFIED. It assumed the X marker is ground truth.
     Direct inspection of one disagreeing sequence: 20 trajectories, 11 marked X1 and 9 marked O, and ALL 20
     printed the exact target structure at full length. Trajectory 5 visited it and then moved on to
     `...(....)...`, ending O. So --stop does not fire on every visit to the stop structure under --grow.
     The parser sees a visit that the marker missed; reach_0 is higher than the marker on 27 of 60 sequences,
     never lower, mean +0.10.

     Consequence, and it reaches back over everything: EVERY Kinfold number in this project undercounts,
     because all of them were the marker. Whether the +16.1 point calibration gap survives a visit-based
     judge is a separate re-run, not something this file may assume.

  2. DYNAMIC RANGE, and only then. On the >=100 nt constructs where the old judge was pinned at zero,
     does reach_d actually vary across sequences? A judge that returns 1.0 for everything is as useless as
     one that returns 0.0 for everything, so BOTH degenerate directions are reported.

  3. THE NULLS MUST SURVIVE. Buying dynamic range with precision is the failure mode here. That check lives
     with the decoy suites, not in this file, and is stated in the report as a condition on using this judge
     for anything -- not as something this file establishes.

Nothing about the criterion changes here. This file changes only the instrument that judges it.
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
sys.path.insert(0, HERE)

KINFOLD = os.environ.get("KINFOLD", "/home/ubuntu/miniconda3/envs/vpk/bin/Kinfold")
SIM_TIME = 8000.0
GROW = 20.0
D_LIST = (0, 1, 2, 3, 5, 8, 12)


def pairs(struct: str):
    out, st = set(), []
    for i, ch in enumerate(struct):
        if ch == "(":
            st.append(i)
        elif ch == ")" and st:
            out.add((st.pop(), i))
    return out


def kinfold_neighbourhood(seq: str, target: str, traj: int = 20, grow: float = GROW,
                          sim_time: float = SIM_TIME, timeout_s: float = 900.0):
    """Smallest base-pair distance to the target visited at FULL LENGTH, per trajectory.

    Only full-length structures count. While the chain is still growing it cannot possibly contain the
    target's 3' pairs, so scoring partial structures would report a distance that is an artefact of
    transcription progress rather than of folding.

    Streamed rather than collected: one 46 nt sequence produces ~11k lines per trajectory, so holding 20
    trajectories of a 284 nt molecule in memory is not an option.
    """
    if not os.path.exists(KINFOLD):
        return None
    tgt = pairs(target)
    n = len(target)
    n_tgt = len(tgt)
    # A 284 nt molecule emits on the order of a million accepted moves, and building a pair set for every one
    # of them is O(n) in Python per line -- the first version simply never finished. Almost every visited
    # structure is far from the target, and |#pairs(s) - #pairs(target)| is a LOWER BOUND on base-pair
    # distance costing one C-level str.count. So the expensive set is built only for lines that could beat
    # the running best.
    best, per_traj, ended, exact = None, [], 0, 0
    scored = 0
    try:
        with tempfile.TemporaryDirectory() as d:
            p = subprocess.Popen(
                [KINFOLD, "--stop", "--fpt", "--grow", str(grow), "--glen", "10",
                 "--num", str(traj), "--time", str(sim_time)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, cwd=d)
            p.stdin.write(f"{seq}\n{target}\n")
            p.stdin.close()
            deadline = time.time() + timeout_s
            for line in p.stdout:
                if time.time() > deadline:
                    p.kill()
                    return None
                f = line.split()
                if not f:
                    continue
                s = f[0]
                if len(s) == n and s[0] in ".()":
                    lb = abs(s.count("(") - n_tgt)
                    if best is None or lb < best:
                        dist = len(pairs(s) ^ tgt)
                        best = dist if best is None else min(best, dist)
                        scored += 1
                if len(f) >= 4 and (f[-1] == "O" or f[-1].startswith("X")):
                    ended += 1
                    exact += f[-1].startswith("X")
                    per_traj.append(best if best is not None else 10 ** 6)
                    best = None
            p.wait(timeout=30)
    except Exception:                                                          # noqa: BLE001
        return None
    if ended == 0:
        return None
    out = {"n_traj": ended, "exact_by_marker": exact / ended, "n_scored": scored,
           "min_dist_median": sorted(per_traj)[len(per_traj) // 2]}
    for dd in D_LIST:
        out[f"reach_{dd}"] = sum(1 for x in per_traj if x <= dd) / ended
    return out


def work(job):
    kind, tid, seq, target, old = job
    r = kinfold_neighbourhood(seq, target, traj=(20 if kind == "short" else 4),
                              timeout_s=(300.0 if kind == "short" else 1800.0))
    return {"kind": kind, "task_id": tid, "n": len(target), "old_reach": old, "nb": r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-short", type=int, default=60)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(HERE, "neighbour_judge.json"))
    a = ap.parse_args()

    jobs = []
    # --- population 1: where the old judge worked, to prove d=0 reproduces it -------------------------
    from rna_harness.benchmarks.bprna_nested import load_bprna_nested
    big = json.load(open(os.path.join(HERE, "kinetic_calibrate_big.json")))
    tmap = {str(r["id"]): str(r["target"]) for r in load_bprna_nested(big.get("pool", 400))}
    seen = set()
    for r in big["rows"]:
        if r.get("error") or r["task_id"] in seen:
            continue
        tg = tmap.get(r["task_id"])
        for c in r["candidates"]:
            if c.get("kinfold") and tg and len(tg) == len(c["seq"]):
                jobs.append(("short", r["task_id"], c["seq"], tg, c["kinfold"]["reach"]))
                seen.add(r["task_id"])
                break
        if len(seen) >= a.n_short:
            break
    # --- population 2: where it was pinned at zero ----------------------------------------------------
    from experiment_check import parse_rdat
    ec = json.load(open(os.path.join(HERE, "experiment_check.json")))
    nc = json.load(open(os.path.join(HERE, "native_coherence.json")))
    dead = set(nc.get("flat_judge") or [])
    for r in ec["rows"]:
        if r.get("skipped") or r["rmdb_id"] not in dead:
            continue
        rec = parse_rdat(open(os.path.join(HERE, "rmdb_cache", f"{r['rmdb_id']}.rdat"),
                              encoding="utf8", errors="ignore").read())
        jobs.append(("long", r["rmdb_id"], rec["sequence"], r["target"], 0.0))
    print(f"邻域判决:{sum(1 for j in jobs if j[0]=='short')} 条短的(25–70 nt,老判决有效)"
          f" + {sum(1 for j in jobs if j[0]=='long')} 条长的(老判决恒为 0)\n"
          f"  先证 d=0 复现老判决,再看 d>0 是否恢复动态范围\n", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    R, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, f in enumerate(as_completed([ex.submit(work, j) for j in jobs]), 1):
            R.append(f.result())
            if n % 10 == 0 or n == len(jobs):
                print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)

    import numpy as np
    short = [r for r in R if r["kind"] == "short" and r["nb"]]
    long_ = [r for r in R if r["kind"] == "long" and r["nb"]]

    print(f"\n=== 检查 1:d=0 是否复现老判决({len(short)} 条)===")
    if short:
        d0 = np.array([r["nb"]["reach_0"] for r in short])
        mk = np.array([r["nb"]["exact_by_marker"] for r in short])
        old = np.array([r["old_reach"] for r in short])
        print(f"  本文件解析的 reach_0        vs  Kinfold 自己的 X 标记:"
              f"最大差 {np.abs(d0-mk).max():.4f}   完全相等 {int((np.abs(d0-mk)<1e-9).sum())}/{len(d0)}")
        print(f"  本文件解析的 reach_0        vs  归档里的老 reach   :"
              f"平均差 {np.abs(d0-old).mean():+.4f}   相关 {np.corrcoef(d0, old)[0,1]:.3f}")
        ok1 = bool(np.abs(d0 - mk).max() < 1e-9)
        print(f"  -> {'解析正确,d=0 与老判决同一件事' if ok1 else '不一致 —— 解析器有问题,后面的都不作数'}")
    else:
        ok1 = False
        print("  无短样本")

    print(f"\n=== 检查 2:长序列上动态范围是否恢复({len(long_)} 条,老判决全部 0)===")
    tab = []
    for dd in D_LIST:
        v = np.array([r["nb"][f"reach_{dd}"] for r in long_]) if long_ else np.array([])
        if v.size:
            tab.append({"d": dd, "mean": float(v.mean()), "sd": float(v.std()),
                        "n_zero": int((v == 0).sum()), "n_one": int((v == 1).sum())})
            print(f"  d={dd:>2}  均值 {v.mean():.3f}  标准差 {v.std():.3f}  "
                  f"恒 0 的 {int((v==0).sum())}/{len(v)}  恒 1 的 {int((v==1).sum())}/{len(v)}")
    # "not all zero" is far too weak: one construct out of seventeen being nonzero gives sd 0.059 and would
    # be reported as dynamic range recovered. A judge is usable only if a real share of the population sits
    # strictly between the two degenerate ends.
    good = [t for t in tab
            if t["sd"] > 0.05 and (len(long_) - t["n_zero"] - t["n_one"]) >= max(3, len(long_) // 3)]
    print(f"\n  两种退化都要报:全 0 无用,全 1 同样无用。")
    if good:
        b = min(good, key=lambda t: t["d"])
        print(f"  -> 最小的可用邻域是 d={b['d']}(标准差 {b['sd']:.3f})—— 动态范围恢复")
    else:
        print("  -> 没有一个 d 给出动态范围。换判决救不了这个总体,照实报。")
    if long_:
        md = [r["nb"]["min_dist_median"] for r in long_]
        print(f"  这些长序列实际走到的最近距离:中位 {sorted(md)[len(md)//2]} 个碱基对")

    json.dump({"agree_d0": ok1, "d_list": list(D_LIST), "long_table": tab,
               "usable_d": (min(t["d"] for t in good) if good else None),
               "rows": R}, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
