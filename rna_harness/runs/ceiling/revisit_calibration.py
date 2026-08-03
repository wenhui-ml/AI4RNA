#!/usr/bin/env python3
"""Does the second ruler's headline survive a judge that does not undercount?

WHAT CHANGED UNDER US. Every Kinfold number in this project read the X marker that --stop writes at the end of
a trajectory. Direct inspection while building the neighbourhood judge showed that marker is incomplete: on
one 60 nt sequence all 20 trajectories printed the exact target structure at full length, yet only 11 were
marked X1 -- the other 9 visited it, moved on, and ended O. Across 60 sequences the visit-based count is
higher on 27 and lower on none, mean +0.10.

So the judge behind the +16.1 point calibration gap UNDERCOUNTS, and there is no argument from first
principles about whether that matters: an undercount that hits both arms equally leaves the gap alone, and one
that hits the accepted arm harder shrinks it. Which it is has to be measured.

This re-runs the same population with reach defined as A VISIT rather than as a marker, and redoes the two
controls that mattered:

    within-target pairing   identical length and structure, so not a length effect
    GC pinned per stratum   same target AND same composition to four decimals, the control that answers
                            "does it collapse to GC", and the one that needed 79 strata to become established

DECLARED BEFORE THE RUN: the ruler survives if both controls keep the same sign and stay significant. If the
gap shrinks materially, the published +16.1 is withdrawn and replaced, exactly as +26.5 was.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from neighbour_judge import kinfold_neighbourhood                             # noqa: E402


def work(job):
    tid, seq, target, gc, tf, acc, old = job
    r = kinfold_neighbourhood(seq, target, traj=20, timeout_s=300.0)
    if not r:
        return None
    return {"task_id": tid, "n": len(target), "gc": gc, "trap_fraction": tf, "accepted": acc,
            "old_reach": old, "visit_reach": r["reach_0"], "marker_reach": r["exact_by_marker"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=700)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--src", default=os.path.join(HERE, "kinetic_calibrate_big.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "revisit_calibration.json"))
    a = ap.parse_args()

    from rna_harness.benchmarks.bprna_nested import load_bprna_nested
    d = json.load(open(a.src))
    tmap = {str(r["id"]): str(r["target"]) for r in load_bprna_nested(d.get("pool", 400))}
    jobs = []
    for r in d["rows"]:
        if r.get("error"):
            continue
        tg = tmap.get(r["task_id"])
        if not tg:
            continue
        for c in r["candidates"]:
            if c.get("kinfold") and len(tg) == len(c["seq"]):
                jobs.append((r["task_id"], c["seq"], tg, c["gc"], c["trap_fraction"],
                             c["accepted"], c["kinfold"]["reach"]))
    # keep every target represented rather than truncating the list, so the paired controls do not lose
    # whole targets to an arbitrary cut
    by = defaultdict(list)
    for j in jobs:
        by[j[0]].append(j)
    picked, i = [], 0
    while len(picked) < a.n and any(len(v) > i for v in by.values()):
        for k in list(by):
            if len(by[k]) > i and len(picked) < a.n:
                picked.append(by[k][i])
        i += 1
    print(f"用【访问过】而不是【标记】重判 {len(picked)} 条候选,来自 {len({p[0] for p in picked})} 个靶标\n",
          flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    R, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, f in enumerate(as_completed([ex.submit(work, j) for j in picked]), 1):
            r = f.result()
            if r:
                R.append(r)
            if n % 100 == 0 or n == len(picked):
                print(f"    {n}/{len(picked)}  {(time.time()-t0)/60:.1f} min", flush=True)

    import numpy as np
    from scipy.stats import spearmanr, wilcoxon
    vr = np.array([r["visit_reach"] for r in R])
    mr = np.array([r["marker_reach"] for r in R])
    print(f"\n=== 判决口径的差 ===")
    print(f"  访问式 {vr.mean():.3f}   标记式 {mr.mean():.3f}   平均低估 {(vr-mr).mean():+.3f}")
    print(f"  访问式更高的 {int((vr>mr).sum())}/{len(R)}   更低的 {int((vr<mr).sum())}")

    def controls(key):
        pairs_, strata = [], defaultdict(lambda: {"a": [], "r": []})
        byt = defaultdict(lambda: {"a": [], "r": []})
        for r in R:
            byt[r["task_id"]]["a" if r["accepted"] else "r"].append(r[key])
            strata[(r["task_id"], round(r["gc"], 4))]["a" if r["accepted"] else "r"].append(r[key])
        for v in byt.values():
            if v["a"] and v["r"]:
                pairs_.append((st.mean(v["a"]), st.mean(v["r"])))
        use = [v for v in strata.values() if v["a"] and v["r"]]
        gc_ = [(st.mean(v["a"]), st.mean(v["r"])) for v in use]
        out = {}
        for name, P in (("within_target", pairs_), ("gc_pinned", gc_)):
            if len(P) >= 6:
                A = np.array([p[0] for p in P]); B = np.array([p[1] for p in P])
                out[name] = {"n": len(P), "gap": float((A - B).mean()),
                             "p": float(wilcoxon(A, B, alternative="greater")[1])}
            else:
                out[name] = {"n": len(P), "gap": None, "p": None}
        rho, pr = spearmanr([r["trap_fraction"] for r in R], [r[key] for r in R])
        out["spearman"] = {"rho": float(rho), "p": float(pr)}
        return out

    old = controls("old_reach")
    new = controls("visit_reach")
    print(f"\n=== 两个控制,旧口径 vs 新口径 ===")
    for name, label in (("within_target", "同 target 内配对"), ("gc_pinned", "GC 逐条钉死")):
        o, n_ = old[name], new[name]
        if o["gap"] is None or n_["gap"] is None:
            print(f"  {label}:样本不足(旧 n={o['n']} 新 n={n_['n']})")
            continue
        print(f"  {label:16s} 旧 {o['gap']:+.1%} (p={o['p']:.2g}, n={o['n']})   "
              f"新 {n_['gap']:+.1%} (p={n_['p']:.2g}, n={n_['n']})")
    print(f"  Spearman(trap, reach)  旧 {old['spearman']['rho']:+.3f}   新 {new['spearman']['rho']:+.3f}")

    keep = all(new[k]["gap"] is not None and new[k]["gap"] > 0 and new[k]["p"] < 0.05
               for k in ("within_target", "gc_pinned"))
    print(f"\n  预注册:两个控制都同号且显著 -> "
          f"{'通过 —— 换掉低估的判决之后结论不变' if keep else '未通过 —— 撤回并改报新数字'}")
    json.dump({"n": len(R), "old": old, "new": new, "undercount": float((vr - mr).mean()),
               "survives": bool(keep), "rows": R}, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
