#!/usr/bin/env python3
"""The same retrieval library, through the MUTABLE-START channel instead of the hard-pin channel.

Yesterday's-shape result, for contrast: retrieval injected as DesiRNA's `seq_restr` IUPAC pin was net -21 --
it solved 4 of the 12 escalation targets but BROKE 25 of the 72 the plain cheap tier already solved, and the
canonical prior broke 30 of the same 24-ish targets. Pinning is destructive on its own; the retrieved content
was in fact mildly better than the prior at identical positions. Conclusion drawn there: the information is
real, the channel is wrong.

generators/desirna_launch.py supplies the missing channel by replacing
utils/sequence_utils.initial_sequence_generator (which only decides WHERE the walk starts) while leaving
get_nt_list (which decides what mutations are ALLOWED) untouched. Proof of concept: target 92 needed t=180
cold and solved at t=5 warm, from a start whose own bp (37) was WORSE than what the cold search reached at
t=5 (21), with 29/103 positions changed by the search -- so the start moved the basin, it was not a transplant.

THREE SETS, and C is the one that killed the pin channel:
  A  the 16 targets no tier solves, t=180            capability, upside-only
  B  the 12 escalation targets, cheap tier t=5       efficiency
  C  the 72 the plain cheap tier already solves      DOES A WARM START BREAK THEM? a mutable start should be
                                                     able to walk away from a bad basin; that is a prediction,
                                                     and predictions get measured

MATCHED CONTROL: `prior_warm` assembles the start with retrieval.py's canonical priors for EVERY unit -- an
EMPTY library run through the identical warm_start() assembly -- so it differs from `warm` only in whether the
unit sequences came from other solved targets. Same procedure, same positions, no transferred content.

Libraries are reused from runs/ceiling/retrieval_arm/_lib (built only from our own 84 verified solves, with
leave-one-out for every within-benchmark evaluation).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
OUT = os.path.join(HERE, "warmstart_arm")
LIB = os.path.join(HERE, "retrieval_arm", "_lib")
LAUNCH = os.path.join(REPO, "rna_harness", "generators", "desirna_launch.py")


def make_start(target, libpath, mode):
    from rna_harness.experiments.retrieval import RetrievalLibrary
    lib = RetrievalLibrary() if mode == "prior_warm" else RetrievalLibrary.load(libpath)
    seq, cov, meta = lib.warm_start(target, clamp_hits=(mode == "warm_clamp"))
    return seq, cov, meta


def _cell(pid, mode, tl, seed):
    """Seed 0 keeps the original name so the single-seed run already on disk is reused, not repeated."""
    return f"p{pid}_{mode}_t{tl}.json" if seed == 0 else f"p{pid}_{mode}_t{tl}_s{seed}.json"


def work(item):
    pid, target, mode, tl, libpath, seed = item
    dest = os.path.join(OUT, _cell(pid, mode, tl, seed))
    if os.path.exists(dest):
        return dest
    from rna_harness.cognitive.cost import solver_cost_usd
    from rna_harness.task import DesignTask, OracleConfig
    start, cov, meta = make_start(target, libpath, mode)
    os.environ["DESIRNA_CMD"] = f"{sys.executable} {LAUNCH}"
    os.environ["DESIRNA_INIT_SEQ"] = start
    from rna_harness.generators.desirna import DesiRNAGenerator
    t0 = time.time()
    try:
        res = DesiRNAGenerator(replicas=10, timelimit_s=tl, results_number=20).design(
            DesignTask(task_id=str(pid), target=target,
                       oracle=OracleConfig(param_set="turner2004", temperature=37.0, dangles=2)), seed=seed)
        w = time.time() - t0
        seq = res.best_sequence or None
        rec = {"id": pid, "n": len(target), "mode": mode, "t": tl, "seed": seed,
               "solved": bool(res.solved),
               "bp": int(res.best_bp_distance), "wall_s": round(w, 1),
               "cost_usd": solver_cost_usd(w, 10), "unit_cov": round(cov, 3),
               "start_bp": meta.get("assembled_bp"), "n_units": meta.get("n_units"),
               "n_hit": meta.get("n_hit"),
               "moved": (sum(1 for a, b in zip(seq, start) if a != b) if seq else None),
               "seq": seq}
    except Exception as e:                                                       # noqa: BLE001
        rec = {"id": pid, "mode": mode, "t": tl, "seed": seed, "error": repr(e)[:200]}
    with open(dest, "w") as f:
        json.dump(rec, f)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--hard-tier", type=int, default=180)
    ap.add_argument("--cheap-tier", type=int, default=5)
    ap.add_argument("--sets", default="A,B,C")
    ap.add_argument("--seeds", default="0", help="comma list; 0 reuses the original single-seed cells")
    ap.add_argument("--modes", default="warm,warm_clamp,prior_warm")
    ap.add_argument("--analyse-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    base = collections.defaultdict(dict)
    for f in glob.glob(os.path.join(HERE, "sota_curve", "*.json")):
        d = json.load(open(f))
        base[d["t"]][d["id"]] = d
    B = sorted(base)
    ids = sorted(base[B[0]])
    from rna_harness.benchmarks.eterna100 import load_eterna100
    tgt = {p["id"]: p["target"] for p in load_eterna100(version="V2")}
    solved_at = {}
    for i in ids:
        for t in B:
            if base[t][i].get("solved"):
                solved_at[i] = t
                break
    HOPELESS = [i for i in ids if i not in solved_at]
    ESCALATE = [i for i in ids if solved_at.get(i, 0) > B[0]]
    ALREADY = [i for i in ids if solved_at.get(i) == B[0]]
    # Set D closes the only downside risk left for warm-starting the LAST rung. In a cascade a target reaches
    # t=180 only after every cheaper rung failed, so there is no cheap-tier solve to break -- but cold t=180
    # itself solves 2 targets, and a warm start could break THOSE. Without D, "+2 at no risk" is unmeasured.
    AT_HARD = [i for i in ids if solved_at.get(i) == a.hard_tier]
    MODES = tuple(a.modes.split(","))
    SEEDS = [int(x) for x in a.seeds.split(",")]
    sets = set(a.sets.split(","))
    print(f"warm-start arm: A={len(HOPELESS)} hopeless @t={a.hard_tier}, B={len(ESCALATE)} escalation "
          f"@t={a.cheap_tier}, C={len(ALREADY)} already-solved @t={a.cheap_tier}; modes {MODES}")

    jobs = []
    if "A" in sets:
        jobs += [(i, tgt[i], m, a.hard_tier, os.path.join(LIB, "libA.json"), sd)
                 for i in HOPELESS for m in MODES for sd in SEEDS]
    if "B" in sets:
        jobs += [(i, tgt[i], m, a.cheap_tier, os.path.join(LIB, f"libB_{i}.json"), sd)
                 for i in ESCALATE for m in MODES for sd in SEEDS]
    if "D" in sets:
        from rna_harness.experiments.retrieval import RetrievalLibrary
        donors = [(i, base[solved_at[i]][i].get("seq")) for i in solved_at]
        donors = [(i, s_) for i, s_ in donors if s_]
        for i in AT_HARD:
            lp = os.path.join(LIB, f"libD_{i}.json")
            if not os.path.exists(lp):
                L = RetrievalLibrary()
                L.forbid([tgt[i]])
                for j, s_ in donors:
                    if j != i:
                        L.add_solution(tgt[j], s_, source="sota_curve")
                L.save(lp)
        jobs += [(i, tgt[i], m, a.hard_tier, os.path.join(LIB, f"libD_{i}.json"), sd)
                 for i in AT_HARD for m in MODES for sd in SEEDS]
    if "C" in sets:
        jobs += [(i, tgt[i], m, a.cheap_tier, os.path.join(LIB, f"libC_{i}.json"), sd)
                 for i in ALREADY for m in MODES for sd in SEEDS]
    if not a.analyse_only:
        todo = [j for j in jobs if not os.path.exists(os.path.join(OUT, _cell(j[0], j[2], j[3], j[5])))]
        print(f"  {len(todo)} cells to run, {a.workers} workers", flush=True)
        from concurrent.futures import ProcessPoolExecutor, as_completed
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(work, j) for j in todo]
            for n, _f in enumerate(as_completed(futs), 1):
                if n % 20 == 0 or n == len(todo):
                    el = time.time() - t0
                    print(f"    {n}/{len(todo)}  {el/60:.1f} min  "
                          f"eta {el/max(n,1)*(len(todo)-n)/60:.1f} min", flush=True)

    got = collections.defaultdict(lambda: collections.defaultdict(dict))
    for f in glob.glob(os.path.join(OUT, "*.json")):
        d = json.load(open(f))
        if "error" not in d:
            got[(d["id"], d["t"])][d["mode"]][d.get("seed", 0)] = d

    def arm(label, S, tier, baseline_txt):
        print(f"\n=== {label} ===")
        print(f"  {baseline_txt}")
        res = {}
        for m in MODES:
            cells = [got[(i, tier)][m].get(SEEDS[0]) for i in S]
            cells = [c for c in cells if c]
            if not cells:
                continue
            w = [c["id"] for c in cells if c["solved"]]
            mv = [c["moved"] for c in cells if c.get("moved") is not None]
            print(f"  {m:<12} {len(w):>3}/{len(cells):<3} solved   "
                  f"median start_bp {sorted(c['start_bp'] for c in cells)[len(cells)//2]:>4}   "
                  f"median moved {sorted(mv)[len(mv)//2] if mv else '-':>4}   "
                  f"{('-> ' + str(w[:12])) if w and len(w) <= 12 else ''}")
            res[m] = set(w)
        return res

    rA = arm(f"A. capability: {len(HOPELESS)} targets no tier solves, t={a.hard_tier}",
             HOPELESS, a.hard_tier, f"cold baseline: 0/{len(HOPELESS)}; hard-pin retrieval got 1 (target 97)")
    rB = arm(f"B. efficiency: {len(ESCALATE)} escalation targets at t={a.cheap_tier}",
             ESCALATE, a.cheap_tier,
             f"cold at t={a.cheap_tier}: 0/{len(ESCALATE)} by construction; hard-pin retrieval got 4, prior 2")
    rC = arm(f"C. breakage: {len(ALREADY)} targets the plain t={a.cheap_tier} tier already solves",
             ALREADY, a.cheap_tier,
             f"cold: {len(ALREADY)}/{len(ALREADY)} by construction; hard-pin retrieval BROKE 25, prior BROKE 30")

    if "D" in sets:
        rD = arm(f"D. breakage at the LAST rung: {len(AT_HARD)} targets cold t={a.hard_tier} solves",
                 AT_HARD, a.hard_tier,
                 f"cold t={a.hard_tier}: {len(AT_HARD)}/{len(AT_HARD)} by construction -> {AT_HARD}")
        print(f"\n=== the shippable design: warm-start ONLY the last rung ===")
        print(f"  a target reaching t={a.hard_tier} has already failed every cheaper rung, so the cheap-tier")
        print(f"  breakage measured in C cannot occur; the only risk is D.")
        for m in MODES:
            if m in rA and m in rD:
                gain, broke = len(rA[m]), len(AT_HARD) - len(rD[m])
                print(f"  {m:<12} A gains {gain}, D breaks {broke}  ->  net {gain-broke:+d} solves "
                      f"at ZERO extra cost (the rung is paid anyway)")

    if len(SEEDS) > 1:
        print(f"\n=== SEED REPLICATION: is the +1 stable, or was it three lucky events? ===")
        print(f"  decision rule declared BEFORE the run: ship iff A >= 2 on EVERY seed and D breaks <= 1 "
              f"on every seed.")
        print(f"  {'mode':<12} {'seed':>5} {'A solved/16':>12} {'D broken/2':>11} {'net':>5}")
        verdict = {}
        for m in MODES:
            nets = []
            for sd in SEEDS:
                ca = [got[(i, a.hard_tier)][m].get(sd) for i in HOPELESS]
                ca = [c for c in ca if c]
                cd = [got[(i, a.hard_tier)][m].get(sd) for i in AT_HARD]
                cd = [c for c in cd if c]
                if not ca or not cd:
                    continue
                gain = sum(1 for c in ca if c["solved"])
                broke = sum(1 for c in cd if not c["solved"])
                nets.append((sd, gain, broke))
                print(f"  {m:<12} {sd:>5} {gain:>12} {broke:>11} {gain-broke:>+5d}")
            if nets:
                ok = all(g >= 2 and b <= 1 for _s, g, b in nets)
                verdict[m] = (ok, nets)
                print(f"  {m:<12} -> {'SHIP' if ok else 'DO NOT SHIP'}   "
                      f"gains {[g for _s, g, _b in nets]}, breaks {[b for _s, _g, b in nets]}, "
                      f"nets {[g-b for _s, g, b in nets]}")

    print(f"\n=== channel vs channel, net at the cheap tier ===")
    print(f"  {'channel':<26} {'B gained':>9} {'C broken':>9} {'net':>6}")
    print(f"  {'hard pin, retrieval':<26} {4:>9} {25:>9} {4-25:>6}")
    print(f"  {'hard pin, prior':<26} {2:>9} {30:>9} {2-30:>6}")
    for m in MODES:
        if m in rB and m in rC:
            broke = len(ALREADY) - len(rC[m])
            print(f"  {'mutable start, ' + m:<26} {len(rB[m]):>9} {broke:>9} {len(rB[m])-broke:>6}")
    if "warm" in rB and "prior_warm" in rB:
        wo = rB["warm"] - rB["prior_warm"]
        po = rB["prior_warm"] - rB["warm"]
        from math import comb
        n = len(wo) + len(po)
        pv = min(1.0, 2.0 * sum(comb(n, i) for i in range(min(len(wo), len(po)) + 1)) / 2.0 ** n) if n else 1.0
        print(f"\n  B, retrieval vs its matched prior control: warm-only {sorted(wo)}, "
              f"prior-only {sorted(po)}, exact McNemar p = {pv:.3f}")

    json.dump({"hopeless": HOPELESS, "escalate": ESCALATE, "already": ALREADY, "modes": list(MODES),
               "cells": {f"{k[0]}_{k[1]}_{m}": v for k, d in got.items() for m, v in d.items()}},
              open(os.path.join(HERE, "warmstart_arm.json"), "w"), indent=1)
    print(f"\n  wrote rna_harness/runs/ceiling/warmstart_arm.json")


if __name__ == "__main__":
    main()
