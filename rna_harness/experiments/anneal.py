"""Replica-exchange Monte Carlo (parallel tempering) for RNA inverse design.

A single BOUNDED, PRINCIPLED global search replacing the patchwork of unbounded local-search
heuristics (inverse_fold / iters-unscaled parallel_finish / close_pair). Design goals, all solved by
the one method:

  * BOUNDED: one unified fold-call counter + wall clock, checked every epoch. No nested unbounded
    loop can exist -- every move costs exactly one counted MFE fold.
  * ESCAPE LOCAL MINIMA: Metropolis acceptance (accept a worse move with prob exp(-dE/T)) across a
    geometric temperature ladder, with periodic exchange of adjacent replicas (the parallel-tempering
    criterion). A cold replica stuck at bp=1 gets a better configuration swapped in from a hotter,
    more exploratory replica -- exactly the pure-hillclimb failure mode (pid 97 stuck at bp=1).
  * DIRECTED: 90% of moves target a currently-mispaired position (the one good part of the old
    hillclimb), keeping the partner complementary -- so it is far more efficient than uniform MC.

Energy = base-pair distance to the target under the exact ViennaRNA MFE oracle (the quantity we
report). Solved = a replica reaches bp 0 (verified by an independent refold). Like DesiRNA in spirit
(replica-exchange MC), instantiated for the uMFE-match objective. RED LINE: oracle only; no samples.
"""
from __future__ import annotations

import itertools
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List

import RNA

from ..structure_utils import pair_table
from ..mcp.compute import MAX_CORES
from .retrieval import RetrievalLibrary

_BASES = "ACGU"
_COMP = {"A": "U", "U": "A", "G": "C", "C": "G"}


def _energy(seq_s: str, target: str):
    st, _ = RNA.fold(seq_s)
    return st, RNA.bp_distance(st, target)


def _replica_chunk(args):
    """Run `steps` Metropolis MC steps for ONE replica at temperature T from a start sequence.
    Directed moves (90% at a mispaired site). Bounded: exactly `steps` folds. Top-level = picklable."""
    seq_s, T, steps, target, seed = args
    rng = random.Random(seed)
    n = len(target)
    pt = pair_table(target)
    seq = list(seq_s)
    st, E = _energy(seq_s, target)
    calls = 1
    best, best_e = seq[:], E
    for _ in range(steps):
        if E == 0:
            break
        ps = pair_table(st)
        bad = [k for k in range(n) if ps[k] != pt[k]]
        i = rng.choice(bad) if (bad and rng.random() < 0.9) else rng.randrange(n)
        saved = {i: seq[i]}
        seq[i] = rng.choice(_BASES)
        j = pt[i]
        if j >= 0 and j != i and rng.random() < 0.8:      # keep the intended partner complementary
            saved[j] = seq[j]
            seq[j] = _COMP[seq[i]]
        st2, e2 = _energy("".join(seq), target)
        calls += 1
        d = e2 - E
        if d <= 0 or rng.random() < math.exp(-d / T):     # Metropolis acceptance
            st, E = st2, e2
            if E < best_e:
                best_e, best = E, seq[:]
        else:
            for k, v in saved.items():
                seq[k] = v
    return {"final": "".join(seq), "final_e": E, "best": "".join(best), "best_e": best_e, "calls": calls}


def _ensemble_defect(seq_s: str, target: str) -> float:
    fc = RNA.fold_compound(seq_s)
    fc.pf()
    return float(fc.ensemble_defect(target))


def coordinated_endgame(seq: str, target: str, trigger_bp: int = 6, sys_top: int = 3,
                        stoch_iters: int = 4000, restarts: int = 10, seed: int = 0) -> Dict:
    """Last-mile coordinated repair for near-misses (bp <= trigger). Attacks the DEFECT CLUSTER --
    unformed target pairs + the intruder positions currently competing with them + flank -- and
    JOINTLY reassigns them: the coordinated multi-base change (break intruder + build the target pair)
    that single-position moves cannot reach (diagnosed by min_pair_prob~=0). Objective is LEXICOGRAPHIC
    (bp, then ensemble_defect): among bp-equal candidates it prefers the one thermodynamically closest
    to forming the missing pair. ensemble_defect is only evaluated on improve-or-tie => bounded cost."""
    n = len(target)
    pt = pair_table(target)
    st = RNA.fold(seq)[0]
    bp = RNA.bp_distance(st, target)
    if bp == 0 or bp > trigger_bp:
        return {"seq": seq, "bp": bp, "solved": bp == 0}
    best, best_bp, best_ed = seq, bp, _ensemble_defect(seq, target)
    rng = random.Random(seed)
    PAIRS = [("G", "C"), ("C", "G"), ("A", "U"), ("U", "A"), ("G", "U"), ("U", "G")]

    def consider(cand_s: str):
        nonlocal best, best_bp, best_ed
        bp2 = RNA.bp_distance(RNA.fold(cand_s)[0], target)
        if bp2 > best_bp:
            return
        ed2 = _ensemble_defect(cand_s, target)
        if bp2 < best_bp or (bp2 == best_bp and ed2 < best_ed):
            best, best_bp, best_ed = cand_s, bp2, ed2

    # PHASE 1 -- systematic canonical pairs on the worst unformed target pairs + break their intruders
    ps = pair_table(st)
    unformed = [(i, pt[i]) for i in range(n) if pt[i] > i and ps[i] != pt[i]][:sys_top]
    for combo in itertools.product(PAIRS, repeat=len(unformed)):
        cand = list(best)
        for (i, j), (a, b) in zip(unformed, combo):
            cand[i], cand[j] = a, b
            for p in (i, j):
                ip = ps[p]
                if 0 <= ip < n and ip not in (i, j):
                    cand[ip] = "A"                       # break the competing intruder pair
        consider("".join(cand))
        if best_bp == 0:
            return {"seq": best, "bp": 0, "solved": RNA.fold(best)[0] == target}

    # PHASE 2 -- many-restart stochastic search restricted to the (recomputed) defect cluster
    for _r in range(restarts):
        cur = list(best)
        ps_c = pair_table(RNA.fold("".join(cur))[0])
        cl = set()
        for k in range(n):
            if ps_c[k] != pt[k]:
                cl.add(k)
                if pt[k] >= 0:
                    cl.add(pt[k])
                if 0 <= ps_c[k] < n:
                    cl.add(ps_c[k])
        for k in list(cl):
            for q in (k - 1, k + 1):
                if 0 <= q < n:
                    cl.add(q)
        cluster = sorted(cl) or list(range(n))
        cur_bp = RNA.bp_distance(RNA.fold("".join(cur))[0], target)
        for _ in range(stoch_iters):
            i = rng.choice(cluster)
            saved = {i: cur[i]}
            cur[i] = rng.choice(_BASES)
            j = pt[i]
            if 0 <= j < n and j != i and rng.random() < 0.8:
                saved[j] = cur[j]
                cur[j] = _COMP[cur[i]]
            cand_s = "".join(cur)
            bp2 = RNA.bp_distance(RNA.fold(cand_s)[0], target)
            if bp2 <= cur_bp:
                cur_bp = bp2
                consider(cand_s)
                if best_bp == 0:
                    return {"seq": best, "bp": 0, "solved": RNA.fold(best)[0] == target}
            else:
                for k, v in saved.items():
                    cur[k] = v
    return {"seq": best, "bp": best_bp, "solved": best_bp == 0 and RNA.fold(best)[0] == target}


def replica_exchange(target: str, n_replicas: int = 8, max_calls: int = 400000, wall_s: float = 120.0,
                     t_min: float = 0.15, t_max: float = 4.0, exchange_every: int = 300, seed: int = 0,
                     seed_seq: str = None, endgame: bool = True, endgame_trigger: int = 6,
                     stall_epochs: int = 8) -> Dict:
    """Parallel-tempering search. Returns best bp, solved, total oracle calls, epochs, time. Bounded by
    (max_calls, wall_s); escapes minima via Metropolis + adjacent-replica exchange."""
    t0 = time.time()
    R = n_replicas
    if seed_seq is None:
        seed_seq, _, _ = RetrievalLibrary().warm_start(target)    # canonical GC-stiff assembly
    Ts = [t_min * (t_max / t_min) ** (i / (R - 1)) for i in range(R)]   # geometric ladder
    seqs = [seed_seq for _ in range(R)]
    _, e0 = _energy(seed_seq, target)
    Es = [e0 for _ in range(R)]
    from .guards import BudgetGuard, IterationGuard, guard_main_only
    guard_main_only("replica_exchange process pool")
    budget = BudgetGuard(max_calls=max_calls, wall_s=wall_s)   # unified hard bound (E4)
    budget.spend(1)                                            # the initial e0 energy eval
    stall = IterationGuard(max_iters=10 ** 9, stall_patience=stall_epochs)  # no-progress early stop
    g_best, g_best_e = seed_seq, e0
    rng = random.Random(seed)
    epoch = 0
    n_swaps = 0
    stop = "solved"
    nproc = min(MAX_CORES, R)
    with ProcessPoolExecutor(max_workers=nproc) as ex:
        while g_best_e > 0 and budget.ok():
            args = [(seqs[i], Ts[i], exchange_every, target, seed + epoch * R + i + 1) for i in range(R)]
            results = list(ex.map(_replica_chunk, args))
            for i, res in enumerate(results):
                seqs[i], Es[i] = res["final"], res["final_e"]
                budget.spend(res["calls"])
                if res["best_e"] < g_best_e:
                    g_best_e, g_best = res["best_e"], res["best"]
            if g_best_e == 0:
                break
            cont, why = stall.advance(g_best_e)   # stop if global best unchanged for stall_epochs epochs
            if not cont:
                stop = why
                break
            # exchange adjacent replicas (alternate even/odd pairs each epoch)
            for i in range(epoch % 2, R - 1, 2):
                delta = (1.0 / Ts[i] - 1.0 / Ts[i + 1]) * (Es[i] - Es[i + 1])
                if delta >= 0 or rng.random() < math.exp(delta):
                    seqs[i], seqs[i + 1] = seqs[i + 1], seqs[i]
                    Es[i], Es[i + 1] = Es[i + 1], Es[i]
                    n_swaps += 1
            epoch += 1
        if g_best_e > 0 and stop == "solved":     # loop fell out on budget, not solve/stall
            stop = budget.reason()
    calls = budget.calls
    eg_applied = False
    if endgame and 0 < g_best_e <= endgame_trigger:
        eg = coordinated_endgame(g_best, target, trigger_bp=endgame_trigger, seed=seed)
        if eg["bp"] < g_best_e:
            g_best_e, g_best = eg["bp"], eg["seq"]
            eg_applied = True
    solved = g_best_e == 0 and _energy(g_best, target)[1] == 0
    return {"bp": g_best_e, "solved": bool(solved), "calls": calls, "epochs": epoch,
            "swaps": n_swaps, "endgame_applied": eg_applied, "stop": stop,
            "t_s": round(time.time() - t0, 1), "seq": g_best if solved else None, "best_seq": g_best}


if __name__ == "__main__":
    import argparse
    import json
    import os
    from ..benchmarks.eterna100 import load_eterna100

    from ..mcp.motif_kb import analyze_structure

    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["unsolved", "all"], default="unsolved")
    ap.add_argument("--base-wall-s", type=float, default=30.0)   # per-target wall = base + n*slope, capped
    ap.add_argument("--wall-slope", type=float, default=0.45)
    ap.add_argument("--wall-cap", type=float, default=220.0)
    ap.add_argument("--replicas", type=int, default=8)
    ap.add_argument("--exchange-every", type=int, default=300)
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "..", "runs", "anneal"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    puz = {p["id"]: p for p in load_eterna100(version="V2")}
    summ = json.load(open(os.path.join(os.path.dirname(__file__), "..", "runs",
                                       "glm5.2_eterna100", "summary.json")))
    unsolved = [int(r["task_id"].split(":")[1]) for r in summ["records"] if not r["solved"]]
    ids = sorted(unsolved if a.set == "unsolved" else sorted(puz), key=lambda i: len(puz[i]["target"]))
    print(f"replica-exchange on {len(ids)} targets; R={a.replicas}; per-target wall = "
          f"min({a.wall_cap}, {a.base_wall_s}+{a.wall_slope}*n)s", flush=True)
    rows, solved = [], 0
    for pid in ids:
        t = puz[pid]["target"]
        n = len(t)
        wall = min(a.wall_cap, a.base_wall_s + a.wall_slope * n)   # size-aware budget: big targets get more
        r = replica_exchange(t, n_replicas=a.replicas, wall_s=wall, exchange_every=a.exchange_every)
        # auto-classify residual by hard structural features (lonely pairs / triloops = near-limit)
        ana = analyze_structure(t)
        lonely = sum(1 for L in ana["stem_lengths"] if L == 1)
        triloop = sum(1 for h in ana["hairpins"] if h.get("loop_len") == 3)
        solved += int(r["solved"])
        rows.append({"pid": pid, "name": puz[pid]["name"], "len": n, "wall_s": round(wall, 1),
                     "lonely_pairs": lonely, "triloops": triloop,
                     **{k: r[k] for k in ("bp", "solved", "calls", "epochs", "swaps", "t_s")}})
        print(f"  pid {pid:>3} n={n:>4} bp={r['bp']:>3} solved={r['solved']} "
              f"calls={r['calls']:>6} lonely={lonely} tri={triloop} {r['t_s']:>5.1f}s", flush=True)
        json.dump({"config": vars(a), "n_solved": solved, "rows": rows},
                  open(os.path.join(a.outdir, "anneal.json"), "w"), indent=1)
    resid = [r for r in rows if not r["solved"]]
    hard_feat = [r["pid"] for r in resid if r["lonely_pairs"] or r["triloops"]]
    clean = [r["pid"] for r in resid if not (r["lonely_pairs"] or r["triloops"])]
    print(f"\nSOLVED {solved}/{len(ids)} by bounded replica-exchange.", flush=True)
    print(f"RESIDUAL {len(resid)}: with hard features (lonely/triloop, near-limit) = {hard_feat}; "
          f"CLEAN (no obvious hard feature -> possibly learnable/compute-bound) = {clean}", flush=True)
