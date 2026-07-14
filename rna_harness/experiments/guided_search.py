"""AF1-style GUIDED, HARD-BOUNDED inverse-design search.

AF1 replaced a rugged physics search with descent on a SMOOTH potential -> fast convergence. Phase-A
(no learning) uses the exact physics potential's smooth signal: the partition-function base-pair
probabilities (bpp from RNA.pf) tell us WHICH target pairs are thermodynamically weakest, so we DIRECT
sequence moves at the weakest/most-defective sites instead of undirected random hillclimbing. The
search is HARD-BOUNDED (anytime; stops at an oracle-call budget or wall clock) so it CANNOT hang --
directly fixing the frozen-budget run's 45-min unbounded refine (root cause: RNA.inverse_fold has no
internal bound; this module never calls it). Efficiency metric = oracle-calls-to-solve (fewer = a
stronger search, the AF1-style "reach the answer with less compute" claim, and gaming-proof).

The trajectories this produces on the hard tail are the training data for Phase-B: a LEARNED
per-position nucleotide policy (GRPO) that guides the search like AF1's neural potential. This search
strengthening SERVES the learned agent (A_theta) + trajectory generation, NOT a standalone CPU solver.

RED LINE: designs under the folding oracle only; never reads a sample solution.
"""
from __future__ import annotations

import random
import time
from typing import Dict, List, Optional

import RNA

from ..structure_utils import pair_table
from ..mcp.motif_kb import decompose_structure

_BASES = "ACGU"
_COMP = {"A": "U", "U": "A", "G": "C", "C": "G"}
_PAIRS = [("G", "C"), ("C", "G"), ("A", "U"), ("U", "A"), ("G", "U"), ("U", "G")]


def guided_search(target: str, warm: Optional[str] = None, max_calls: int = 20000,
                  wall_s: float = 60.0, pf_every: int = 250, seed: int = 0,
                  p_weak: float = 0.7) -> Dict:
    """Bounded, bpp-guided anytime search. Counts every folding-oracle evaluation (mfe + pf). Returns
    best sequence, bp_distance, solved, and the oracle-call cost. Never hangs: hard caps on calls+wall.
    """
    t0 = time.time()
    n = len(target)
    pt = pair_table(target)
    rng = random.Random(seed)
    cur = list(warm) if (warm and len(warm) == n) else [rng.choice(_BASES) for _ in range(n)]
    mfe_calls = 0
    pf_calls = 0

    def fold_bp(seq_list):
        nonlocal mfe_calls
        mfe_calls += 1
        st, _ = RNA.fold("".join(seq_list))
        return st, RNA.bp_distance(st, target)

    def weak_pairs(seq_list):
        """Target pairs ranked weakest-first by partition-function probability (the guidance signal)."""
        nonlocal pf_calls
        pf_calls += 1
        fc = RNA.fold_compound("".join(seq_list))
        fc.pf()
        bpp = fc.bpp()  # 1-indexed
        probs = [(bpp[i + 1][pt[i] + 1], i, pt[i]) for i in range(n) if pt[i] > i]
        probs.sort()
        return [(i, j) for _p, i, j in probs]

    cur_st, cur_bp = fold_bp(cur)
    best, best_bp = cur[:], cur_bp
    weak_order: List = []
    since_pf = pf_every  # force a pf on the first iteration

    def budget_left():
        return (mfe_calls + pf_calls) < max_calls and (time.time() - t0) < wall_s

    while cur_bp > 0 and budget_left():
        if since_pf >= pf_every:
            weak_order = weak_pairs(cur)
            since_pf = 0
        since_pf += 1
        ps = pair_table(cur_st)
        unformed = [(i, j) for (i, j) in weak_order if ps[i] != pt[i] or ps[j] != pt[j]]
        saved: Dict[int, str] = {}
        if unformed and rng.random() < p_weak:
            # DIRECTED move: strengthen a thermodynamically weak, still-unformed target pair
            i, j = unformed[0] if rng.random() < 0.5 else rng.choice(unformed[:max(1, len(unformed) // 2)])
            a, b = rng.choice(_PAIRS)
            saved = {i: cur[i], j: cur[j]}
            cur[i], cur[j] = a, b
        else:
            # local repair: mutate a mispaired position (and its intended partner)
            bad = [k for k in range(n) if ps[k] != pt[k]] or list(range(n))
            i = rng.choice(bad)
            saved = {i: cur[i]}
            cur[i] = rng.choice(_BASES)
            j = pt[i]
            if j >= 0 and j != i:
                saved[j] = cur[j]
                cur[j] = _COMP[cur[i]]
        st, bp = fold_bp(cur)
        if bp <= cur_bp:                       # accept equal-or-better (anytime descent)
            cur_st, cur_bp = st, bp
            if bp < best_bp:
                best_bp, best = bp, cur[:]
        else:
            for k, v in saved.items():
                cur[k] = v
    bs = "".join(best)
    solved = best_bp == 0 and RNA.fold(bs)[0] == target
    return {"seq": bs, "bp_distance": best_bp, "solved": bool(solved),
            "mfe_calls": mfe_calls, "pf_calls": pf_calls, "calls": mfe_calls + pf_calls,
            "t_s": round(time.time() - t0, 1)}


def guided_solve(target: str, max_calls: int = 60000, wall_s: float = 90.0, seed: int = 0,
                 decompose_over: int = 120) -> Dict:
    """Divide-and-conquer wrapper: for long targets, guided-search each leaf unit (small, bounded),
    assemble with G-C seam pairs, then guided-search the full assembly for the remaining budget. Keeps
    big targets TRACTABLE and BOUNDED (no unbounded per-unit inverse_fold -> no hang). Anytime."""
    t0 = time.time()
    n = len(target)
    if n <= decompose_over:
        r = guided_search(target, max_calls=max_calls, wall_s=wall_s, seed=seed)
        r["mode"] = "monolithic"
        return r
    pt = pair_table(target)
    dec = decompose_structure(target)
    seq = ["A"] * n
    covered = set()
    unit_calls = 0
    per_unit_cap = max(2000, max_calls // (2 * max(1, len(dec.get("units", [])))))
    for u in dec.get("units", []):
        a, b = u["positions"]
        left = max_calls - unit_calls
        if left <= 0 or time.time() - t0 > wall_s * 0.6:
            break
        ur = guided_search(u["sub_target"], max_calls=min(per_unit_cap, left),
                           wall_s=max(3.0, wall_s * 0.5 / max(1, len(dec["units"]))), seed=seed)
        unit_calls += ur["calls"]
        us = ur["seq"]
        if len(us) == (b - a + 1):
            for k, ch in enumerate(us):
                seq[a + k] = ch
            covered.update(range(a, b + 1))
    tog = 0
    for i in range(n):
        j = pt[i]
        if j > i and (i not in covered or j not in covered):
            g = ("G", "C") if tog % 2 == 0 else ("C", "G")
            seq[i], seq[j] = g
            tog += 1
    # guided seam repair on the full assembly, remaining budget
    left = max_calls - unit_calls
    r = guided_search(target, warm="".join(seq), max_calls=max(2000, left),
                      wall_s=max(5.0, wall_s - (time.time() - t0)), seed=seed)
    r["mode"] = "decompose"
    r["unit_calls"] = unit_calls
    r["calls"] += unit_calls
    return r
