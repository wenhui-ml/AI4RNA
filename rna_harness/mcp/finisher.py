"""Strong inverse-folding PRIMITIVES exposed to the agent as safe, bounded MCP tools.

The agent (the meta_runner "layman-expert") ORCHESTRATES these — it decides when to decompose,
which sub-target to search, and how to close a near-miss. Meta-learning distills the ORCHESTRATION
STRATEGY (not answers). These are primitives, NOT a god-solver:
  - inverse_fold_search(target): many-restart RNA.inverse_fold on a (sub)target -> best seq.
  - finish(candidate, target): diagnosis-guided SA hill-climb from a near-miss -> closes last pairs.

Bounded (fixed restarts/iters, fork pool <= MAX_CORES) so the agent can't spawn runaway pools.
No cross-oracle / spawn -> no fork-bomb path.
"""
from __future__ import annotations

import random
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional

import RNA

from .compute import MAX_CORES
from .motif_kb import pair_table

_COMP = {"A": "U", "U": "A", "G": "C", "C": "G"}
_BASES = "ACGU"


def _fold(seq: str) -> str:
    return RNA.fold_compound(seq).mfe()[0]


def _mispaired(struct: str, target: str) -> List[int]:
    ps, pt = pair_table(struct), pair_table(target)
    return [i for i in range(len(target)) if ps[i] != pt[i]]


def _inv_worker(arg):
    target, seed, tries, warm = arg
    rng = random.Random(seed)
    n = len(target)
    best, bd = (warm, RNA.bp_distance(_fold(warm), target)) if warm else (None, 99)
    for k in range(tries):
        start = list(warm) if (warm and k % 3 == 0) else [rng.choice(_BASES) for _ in range(n)]
        for _ in range(rng.randint(0, 4)):
            start[rng.randrange(n)] = rng.choice(_BASES)
        try:
            seq, _ = RNA.inverse_fold("".join(start), target)
        except Exception:
            continue
        d = RNA.bp_distance(_fold(seq), target)
        if d < bd:
            bd, best = d, seq
        if bd == 0:
            break
    return {"seq": best, "bp_distance": bd}


def parallel_inverse_fold(target: str, warm: Optional[str] = None, n_restarts: int = 0, tries: int = 40) -> Dict:
    n_restarts = n_restarts if n_restarts > 0 else MAX_CORES
    nproc = min(MAX_CORES, n_restarts)
    args = [(target, 104729 * (k + 1), tries, warm) for k in range(n_restarts)]
    with ProcessPoolExecutor(max_workers=nproc) as ex:
        res = [r for r in ex.map(_inv_worker, args) if r.get("seq")]
    res.sort(key=lambda r: r["bp_distance"])
    best = res[0] if res else {"seq": warm or "", "bp_distance": 99}
    best["solved"] = (best["bp_distance"] == 0 and best["seq"] and _fold(best["seq"]) == target)
    return best


def _hillclimb(seq: str, target: str, iters: int, seed: int) -> Dict:
    rng = random.Random(seed)
    n = len(target)
    pt_t = pair_table(target)
    cur = list(seq)
    cur_st = _fold("".join(cur)); cur_d = RNA.bp_distance(cur_st, target)
    best, best_d, stall = cur[:], cur_d, 0
    for _ in range(iters):
        if best_d == 0:
            break
        bad = _mispaired(cur_st, target)
        i = rng.choice(bad) if (bad and rng.random() < 0.85) else rng.randrange(n)
        j = pt_t[i]
        saved = {i: cur[i]}
        cur[i] = rng.choice(_BASES)
        if j >= 0 and j != i and rng.random() < 0.75:
            saved[j] = cur[j]; cur[j] = _COMP[cur[i]]
        st = _fold("".join(cur)); d = RNA.bp_distance(st, target)
        if d < cur_d or (d == cur_d and rng.random() < 0.35):
            cur_d, cur_st = d, st
            if d < best_d:
                best_d, best, stall = d, cur[:], 0
            else:
                stall += 1
        else:
            for k, v in saved.items():
                cur[k] = v
            stall += 1
        if stall > 800:
            cur = best[:]
            for _ in range(rng.randint(1, 3)):
                cur[rng.randrange(n)] = rng.choice(_BASES)
            cur_st = _fold("".join(cur)); cur_d = RNA.bp_distance(cur_st, target); stall = 0
    out = "".join(best)
    return {"seq": out, "bp_distance": RNA.bp_distance(_fold(out), target)}


def _hc_worker(arg):
    return _hillclimb(*arg)


def parallel_finish(seq: str, target: str, n_restarts: int = 0, iters: int = 20000) -> Dict:
    n_restarts = n_restarts if n_restarts > 0 else MAX_CORES
    nproc = min(MAX_CORES, n_restarts)
    args = [(seq, target, iters, 7919 * (k + 1)) for k in range(n_restarts)]
    with ProcessPoolExecutor(max_workers=nproc) as ex:
        res = list(ex.map(_hc_worker, args))
    res.sort(key=lambda r: r["bp_distance"])
    best = res[0]
    best["solved"] = (best["bp_distance"] == 0 and _fold(best["seq"]) == target)
    return best


def decompose_solve(target: str, n_restarts: int = 0, per_unit_tries: int = 40,
                    seam_iters: int = 0, warm_full: bool = True, per_unit_restarts: int = 0) -> Dict:
    """DETERMINISTIC pipeline for LONG targets where whole-target search does not converge:
    decompose -> solve each SHORT stem-loop unit's sub_target by inverse_fold -> place each at its
    positions -> assign inter-unit/multiloop seam base-pairs G-C (alternating) & linkers A -> a
    seam-focused hill-climb repair. This removes the LLM's error-prone manual assembly/bookkeeping
    (drift, mis-placement) that capped the crutched agent at 4/14 on >250nt targets -- the assembly
    is now exact code; the LLM only orchestrates + handles a residual near-miss.
    Returns best sequence + bp_distance + per-unit convergence, so the caller sees WHERE it stands."""
    import time as _t
    from .motif_kb import decompose_structure
    n = len(target)
    pt = pair_table(target)
    dec = decompose_structure(target)
    seq = ["A"] * n
    covered = set()
    per_unit = []
    pur = per_unit_restarts if per_unit_restarts > 0 else min(8, MAX_CORES)  # short units don't need 24 restarts
    _tu = _t.time()
    for u in dec.get("units", []):
        a, b = u["positions"]
        r = parallel_inverse_fold(u["sub_target"], n_restarts=pur, tries=per_unit_tries)
        us = r.get("seq") or ""
        if len(us) == (b - a + 1):
            for k, ch in enumerate(us):
                seq[a + k] = ch
            covered.update(range(a, b + 1))
        per_unit.append({"positions": [a, b], "len": b - a + 1, "bp": r.get("bp_distance")})
    # seam base-pairs (multiloop-closing helices / inter-unit stems not inside a leaf unit): strong
    # G-C, alternating orientation to reduce register-slip between identical repeats. Unpaired -> A.
    gc = ("G", "C")
    tog = 0
    for i in range(n):
        j = pt[i]
        if j > i and (i not in covered or j not in covered):
            a1, a2 = (gc if tog % 2 == 0 else gc[::-1])
            seq[i], seq[j] = a1, a2
            tog += 1
    t_units = round(_t.time() - _tu, 1)
    assembled = "".join(seq)
    assembled_bp = RNA.bp_distance(_fold(assembled), target)
    it = seam_iters if seam_iters > 0 else max(20000, 120 * n)
    _tf = _t.time()
    best = parallel_finish(assembled, target, n_restarts=n_restarts, iters=it)
    # fallback: warm-start a FULL inverse_fold from the polished assembly if seam-repair stalled
    if warm_full and not best.get("solved"):
        wr = parallel_inverse_fold(target, warm=(best.get("seq") or assembled),
                                   n_restarts=n_restarts, tries=per_unit_tries)
        if wr.get("bp_distance", 99) < best.get("bp_distance", 99):
            best = wr
    t_finish = round(_t.time() - _tf, 1)
    return {"seq": best.get("seq"), "bp_distance": best.get("bp_distance"),
            "solved": bool(best.get("solved")), "assembled_bp": assembled_bp,
            "n_units": dec.get("n_units"), "per_unit": per_unit,
            "t_units": t_units, "t_finish": t_finish}
