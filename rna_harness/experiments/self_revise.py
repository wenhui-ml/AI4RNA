"""Self-revising, cognitive-core-structured solver loop for RNA inverse design.

Design principle (Karpathy Cognitive Core + Wang&Buehler self-revising discovery, arXiv 2606.01444):
the agent's next action is conditioned on RICH STATE (C_t = per-position defect + weak stems +
ensemble defect + what the sequence WANTS to fold into) and a VERIFIER (V = ViennaRNA), and it
carries a library of transferable OPERATORS (A_theta). A single (representation, operator-pref)
pair is a REGIME; searching within it is `search`. When the verifier shows a DEFECT PLATEAU
(no residual reduction for K steps), the loop performs a *verified regime transition*: it re-types
the problem, transports the already-correct substructure into the new regime (warm-start = the
left-Kan-transport analog), and ACCEPTS the transition only if V shows residual defect below the
old plateau (the MDL/AIC-style discovery gate). This is orchestration value, NOT more brute force:
every controller gets the SAME per-op budget and the SAME total op count, so any lift is the
sequencing of representations, not extra compute.

Controllers (the decision policy = who plays A_theta's role):
  - blind      : one regime (inverse_fold+finish), no diagnosis, no transitions   [BASELINE]
  - roundrobin : cycle regimes on plateau, diagnosis-BLIND choice                  [ABLATION]
  - greedy     : choose the next regime/operator FROM the diagnosis                [MECHANISM]
  - (agent)    : an LLM picks, given the same diagnosis  -- added later            [SCIENCE]

RED LINE: designs from scratch under the folding oracle only. Never reads any sample solution.
No in-vivo claim. Bounded pools (<= per-op restarts) so it stays a polite CPU neighbour.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Tuple

import RNA

from ..mcp.finisher import parallel_inverse_fold, parallel_finish, decompose_solve, _fold
from ..mcp.motif_kb import pair_table

_BASES = "ACGU"
_COMP = {"A": "U", "U": "A", "G": "C", "C": "G"}


# ----------------------------------------------------------------------------- diagnosis (C_t)
def diagnose(seq: str, target: str) -> Dict:
    """Rich state the controller conditions on. Uses the partition function (bpp) so we see WHICH
    target features are thermodynamically weak, not just pass/fail. All exact under ViennaRNA."""
    n = len(target)
    pt = pair_table(target)
    st = _fold(seq)
    ps = pair_table(st)
    bp = RNA.bp_distance(st, target)
    fc = RNA.fold_compound(seq)
    fc.pf()
    bpp = fc.bpp()  # 1-indexed (n+1)x(n+1)
    # per-target-pair probability of being formed; low => weak stem the design fails to enforce
    pair_probs: List[Tuple[int, int, float]] = []
    for i in range(n):
        j = pt[i]
        if j > i:
            pair_probs.append((i, j, float(bpp[i + 1][j + 1])))
    weak_pairs = sorted(pair_probs, key=lambda x: x[2])  # weakest first
    mispaired = [i for i in range(n) if ps[i] != pt[i]]
    # group contiguous mispaired runs -> "defect regions" (a region ~ one broken stem/loop)
    regions: List[Tuple[int, int]] = []
    if mispaired:
        s = p = mispaired[0]
        for k in mispaired[1:]:
            if k == p + 1:
                p = k
            else:
                regions.append((s, p)); s = p = k
        regions.append((s, p))
    ed = fc.ensemble_defect(target) if hasattr(fc, "ensemble_defect") else None
    return {
        "bp_distance": bp, "solved": bp == 0 and st == target, "struct": st,
        "n_mispaired": len(mispaired), "defect_regions": regions,
        "weak_pairs": weak_pairs[:6], "min_pair_prob": (weak_pairs[0][2] if weak_pairs else 1.0),
        "ensemble_defect": (round(ed, 3) if ed is not None else None),
    }


# ----------------------------------------------------------------------------- operators (A_theta)
def _rand_seq(n: int, seed: int) -> str:
    import random
    rng = random.Random(seed)
    return "".join(rng.choice(_BASES) for _ in range(n))


def op_inverse_fold(target: str, warm: Optional[str], budget: Dict) -> Dict:
    return parallel_inverse_fold(target, warm=warm, n_restarts=budget["restarts"], tries=budget["tries"])


def op_finish(seq: str, target: str, budget: Dict) -> Dict:
    return parallel_finish(seq, target, n_restarts=budget["restarts"], iters=budget["iters"])


def op_decompose(target: str, budget: Dict) -> Dict:
    r = decompose_solve(target, n_restarts=budget["restarts"], per_unit_tries=budget["tries"],
                        seam_iters=budget["iters"], warm_full=False)
    return {"seq": r.get("seq"), "bp_distance": r.get("bp_distance"), "solved": bool(r.get("solved"))}


def op_gc_strengthen(seq: str, target: str, weak_pairs, budget: Dict) -> Dict:
    """Force the weakest target pairs to strong G-C (alternating), then local-repair. A cheap
    'the stem won't hold' fix that blind restart-search rarely targets."""
    s = list(seq)
    tog = 0
    for i, j, _p in weak_pairs:
        a, b = ("G", "C") if tog % 2 == 0 else ("C", "G")
        s[i], s[j] = a, b
        tog += 1
    return parallel_finish("".join(s), target, n_restarts=budget["restarts"], iters=budget["iters"])


def op_close_pair(seq: str, target: str, diag: Optional[Dict] = None, budget: Optional[Dict] = None) -> Dict:
    """ENDGAME operator for near-misses (bp<=~3): the failures cluster at bp=1 where blind restart
    search stalls one pair from the target. Fix ONLY the unformed target pairs + break register-slip
    intruders inside a small focus window, folding each candidate — the exact local fix a global
    hill-climb rarely lands. (1) systematic: assign the worst unformed pairs each of the 6 canonical
    pairs; (2) focused partner-aware stochastic repair restricted to the defect neighbourhood."""
    import itertools
    import random
    n = len(target)
    pt = pair_table(target)
    d = diag or diagnose(seq, target)
    if d["solved"]:
        return {"seq": seq, "bp_distance": 0, "solved": True}
    ps = pair_table(d["struct"])
    unformed = [(i, pt[i]) for i in range(n) if pt[i] > i and ps[i] != pt[i]]
    if not unformed:
        return {"seq": seq, "bp_distance": d["bp_distance"], "solved": False}
    focus = set()
    for i, j in unformed:
        for p in (i, j):
            for q in (p - 1, p, p + 1):
                if 0 <= q < n:
                    focus.add(q)
            if 0 <= ps[p] < n:  # the intruder p currently mispairs with
                focus.add(ps[p])
    best = list(seq)
    best_d = d["bp_distance"]

    def _score(cand):
        s = "".join(cand)
        return s, RNA.bp_distance(_fold(s), target)

    PAIRS = [("G", "C"), ("C", "G"), ("A", "U"), ("U", "A"), ("G", "U"), ("U", "G")]
    for combo in itertools.product(PAIRS, repeat=min(len(unformed), 3)):
        cand = best[:]
        for (i, j), (a, b) in zip(unformed[:3], combo):
            cand[i], cand[j] = a, b
        s, dd = _score(cand)
        if dd < best_d:
            best_d, best = dd, cand[:]
        if dd == 0:
            return {"seq": s, "bp_distance": 0, "solved": _fold(s) == target}
    rng = random.Random(20260710)
    cur, cur_d = best[:], best_d
    flist = sorted(focus)
    for _ in range(min((budget or {}).get("iters", 6000), 4000)):
        i = rng.choice(flist)
        saved = {i: cur[i]}
        cur[i] = rng.choice(_BASES)
        j = pt[i]
        if j >= 0 and j != i:
            saved[j] = cur[j]
            cur[j] = _COMP[cur[i]]
        s, dd = _score(cur)
        if dd <= cur_d:
            cur_d = dd
            if dd < best_d:
                best_d, best = dd, cur[:]
            if dd == 0:
                return {"seq": s, "bp_distance": 0, "solved": _fold(s) == target}
        else:
            for k, v in saved.items():
                cur[k] = v
    bs = "".join(best)
    return {"seq": bs, "bp_distance": best_d, "solved": best_d == 0 and _fold(bs) == target}


# ----------------------------------------------------------------------------- regimes
# a regime = a way to (re)type the problem + which operator drives it. Transport = warm-start.
REGIMES = ["monolithic", "hierarchical", "gc_targeted", "close_pair"]


def _run_regime(regime: str, target: str, warm: Optional[str], diag: Optional[Dict], budget: Dict) -> Dict:
    if regime == "monolithic":
        r = op_inverse_fold(target, warm, budget)
        if not r.get("solved"):
            f = op_finish(r.get("seq") or (warm or _rand_seq(len(target), 1)), target, budget)
            if f["bp_distance"] < r.get("bp_distance", 99):
                r = f
        return r
    if regime == "hierarchical":
        r = op_decompose(target, budget)
        if warm and not r.get("solved"):
            f = op_finish(warm, target, budget)
            if f["bp_distance"] < r.get("bp_distance", 99):
                r = f
        return r
    if regime == "gc_targeted":
        base = warm or op_inverse_fold(target, None, budget).get("seq") or _rand_seq(len(target), 2)
        wp = (diag or diagnose(base, target))["weak_pairs"]
        return op_gc_strengthen(base, target, wp, budget)
    if regime == "close_pair":
        base = warm or op_inverse_fold(target, None, budget).get("seq")
        if not base:
            return {"seq": None, "bp_distance": 99, "solved": False}
        return op_close_pair(base, target, diag, budget)
    raise ValueError(regime)


# ----------------------------------------------------------------------------- controllers
def _pick_greedy(diag: Dict, tried: List[str]) -> str:
    """Choose the next regime FROM the diagnosis. Near-miss (small bp) -> close_pair endgame;
    weak stems (low pair prob) -> gc_targeted; many separate defect regions -> hierarchical
    (re-type into units); else monolithic. This diagnosis-timed ordering is greedy's ONLY edge
    over roundrobin (which cycles the same regimes blind)."""
    order = []
    if diag.get("bp_distance", 99) <= 3:
        order.append("close_pair")
    if diag.get("min_pair_prob", 1.0) < 0.5:
        order.append("gc_targeted")
    if len(diag.get("defect_regions", [])) >= 3:
        order.append("hierarchical")
    order += ["close_pair", "monolithic", "hierarchical", "gc_targeted"]
    for r in order:
        if r not in tried:
            return r
    return order[0]


def solve(target: str, controller: str = "greedy", seed: int = 0, n_ops: int = 10,
          restarts: int = 6, tries: int = 40, iters: int = 6000, plateau_k: int = 2,
          wall_s: float = 150.0) -> Dict:
    """Run the self-revising loop under one controller. All controllers get the same n_ops budget
    and per-op (restarts,tries,iters), so a difference is orchestration, not compute."""
    t0 = time.time()
    budget = {"restarts": restarts, "tries": tries, "iters": iters}
    n = len(target)
    best_seq, best_bp = None, 99
    transitions = 0
    trace: List[Dict] = []

    if controller == "blind":
        # one regime, alternate inverse_fold / finish, no diagnosis, no transition
        warm = _rand_seq(n, seed)
        for k in range(n_ops):
            if time.time() - t0 > wall_s:
                break
            r = op_inverse_fold(target, warm if k else None, budget) if k % 2 == 0 \
                else op_finish(warm, target, budget)
            if r.get("bp_distance", 99) < best_bp:
                best_bp, best_seq = r["bp_distance"], r["seq"]
            warm = r.get("seq") or warm
            trace.append({"op": k, "regime": "monolithic", "bp": r.get("bp_distance")})
            if best_bp == 0:
                break
        return _result(target, best_seq, best_bp, transitions, trace, t0, controller, seed)

    # self-revising controllers (roundrobin / greedy)
    regime = "monolithic"
    tried: List[str] = []
    warm: Optional[str] = None
    plateau_bp = 99
    stall = 0
    for k in range(n_ops):
        if time.time() - t0 > wall_s:
            break
        diag = diagnose(warm, target) if warm else None
        r = _run_regime(regime, target, warm, diag, budget)
        bp = r.get("bp_distance", 99)
        improved = bp < best_bp
        if improved:
            best_bp, best_seq = bp, r["seq"]
        # transport: carry best-so-far into the next regime as warm-start
        if r.get("seq") and bp <= best_bp + 1:
            warm = r["seq"]
        trace.append({"op": k, "regime": regime, "bp": bp, "best": best_bp})
        if best_bp == 0:
            break
        # plateau detection on the VERIFIER's best -> verified regime transition
        if bp >= plateau_bp:
            stall += 1
        else:
            stall = 0
        plateau_bp = min(plateau_bp, bp)
        if stall >= plateau_k and k < n_ops - 1:
            tried.append(regime)
            nxt_diag = diagnose(warm, target) if warm else {"min_pair_prob": 1.0, "defect_regions": []}
            if controller == "greedy":
                regime = _pick_greedy(nxt_diag, tried)
            else:  # roundrobin: diagnosis-blind
                remaining = [x for x in REGIMES if x not in tried] or REGIMES
                regime = remaining[transitions % len(remaining)]
            transitions += 1
            stall = 0
            plateau_bp = best_bp  # accept-gate: new regime must beat the old plateau to matter
    return _result(target, best_seq, best_bp, transitions, trace, t0, controller, seed)


def _result(target, seq, bp, transitions, trace, t0, controller, seed) -> Dict:
    solved = bool(seq) and bp == 0 and _fold(seq) == target
    return {"controller": controller, "seed": seed, "solved": solved, "bp_distance": bp,
            "transitions": transitions, "n_ops": len(trace), "t_s": round(time.time() - t0, 1),
            "seq": seq if solved else None, "trace": trace}


def diag_state(warm: Optional[str], target: str, k: int, n_ops: int, ops_used: List[str]) -> Dict:
    """The exact rich state handed to the AGENT each step (what it conditions its A_theta on).
    Facts only — never the answer sequence (red line)."""
    d = diagnose(warm, target) if warm else None
    return {
        "op_index": k, "budget_left": n_ops - k, "length": len(target),
        "have_candidate": warm is not None,
        "bp_distance": (d["bp_distance"] if d else None),
        "n_mispaired": (d["n_mispaired"] if d else None),
        "n_defect_regions": (len(d["defect_regions"]) if d else None),
        "min_pair_prob": (round(d["min_pair_prob"], 3) if d else None),
        "weak_pairs": ([[i, j, round(p, 3)] for i, j, p in d["weak_pairs"][:4]] if d else None),
        "ensemble_defect": (d["ensemble_defect"] if d else None),
        "ops_used_so_far": ops_used,
    }, d


def agent_solve(target: str, choose_fn, seed: int = 0, n_ops: int = 6, restarts: int = 6,
                tries: int = 40, iters: int = 6000, wall_s: float = 220.0) -> Dict:
    """AGENT-DRIVEN controller: the LLM (choose_fn) picks the operator each step FROM the diagnosis.
    Identical operator library + budget as solve('greedy'); ONLY the selection policy differs
    (LLM A_theta vs the hand-coded _pick_greedy). Measures whether the agent EXERCISES situation-aware
    A_theta selection to match a competent policy. choose_fn(state)->regime-name (or None on failure)."""
    t0 = time.time()
    budget = {"restarts": restarts, "tries": tries, "iters": iters}
    best_seq, best_bp = None, 99
    warm = None
    ops_log: List[str] = []
    n_invalid = 0
    for k in range(n_ops):
        if time.time() - t0 > wall_s:
            break
        state, diag = diag_state(warm, target, k, n_ops, ops_log)
        pick = choose_fn(state)
        if pick in REGIMES:
            regime = pick
        else:                                   # invalid / empty LLM decision -> honest fallback
            n_invalid += 1
            regime = "monolithic"
        ops_log.append(regime if pick in REGIMES else f"<invalid:{pick}>->monolithic")
        r = _run_regime(regime, target, warm, diag, budget)
        bp = r.get("bp_distance", 99)
        if bp < best_bp:
            best_bp, best_seq = bp, r["seq"]
        if r.get("seq") and bp <= best_bp + 1:
            warm = r["seq"]
        if best_bp == 0:
            break
    solved = bool(best_seq) and best_bp == 0 and _fold(best_seq) == target
    return {"controller": "agent", "seed": seed, "solved": solved, "bp_distance": best_bp,
            "n_ops": len(ops_log), "n_invalid_decisions": n_invalid, "ops": ops_log,
            "t_s": round(time.time() - t0, 1), "seq": best_seq if solved else None}
