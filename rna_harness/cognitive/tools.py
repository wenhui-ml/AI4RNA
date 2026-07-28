"""Uniform tool interface. Every solver is a Tool; `tool.attempt(target, seed)` returns a ToolResult
carrying a COST. Verification uses the CORRECT oracle (pKiss for pseudoknots, ViennaRNA MFE otherwise)
so 'solved' is never a false positive. ALL domain knowledge lives here (in the wrapped tools); the
router/core above is domain-agnostic and only ever sees features + (solved, cost)."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Optional

from ..structure_utils import has_pseudoknot
from ..task import DesignTask, OracleConfig
from .cost import solver_cost_usd

_TURNER2004 = OracleConfig(param_set="turner2004", temperature=37.0, dangles=2)
# 'pfe' strictness: the target's Boltzmann probability must clear this to count as solved (robust fold,
# not a knife-edge single-structure match). Declarable via env; only used when strictness == "pfe".
_PFE_MIN = float(os.environ.get("COGCORE_PFE_MIN", "0.90"))


@dataclass
class ToolResult:
    tool: str
    solved: bool
    bp: int
    wall_s: float
    cores: int
    cost_usd: float             # solver CPU cost + oracle/verify cost (the FULL price of the attempt)
    oracle: str                 # 'mfe' | 'mfe:umfe' | 'mfe:pfe' | 'pkiss' | 'none'
    seq: Optional[str]          # only set when solved (verified)
    available: bool = True
    oracle_s: float = 0.0       # wall spent in verification (negligible for MFE; can dominate for pKiss)
    stats: Optional[dict] = None  # optional per-attempt budget accounting (oracle calls, iterations, ...)


def verify(seq: Optional[str], target: str, strictness: str = "match"):
    """Authoritative verification: the DECLARED decidable predicate evaluated on `seq`. The truth value
    of "does seq solve target?" is obtained by EVALUATION (fold once, compare) -- SAT-analogous: fold is
    the total computable function, `==` the decidable check. Returns (solved, bp, oracle_name).

    Three axes of the verifier live here:
      * WHICH oracle (axis 1): pseudoknot target -> pKiss; nested -> ViennaRNA MFE. Picking wrong (MFE on
        a pseudoknot) is a category error -- MFE literally cannot emit '[', so it is INCAPABLE of the truth.
      * HOW STRICT (axis 2), MFE targets only -- `strictness`:
          'match' fold(seq)==target                        (exact match; default, knife-edge)
          'umfe'  match AND target is the UNIQUE MFE        (closes the degenerate-co-optimal loophole)
          'pfe'   P(target) in the Boltzmann ensemble >= _PFE_MIN  (robust fold, not a fragile 1-of-many)
      * NO oracle (axis 3): unjudgeable input (e.g. length mismatch, or -- outside this fn -- a functional
        spec with no computable predicate) returns oracle 'none' -> the core must ABSTAIN, never claim solved.
    Pseudoknot targets are always judged by pKiss; `strictness` applies only to the ViennaRNA MFE path."""
    if not seq or len(seq) != len(target):
        return False, len(target), "none"
    if has_pseudoknot(target):
        from ..benchmarks.rnainvbench import verify_pk    # pKiss; pseudoknot-aware
        r = verify_pk(seq, target)
        return bool(r.get("solved")), int(r.get("bp", len(target))), "pkiss"
    import RNA
    st, _ = RNA.fold(seq)
    bp = RNA.bp_distance(st, target)
    match = (st == target)
    if strictness == "match":
        return match, bp, "mfe"
    if strictness == "umfe":
        fc = RNA.fold_compound(seq)
        _, mfe_e = fc.mfe()
        # uMFE = target is the sole ground state. We count ONLY structures at the MFE energy, so a
        # narrow window (0.01 kcal/mol = MFE +/- rounding) is both correct and BOUNDED. A wide band
        # (e.g. subopt(50) = 0.5 kcal/mol) enumerates an exponential set on large/degenerate seqs --
        # all discarded by the <1e-6 filter -- and hangs the scaled multi-seed run. subopt(1) yields
        # a bit-identical n_coopt (same filtered set) without the blow-up.
        n_coopt = sum(1 for s in fc.subopt(1) if abs(s.energy - mfe_e) < 1e-6)  # co-optimal @ MFE
        return (match and n_coopt == 1), bp, "mfe:umfe"
    if strictness == "pfe":
        fc = RNA.fold_compound(seq)
        _, mfe_e = fc.mfe()
        fc.exp_params_rescale(mfe_e)
        fc.pf()
        return (fc.pr_structure(target) >= _PFE_MIN), bp, "mfe:pfe"
    raise ValueError(f"unknown strictness {strictness!r}; use 'match' | 'umfe' | 'pfe'")


class Tool:
    name = "tool"
    cores = 1
    budget_s = 0.0

    def nominal_cost_usd(self) -> float:
        """A-priori worst-case cost of firing this tool = cores x its budget seconds. The router
        orders actions cheapest-first by this before it has outcome data."""
        return solver_cost_usd(self.budget_s, self.cores)

    def attempt(self, target: str, seed: int = 0) -> ToolResult:  # pragma: no cover - interface
        raise NotImplementedError


class RemcTool(Tool):
    """Our replica-exchange MC (anneal.py), bounded by a wall-second budget."""

    def __init__(self, wall_s: float, n_replicas: int = 8):
        self.wall_s_budget = float(wall_s)
        self.budget_s = float(wall_s)
        self.cores = int(n_replicas)
        self.name = f"remc@{int(wall_s)}s"

    def attempt(self, target: str, seed: int = 0) -> ToolResult:
        from ..experiments.anneal import replica_exchange
        # REMC optimises the ViennaRNA MFE structure, which CANNOT represent pseudoknots. On a pk target
        # it designs the nested SKELETON (its best achievable) -- anneal/warm-start assume nested and would
        # crash on '[' otherwise -- and pKiss then judges the FULL target, so REMC fails GRACEFULLY on a
        # pseudoknot (the crossing stem is out of its model) instead of raising.
        solve_target = "".join(c if c in "()." else "." for c in target) if has_pseudoknot(target) else target
        t0 = time.time()
        r = replica_exchange(solve_target, n_replicas=self.cores, wall_s=self.wall_s_budget, seed=seed)
        wall = time.time() - t0
        seq = r.get("best_seq")
        v0 = time.time()
        solved, bp, oracle = verify(seq, target)      # re-verify (anneal optimises MFE; pk => pKiss)
        ov = time.time() - v0
        cost = solver_cost_usd(wall, self.cores) + solver_cost_usd(ov, cores=1)   # solver + oracle
        return ToolResult(self.name, solved, bp, round(wall, 3), self.cores,
                          cost, oracle, seq if solved else None, oracle_s=round(ov, 3))


class DesirnaTool(Tool):
    """External SOTA DesiRNA via the oracle-verified subprocess adapter, bounded by its -t timelimit."""

    def __init__(self, timelimit_s: float, replicas: int = 10, results_number: int = 20,
                 panel_screen: bool = False):
        self.tl = int(timelimit_s)
        self.budget_s = float(timelimit_s)
        self.cores = int(replicas)
        self.results_number = int(results_number)
        # DesiRNA emits `results_number` candidates per call and this tool has always kept ONE. The
        # candidate loop below screens the rest with `verify`, which on a pseudoknot target means pKiss
        # alone -- the same in-loop oracle DesiRNA is effectively optimising -- and stops at the first
        # hit. Measured consequence (runs/ceiling/who_rejects.py, and desirna_seqs/panel_always_sota.json):
        # DesiRNA reaches 43/60 under pKiss and 0/60 under the three-oracle panel.
        #
        # With panel_screen=True every candidate is put to the FULL panel and the loop keeps going past a
        # pKiss hit. This is the cheapest available form of "add an information source": ShapeKnots costs
        # 65 ms, so screening 20 candidates costs about 1.3 s against a 180 s design budget -- under 1%.
        # It is also the form the evidence favours: ranking candidates by a cheap surrogate and taking the
        # best gave 0.187 where screening all of them gave 0.580 (runs/ceiling/disagree.py). Screening
        # beats steering, because screening is cheap.
        self.panel_screen = bool(panel_screen)
        self.name = f"desirna@{int(timelimit_s)}s" + ("+panel" if panel_screen else "")

    def attempt(self, target: str, seed: int = 0) -> ToolResult:
        from ..generators.desirna import DesiRNAGenerator, desirna_available
        if not desirna_available():
            return ToolResult(self.name, False, len(target), 0.0, self.cores, 0.0, "none", None,
                              available=False)
        t0 = time.time()
        res = DesiRNAGenerator(replicas=self.cores, timelimit_s=self.tl,
                               results_number=self.results_number).design(
            DesignTask(task_id="cog", target=target, oracle=_TURNER2004), seed=seed)
        wall = time.time() - t0
        best_seq = res.best_sequence or None
        ov = 0.0
        v0 = time.time()
        solved, bp, oracle = verify(best_seq, target)
        ov += time.time() - v0
        cands = list(getattr(res, "candidates", None) or [])
        n_scr = n_unan = 0
        if self.panel_screen and has_pseudoknot(target):
            from ..oracles_pk import forms_target, pkiss_pairs, probknot_pairs, shapeknots_pairs
            # The panel is a CONJUNCTION, so a short-circuit should test cheapest-first. Which oracle is
            # cheapest DEPENDS ON LENGTH, and getting that wrong is expensive: a fixed ShapeKnots-first
            # order was 2.7-4.1x cheaper on two smoke-test targets and 3.3x WORSE on a third.
            #
            # runs/ceiling/oracle_cost.py, 14 targets over 21-62 nt, ms = a*(n/30)^b:
            #     pKiss       146 ms at n=30, exponent 0.14   (essentially flat)
            #     ProbKnot     95 ms at n=30, exponent 0.26   (essentially flat)
            #     ShapeKnots   60 ms at n=30, exponent 1.90   (QUADRATIC -- 223 ms at 60, 588 at 100)
            #
            # So ShapeKnots is the cheapest oracle only below about 45 nt and the most expensive above it.
            # An earlier claim of mine, "ShapeKnots costs 0.31x pKiss", came from a 21-42 nt sample and is
            # not general. Ordering from the fitted curve rather than from a constant.
            def _ms(n, a, b):
                return a * (max(n, 1) / 30.0) ** b
            _n = len(target)
            ORDER = tuple(x[1:] for x in sorted(
                ((_ms(_n, 59.9, 1.90), "shapeknots", shapeknots_pairs),
                 (_ms(_n, 94.7, 0.26), "probknot", probknot_pairs),
                 (_ms(_n, 146.0, 0.14), "pkiss", pkiss_pairs))))
            n_calls = {k: 0 for k, _ in ORDER}
            win = None
            for c in ([best_seq] if best_seq else []) + [c for c in cands if c != best_seq]:
                v0 = time.time()
                n_scr += 1
                ok = True
                for k, fn in ORDER:
                    n_calls[k] += 1
                    if not forms_target(fn(c), target):
                        ok = False
                        break
                ov += time.time() - v0
                if ok:
                    n_unan += 1
                    if win is None:
                        win = c
            solved, best_seq, oracle = (win is not None), win, "panel:pkiss+probknot+shapeknots"
            bp = 0 if win is not None else bp
        elif has_pseudoknot(target) and not solved:
            # legacy path: screens with the IN-LOOP oracle only (pKiss) and stops at the first hit
            for c in cands:
                v0 = time.time()
                s, b, o = verify(c, target)
                ov += time.time() - v0
                if b < bp:
                    solved, bp, oracle, best_seq = s, b, o, c
                if solved:
                    break
        cost = solver_cost_usd(wall, self.cores) + solver_cost_usd(ov, cores=1)   # solver + oracle (pKiss dominates)
        return ToolResult(self.name, solved, bp, round(wall, 3), self.cores,
                          cost, oracle, best_seq if solved else None, oracle_s=round(ov, 3),
                          stats={"n_candidates": len(cands), "n_screened": n_scr,
                                 "n_unanimous": n_unan, "panel_screen": self.panel_screen,
                                 **({"oracle_calls": n_calls} if self.panel_screen
                                    and has_pseudoknot(target) else {})})


def default_toolbox() -> List[Tool]:
    """The action set the router chooses from: cheap->expensive, our-solver + SOTA. Each (tool,budget)
    is one action; the router learns which to fire (and in what order) per problem, to minimise cost."""
    return [
        RemcTool(wall_s=5),
        RemcTool(wall_s=30),
        RemcTool(wall_s=120),
        DesirnaTool(timelimit_s=60),
        DesirnaTool(timelimit_s=180),
    ]
