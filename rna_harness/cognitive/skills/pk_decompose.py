"""DecomposePKTool -- a distilled PROCEDURE for pseudoknot design: decompose -> compose -> pKiss-guided repair.

WHY THIS IS A NEW CAPABILITY (not a reordering). The baseline toolbox is {REMC, DesiRNA}, and BOTH optimise
the ViennaRNA MFE energy -- neither ever calls the pseudoknot-aware oracle (pKiss) DURING search. So on a
pseudoknot target the crossing stem is only ever satisfied incidentally, and the last base-pair is never
repaired against the true oracle. On the full-251 Pseudobase++ baseline DesiRNA leaves 71 residual, of which
35 are bp<=2 near-misses (23 at bp=1). This tool attacks exactly that gap with the ONE move the toolbox lacks:
optimise directly against pKiss.

THE PROCEDURE (transferable, zero memorised answers):
  1. DECOMPOSE  target T -> nested skeleton round_only(T) (the '()' part) + crossing pairs (the '[]{}' layers).
     By construction of extended dot-bracket every position belongs to at most one bracket family, so the
     crossing positions are DISJOINT from the nested ones -- they can be set without disturbing the skeleton pairing.
  2. SUB-SOLVE  the nested skeleton cheaply with REMC (bounded wall).
  3. COMPOSE    enforce Watson-Crick complementarity on each crossing pair (seed-varied) -- the constraint MFE
                cannot optimise but the pseudoknot needs.
  4. VERIFY     fold the FULL T with pKiss (the correct oracle). solved <=> pk_bp_distance == 0.
  5. REPAIR     pKiss-in-the-loop stochastic hill-climb: localise the mismatch (symmetric difference of folded
                vs target pair SETS), mutate only wrong positions (a target-paired position is mutated together
                with its partner to keep complementarity), accept on bp decrease. Bounded by a wall budget.

Cost = REMC CPU-seconds + pKiss oracle-seconds (pKiss dominates), via cost.solver_cost_usd -- so this slots
directly into the cost-to-solve metric alongside REMC/DesiRNA. On a NON-pseudoknot target the tool abstains
cheaply (it has nothing to add). All domain knowledge stays here; the router/core never see a nucleotide.
"""
from __future__ import annotations

import random
import time
from typing import Dict, List, Optional, Set, Tuple

from ..cost import solver_cost_usd
from ..tools import Tool, ToolResult

_WC = [("G", "C"), ("C", "G"), ("A", "U"), ("U", "A")]


def _partner_map(pairs: Set[Tuple[int, int]]) -> Dict[int, int]:
    m: Dict[int, int] = {}
    for i, j in pairs:
        m[i] = j
        m[j] = i
    return m


def _wrong_positions(folded_pairs: Set[Tuple[int, int]], target_pairs: Set[Tuple[int, int]],
                     n: int, halo: int = 2) -> List[int]:
    """Positions to mutate to fix the mismatch = positions touched by a pair present in exactly one of
    {folded, target}, PLUS a +/-`halo` window around each. The halo matters: a single missing terminal
    pair (i,j) with no spurious pair leaves the raw mismatch = {i,j}; but the FIX is usually to
    strengthen the adjacent stack or weaken a competing fold NEAR the pair, so mutating only i,j is a
    dead end. Widening to the local context is what lets the walk actually close the last base-pair."""
    core: Set[int] = set()
    for i, j in folded_pairs ^ target_pairs:
        core.add(i)
        core.add(j)
    pos: Set[int] = set()
    for p in core:
        for q in range(max(0, p - halo), min(n, p + halo + 1)):
            pos.add(q)
    return sorted(pos)


class DecomposePKTool(Tool):
    """Pseudoknot-only skill. Cheap-abstains on nested targets (nothing to add there)."""

    def __init__(self, skel_wall_s: float = 15.0, repair_wall_s: float = 60.0,
                 max_repair_iters: int = 400, skel_restarts: int = 6, n_replicas: int = 1,
                 ablate: str = "none"):
        self.skel_wall_s = float(skel_wall_s)
        self.repair_wall_s = float(repair_wall_s)
        self.max_repair_iters = int(max_repair_iters)
        self.skel_restarts = int(skel_restarts)
        # P2 component ablation: 'none' (full) | 'norepair' (no pKiss-guided repair loop) |
        # 'nocompose' (skip WC on crossing pairs) | 'nodecompose' (random init, no CF skeleton).
        self.ablate = str(ablate)
        # single-core: the nested skeleton is solved by ViennaRNA inverse_fold (no process pool -> safe to
        # run MANY of these under an OUTER pool; n_replicas kept only for signature compatibility).
        self.cores = 1
        # a-priori worst-case wall for cold-start ordering (skeleton solve + repair budget)
        self.budget_s = self.skel_wall_s + self.repair_wall_s
        self.name = f"pkdecomp@{int(self.repair_wall_s)}s"

    # -- helpers ------------------------------------------------------------
    def _abstain(self, target: str) -> ToolResult:
        """Non-pseudoknot target: this skill has nothing to offer. Fail cheaply (near-zero cost)."""
        return ToolResult(self.name, False, len(target), 0.0, 1, 0.0, "none", None, available=True)

    # -- main ---------------------------------------------------------------
    def attempt(self, target: str, seed: int = 0, mfe_screen=None, fixed=None) -> ToolResult:
        """Optional (backward-compatible, default off):
          * mfe_screen (int): F5 verifier-cost routing -- before each EXPENSIVE pKiss call in repair, screen
            the candidate with the CHEAP MFE oracle; skip pKiss if its nested fold is >mfe_screen bp from the
            skeleton (a broken nested part can't be a real pseudoknot solve). Cuts oracle cost.
          * fixed (dict pos->base): E2 constrained design -- IUPAC sequence restraints; these positions are
            pinned (set at compose, never mutated in repair)."""
        import RNA
        from ...structure_utils import has_pseudoknot
        from ...benchmarks.rnainvbench import pk_pairs, verify_pk, pkiss_available
        fixed = fixed or {}
        fixed_set = set(fixed)

        if not has_pseudoknot(target):
            return self._abstain(target)
        if not pkiss_available():
            return ToolResult(self.name, False, len(target), 0.0, self.cores, 0.0, "none", None,
                              available=False)

        n = len(target)
        rng = random.Random(seed)
        target_pairs = pk_pairs(target)
        skeleton = "".join(c if c in "()." else "." for c in target)   # the nested '()' reduction
        skel_pairs = pk_pairs(skeleton)
        cross_pairs = sorted(target_pairs - skel_pairs)                # the crossing '[]{}' base pairs
        partner = _partner_map(target_pairs)

        # 1-2. sub-solve the nested skeleton with ViennaRNA inverse_fold (the skeleton is pure nested, so
        #      this is exactly the right tool -- single-process, no pool, safe under an outer pool). A few
        #      random restarts; an approximate skeleton is fine since the pKiss repair below finishes the job.
        t0 = time.time()
        if self.ablate == "nodecompose":
            best_skel = "".join(rng.choice("ACGU") for _ in range(n))   # ablate: no CF-skeleton guidance
        else:
            best_skel, best_dist = "A" * n, float("inf")
            for _ in range(max(1, self.skel_restarts)):
                start = "".join(rng.choice("ACGU") for _ in range(n))
                s, dist = RNA.inverse_fold(start, skeleton)
                if dist < best_dist:
                    best_skel, best_dist = s, dist
                if best_dist <= 0 or (time.time() - t0) > self.skel_wall_s:
                    break
        seq = list(best_skel)
        if len(seq) != n:                                              # defensive: pad/trim to length
            seq = (seq + ["A"] * n)[:n]

        # 3. compose: enforce Watson-Crick complementarity on every crossing pair (seed-varied choice)
        if self.ablate not in ("nocompose", "nodecompose"):
            for k, (i, j) in enumerate(cross_pairs):
                a, b = _WC[(seed + k) % 4]
                seq[i], seq[j] = a, b
        for p, base in fixed.items():                                  # E2: pin the IUPAC-fixed positions
            if 0 <= p < n:
                seq[p] = base
        best_seq = "".join(seq)
        solver_wall = time.time() - t0
        n_pkiss = n_screened = 0                                       # F5 accounting

        # 4. verify with the correct oracle (pKiss)
        oracle_s = 0.0
        v0 = time.time()
        v = verify_pk(best_seq, target)
        oracle_s += time.time() - v0
        n_pkiss += 1
        best_bp = v["bp"] if v.get("bp") is not None else n
        solved = bool(v.get("solved"))
        # walker state: cur_* is the incumbent we random-walk from; best_* is the global best returned.
        cur_seq, cur_bp, cur_folded = best_seq, best_bp, v.get("folded")

        # 5. targeted pKiss-guided repair (the move the MFE-only toolbox lacks). Best-frontier random walk:
        #    accept any move that does not WORSEN bp (downhill always; sideways on the plateau) so we
        #    explore many bp=1 states -- escaping the local optimum a strict hill-climb gets stuck in.
        it = 0
        t_rep = time.time()
        n_cross = len(cross_pairs)
        while (self.ablate != "norepair" and not solved and cur_bp and cur_bp > 0
               and it < self.max_repair_iters and (time.time() - t_rep) < self.repair_wall_s):
            it += 1
            cand = list(cur_seq)
            # occasional restart move: re-randomise ALL crossing-stem pairs (the pseudoknot-specific part)
            if n_cross and rng.random() < 0.10:
                for i, j in cross_pairs:
                    a, b = rng.choice(_WC)
                    cand[i], cand[j] = a, b
            else:
                fpairs = pk_pairs(cur_folded) if cur_folded else set()
                wrong = _wrong_positions(fpairs, target_pairs, n) or list(range(n))
                pos = rng.choice(wrong) if rng.random() < 0.85 else rng.randrange(n)  # 15% global disruption
                if pos in fixed_set or partner.get(pos) in fixed_set:  # E2: never mutate a pinned position
                    continue
                if pos in partner:                                     # target-paired -> mutate the whole pair
                    j = partner[pos]
                    a, b = rng.choice(_WC)
                    cand[pos], cand[j] = a, b
                else:                                                  # unpaired -> A/U-biased single mutation
                    cand[pos] = rng.choice(["A", "U", "A", "U", "G", "C"])
            for p, base in fixed.items():                              # E2: re-pin (crossing-restart may hit them)
                if 0 <= p < n:
                    cand[p] = base
            cand_seq = "".join(cand)
            if mfe_screen is not None:                                 # F5: cheap MFE pre-screen before pKiss
                st, _ = RNA.fold(cand_seq)
                if RNA.bp_distance(st, skeleton) > mfe_screen:
                    n_screened += 1
                    continue                                           # nested part too broken -> skip pKiss
            v0 = time.time()
            v2 = verify_pk(cand_seq, target)
            oracle_s += time.time() - v0
            n_pkiss += 1
            bp2 = v2["bp"] if v2.get("bp") is not None else n
            if bp2 <= cur_bp:                                          # accept downhill + sideways (plateau walk)
                cur_seq, cur_bp, cur_folded = cand_seq, bp2, v2.get("folded")
                if bp2 < best_bp:
                    best_seq, best_bp = cand_seq, bp2
                    solved = bool(v2.get("solved"))

        cost = solver_cost_usd(solver_wall, self.cores) + solver_cost_usd(oracle_s, cores=1)
        # Realized-budget accounting, mirroring DecomposePKPanelTool, so the gamer-vs-panel head-to-head
        # can report realized compute per arm and not just the matched nominal cap.
        stats = {"n_pkiss": n_pkiss, "n_probknot": 0, "n_iters": it,
                 "n_candidates": it + 1, "n_screened": n_screened,
                 "oracle_s": round(oracle_s, 3), "repair_wall_s": round(time.time() - t_rep, 1),
                 "skel_wall_s": round(solver_wall, 2), "ablate": self.ablate}
        return ToolResult(self.name, solved, int(best_bp), round(solver_wall, 3), self.cores,
                          cost, "pkiss", best_seq if solved else None, oracle_s=round(oracle_s, 3),
                          stats=stats)
