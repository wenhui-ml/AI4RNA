"""DecomposeProbKnotTool -- the SYMMETRY control for the oracle-gaming finding (P4, leave-one-oracle-out).

Identical decompose->skeleton->compose->repair procedure as DecomposePKTool, but the in-loop fitness is
ProbKnot (RNAstructure) instead of pKiss. Purpose: check whether single-oracle gaming is SYMMETRIC. The
pKiss-optimizing operator scored 43/60 under pKiss but only 2/43 under ProbKnot. If a ProbKnot-optimizing
operator symmetrically scores high under ProbKnot but collapses under a held-out pKiss, then oracle-gaming is
a general property of optimizing any single heuristic predictor -- not a pKiss quirk. Autonomy is structural
(no external pseudoknot solver). ShapeKnots is too slow to optimize in-loop, so the tractable LOOO cells are:
optimize pKiss -> test ProbKnot (=DecomposePKTool, 2/43); optimize ProbKnot -> test pKiss (this tool);
optimize both -> test ShapeKnots (=DecomposePKPanelTool, 8/60).
"""
from __future__ import annotations

import random
import time

from ..cost import solver_cost_usd
from ..tools import Tool, ToolResult
from .pk_decompose import _WC, _partner_map, _wrong_positions


class DecomposeProbKnotTool(Tool):
    def __init__(self, skel_wall_s: float = 12.0, repair_wall_s: float = 90.0,
                 max_repair_iters: int = 400, skel_restarts: int = 6):
        self.skel_wall_s = float(skel_wall_s)
        self.repair_wall_s = float(repair_wall_s)
        self.max_repair_iters = int(max_repair_iters)
        self.skel_restarts = int(skel_restarts)
        self.cores = 1
        self.budget_s = self.skel_wall_s + self.repair_wall_s
        self.name = f"pbdecomp@{int(self.repair_wall_s)}s"

    def _abstain(self, target: str) -> ToolResult:
        return ToolResult(self.name, False, len(target), 0.0, 1, 0.0, "none", None, available=True)

    def attempt(self, target: str, seed: int = 0) -> ToolResult:
        import RNA
        from ...structure_utils import has_pseudoknot
        from ...benchmarks.rnainvbench import pk_pairs, pkiss_available
        from ...oracles_pk import probknot_pairs, rnastructure_available

        if not has_pseudoknot(target):
            return self._abstain(target)
        if not rnastructure_available() or not pkiss_available():
            return ToolResult(self.name, False, len(target), 0.0, self.cores, 0.0, "none", None,
                              available=False)

        n = len(target)
        rng = random.Random(seed)
        target_pairs = pk_pairs(target)
        skeleton = "".join(c if c in "()." else "." for c in target)
        skel_pairs = pk_pairs(skeleton)
        cross_pairs = sorted(target_pairs - skel_pairs)
        partner = _partner_map(target_pairs)
        oracle_s = 0.0

        def pb_eval(seq_str):
            """(bp, folded_pairs, solved) under ProbKnot; bp = symmetric-difference size to the target."""
            nonlocal oracle_s
            t = time.time()
            pbp = probknot_pairs(seq_str)
            oracle_s += time.time() - t
            if pbp is None:
                return n, set(), False
            bp = len(target_pairs ^ pbp)
            return bp, pbp, (bp == 0)

        # 1-2. skeleton sub-solve (pure nested -> ViennaRNA)
        t0 = time.time()
        best_skel, best_dist = "A" * n, float("inf")
        for _ in range(max(1, self.skel_restarts)):
            start = "".join(rng.choice("ACGU") for _ in range(n))
            s, dist = RNA.inverse_fold(start, skeleton)
            if dist < best_dist:
                best_skel, best_dist = s, dist
            if best_dist <= 0 or (time.time() - t0) > self.skel_wall_s:
                break
        seq = list(best_skel)
        if len(seq) != n:
            seq = (seq + ["A"] * n)[:n]
        # 3. compose crossing pairs (WC)
        for k, (i, j) in enumerate(cross_pairs):
            a, b = _WC[(seed + k) % 4]
            seq[i], seq[j] = a, b
        cur_seq = "".join(seq)
        solver_wall = time.time() - t0

        cur_bp, cur_fpairs, solved = pb_eval(cur_seq)
        best_seq, best_bp = cur_seq, cur_bp

        # 4. ProbKnot-guided repair (mirror of the pKiss walk)
        it = 0
        t_rep = time.time()
        n_cross = len(cross_pairs)
        while (not solved and cur_bp and cur_bp > 0 and it < self.max_repair_iters
               and (time.time() - t_rep) < self.repair_wall_s):
            it += 1
            cand = list(cur_seq)
            if n_cross and rng.random() < 0.10:
                for i, j in cross_pairs:
                    a, b = rng.choice(_WC)
                    cand[i], cand[j] = a, b
            else:
                wrong = _wrong_positions(cur_fpairs, target_pairs, n) or list(range(n))
                pos = rng.choice(wrong) if rng.random() < 0.85 else rng.randrange(n)
                if pos in partner:
                    j = partner[pos]
                    a, b = rng.choice(_WC)
                    cand[pos], cand[j] = a, b
                else:
                    cand[pos] = rng.choice(["A", "U", "A", "U", "G", "C"])
            cand_seq = "".join(cand)
            bp2, fp2, s2 = pb_eval(cand_seq)
            if bp2 <= cur_bp:
                cur_seq, cur_bp, cur_fpairs = cand_seq, bp2, fp2
                if bp2 < best_bp:
                    best_seq, best_bp = cand_seq, bp2
                    solved = s2

        cost = solver_cost_usd(solver_wall, self.cores) + solver_cost_usd(oracle_s, cores=1)
        return ToolResult(self.name, solved, int(best_bp), round(solver_wall, 3), self.cores,
                          cost, "probknot", best_seq if solved else None, oracle_s=round(oracle_s, 3))
