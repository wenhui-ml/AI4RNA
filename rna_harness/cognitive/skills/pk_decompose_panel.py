"""DecomposePKPanelTool -- a PANEL-optimizing variant of DecomposePKTool.

WHY. The base operator (pk_decompose.py) optimizes a SINGLE pseudoknot oracle (pKiss) in the repair loop and,
as the chomsky_climb ascent experiment showed, learns that oracle's blind spots: 43/60 under pKiss but only
2/43 confirmed by an INDEPENDENT predictor (ProbKnot). That is oracle-level reward hacking -- a pKiss
artifact, not a model-robust capability.

THIS VARIANT optimizes PANEL AGREEMENT instead. A design counts as solved only when BOTH pKiss AND ProbKnot
fold it to the exact target, and the repair walk minimizes the pair (pKiss_bp, ProbKnot_bp) LEXICOGRAPHICALLY
(drive pKiss to 0 first, then close ProbKnot without breaking pKiss). ProbKnot is a subprocess call, so it is
evaluated ONLY on near-pKiss-solved candidates (pk_bp <= gate) and under a per-target call budget, keeping
cost bounded. The question this answers: can optimizing oracle AGREEMENT (rather than one oracle) yield
model-robust crossing designs that survive the panel, where the single-oracle operator (2/43) does not?

Autonomy is still structural: it never calls an external pseudoknot SOLVER (DesiRNA); it uses the two
predictors only as verifiers/fitness. Cost = skeleton CPU + pKiss + ProbKnot oracle-seconds.

ABLATION MODES (reviewer R4: the panel operator's own per-component attribution + a matched-budget
undirected-search control, previously left to future work). `ablate` selects:
  none        -- the full operator (default; unchanged behaviour)
  randwalk    -- MATCHED-BUDGET UNDIRECTED SEARCH. Identical initialization, identical mutation proposal
                 distribution, identical iteration/wall/oracle-call budgets and identical ProbKnot gating,
                 but the lexicographic acceptance test is REMOVED: every proposal is accepted, so the walk
                 is undirected. The verifier still recognizes a hit (that is what "solved" means), but it
                 no longer GUIDES. This isolates the panel-agreement objective from the mere availability
                 of the same budget and the same crossing-aware oracle access.
  blindrand   -- as randwalk, and additionally the proposal distribution is made ORACLE-BLIND: mutation
                 sites are drawn uniformly instead of being biased toward positions where the current fold
                 disagrees with the target. This is the fully undirected matched-budget random search: the
                 oracles are consulted only to recognize a hit, never to steer.
  norepair    -- skeleton sub-solve + complementarity composition only; no repair walk at all.
  nodecompose -- random initial sequence instead of the CF-skeleton inverse fold (keeps composition+repair).
  nocompose   -- crossing positions left as the skeleton left them (keeps decomposition+repair).
"""
from __future__ import annotations

import random
import time
from typing import Optional

from ..cost import solver_cost_usd
from ..tools import Tool, ToolResult
from .pk_decompose import _WC, _partner_map, _wrong_positions


class DecomposePKPanelTool(Tool):
    """Panel-optimizing pseudoknot operator. Cheap-abstains on nested targets."""

    ABLATIONS = ("none", "randwalk", "blindrand", "norepair", "nodecompose", "nocompose")

    def __init__(self, skel_wall_s: float = 12.0, repair_wall_s: float = 90.0,
                 max_repair_iters: int = 800, skel_restarts: int = 6,
                 panel_gate: int = 3, max_probknot: int = 120, ablate: str = "none",
                 deterministic: bool = False, max_pkiss: int = 150):
        self.skel_wall_s = float(skel_wall_s)
        self.repair_wall_s = float(repair_wall_s)
        self.max_repair_iters = int(max_repair_iters)
        self.skel_restarts = int(skel_restarts)
        # R7 M5: see DecomposePKTool. With deterministic=True no wall-clock term enters a stopping rule,
        # so a nominal seed fixes the run bit-for-bit and the execution-level variance that dominates
        # Table 9 is removed by construction rather than averaged over.
        self.deterministic = bool(deterministic)
        # Counted compute allowance used in deterministic mode. 150 exceeds the largest pKiss use
        # realized by any arm under the wall-clock regime (142.7), so switching to it hands no arm
        # less budget than it previously took, while making the allowance identical across arms.
        self.max_pkiss = int(max_pkiss)
        self.panel_gate = int(panel_gate)        # evaluate ProbKnot only when pKiss bp <= gate
        self.max_probknot = int(max_probknot)     # per-target ProbKnot call budget
        if ablate not in self.ABLATIONS:
            raise ValueError(f"ablate must be one of {self.ABLATIONS}, got {ablate!r}")
        self.ablate = ablate
        self.cores = 1
        self.budget_s = self.skel_wall_s + self.repair_wall_s
        self.name = f"pkpanel@{int(self.repair_wall_s)}s" + ("" if ablate == "none" else f"-{ablate}")

    def _abstain(self, target: str) -> ToolResult:
        return ToolResult(self.name, False, len(target), 0.0, 1, 0.0, "none", None, available=True)

    def attempt(self, target: str, seed: int = 0) -> ToolResult:
        import RNA
        from ...structure_utils import has_pseudoknot
        from ...benchmarks.rnainvbench import pk_pairs, verify_pk, pkiss_available
        from ...oracles_pk import probknot_pairs, rnastructure_available

        if not has_pseudoknot(target):
            return self._abstain(target)
        if not pkiss_available() or not rnastructure_available():
            return ToolResult(self.name, False, len(target), 0.0, self.cores, 0.0, "none", None,
                              available=False)

        n = len(target)
        rng = random.Random(seed)
        if self.deterministic:
            # ViennaRNA's inverse_fold is a stochastic search on ViennaRNA's OWN global RNG, which
            # advances across calls and is not touched by random.Random(seed). Without this line two
            # attempts at the same (target, seed) diverge even on an idle machine in one process --
            # measured, and the actual reason the operator was never reproducible. Removing the
            # wall-clock caps alone does not fix it.
            RNA.init_rand(seed)
        target_pairs = pk_pairs(target)
        skeleton = "".join(c if c in "()." else "." for c in target)
        skel_pairs = pk_pairs(skeleton)
        cross_pairs = sorted(target_pairs - skel_pairs)
        partner = _partner_map(target_pairs)
        pb_budget = [self.max_probknot]
        oracle_s = 0.0
        n_pkiss = n_pb = 0

        def panel_eval(seq_str):
            """(pk_bp, pb_bp, pk_folded, pk_solved). ProbKnot only if pk_bp<=gate and budget remains;
            pb_bp is a sentinel (n) when ProbKnot is not consulted, so lexicographic order uses pk_bp first."""
            nonlocal oracle_s, n_pkiss, n_pb
            t0 = time.time()
            v = verify_pk(seq_str, target)
            oracle_s += time.time() - t0
            n_pkiss += 1
            pk_bp = v["bp"] if v.get("bp") is not None else n
            pk_folded = v.get("folded")
            pk_solved = bool(v.get("solved"))
            pb_bp = n
            if pk_bp <= self.panel_gate and pb_budget[0] > 0:
                t1 = time.time()
                pbp = probknot_pairs(seq_str)
                oracle_s += time.time() - t1
                pb_budget[0] -= 1
                n_pb += 1
                pb_bp = len(target_pairs ^ pbp) if pbp is not None else n
            return pk_bp, pb_bp, pk_folded, pk_solved

        # 1-2. skeleton sub-solve (pure nested -> ViennaRNA inverse_fold)
        t0 = time.time()
        if self.ablate == "nodecompose":
            seq = [rng.choice("ACGU") for _ in range(n)]      # random init: no CF skeleton
        else:
            best_skel, best_dist = "A" * n, float("inf")
            for _ in range(max(1, self.skel_restarts)):
                start = "".join(rng.choice("ACGU") for _ in range(n))
                s, dist = RNA.inverse_fold(start, skeleton)
                if dist < best_dist:
                    best_skel, best_dist = s, dist
                if best_dist <= 0 or (not self.deterministic and (time.time() - t0) > self.skel_wall_s):
                    break
            seq = list(best_skel)
            if len(seq) != n:
                seq = (seq + ["A"] * n)[:n]

        # 3. compose crossing pairs (WC, seed-varied)
        if self.ablate != "nocompose":
            for k, (i, j) in enumerate(cross_pairs):
                a, b = _WC[(seed + k) % 4]
                seq[i], seq[j] = a, b
        cur_seq = "".join(seq)
        solver_wall = time.time() - t0

        # 4. initial panel evaluation
        cur_pk, cur_pb, cur_folded, pk_solved = panel_eval(cur_seq)
        best = (cur_pk, cur_pb, cur_seq)
        solved = pk_solved and cur_pb == 0

        # 5. lexicographic (pk_bp, pb_bp) best-frontier repair walk.
        # Under `randwalk`/`blindrand` the acceptance test is dropped (every proposal is accepted), so the
        # walk is undirected at a MATCHED budget; under `blindrand` the proposal sites are additionally
        # drawn oracle-blind. `norepair` skips the loop entirely.
        undirected = self.ablate in ("randwalk", "blindrand")
        blind = self.ablate == "blindrand"
        it = 0
        t_rep = time.time()
        n_cross = len(cross_pairs)
        while (self.ablate != "norepair" and not solved and (cur_pk > 0 or cur_pb > 0)
               and it < self.max_repair_iters and n_pkiss < self.max_pkiss
               and (self.deterministic or (time.time() - t_rep) < self.repair_wall_s)):
            it += 1
            cand = list(cur_seq)
            if n_cross and rng.random() < 0.10:
                for i, j in cross_pairs:
                    a, b = rng.choice(_WC)
                    cand[i], cand[j] = a, b
            else:
                if blind:
                    wrong = list(range(n))          # oracle-blind: uniform site choice
                else:
                    fpairs = pk_pairs(cur_folded) if cur_folded else set()
                    wrong = _wrong_positions(fpairs, target_pairs, n) or list(range(n))
                pos = rng.choice(wrong) if rng.random() < 0.85 else rng.randrange(n)
                if pos in partner:
                    j = partner[pos]
                    a, b = rng.choice(_WC)
                    cand[pos], cand[j] = a, b
                else:
                    cand[pos] = rng.choice(["A", "U", "A", "U", "G", "C"])
            cand_seq = "".join(cand)
            cpk, cpb, cfolded, cpk_solved = panel_eval(cand_seq)
            # accept on lexicographic non-worsening of (pk_bp, pb_bp): drive pKiss to 0, then ProbKnot
            if undirected or (cpk, cpb) <= (cur_pk, cur_pb):
                cur_seq, cur_pk, cur_pb, cur_folded = cand_seq, cpk, cpb, cfolded
                if (cpk, cpb) < (best[0], best[1]):
                    best = (cpk, cpb, cand_seq)
                solved = cpk_solved and cpb == 0

        best_pk, best_pb, best_seq = best
        cost = solver_cost_usd(solver_wall, self.cores) + solver_cost_usd(oracle_s, cores=1)
        # Budget accounting (reviewer R5 §4): "identical wall-clock + identical iteration cap" does NOT
        # imply identical realized compute, because arms differ in how many oracle calls they actually make
        # and therefore in how many candidates they explore inside the same wall. Report the realized budget
        # per attempt so matched-NOMINAL-budget can be told apart from matched-REALIZED-budget.
        stats = {"n_pkiss": n_pkiss, "n_probknot": n_pb, "n_iters": it,
                 "n_candidates": it + 1,           # initial evaluation + one proposal per iteration
                 "oracle_s": round(oracle_s, 3), "repair_wall_s": round(time.time() - t_rep, 1),
                 "skel_wall_s": round(solver_wall, 2), "ablate": self.ablate}
        # 'bp' reported is the panel distance pk_bp+pb_bp (0 iff both oracles exact)
        return ToolResult(self.name, solved, int(best_pk + best_pb), round(solver_wall, 3), self.cores,
                          cost, "panel:pkiss+probknot", best_seq if solved else None,
                          oracle_s=round(oracle_s, 3), stats=stats)
