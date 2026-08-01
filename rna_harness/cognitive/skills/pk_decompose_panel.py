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

import math
import random
import time
from typing import Optional

from ..cost import solver_cost_usd
from ..tools import Tool, ToolResult
from .pk_decompose import _WC, _partner_map, _wrong_positions

# variants whose objective sums over both predictors, so both must be evaluated every candidate
_UNGATED = ("ungated", "scalar1", "scalar2", "anneal")


class DecomposePKPanelTool(Tool):
    """Panel-optimizing pseudoknot operator. Cheap-abstains on nested targets."""

    ABLATIONS = ("none", "randwalk", "blindrand", "norepair", "nodecompose", "nocompose")
    # ungated  : consult ProbKnot on every candidate instead of only when pKiss is already within `gate`.
    #            The agent operators do this, and ProbKnot is the least-correlated panel member (kappa 0.27,
    #            0.12), so this transplants "let the independent voice steer the whole search".
    # scalar1  : replace lexicographic (pk_bp, pb_bp) acceptance with a SUMMED cost, symmetric weights.
    #            Isolates scalar-vs-lexicographic on its own.
    # scalar2  : summed cost with spurious pairs weighted 2x, as the agent operators write it. Against
    #            scalar1 this isolates the false-positive ASYMMETRY.
    # anneal   : add Metropolis acceptance (T0=2.0, cooling 0.997) to the baseline rule.
    # bigpk    : 3x the pKiss allowance and nothing else -- the compute control. If only this moves
    #            carry-over, the agents bought their advantage with compute and it is search, not a principle.
    # tinypk   : the compute control run the OTHER way, and the one the budget measurement asked for.
    #            agent_budget.py finds the frozen agent operators realize ~28 and ~13 pKiss calls per target
    #            against this operator's 128 in the same regime, and one of them writes an explicit
    #            `budget = [30]` into its own source. So the live hypothesis is no longer "the agents searched
    #            harder" but its inverse: a search forced to commit early may land in a different region of
    #            sequence space, one that a predictor outside the loop happens to like. This arm tests it by
    #            cutting the pKiss allowance to 30 -- kimi-k3's self-imposed number -- and changing nothing
    #            else. If carry-over rises while r_in does not collapse, that is the first mechanism in this
    #            paper to move the statistic.
    # Each variant differs from the one above it by exactly one mechanism, so the chain
    #   none -> ungated -> scalar1 -> scalar2 -> anneal
    # attributes the gap one step at a time; `bigpk` sits outside the chain as the compute control.
    VARIANTS = ("none", "ungated", "scalar1", "scalar2", "anneal", "bigpk", "tinypk")

    def __init__(self, skel_wall_s: float = 12.0, repair_wall_s: float = 90.0,
                 max_repair_iters: int = 800, skel_restarts: int = 6,
                 panel_gate: int = 3, max_probknot: int = 120, ablate: str = "none",
                 deterministic: bool = False, max_pkiss: int = 150, variant: str = "none"):
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
        # R8 attribution: each VARIANT transplants exactly ONE mechanism that the frozen agent-written
        # operators have and this one does not, so the carry-over gap (0.302 vs ~0.108) can be attributed
        # to a mechanism instead of left as "we cannot yet say why". Additions, not removals -- hence a
        # separate axis from `ablate`.
        if variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {self.VARIANTS}, got {variant!r}")
        self.variant = variant
        if variant in _UNGATED:
            # ProbKnot on EVERY candidate needs a matching call allowance, else the 120-call cap silently
            # re-gates it and the variant measures the cap rather than the mechanism.
            self.max_probknot = max(self.max_probknot, self.max_pkiss)
        if variant == "bigpk":
            self.max_pkiss = self.max_pkiss * 3      # the pure-compute control; deliberately NOT matched
        if variant == "tinypk":
            self.max_pkiss = 30                      # kimi-k3's self-imposed budget, transplanted verbatim
            self.max_probknot = min(self.max_probknot, 30)
        self.cores = 1
        self.budget_s = self.skel_wall_s + self.repair_wall_s
        self.name = (f"pkpanel@{int(self.repair_wall_s)}s" + ("" if ablate == "none" else f"-{ablate}")
                     + ("" if variant == "none" else f"+{variant}"))

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
            miss_pb = extra_pb = n
            # Every variant that uses a SUMMED objective must evaluate both predictors on every candidate:
            # under the gate an unconsulted ProbKnot contributes 0, so a summed cost would reward staying
            # away from the gate rather than satisfying ProbKnot. Gating is therefore lifted for all of them.
            gate_open = True if self.variant in _UNGATED else (pk_bp <= self.panel_gate)
            if gate_open and pb_budget[0] > 0:
                t1 = time.time()
                pbp = probknot_pairs(seq_str)
                oracle_s += time.time() - t1
                pb_budget[0] -= 1
                n_pb += 1
                if pbp is not None:
                    pb_bp = len(target_pairs ^ pbp)
                    miss_pb, extra_pb = len(target_pairs - pbp), len(pbp - target_pairs)
            # the scalar variants need pKiss's miss/extra split too, which the bp distance hides
            fp = pk_pairs(pk_folded) if pk_folded else set()
            miss_pk, extra_pk = len(target_pairs - fp), len(fp - target_pairs)
            return pk_bp, pb_bp, pk_folded, pk_solved, (miss_pk, extra_pk, miss_pb, extra_pb)

        def scalar_cost(parts):
            """Summed cost over both predictors. `w` is the weight on SPURIOUS pairs: 1 reproduces a
            symmetric objective, 2 reproduces the asymmetry the agent operators wrote."""
            miss_pk, extra_pk, miss_pb, extra_pb = parts
            w = 2 if self.variant in ("scalar2", "anneal") else 1
            pb_terms = (miss_pb + w * extra_pb) if miss_pb < n else 0
            return miss_pk + w * extra_pk + pb_terms

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
        cur_pk, cur_pb, cur_folded, pk_solved, cur_parts = panel_eval(cur_seq)
        best = (cur_pk, cur_pb, cur_seq)
        cur_cost = scalar_cost(cur_parts)
        best_cost = cur_cost
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
            cpk, cpb, cfolded, cpk_solved, cparts = panel_eval(cand_seq)
            ccost = scalar_cost(cparts)
            # Acceptance. Baseline: lexicographic non-worsening of (pk_bp, pb_bp) -- drive pKiss to 0, then
            # ProbKnot. scalar*: non-worsening of the summed cost instead. anneal: Metropolis on the summed
            # cost. undirected: accept everything (the R4 control).
            if undirected:
                accept = True
            elif self.variant in ("scalar1", "scalar2"):
                accept = ccost <= cur_cost
            elif self.variant == "anneal":
                temp = max(2.0 * (0.997 ** it), 0.01)
                accept = (ccost <= cur_cost
                          or rng.random() < math.exp(-(ccost - cur_cost) / temp))
            else:
                accept = (cpk, cpb) <= (cur_pk, cur_pb)
            if accept:
                cur_seq, cur_pk, cur_pb, cur_folded, cur_cost = cand_seq, cpk, cpb, cfolded, ccost
                # track the best under whichever objective this variant is actually optimizing
                if self.variant in ("scalar1", "scalar2", "anneal"):
                    if ccost < best_cost:
                        best, best_cost = (cpk, cpb, cand_seq), ccost
                elif (cpk, cpb) < (best[0], best[1]):
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
