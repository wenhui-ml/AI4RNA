# Pushing the RNA inverse-design ceiling — grounded plan

Branch `rna-ceiling`, opened off `chomsky-climb` at 4a207b9.
Goal: raise the **hardest-problem solve ceiling** and the **cost per solve**. Not an audit paper.

## 1. Where the ceiling actually is (measured, not assumed)

Best arm on Eterna100 (`_archive_runs/eterna_paired/memon`, 100 tasks, best-of-reps):

| bucket | tasks | solved | rate |
|---|---|---|---|
| <60 nt | 21 | 19 | 0.90 |
| 60–100 | 18 | 13 | 0.72 |
| 100–150 | 27 | 18 | 0.67 |
| 150–250 | 9 | 6 | 0.67 |
| **≥250** | **25** | **8** | **0.32** |

**64/100.** The 36 residuals split into two groups with *different* signatures:

| group | n | med len | med max junction degree | med #multiloops | % with a 4+-way junction |
|---|---|---|---|---|---|
| solved | 64 | 100 | 3.0 | 1.0 | 36% |
| unsolved <150 nt | 16 | 100 | 3.0 | 2.0 | **12%** |
| unsolved ≥250 nt | 17 | 389 | 4.0 | 5.0 | **71%** |

Two readings, and the second is the surprise:

* **Long residuals are the branching regime.** Length and junction count compound: median 389 nt with five multiloops. This is a *composition* problem — many locally-easy units that must hold simultaneously.
* **Short residuals are structurally unremarkable.** Same median length and junction degree as the solved set, and *fewer* 4+-way junctions (12% vs 36%). Whatever defeats `Bug 18` (53 nt, max degree 2, zero multiloops) is not branching and not size. The obvious hypotheses are competing-fold degeneracy and outright undesignability.

## 2. The blocking gap: we cannot tell "search failed" from "no sequence exists"

The archived Eterna rows record `solved` and a boolean `recall`, and `rna_calls` is stuck at 1. **The residual base-pair distance on failures is not logged**, so the 36 residuals cannot be triaged. `ToolResult.bp` already carries the number; this is a logging gap, not a measurement gap.

This is the first task, because it decides everything after it. A residual that stalls at `bp = 1` across independent optimizers is the classic undesignability signature — the same wall the RSSD study hit — and chasing it with more search is wasted budget. A residual sitting at `bp = 8` is a search problem and is worth attacking.

## 3. What the finished work already tells us — carry these forward

From `chomsky-climb`, all measured, all reusable:

* **Search *guidance* is not load-bearing.** At matched budget an undirected walk matches the full guided operator (paired *p* = 0.57, point estimate slightly *against* guidance) on the metric that is not the operator's own objective. **Do not spend effort on smarter search heuristics.**
* **LLM-written operators beat the hand-built one 4×** on carry-over to a never-optimized predictor (0.302 [0.275, 0.330] vs 0.091 [0.068, 0.118], non-overlapping), and are **bit-reproducible** where ours is not. They have only ever been tested on pseudoknots. Pointing the invention loop at the nested hard set is the cheapest untested shot at the ceiling.
* **External tool composition works where our search does not** — DesiRNA cracked A-class residuals our own optimizer could not.
* **Compositional priors beat retrieved per-unit sequences** (GC-stiff priors compose; per-unit retrieval does not). This is the lever aimed straight at the long/branched regime.
* **Efficiency is already solved and transferable**: learned abstention cut SOTA cost 8–13× at identical held-out quality, zero solves lost.
* **Determinism is now available**: `deterministic=True` on both operators removes wall-clock stopping and seeds ViennaRNA's own RNG, giving bit-reproducible runs under load, plus an identical counted oracle-call allowance across arms.

## 4. Plan, in dependency order

**(0) Instrument failure.** Log residual `bp`, the best structure reached, and realized oracle calls on every failure. One-line change; unblocks everything.

**(1) Triage the 36 residuals into three classes.**
  * *undesignable* — certify with the exact construction of Zhou et al.; these leave the denominator and the certificate is itself a result;
  * *long/branched* — 17 targets, attack with composition;
  * *short/designable* — the true frontier, and the interesting one.

**(2) Attack the long/branched regime with compositional fidelity.** Decompose to junction-anchored units, design units against stiff priors rather than retrieved sequences, and measure **where composition breaks** (unit-level solve rate vs assembled solve rate). The gap between those two numbers is the whole problem and we have never measured it.

**(3) Point the invention loop at the nested hard set.** Frozen agent-written operators are already 4× better on the one cross-model metric we have, and cost nothing but CPU to evaluate. Run them on the 36 residuals directly, before writing any new operator by hand.

**(4) Efficiency, in the same currency.** Report solves per oracle-call and per dollar, not just solve counts, and carry the learned-abstention gate over so the cost claim is measured on this benchmark too.

## 5. What would make this a result rather than an engineering diary

A ceiling claim needs the same discipline the audit paper earned the hard way: paired comparisons inside one execution, replication across seeds *and* executions, and an honest denominator. The undesignability certificates matter here — "we solve 78/100" and "we solve 78 of the 84 that are designable, and prove the other 16 are not" are very different claims, and only the second is stable.
