# rna-ceiling: stage summary and next steps

Written at the end of a long session. Ordered by what a reader needs, not by chronology.

## 1. The single most important result, and it is not ours

DesiRNA emits about 200 candidate sequences per call. `DesirnaTool` has always kept ONE, screened the rest
with `verify` -- which on a pseudoknot target means pKiss alone -- and stopped at the first hit. Screening
every candidate against the full three-oracle conjunction instead:

    arm     seed   unanimous/60   claimed   false positives   cpu-s per pass
    plain    900              0        43                43            18231
    plain    901              0        43                43            18127
    plain    902              1        43                42            18222
    plain    903              0        43                43            18065
    panel    900             28        28                 0            20378
    panel    901             27        27                 0            20234
    panel    902             22        22                 0            20117
    panel    903             24        24                 0            19943

    plain   0.25/60 per execution, $0.2522/pass, $1.0090 per solve
    panel  25.25/60 per execution, $0.2801/pass, $0.0111 per solve

Exact McNemar on 240 paired (target, seed) cells: panel wins 100, plain wins 0, **p = 1.58e-30** -- the
strongest statistic on this branch by twenty orders of magnitude. 101x capability at +11% compute, 91x
cheaper per solve, and every one of the plain arm's 43 solve claims per pass is a false positive.

On the shared surface, which is the only fair comparison since budgets differ:

    system                             $/pass   solves/exec   $/solve   basis
    our operator, flat policy          0.0933          5.50    0.0170   measured
    our operator, optimal allocation   0.0933         17.42    0.0054   modelled S*(B)
    our operator, optimal, matched $   0.2746         21.25    0.0129   modelled S*(B)
    DesiRNA + panel screen, flat       0.2746         25.67    0.0107   measured

DesiRNA+screen with a FLAT policy beats our operator's MODELLED OPTIMAL allocation at the same budget, and
its ceiling is 32/60 after 4 seeds against our 22/60 after 8. **The contribution is the specification
wrapper, not our operator** -- which is the outcome I pre-committed to calling the bigger one, because
improving SOTA is worth more than beating a mis-specified SOTA.

Caveat that must travel with it: the three-oracle panel is OUR choice. Under the single-oracle criterion the
field actually uses, DesiRNA already reaches 43/60 and needs nothing from us. What is not our choice is that
the three predictors disagree on >90% of sequences -- that is the unsettled state of pseudoknot prediction.

## 2. Validated, with intervals

| change | effect | statistics |
|---|---|---|
| ShapeKnots gate (in-loop, crossing) | 3.25 -> 5.50/60 per exec; 12 -> 22/60 ever; false positives 253 -> 0 | McNemar 18/0, p = 7.6e-06; ratio 1.69 [1.27, 2.26] |
| compute-matched falsifier `off_400` | 2.5x the pKiss budget changed **nothing** -- 279/26/253 identical, 0 discordant cells | p = 1 |
| trajectory allocation | efficiency 1.6-2.6x, all CIs excluding 1 | dose-response 11/0 p = 0.0010, 13/0 p = 0.0002 |
| Eterna ensemble objective (`inverse_pf_fold`) | median P(target) 0.0594 -> 0.8077, 86% still MFE-matching | 13.59x [8.12, 22.50], 62/64, p = 6.5e-12 |
| oracle cost scaling | ShapeKnots n^1.90 against pKiss n^0.14, ProbKnot n^0.26 | 14 targets, 21-62 nt |
| S(B)'s independence precondition | p-hat from 8 trajectories predicts the 2- and 4-trajectory arms to within 4% | inside bootstrap CIs |

## 3. Refuted -- six of my own ideas, none survived

1. **Extreme-value extrapolation** to the true margin: AUC 0.818 against 0.875 for the raw minimum. Where
   the curve is flat (fitted alpha 16-23) the endpoint is unidentified while R^2 stays at 0.998.
2. **Weibull hazard in calls + KKT allocation**: miscalibrated 1.6x out of sample, and k > 1 makes the
   objective non-concave so the allocator returned budget-infeasible allocations (flat E[solves] across a
   10x budget range was the tell).
3. **Cure mixture** to fix it: still 1.57x out.
4. **Two-sided Delta band**: refuted on its own fit split -- the relation is monotone there and the bottom
   6 targets are the BEST. I had built the mechanism on Poisson noise.
5. **p0, the measure of the satisfying set**: Spearman +0.004, p = 0.989.
6. **Single-model ensemble uncertainty predicting cross-model disagreement**: entropy 0.458, diversity
   0.461, both at chance; and my own declared control (skeleton MFE gap, 0.715) tied the winner (0.725), so
   by the criterion written into the script nothing new was found. It also fails on economics: ranking k
   candidates by the proxy gives 0.187 where just running ShapeKnots on all k gives 0.580. **Screening beats
   steering, because screening is cheap.**

The six share a structure worth more than a seventh guess: every one computes a new statistic on quantities
the system already produces. The three changes that worked all added an information source.

    re-analysing existing signals    0 / 6
    adding an information source     3 / 3

## 4. My own errors, corrected

1. Presented **oracle skylines** (5.53x, 7.06x) as achievable policy values. They are upper bounds
   requiring outcome knowledge.
2. **k * p** where the within-execution cap makes it 1 - (1-p)^k, which inflated a headroom estimate 7-fold.
3. **"ShapeKnots costs 0.31x pKiss"** -- from a 21-42 nt sample. It scales as n^1.90, so it is cheapest only
   below ~45 nt and would cost ~7.5 s per call at Eterna's 400 nt.
4. **"The honest Eterna number is 55/100, not 64"** -- wrong. 64/100 under `match` IS the benchmark's
   declared criterion. `umfe` and `pfe` are harder problems I invented, not corrections. The benchmark ships
   a different sample solution per folding engine and 59 of 98 disagree under ViennaRNA 2.7.2, so
   model-relativity is the task's definition, not a defect.
5. **"fold-calls" as a budget axis** -- invalid across lengths, since one fold costs 1 ms at 30 nt and
   179 ms at 400 nt. The sweep meant to measure it was killed after 72 minutes when process inspection
   showed cells pinned at the 600 s subprocess timeout rather than at their call budget. It also lost all
   its data, because the script aggregated only at the end with no incremental persistence.

## 5. Where the project stands against the north star

The north star (weekly report 2026-07-10) is to accumulate transferable A-theta gated by an exact verifier
V so that solving compounds. Three things clarified today:

* **V is exact.** R_det ~ 1 holds. `verify(seq, target)` computes a model-relative predicate by dynamic
  programming with no error. What it does not do is speak about a test tube, and the benchmark itself
  concedes that by shipping one answer per engine.
* **The crossing benchmark is a poor A-theta microscope.** Three pseudoknot predictors agree on under 10%
  of sequences, so any Delta measured there is confounded by which predictor was chosen. Eterna under
  `match` is the right instrument.
* **GLM-5.2 has already authored working operators**, and the economics were never charged:

        crossing (125 held-out)   skill 91/125 at $0.087   vs SOTA 96/125 at $5.42   (62x, 5 lost)
        nested   (37 held-out)    skill 24/37  at $0.038   vs SOTA 37/37  at $0.955  (25x, 13 lost)
        invention capex           4 calls, 16336 in / 6385 out = ~$0.017

  Amortised over 33 targets that capex is $0.0005/target. **An agent-authored operator already wins
  economically; what was missing was a framework for charging the token cost, which cost(N) = c_cpu +
  C_invent/N now provides.**

And the nested result has a specific defect worth pursuing: `budget_hit = 13` equals `solves_lost = 13`.
Thirteen held-out targets ran out of budget rather than proving unsolvable, so 24/37 may be the budget we
chose rather than the operator the agent wrote.

## 6. Next steps

1. **Fix the budget axis, then measure it.** Per-target CPU-seconds, not fold-calls, because fold cost
   varies 179x with length. Add incremental per-cell persistence so a kill cannot lose the run again.
2. **Run `agent_surface.py`** on the corrected axis: the objective becomes (S_inf, U, -cost) over a budget
   sweep instead of solves-at-one-budget, and the prompt tells the agent it will be run at several budgets
   without knowing which. **The objective scores the agent's own skill only** -- with a SOTA escape hatch
   inside it, "raise S_inf" has the trivial gamed optimum of always calling SOTA. This is the first time
   today's evaluation axis enters the agent's own optimisation loop.
3. **Eterna's ceiling is a length problem**, not a luck problem: 8 seeds buy only 1.09-1.17x over one shot
   (crossing buys 3.7x), and above 250 nt only 9 of 25 targets are reachable against 53 of 75 below. The
   unused asset aimed exactly there is the retrieval warm-start library -- built and tested 10/10, never
   wired into the loop.
4. **Do not transplant the panel screen to Eterna.** Under a single exact criterion a candidate either
   works or does not; there is no conjunction to screen against. The nested lever is different.

## 7. What is not claimed

Two benchmarks, one operator family, short crossing targets. Of roughly 33 hypothesis tests on this branch
four survive Bonferroni: the panel screen (1.58e-30), the ShapeKnots gate (7.6e-06), the trajectory
dose-response (2e-04 and 1e-03), and the ensemble-objective result (6.5e-12). The trajectory arm's headline
solve ratio (p = 0.0225) does not. S_inf estimated from 8 seeds is a lower bound -- a target with true
p = 0.10 shows 0/8 with probability 0.43 -- and the two coordinates have opposite sampling needs, so one
sweep cannot estimate both well.
