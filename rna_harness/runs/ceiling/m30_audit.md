# M30 audit: the selection rule we never checked, and what it does to the headline

Declaration `m30_selection_audit_declaration.json`, frozen and pushed before the code existed.
Offline, $0. Artifacts `m30_selection_audit.json` / `.log`.

## Why this run exists

An outside reading of the OpenKnotAIDesignData repository found a sentence we should have found
ourselves. Of the mutate-and-map-rescue experiments specifically — our corpus — it says:

> The top OpenKnot score design for each method for each target were selected amongst those with
> `RNet_F1` greater than equal to 80. (If no designs passed this secondary structure accuracy
> cutoff, the top score design was still tested.)

`RNet_F1` is RibonanzaNet's own agreement with the target structure. Read literally, our corpus
is conditioned on a neural model's own prediction, and no sentence of the manuscript said so.
That is a defect in the manuscript whether or not it moves a number.

## What was staked, and what happened

| staked in advance | outcome |
|---|---|
| 1. published rule reproduces at ≥ 0.90 | **FAILED — 0.8643** |
| 2. pool pass rate ≥ 0.70 | held (0.8303 this round, 0.7629 all rounds) |
| 3. control arm powered (≥ 20 cells, ≥ 300 interventions) | held (30 cells, 727 interventions) |
| 4. fallback-cell contrast same sign | held (+0.02284 vs +0.05393) |
| 5. difference CI contains zero | held ([−0.02808, +0.0887]) |
| 6. not saturated | held (28.1% above 0.99) |
| 7. skill does not rise with RNet_F1 | held (−0.0637, CI [−0.2786, +0.1184]) |

Six of seven. The miss is the interesting one.

## The miss, reviewed rather than accepted

All 160 tested designs link to the public pool by exact design-sequence match, all in round 3.
The published rule reproduces 121/140 non-control cells. Reviewing all 19 misses one by one:

- **135/140 tested designs are the top-`target_openknot_score` design in their cell with the
  `RNet_F1` filter simply not applied** (agreement 0.9643).
- In **14** of the 19 misses, eligible designs existed — up to 18 of 20 candidates passed the
  cutoff — and the top-score design was tested anyway despite failing it. The published filter
  demonstrably did not bind in those cells.
- Of the 5 cells that "top score, no filter" also misses: three (P03/P10/P15 Rosetta) skip a
  top-score design whose `RNet_F1` is exactly 0.0 and test the runner-up, so some F1 influence is
  visible at the degenerate end; one (P06/Eterna) has 15 designs tied at score 100.0 and tests a
  design at 99.3333; one (P06/MPNN-fixbb) tests rank 3 behind a tie of 2.

**The variable that actually selected our corpus is the experimental OpenKnot score, not the
neural model's prediction.** No third rule was fitted to close the residual 5 — that would be
overfitting the audit.

## The consequence, and the arm it forced

A control aimed at the wrong variable is not a control. S3/S4 were pre-registered against
`RNet_F1` because the repository names it; S1 showed the real variable is the OpenKnot score, so
the same two arms were re-run against it (S3b/S4b, marked post-hoc in the artifact; nothing
pre-registered was altered).

1. **Selection status is constant inside every stratum.** The stratum is
   `(design, min(depth,3), crossed)` and selection acts on `design` alone: 0 of 970 strata are
   mixed. Selection is conditioned out by construction, not by argument.
2. **The identity effect survives where the filter did not bind.** In the 30 cells whose tested
   design fails the published cutoff: +0.02284, same sign as the full corpus (+0.05393, weight
   902, 553 strata — reproducing M21b bit-for-bit). Difference [−0.02808, +0.0887].
3. **Measured skill runs *against* the selection variable.** Cell-level rank skill vs the tested
   design's OpenKnot score: Spearman −0.2455, CI [−0.3773, −0.0929], excluding zero. Cells the
   selection favoured show *less* model skill, not more. On slopes: +0.049 in high-score cells
   vs +0.174 in low-score cells.
4. **Skill is unrelated to `RNet_F1`** on both statistics (slope-based −0.0637 CI
   [−0.2786, +0.1184]; rank-based −0.0241 CI [−0.2307, +0.1372]).

Point 3 is a slope comparison across two populations — the comparison I made in M28 and
retracted under rule 68. It is repeated here on a rank statistic and survives, and the two arms'
`P_model` dispersion differs by a factor of 1.004, so the scale objection does not apply to this
one either. Both are reported.

## Dynamic range (D)

The outside reading named a kill condition: a majority of interventions above P = 0.99 would mean
the score is saturated and M26's calibration reading is withdrawn.

| | min | p10 | median | p90 | >0.99 | <0.08 | between |
|---|---|---|---|---|---|---|---|
| ViennaRNA | 0.0000 | 0.0000 | 0.7451 | 0.9996 | 28.1% | 33.1% | 38.8% |
| RibonanzaNet-SS | 0.0000 | 0.0136 | 0.9839 | 0.9994 | 44.1% | 13.4% | 42.5% |

K3 does not fire for either predictor.

## Verdict

`selection_real_disclosed_and_conditioned_out`. K0, K1, K2, K3 all false.

The selection is real, it was undisclosed by us, and it is now measured: it acts on a variable
that is negatively associated with what we report, it is absorbed by the stratification, and the
effect survives in the cells where the published filter did not bind. The manuscript states the
rule, states that we did not check it before an outside reading pointed at it, and states that
the rule as published is not the rule the data shows.

## What this run does NOT cover

The exposure-graded leakage audit against the public 2,150,401-sequence Ribonanza library, whose
final pre-training stage pseudo-labelled all training *and* test sequences. Our 14-deposit,
861,916-sequence audit does not address the "sequence seen, experimental label not seen" tier.
That needs a download we do not have. The manuscript says so rather than implying otherwise.
