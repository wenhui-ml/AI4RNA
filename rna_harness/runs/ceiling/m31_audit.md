# M31 audit: a free baseline ties the state of the art, and the model's increment is real but small

Declaration `m31_which_flips_rescue_declaration.json`, frozen and pushed before the code existed.
Review `m31r_review.py`. Both offline, $0, no new data.

## The question

Given one design and a budget, which compensatory flips should be built? That makes the predictor
the object under test, and it needs a different conditioning set from the causal contrast: the
causal question conditions on helix depth because depth confounds pair type; the prediction
question must not, because depth is information the experimenter already holds.

## M31 returned seven of seven, and its verdict was withheld

I wrote K2 to refuse the reframe if no predictor could beat a free helix-depth baseline, and then
coded it as a comparison of two point estimates with no interval on the difference — the error
this project retracted in M28. The run is committed unchanged; the verdict was not read.

## What the review found

**The free baseline ties the state of the art.** Within-design Spearman against measured rescue,
158 cells, all 4317 interventions:

| predictor | rho | 95% CI |
|---|---|---|
| RibonanzaNet | +0.32947 | [0.271, 0.390] |
| **helix depth (free)** | **+0.32252** | **[0.270, 0.375]** |
| ViennaRNA | +0.25570 | [0.192, 0.322] |

Paired on the same resample: neural − depth = **+0.00695 [−0.074, +0.087]**, vienna − depth =
−0.06682 [−0.165, +0.029]. Both contain zero. **K2 reissued = True: the claim that a predictor
beats a free baseline is refused.** At a rescue threshold of 0.7 the baseline overtakes both
outright (AUROC 0.727 against 0.679 and 0.637).

**Each model's agreement with the outcome is close to its agreement with depth.** Within-cell
Spearman against depth: neural +0.33410 (against +0.32947 with the outcome), ViennaRNA +0.25195
(against +0.25570). To three decimals, what each predictor knows about which flips rescue is what
it knows about how deep the pair sits.

**But depth does not exhaust them.** Partial rank correlation given depth, no parameters fitted,
permutation done inside (cell, depth) blocks:

| predictor | partial rho \| depth | 95% CI | perm p |
|---|---|---|---|
| RibonanzaNet | +0.23799 | [0.174, 0.307] | 0.0005 |
| ViennaRNA | +0.18198 | [0.107, 0.271] | 0.0005 |

**And the increment converts into a decision, honestly and modestly.** Ranking by depth alone and
building the top 25% of a design's flips recovers 34.854% of the rescuing ones. Using the model
only to break depth's ties — no fitting, no weights — gives 35.968% (ViennaRNA, +1.11 points,
CI [+0.001, +0.025]) and 36.817% (RibonanzaNet, +1.96 points, CI [+0.011, +0.031]). Depth is
concentrated at 0–2 (1487 / 1340 / 929 of 4317), so there are many ties for the model to break.

**Both control populations hold.** Partial rho given depth on the 29 cells failing M30's published
selection cutoff: +0.407 and +0.360. On the 144 cells surviving M26d's leakage cut: +0.247 and
+0.198.

## Verdict

`reframe_supported_as_increment_over_depth`.

Two sentences may be written and a third may not:

- **May:** a free integer read off the declared target structure ranks which compensatory flips
  rescue as well as a transformer trained on two million chemical-mapping profiles.
- **May:** with depth held fixed, both predictors still rank at +0.238 and +0.182, and using a
  predictor to break depth's ties raises top-quartile recovery by 1.96 and 1.11 points.
- **May not:** that a predictor beats, or outperforms, a depth baseline. It does not.

## What this changes about how the field reads such results

Depth is free, it is computable from the declared target alone, and it is not reported as a
baseline anywhere we have found. Any claim that a structure predictor "knows which mutations
rescue" has to be stated as an increment over it.
