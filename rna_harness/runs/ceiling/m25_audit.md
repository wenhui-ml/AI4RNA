# M25 audit — the estimator benchmark and the readout comparison

Declaration `m25_estimator_benchmark_declaration.json`, frozen and pushed at `56c969a4`
**before** the runner existed. Offline, $0.

Both arms exist because the target venue asks for a head-to-head comparison with what people
currently do. Neither is answered by argument here: arm A has a controlled truth, arm B is the
same molecules and the same pairs.

## Arm A — four estimators against a known effect

The real `(pt, depth, crossed, design)` is kept, so the identity-to-depth confounding is real; only
the outcome is replaced by `tau * 1[CG] + f(depth) + noise`. The within-design depth response is
`beta = +0.06759` with curvature `c = -0.02125` and residual SD `0.2228`, all fitted from the data;
the within-design depth gap is `g = -0.3487`, so a marginal estimator should be biased by
`beta * g = -0.02357`. The array implementation is cross-checked against the readable definitions
to `9.7e-17` before the arm starts.

| truth | tau | E1 marginal | E2 regression | E3 IPW | E4 matched |
|---|---|---|---|---|---|
| linear | 0.00 | **-0.0216** | +0.0015 | +0.0008 | +0.0011 |
| linear | 0.02 | **-0.0211** | +0.0021 | -0.0094 | +0.0021 |
| linear | 0.05 | **-0.0239** | -0.0008 | -0.0256 | -0.0008 |
| linear | 0.10 | **-0.0233** | -0.0003 | **-0.0507** | -0.0018 |
| non-linear | 0.05 | **-0.0209** | -0.0010 | -0.0258 | -0.0014 |
| non-linear | 0.10 | **-0.0206** | -0.0007 | **-0.0500** | -0.0005 |

(bias = mean estimate − tau, 40 draws each; type-I at tau = 0 is 0.00 for all four.)

**Three readings.**

1. **The marginal contrast is biased by exactly the predicted amount.** −0.021 to −0.024 against
   `beta * g = -0.0236`, at every tau and under both truths. That is the cost of not conditioning
   on placement, and it is now a measured quantity rather than an inference.
2. **Propensity weighting is worse than doing nothing careful.** Its bias grows with the effect
   (−0.009, −0.026, −0.051 at tau = 0.02, 0.05, 0.10), roughly −0.5 tau, because strata where one
   pair type dominates produce extreme weights.
3. **K2 fires.** Regression adjustment ties the matched estimator under *both* truths. The
   declaration wrote down what that costs before the run: **matching is not necessary here.** The
   claim the paper may make is *"the placement must be conditioned on; matching and regression
   adjustment both do it"* — not *"matching is required"*.

A curvature stress test (**not pre-registered**) asks how much non-linearity it takes to separate
them: at the curvature the data actually has, regression wins by `5e-5`; at 5x, 10x and 20x that
curvature, matching wins (E2 bias −0.0024, −0.0025, −0.0056 against E4's −0.0014, −0.0003,
+0.0002). So the two are separable in principle and are not separable here, which is the honest
statement.

## Arm B — the same pairs, two readouts

`158` designs carry both a compensatory-mutagenesis readout and a probing-guided inferred
structure. **The first version of this arm compared 892 pair-weights of rescue against 1982
pair-weights of inferred structure** — the rescue readout exists only where a compensatory double
mutant was made, the inferred one exists for every target pair. That is not the same comparison and
it is exactly what probe P4 was declared to prevent; P4 had been written into the declaration and
not implemented. Both readouts are now keyed by `(design, i, j)` and intersected, and the assertion
is in the code: `4293` common pairs, `892` pair-weights, `549` strata, byte-identical between arms.

| readout | identity contrast | p |
|---|---|---|
| rescue (interventional) | **+0.0527** | 0.0025 |
| inferred structure (conventional) | **+0.0198** | 0.0200 |

Paired, by design cluster bootstrap: **R1 − R2 = +0.0329, [0.0024, 0.0638]**, excluding zero.
**K3 does not fire: the choice of readout changes the answer.**

**Why, and it is not the reason predicted.** Staked prediction (5) said the conventional readout
would *inflate* the contrast, because structure inference runs on nearest-neighbour thermodynamics
and would carry its own preference for G–C. It does the opposite on these pairs, and the mechanism
is visible: on the compensatory subset the inferred structure calls **89.8%** of C–G target pairs
and **90.8%** of A–U ones. It is **saturated exactly where the interventional readout has its
resolution**, so there is no dynamic range left for identity to act on. Predictions (5) and (6) are
recorded as wrong.

**And the conventional readout is not a stable measurement of this quantity.** On the *full* target
pair set of the same 158 designs — not P4-compliant, reported as a contrast — the same inferred
readout gives **+0.0891** with hit rates C–G `0.628` and A–U `0.406`. The same readout on the same
molecules moves by **4.50x** depending on which pairs are looked at. The interventional readout is
defined only where a compensatory mutant exists and has no such degree of freedom.

## Staked predictions

| # | staked | outcome |
|---|---|---|
| 1 | marginal bias ≈ beta × g | **right** (−0.0239 vs −0.0236) |
| 2 | matched bias within ±20% of tau | **right** |
| 3 | regression worse than matching under non-linearity | **wrong** (ties; separable only at ≥5x the real curvature) |
| 4 | matched type-I ≤ 0.01 at tau = 0 | **right** (0.00) |
| 5 | conventional readout inflates the contrast | **wrong** (it deflates it, 0.38x) |
| 6 | inferred structure calls C–G more than A–U | **wrong** (0.898 vs 0.908 on the common pairs) |

Three of six wrong. The two that were wrong about arm B were wrong about the *mechanism* while the
arm's conclusion — the readout choice changes the answer — survived on a paired test that was not
part of the original design and was added because K3 required a paired comparison, not two
separate p-values.

## What this cannot do

Arm A's synthetic outcome is not real rescue; it answers how estimators behave against a known
truth, not what generates rescue. Arm B depends on one structure-inference implementation, so its
conclusion is about *that* conventional path and may not transfer to another. Same laboratory, same
platform, no sequence synthesised.
