# Pre-registration for the allocation experiment

Written after step 0/step 1 and **before step 2 runs**. Everything here is derived from ids 0-29 and the
already-committed 15-cell sweep. Nothing below has seen a step-2 oracle call.

Companion to `METHOD.md` (design and falsifiers, written earlier). Where the two disagree, this file is
the correction and says so.

## 0. What step 0 changed

`ev_margin.py`, `ev_analyze.py`, `pk_ev.py`:

| claim in METHOD.md | outcome |
|---|---|
| the Eterna AUC 0.867 is confounded and "may be lower" | **wrong in magnitude.** Same draws: 0.867 length-dependent `N`, 0.870 at fixed `N=100`. The confound is real but swamped: Δ̂'s `N`-dependence is a few kcal against a 10-200 kcal between-target spread. |
| extreme-value extrapolation `Δ̂_N ≈ Δ* + cN^(−1/α)` is the right correction | **rejected by measurement.** Extrapolated Δ* scores AUC 0.818 vs 0.875 for the raw minimum. Where the curve is flat (fitted α of 16-23) Δ* is unidentified, and the grid search keeps R² ≈ 0.998 while placing Δ* anywhere; it zeroed exactly the hard-and-unsolved targets that carried the signal (Gladius 111.9 → 0.3). **High R² on a monotone decreasing curve is not evidence of a good endpoint estimate.** |
| the fix is to sample more | **unnecessary.** Δ is nearly `N`-free: Eterna AUC 0.862 at `N=5`, 0.875 at `N=200`. What predicts failure is the *location of the whole gap distribution* over the compatible set (median 0.856, mean 0.855, min 0.875), not its infimum; the *spread* predicts far worse (sd 0.767). |
| the slack / information-budget framing (`info_budget.py`) | **dead.** AUC 0.367 pooled, 0.555 / 0.614 within length strata, and **0/100** targets ever violate the budget. `H_avail ∝ length` swamps Δ, so pooling inverts it (Simpson). Dropped; raw Δ is the coordinate. |
| — | **new:** Δ is not length in disguise. Within length terciles AUC(Δ) = 0.839 / 0.878 / 0.888 against AUC(length) = 0.810 / 0.537 / 0.712, and adding Δ to a length-only logistic gives LR = 28.5, p = 9.2e-8 with β(length) flipping to −1.85. Conditional on Δ, longer is if anything easier. |

**A dissociation on the crossing benchmark that the paper's own thesis predicts.** AUC of Δ_bp against two
outcomes, as a function of the number of draws:

| N | 1 | 5 | 8 | 60 | 200 |
|---|---|---|---|---|---|
| pKiss ever solves (the **in-loop** oracle) | 0.601 | 0.736 | 0.806 | 0.863 | **0.951** |
| panel-unanimous (the **honest** metric) | 0.708 | 0.825 | 0.824 | 0.758 | **0.711** |

Driving the search-free probe deeper on the in-loop oracle's own coordinate makes it a near-perfect
certificate of that oracle and a *worse* predictor of oracle-independent success. That is oracle
reward-hacking measured in a statistic that runs no search.

## 1. Declared choices

* **Δ estimator:** minimum over `N = 8` random compatible sequences, normalised by `|P(T)|`.
  `N = 8` is the AUC peak **on ids 0-29 alone** (0.876). Disclosure: I looked at the pooled curve first,
  which peaks at `N = 5`; `N = 5` has the better held-out AUC (0.835 vs 0.785). Using 8 is the
  fit-split-only choice and the conservative direction. Held-out AUC at `N = 8` is **0.785**.
* **Model:** Weibull cure mixture fitted on ids 0-29, `p_i(b) = π_i[1 − exp(−(λ_i b)^k)]`,
  `logit π = −3.124 − 3.291 z`, `log λ = −4.009 + 0.252 z`, `k = 1.539`.
  Cure vs no-cure LR = 8.95, p = 0.011; k free vs k = 1 LR = 6.26, p = 0.012.
* **Δ enters through reachability, not speed:** Δ in π LR = 7.83, **p = 0.005**; Δ in λ LR = 0.57,
  **p = 0.451**. Δ predicts *whether* a target can be hit, not *how fast* once it can.
* **Policy:** rank by `π_i λ_i` (monotone in Δ), fund the top `m`, and split the budget equally among the
  funded. `m` per budget is fixed now: `m = 6, 13, 18, 25` at `B/B0 = 0.10, 0.20, 0.50, 1.00`.
* **Split:** fitted on ids 0-29, applied to ids **30-59**. `B0 = 3684` pKiss calls/cell.
* **Deterministic mode on**, `max_pkiss = b_i`, 15 executions (cell seeds), paired across arms.

## 2. Known defect in the model, stated in advance

The cure model **over-predicts held-out solve counts by 1.57×** (predicted 2.20/cell, observed 1.40; the
no-cure model was 1.62×). The miscalibration is concentrated in the middle Δ tercile: predicted 19.8 wins
against 7 observed, where a smooth logistic π(Δ) assigns π ≈ 0.147 to targets that are in fact ≈ 0.
π(Δ) is sharper than a logistic in `z`.

Therefore **levels are not pre-registered; ratios are.** Predicted levels below carry a ×0.64 calibration
factor.

## 3. The predictions

Measured against the uniform arm at the same budget, same seeds, same targets.

| # | prediction |
|---|---|
| **P1** | The Δ-ranked arm beats uniform at every budget ≤ 0.50·B0. |
| **P2** | At `B = 0.20·B0 = 737`: ratio ≈ **2.25×** (policy 1.63 vs uniform 0.73 solves/cell, calibrated). |
| **P3** | At `B = 0.10·B0 = 368`: ratio ≈ **3.04×** (0.82 vs 0.27). |
| **P4** | At `B = B0` the two arms are **within 10%** (ratio 1.06). The policy does not help at full budget. Stated in advance so the favourable region cannot be reported alone. |
| **P5** | Random abstention on the same `m` does worse than Δ abstention. |
| **P6** | Length-based abstention on the same `m` does worse than Δ abstention. |
| **P7** | Equal-split-among-funded is within 5% of the model's unequal funding, i.e. the deployable policy is model-light: the mathematics earns *which* targets and *how many*, not *how much each*. |
| **P8** | Observed levels ≈ 0.64× the model's raw `E[solves]`. |

**The headline summary statistic is the frontier, not one ratio.** The equal-solves-at-`B0` number
(1.43×, or 1.31× charging Δ's own 240 calls) is the *least* favourable point on it, because both arms
saturate at `B0`. METHOD.md asked for a frontier to prevent cherry-picking a good ratio; the actual risk
turned out to be reporting the one bad ratio.

**Bracket:** uniform = 1.00× (null), oracle skyline = **7.41×** (497 of 3684 calls), policy predicted
1.43× at equal solves and 2.25-3.04× in the 10-20% budget region. A result outside `[1.00, 7.41]` is a
bug, not a finding.

## 4. What kills it

1. The Δ arm fails to dominate uniform at low budget → the efficiency claim dies even though the
   coordinate is real.
2. Random or length abstention matches Δ abstention (P5/P6 fail) → Δ is not what is doing the work.
   Partially pre-answered for length: β(length) flips negative given Δ.
3. `m` does not transfer across the declared split → the policy is fitted, not learned.
4. Observed levels miss even the ×0.64-calibrated prediction by more than 2× → the model is not usable
   for allocation and only the ranking survives.

## 5. What this cannot claim

One benchmark, one operator, 10/30 held-out targets ever solved, 1.40 solves/cell. The frontier will be
coarse and its intervals wide. A search-free coordinate carrying an allocation policy is the claim. Not a
general efficiency law; the second corpus the paper already owes is what would make it one.
