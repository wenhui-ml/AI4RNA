# Turning Δ into an efficiency result — method before code

Branch `rna-ceiling`. Written before the experiment, so the design cannot be chosen after seeing the answer.

## 1. What Δ is, and one flaw in what I already published

For target `T` with pair set `P(T)`, the **compatible set** is

```
S(T) = { x ∈ Σⁿ : x_i x_j ∈ WC ∪ GU  for every (i,j) ∈ P(T) }
```

and the quantity of interest is

```
Δ*(T) = inf over x ∈ S(T) of [ E(x,T) − E(x, MFE(x)) ]        (energy form, nested targets)
Δ*_bp(T) = inf over x ∈ S(T) of |pKiss(x) △ P(T)| / |P(T)|     (structural form, crossing targets)
```

We estimate it by `Δ̂_N = min` over `N` draws from `S(T)`. Three properties follow, and the third is a
defect in work already committed.

**(a) One-sided and consistent from above.** `Δ* ≤ Δ̂_N` always, and `Δ̂_N` is non-increasing in `N`. So Δ̂ is
an *upper bound* on the true margin, never an unbiased estimate of it.

**(b) Extreme-value structure gives the correction.** `Δ̂_N` is the minimum of `N` iid draws from the
distribution of `gap(x)` over `S(T)`. If that distribution is regularly varying near its left endpoint,

```
Δ̂_N  ≈  Δ*  +  c · N^(−1/α)
```

so fitting Δ̂ at several `N` and extrapolating to `N → ∞` estimates `Δ*` and removes the `N` dependence.
This is the right tool, not "take more samples".

**(c) The defect.** `info_budget.py` drew `N = 100 / 50 / 30` **by target length**. Long targets therefore
got fewer draws, inflating their Δ̂, and long targets are also more likely to be unsolved. **The Eterna
AUC 0.867 is confounded by a length–sample-size coupling and must be re-run at fixed `N`.** The crossing
result (`pk_margin.py`, AUC 0.824) and the pool comparison (`pool_margin.py`) both used a uniform `N = 60`
and are unaffected. Fixing this is task zero, and the corrected number may be lower.

## 2. The decision problem, stated

A budget `B` of oracle calls is split across targets. Spending `b` on target `T` succeeds with probability
`p(T, b)`. Δ is a **search-free** predictor of `p`, available before any call is made.

The existing abstention result (8–13× cost cut at identical held-out quality) is *learned*: it needs some
search before it can decide. Δ's contribution is that the decision costs nothing.

**Headroom, measured, not assumed.** On the 15-cell panel-operator sweep:

| | median pKiss calls |
|---|---|
| a success | 23.0 |
| a failure | 79.5 (3.46×) |
| a target never solved in any of 15 cells | 216.0 |
| a target solved at least once | 32.0 (6.75×) |

**86% of all pKiss calls go to the 39 targets that never succeed.** A policy with oracle knowledge of the
outcome would therefore save 86% of the budget at zero cost in solves. That is the skyline; Δ's job is to
approach it while blind to the outcome. Had this ratio been below 1 — doomed targets failing fast and
cheap — the whole experiment would have been pointless, which is why it was checked first.

## 3. Design

* **A frontier, not a point.** Report solves against budget over a sweep of thresholds, plus the two
  numbers people quote: cost at equal solves, and solves at equal cost. One ratio from one threshold is
  a cherry-pick.
* **Bracketed.** Uniform allocation is the null; oracle allocation (skip exactly the never-solved targets)
  is the skyline. A policy that does not sit strictly between them is not doing anything.
* **Paired.** Same targets, same seeds, same executions; only the allocation rule differs. The project's
  own execution-variance result forbids comparing separately-run counts.
* **Declared split.** The Δ threshold is fitted on ids 0–29 and applied to 30–59, stated before running.
  Δ is search-free so the split costs nothing, but an undeclared threshold fits the frontier to its own
  test set.
* **Deterministic mode on** (`deterministic=True`, App. B of the paper). Under the wall-clock regime the
  realized budget is a function of machine load, which makes any cost comparison meaningless.

## 4. What would falsify it

1. **The frontier fails to dominate uniform allocation.** Then Δ predicts solvability but not *where the
   budget goes*, and the efficiency claim is dead even though the coordinate is real.
2. **Δ̂ at fixed `N` loses its predictive power** (item 1c). Then the Eterna result was an artifact and
   only the crossing result survives.
3. **The threshold does not transfer across the declared split.** Then the policy is fitted, not learned.
4. **The saving is not attributable to Δ.** Control: a policy that abstains on the same *number* of targets
   chosen by length, or at random, must do worse. If random abstention matches Δ-abstention, Δ is not what
   is doing the work.

Control 4 is the one most likely to bite, and it is cheap, so it runs alongside rather than afterwards.

## 5. What this cannot claim

The pool gives 21/60 ever-solved and 3.2/60 per cell, so the frontier will be coarse and its confidence
intervals wide. This is a demonstration that a search-free coordinate can carry an allocation policy, on
one benchmark, at one operator. It is not a general efficiency law, and the second corpus the paper
already owes is what would make it one.
