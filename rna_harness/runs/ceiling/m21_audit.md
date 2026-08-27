# M21 audit — is formation a function of the model's own free energy?

Declaration `m21_thermo_mediator_declaration.json`, frozen and pushed at `33c83c4f` **before**
`m21_thermo_mediator.py` existed. Offline, $0.

## Why this experiment exists

Three external reviews converged on one sentence about M16: *"G–C forms more readily than A–U
once depth is controlled"* is what the nearest-neighbour model already says — terminal A–U
penalty plus stronger G–C stacks. Read that way, M16 is a high-throughput reconfirmation of a
textbook fact, and the paper's centre of gravity slides to the corpus-bias result because that is
the only part that looks new. The complaint is correct and no rewording answers it. It needs a
quantity that separates the two.

`dG_local` is that quantity: what the NN model itself assigns to this one pair — its inner stack,
its outer stack, and the terminal AU/GU penalty on whichever side is a helix end — taken from
ViennaRNA 2.7.2's parameter tables (`eval_int_loop`, `param().TerminalAU = 0.50 kcal/mol`), not
hardcoded. Conditioning on it and re-estimating the identity contrast is a controlled direct
effect: what identity does through channels the model does not encode.

## The first run was wrong, and the audit is why it did not stand

The first run reported **retention 153.3%** — the declaration's `shape_1`, which says a retention
above 100% must be recorded as a shape and **not** read as stronger evidence. Two things were
wrong with it, both found by reviewing the code before touching the verdict:

**V1 — `dG_local` was undefined for 5.2% of records.** ViennaRNA's `eval_int_loop` returns a
finite but meaningless number when a "pair" is not canonical (probed: **+1.30 kcal/mol** for an
A–A partner). 233 of 4453 records have a non-canonical target pair on at least one stacking side.
Guarded: a non-canonical partner means no stack, which is also what the NN model does with it.
Largest single change 3.00 kcal/mol.

**V2 — the retention denominator compared two different populations.** The dG-stratified estimate
uses 363 pair-weights; the baseline used 902. Adding a stratifying dimension drops records, so
that ratio mixes **selection** with **mediation**. The denominator has to be the same records with
one fewer dimension.

| | estimate | pair-weights |
|---|---|---|
| full population, no energy dimension | **+0.0539** | 902 |
| retained subset, no energy dimension | **+0.0805** | 367 |
| retained subset, with energy dimension | **+0.0820** | 363 |

**149.3% of the reported 152.1% was selection.** The energy-overlap region — where a C–G pair and
an A–U pair can carry the same NN local free energy — happens to be a region where the identity
contrast is half again as large. That is a fact about the region, not about mediation.

## What stands

**Mediation through the nearest-neighbour model's own local free energy is indistinguishable
from zero.**

- same-population retention **101.9%**, mediated fraction **−1.9% [−10.6%, +6.8%]**
- placebo mediator (same dG values, scrambled within puzzle, same fixed cut points):
  **100.9% ± 4.4%**, real vs placebo **z = +0.22**
- bin count is not doing the work: **100.1% / 101.9% / 100.9%** at 3 / 5 / 8 bins
- the contrast itself survives: **+0.0820**, permutation **p = 0.0005**, cluster CI [0.0488, 0.1201]
- probes: P1 0/12, ladder δ=0 self-check 0.0, top rung 1.0. Detection threshold moves 0.02 → 0.10,
  which is the price of the finer stratification and is reported rather than hidden.
- K5 (rule 80): over 200 permutations the (weight, stratum-count) pair took **exactly one** value.
  Bins are cut from the pooled distribution once, before any contrast, so permuting labels cannot
  rebuild the matched set.

**The linear route would have said the opposite.** Product-of-coefficients through the energy gives
β_dG = −0.02286 per kcal/mol × a 1.7245 kcal/mol gap = **73.1% mediated** — a 75-point disagreement
with the stratified estimate. The assumption it rests on is testable and fails: within-stratum
centred rescue by energy bin is **−0.0085, +0.0214, +0.0122, −0.0019, −0.0222** across
dG = −6.45 → −1.46, which is not monotone. A mediation analysis that assumed linearity here would
have concluded that thermodynamics explains three quarters of the effect. It explains none of it.

**It is not only the terminal A–U penalty.** The contrast at helix ends is +0.0922 (p = 0.0005),
and at depth ≥ 2 — outside the penalty's domain entirely — it is **+0.0608 (p = 0.0005)**.
Depth 1 is null (+0.0124, p = 0.22289) and we cannot explain the non-monotonicity; it is reported.

**The three pair types do not lie on one rescue-versus-ΔG line.** Implied slopes −0.0313 (C–G−A–U),
−0.0439 (C–G−G–U), −0.0553 (A–U−G–U): spread 1.77×, and the directly estimated within-stratum
slope −0.0229 falls outside their range.

**G–U context** (reviewer's point, descriptive, n = 140): C–G−G–U is +0.1711 against G–U pairs
adjacent to another G–U (n = 31) and +0.0893 against isolated G–U (n = 109).

## Predictions staked before the code existed

| # | staked | outcome |
|---|---|---|
| 1 | retention ≥ 50% and p ≤ 0.005 | **right** (101.9%, p = 0.0005) |
| 2 | mediated fraction in 20–60% | **wrong** (−1.9%, interval covers 0) |
| 3 | effect still positive at depth ≥ 2 | **right** (+0.0608, p = 0.0005) |
| 4 | the three types are not collinear | **right** (spread 1.77×, β outside range) |
| 5 | overlap ≥ 200 pair-weights | **right** (363) |

## What this cannot say

`dG_local` is the **local** nearest-neighbour term. Coaxial stacking, dangling ends and tertiary
contacts are not in it, so the licensed reading is *"not the local nearest-neighbour free energy"*,
not *"not thermodynamics"*. The declaration wrote that sentence before the run.

The estimate lives in the energy-overlap region — C–G pairs with unusually weak stacking context
against A–U pairs with unusually strong context (bin 0 holds 776 C–G and **zero** A–U). That is a
positivity limit, it is where the question is answerable at all, and the selection component
(+0.0805 against +0.0539) says the region is not typical of the corpus.

Identity is not randomly assigned and `dG_local` is a function of it, so the CDE's identification
conditions are stronger than the estimator's algebra. We report the estimate and its stratification
and do not claim the conditions hold.

## Files

- `m21_thermo_mediator_declaration.json` — frozen `33c83c4f`
- `m21_thermo_mediator.py`, `.json`, `.log` — the corrected run
- `m21_thermo_mediator_SUPERSEDED_wrong_denominator.{json,log}` — the first run, kept
- `m21v_verify.py`, `.json`, `.log` — V1–V6
