# M26 audit — scoring the model on 4390 interventions

Declaration `m26_model_scored_declaration.json`, frozen and pushed at `97daa3d4` **before** the
runner existed. Offline, $0.

## What made this possible, and it was in the data all along

The compensatory double mutants are a **strict orientation flip**: CG→GC (1635), GC→CG (1426),
AU→UA (588), UA→AU (528), GU→UG (84), UG→GU (56). Not one changes the unordered pair type. And
**every analysable mutant differs from its own wild type at exactly the two mutated positions
inside the design window and is byte-identical everywhere else** (4317 of 4317 admitted, 0
dropped by the P1 assertion).

That is a controlled two-nucleotide intervention on a real synthesised molecule with its own wild
type as control, performed 4317 times. It means the thermodynamic model can be **scored** rather
than conditioned on: for each mutant, ViennaRNA's partition function states the equilibrium
probability that the flipped target pair forms — the model's direct prediction of what rescue
measures, computed over the **whole ensemble**, so coaxial stacking, dangling ends and non-local
competition are all inside it.

This is what removes the boundary M21 wrote into itself: *"not the local nearest-neighbour term,
not 'not thermodynamics'"*.

## D4 — the model is well calibrated, which is what makes D2 mean anything

| bin | n | P_model | rescue | rescue, centred within design |
|---|---|---|---|---|
| 0 | 864 | 0.0001 | +0.2994 | −0.0709 |
| 1 | 863 | 0.0762 | +0.4041 | −0.0021 |
| 2 | 863 | 0.6915 | +0.4375 | +0.0157 |
| 3 | 863 | 0.9791 | +0.4789 | +0.0248 |
| 4 | 864 | 0.9993 | +0.5038 | +0.0326 |

**Monotone.** Staked prediction (2) said it would not be, and it is wrong. That failure is the
best thing in the arm: the model has real, ordered skill here, so holding it fixed is holding
something substantial fixed.

## D1 — the model has skill

| predictor | within-design slope on rescue | permutation p |
|---|---|---|
| P_model, mutant | **+0.1206** | 0.0020 |
| P_model, wild type | +0.1325 | 0.0020 |
| ΔP = mutant − wild type | −0.0064 | 0.6547 |

K3 does not fire.

## D2 — and identity survives it

| | estimate | pair-weights |
|---|---|---|
| full population, no model dimension | +0.0539 | 902 |
| retained subset, no model dimension | **+0.0689** | 577 |
| retained subset, holding P_model fixed | **+0.0663** | 569 |

**Retention 96.3%.** Permutation p = 0.00050, null SD 0.0129, cluster interval [0.0334, 0.1051].
The placebo mediator — the same P_model values permuted within puzzle, cut at the same fixed edges
— retains **98.7% ± 2.8%**, putting the real mediator at **z = −0.87**. Mediated fraction
**+3.7%, [−1.7%, +9.1%]**.

Ladder: δ=0 self-check **0.0**, detection threshold **0.05** against an observed **+0.0663**, so
the arm has power for the effect present.

**The licensed statement is now:** whether a designed base pair forms carries identity information
that this thermodynamic model's *ensemble prediction for that exact molecule* does not encode —
measured on molecules that were built, with a controlled intervention, against a prediction that is
calibrated and monotone.

## D3 — three more model quantities, same machinery

| held fixed | CDE | same-population baseline | retention |
|---|---|---|---|
| P_model of the wild type | +0.0596 | +0.0557 | 106.9% |
| ΔP, mutant − wild type | +0.0531 | +0.0494 | 107.5% |

## D3c — the local cost of the flip, and a declaration number that was wrong

The declaration predicted this arm would be **underpowered rather than null**, on the basis of a
δΔG spread of SD 0.16 recorded in its own BLIND_STATUS. **That number was wrong.** The
exploratory probe that produced it cached ViennaRNA fold compounds by `(puzzle, method)` instead of
by sequence, so every mutant in a cell was evaluated against the first mutant's sequence. Keyed
correctly, **SD(δΔG) = 0.4906** and 38.8% of flips move the local energy by more than
0.3 kcal/mol.

With the correct spread the arm has power: injecting a known slope of −0.01 is recovered in 90% of
draws. The measured within-design slope is **+0.0127 (p = 0.0749)** — the *opposite* sign to the
model's prediction, since a flip that costs local free energy should be rescued less.

**Staked prediction (5) is wrong, and it was wrong because the number it was staked on was wrong.**
The arm is a null with power, not an underpowered arm.

## Staked predictions

| # | staked | outcome |
|---|---|---|
| 1 | the model has skill | **right** (+0.1206, p = 0.0020) |
| 2 | calibration not monotone | **wrong** — it is monotone, and that strengthens D2 |
| 3 | retention ≥ 50% holding the ensemble prediction fixed | **right** (96.3%) |
| 4 | mediated fraction's interval contains zero | **right** ([−1.7%, +9.1%]) |
| 5 | the flip's local-energy arm is underpowered, not null | **wrong** — it has power and is null; the declaration's spread figure came from a cache keyed by cell instead of by sequence |

Two of five wrong, and one of those two was wrong because of a defect in a number the declaration
itself recorded. Both are in the table.

## What this cannot do

**The intervention is not ours.** The dataset's authors flipped every testable pair; we scored a
model on an intervention that had already been performed. The paper may not say we ran a wet
experiment.

**P_model is one implementation and one parameter set.** A second folding model could change the
verdict, and the licensed statement is about *this* model. That is the next thing to add.

The orientation flip does not change the unordered pair type, so this does not deliver a causal
effect of identity; it delivers whether the model's prediction about a performed intervention was
borne out.
