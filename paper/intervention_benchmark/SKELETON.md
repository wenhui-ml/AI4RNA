# Intervention benchmark — skeleton

**Target:** Nature Methods (Article). **Status:** structure fixed 2026-08-27, prose not started.

**One sentence.** Compensatory mutagenesis already performed in chemical-probing corpora
constitutes a controlled two-nucleotide intervention benchmark; scoring structure predictors on it,
rather than on agreement with a reference structure, is unsaturated, free of the pair-set degree of
freedom, and shows that the thermodynamic model has the base-pair-identity axis reversed.

---

## What this paper is not

- Not a claim about pair identity's causal effect. An orientation flip preserves the unordered pair
  type; it changes stacking context. Written into the Methods, not the Discussion.
- Not "a general framework for auditing design-generated corpora". That belongs to the companion.
- Not "all existing models". Two model classes, named, with their parameterisations named.

## Withdrawn before writing begins

Every retention / mediated-fraction number from M26, M26b, M26c, M26d: **96.3%, 100.7%, 100.2%,
95.1%, and the four-parameter panel 96.3–100.9%.** m26r3 calibrated the metric with the real
predictor as mediator and a synthetic outcome of known mediated fraction; it does not track truth
at this corpus's signal-to-noise (neural arm correlation **−0.70**, MAE 0.47; thermodynamic arm MAE
1.74, one sweep point at −842%). The withdrawal is stated in the main text with the calibration
table, not in an appendix. `m26r3_audit.md` is the source.

Consequence: **the paper's second claim is a sign, not a ratio.**

---

## Figures

**Fig 1 — the readout is saturated and it has a free parameter.** (M25 arm B)
- a: inferred-structure hit rate on the target pairs, CG 0.898 / AU 0.908 / GU 0.693
- b: same molecules, same strata, same 4293 pairs — identity contrast +0.0527 (interventional)
  vs +0.0198 (conventional); paired cluster bootstrap of the difference **+0.0329 [0.0024, 0.0638]**
- c: change the pair set, keep everything else — the conventional readout moves **4.50×**; the
  interventional one is undefined off the compensatory arm, so it has no such knob
- d: why a marginal estimator is biased here — predicted β·g = −0.0236, measured −0.021…−0.024 (M25 arm A)

**Fig 2 — the interventions, and what admits them.** (mining criterion + designaudit)
- a: what an orientation flip is; the four inclusion clauses
- b: exclusion counts on OK7a — not_two_positions 9125, pair_not_canonical 73, no_outcome 27,
  admitted 4317 {CG 3061, AU 1116, GU 140}
- c: `designaudit selftest` reproducing 4317 / +0.0539 / 902 / +0.1206 / from the public CSV

**Fig 3 — the instrument has resolution, and it was calibrated before it was used.**
- a: rescue vs binned predicted probability, ViennaRNA — monotone, 0.2994 → 0.5038
- b: within-stratum skill slopes, ViennaRNA **+0.1206** (p 0.0020), RibonanzaNet **+0.2069** (p 0.0020)
- c: leakage ladder — L1 exact 0/14416; design-window ≥0.8 identity 6.37%; ≥0.999 264;
  clean-subset slopes +0.2155 / +0.1288 essentially unchanged

**Fig 4 — the result.** (M26v, pending verification at time of writing)
- a: three matched contrasts on one axis — outcome **+0.0539**, ViennaRNA's own prediction
  **−0.0304**, RibonanzaNet's **+0.0606**
- b: per-design-cell sign of the ViennaRNA contrast, with the sign test
- c: per depth × crossing breakdown — is the sign general or one stratum
- d: product-route upper bounds, 6.8% and 23.2%, labelled as bounds under R4

**Fig 5 — it transfers (M28, pending).** natural coronavirus corpora; usable-flip counts after the
SN cut; two probes as an internal replication.

---

## Sections

1. **Introduction.** Structure benchmarks ask whether the predicted structure is right. Existing
   probing corpora contain perturbations that were performed and measured. Ask instead whether the
   model predicts the response.
2. **Results / The conventional readout on this corpus.** Fig 1.
3. **Results / Recovering the interventions.** Fig 2. Criterion, exclusions, tool.
4. **Results / Scoring predictors.** Fig 3. Calibration before diagnosis. Leakage ladder here, not
   in supplementary.
5. **Results / What the instrument reads out.** Fig 4 + the withdrawal of the retention metric with
   its calibration table.
6. **Results / A second corpus.** Fig 5.
7. **Discussion.** What an orientation flip can and cannot say. Why the interventions not being ours
   is the point. One paragraph, no claims, on the primitive beyond RNA.
8. **Methods.** Mining criterion formally. rescue_factor reproduced from raw reactivity
   (Pearson +1.0000) — the outcome contains no model. Conditioning on placement: regression and
   matching agree (M25 arm A, K2), so **"condition on placement", never "matching is required"**.
   Estimator details cite the companion; nothing here depends on it.

## Companion boundary

Cited, never load-bearing. Nothing in Methods requires the companion to be read. If a reviewer
deletes the citation the paper still stands — that is the test, and it is checked before submission.

## Open before submission

- [ ] M26v: the sign, verified against a second implementation, per cell and per stratum
- [ ] M28: natural corpus, power measured before any claim
- [ ] refs.bib: the probing benchmark's own citation
- [ ] every measured number reaches LaTeX through a macro; `make_check` fails the build on a
      hand-typed decimal
