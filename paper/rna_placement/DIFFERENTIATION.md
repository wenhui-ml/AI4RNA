# R1 — changing the object, not the wording

The two papers were going to overlap on the thing that matters: both rested on the pair-identity
effect, and both needed placement conditioned on. Citing across that boundary does not fix it —
the same empirical claim would appear twice. So R1 changes what it is about.

## New object

**Corpora produced by design algorithms are confounded by their generators — an estimator study.**

RNA is one instance, not the subject. The assets R1 takes are the ones that were never about RNA:

| asset | what it is | RNA-specific? |
|---|---|---|
| M25 arm A | controlled-truth benchmark of four estimators; marginal bias = β·g, predicted −0.0236 vs measured −0.021…−0.024 | no |
| M23c | analytic bound on residual within-caliper imbalance; panel-matched 0.95–3.16% vs panel-free 22.8–37.7% | no |
| R2 / R3 / R4 | same-population denominator; placebo mediator; linearity before a product of coefficients | no |
| the caliper-permutation defect | permuting treatment rebuilds the matched set; null 2–3× too narrow | no |
| M13 / M16 / M20 / M22 / M24 | the RNA instance | yes — demoted to running example |

## What R1 must stop claiming

**Rule 80 is not ours.** The matching literature has it: *"it does not consider whether permuted
versions of treatment would have led to the same matched-pair configuration, leading to substantive
Type I error violations"* (Pimentel et al., matching under rolling enrollment). R1 may report the
**magnitude on a real corpus**; it may not present the defect as a discovery. This was written down
after finding it, and it costs R1 its most distinctive-looking technical claim.

**The pair-identity effect is not R1's headline.** It is the NM paper's. Keeping it here is the
salami-slicing exposure, in the plainest form: one empirical claim, two papers.

## Where the literature actually is

Three neighbours, all qualitative where R1 is quantitative:

- **Chaney, Stewart & Engelhardt** (RecSys 2018), algorithmic confounding — the same structure
  (the data you analyse was produced by an algorithm), but simulation only, no estimator theory.
- **Preference leakage in LLM-as-a-judge** (arXiv 2502.01534) — generator and judge related,
  leakage score 23.6%; reports the contamination, never asks by how much a contrast is attenuated.
- **Protein design benchmarking** — "RFdiffusion favours aromatic-rich sites"; "systematic,
  reproducible deviation in model scoring reflecting training data properties"; ProtDBench
  standardises verifiers. Named as a hazard, never estimated.

A targeted search for the combination — design-generated corpus, attenuation bias, estimator —
returns nothing. **The white space is the estimator layer**, and it is what R1 has.

## Breaking the dependency, concretely

M25 arm A already fired K2: *matching is not necessary; regression adjustment does the same job.*
So the NM paper conditions on placement with the simplest defensible method and writes two
paragraphs of Methods. **It does not need R1's theory.** R1 studies why, and how much.

Test before submission: delete the citation to R1 from the NM manuscript. If anything breaks, the
boundary is wrong.

## Second domain — obtainable, and thinner than it first looked

Zenodo 15722219 (CC-BY-4.0), `final_dataset.csv`, 82 MB, downloaded. 3676 usable designs, 16
targets, experimental binder/non-binder (394/3275), 312 features.

**There is no design-method column.** `source` is the publication or campaign. Reconstructing
method from author (Cao → Rosetta, Watson/Bennett → RFdiffusion) leaves:

| target | crossing |
|---|---|
| IL7Ra | Watson 94 / Cao 77 |
| TrkA | Watson 95 / Cao 33 |
| InsulinR | Watson 95 / Cao 22 |

**3 clean crossed targets, 416 designs.** EGFR was counted at first and should not be: its
"Adaptyv" arm is 302 separate competition entrants (`Reed_HARRISON`, `akshayc`, `alan_blakely`, …)
using unknown tools — a mixture, not a method. Cao's own ids (`bcov_v3_…`, `longxing_…`) are also
several pipelines. FGFR2's 2123 designs are single-method and cannot contribute a contrast.

So: a **corroborating instance**, not a co-equal corpus, and R1 says so in those words. The
alternative — claiming two equal domains — would be the second overstatement in a document written
to remove the first.

## Next action

Estimate β·g on the three crossed targets: does conditioning on the design cell move the contrast
in the direction the theory predicts, on data from a different field? A null there is reportable
and bounded; a positive is what makes R1 cross-domain.
