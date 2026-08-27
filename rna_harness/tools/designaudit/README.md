# designaudit

Score a structure predictor on interventions that have already been performed.

A conventional RNA structure benchmark asks whether a predicted structure agrees with a reference.
On a corpus where the reference is already recovered for ~90% of the pairs being compared, that
question has little dynamic range left, and its answer moves several-fold with the choice of which
pairs enter the comparison. `designaudit` asks a different question, on data that already exists:

> when the molecule was actually perturbed, did the model's prediction move the way the
> measurement moved?

Mutate-map-rescue libraries contain **compensatory double mutants**: a wild type and a mutant
differing at exactly two positions inside the design window, those two positions being a declared
base pair, with the two bases exchanged so the unordered pair type is preserved. That is a
two-nucleotide experiment with its own control, executed and measured, sitting in published data.
This package recovers them and scores predictors on them.

## Install

Pure Python; `numpy` is the only hard dependency. ViennaRNA is optional and only needed for the
bundled `vienna` predictor.

```bash
export PYTHONPATH=/path/to/rna_harness/tools:$PYTHONPATH
python -m designaudit selftest
```

## Use

```bash
# what interventions does this corpus contain?
python -m designaudit mine     --corpus OK7a_M2R_data.v4.5.1.csv

# a pair-type contrast with the design conditioned on
python -m designaudit contrast --corpus OK7a_M2R_data.v4.5.1.csv --levels CG,AU

# score a predictor on the interventions, and read what it does not explain
python -m designaudit score    --corpus OK7a_M2R_data.v4.5.1.csv --predictor vienna --levels CG,AU

# your own predictor: any callable (sequence, i, j) -> float
python -m designaudit score    --corpus ... --predictor mypkg.mymodel:pair_score --levels CG,AU

# reproduce the published headline numbers from the corpus
python -m designaudit selftest
```

As a library:

```python
from designaudit import corpora, mining, scoring, core

constructs = corpora.openknot_m2r("OK7a_M2R_data.v4.5.1.csv")
interventions, report = mining.mine(constructs)
corpora.annotate_context(interventions)
print(report)     # mined 4317 interventions from 13702 constructs (31.5%)
```

## The three readouts

A predictor comes back with three numbers, meant to be read in this order.

| readout | what it asks | why it comes first |
|---|---|---|
| **sensitivity** | does the prediction move with the measurement, inside the design? | if this is null, nothing below can be read as "beyond the model" — the model predicted nothing to be beyond. `Instrument.readable` records that and the repr says so. |
| **calibration** | binned prediction against binned outcome; monotone? | this is what licenses treating the predictor's scale as meaningful rather than merely rank-ordered |
| **residual** | what a named sequence property still explains once the prediction is held fixed | the point of the instrument: a predictor that scores well on the first two and still leaves a residual has been shown to be missing something specific |

## What the mining criterion admits, and what it refuses

Inclusion is fixed. There is no threshold to tune and no argument that increases the yield.

- same length; differ at **exactly two positions inside the declared window**
- those two positions form a pair in the declared target structure
- the wild-type pair is canonical (A-U, C-G, G-U)
- the substitution is an **orientation flip**: `mut[i] == wt[j]` and `mut[j] == wt[i]`

Everything else is excluded under a named reason and counted. A mutant admitting more than one
wild type, or whose two changed positions are explained by more than one declared pair, is
excluded as `ambiguous` rather than resolved by a rule — resolving it would let the corpus choose
which intervention it supplies. Confidence (signal-to-noise, read depth) is carried and reported
but **never decides admission**.

A corpus that yields nothing yields nothing, and `MiningReport` says so rather than degrading
quietly.

### The window is not optional

Constructs in these libraries carry adapters, barcodes and padding outside the designed region.
Two constructs differing only by a compensatory flip inside the window can differ at a dozen
positions outside it. Mining whole sequences returns nothing, and *nothing* is indistinguishable
from *this corpus has no interventions* unless the window is declared. We produced exactly that
false negative on RMDB deposits before catching it; `corpora.py` exists so the next adapter has to
answer the window question explicitly.

## Four rules enforced as code

These are in `core.py` and they are checks, not documentation.

- **R1 — design-fixed randomisation.** When the comparison set is built under calipers, permuting
  the *treatment* rebuilds that set instead of relabelling it, and the resulting null is 2–3x too
  narrow. Permutation is over the **outcome** inside a fixed stratification. `permute="treatment"`
  exists and warns, so the difference can be demonstrated rather than asserted.
- **R2 — same-population denominator.** A mediated fraction is `1 - conditioned/baseline`, and the
  baseline must be computed on the records surviving the extra stratifying dimension. Against the
  full population the ratio mixes selection into mediation; `mediate()` warns with both numbers
  when they diverge. Using the wrong one once turned a 102% retention into 153%.
- **R3 — placebo mediator, always run.** Adding any stratifying dimension shrinks a matched
  contrast mechanically. A mediator drawn from the same marginal, cut at the same edges, unrelated
  to the outcome, measures that shrinkage. `mediate()` will not return a mediated fraction without
  it.
- **R4 — linearity before a product of coefficients.** The stratified and product-of-coefficients
  routes agree only when the conditional mean is affine in the mediator. `mediate()` tests it bin
  by bin and marks the product route inadmissible when it is not. On our corpus the two routes
  disagree by 75 percentage points.

## What this cannot do

- **The interventions are not ours.** They were performed by the corpus authors, uniformly, on
  every measurable pair. This is scoring a model on an executed intervention, not a prospective
  experiment.
- **An orientation flip is narrow.** It changes nearest-neighbour stacking while preserving the
  unordered pair type. It tests whether a model's prediction responds correctly to a substitution;
  it does not by itself give a causal effect of pair identity.
- **Which pairs got compensatory mutants was chosen by the experimenters.** Mining precision is
  not representativeness. That is why every estimate here conditions on the design, and why
  `placement()` exists to report how differently the levels were placed before any outcome is
  touched.
- **One verified corpus.** `corpora.py` ships one adapter because one is what we have verified end
  to end. Transfer is a claim about corpora we have not tested.

## Selftest

`python -m designaudit selftest` re-derives the published headline numbers from the public CSV,
independently of the run scripts, and prints every check whether it passes or fails. Tier 1 needs
only numpy; tier 2 needs ViennaRNA and is skipped **loudly** when absent — a skip is not a pass.
Tolerances are stated in `selftest.py` and are never widened to make a check pass.
