# The estimators, in prose and symbols

Enough to reimplement from the public CSV. No function of ours is named.

## Matched stratified difference in means

For a binary exposure `T in {A,B}` (here: target pair type G-C against A-U) and strata `s`:

    Delta = sum_s w_s * ( mean(Y | T=A, s) - mean(Y | T=B, s) )  /  sum_s w_s
    w_s   = min( n_A(s), n_B(s) )

**Strata missing an arm are dropped, never extrapolated into.** Report how many were dropped and
what weight they carried; the count is part of the estimand.

Two stratification sets are used and they answer different questions:

- **causal set** `s = (design cell, min(depth,3), crossing)` -- holds fixed what a design program
  chose, because placement confounds pair type;
- **prediction set** `s = (design cell)` alone -- for the question an experimenter faces given a
  design and a budget. Depth is information the experimenter already holds, so removing it would
  remove part of the answer.

Using one set to answer the other's question is the single easiest way to mis-audit this paper.

## Inference

- **Permutation**: labels are permuted **within stratum**. Permuting across the matched set
  rebuilds a different matched configuration and narrows the null; the manuscript quantifies that
  error rather than asserting it.
- **Bootstrap**: resample the **20 puzzles** (clusters), not the interventions. One interval in
  the companion manuscript is the exception -- it resamples repeated measurement groups -- and it
  is flagged where it appears.
- **Differences of two quantities** are computed **inside each resample** and the interval is
  taken on the differences. Two separately-resampled intervals do not answer a question about a
  difference. The manuscript retracts one comparison for exactly this reason.

## Within-design rank agreement

Within each design cell holding at least 8 admitted interventions, the Spearman rank correlation
between a score and the measured rescue factor; averaged across cells weighted by cell size;
cluster bootstrap over puzzles; permutation shuffling the outcome **inside the cell**.

## Standardisation before comparing scales

Two contrasts on different outcome columns are **not** comparable raw. Standardise each by the
pooled within-stratum standard deviation of its own outcome before comparing magnitudes. A
recomputation that compares them raw has reproduced a different claim from the one made.

## The quantity this work withdraws

A **mediated fraction** -- a ratio of two contrasts, presented as the share of an effect running
through a mediator. Calibration against an injected known truth shows the estimator's reading
covers only a small part of the range the truth covers. **Any number of that form, anywhere in
either manuscript including a table caption, is a finding.**
