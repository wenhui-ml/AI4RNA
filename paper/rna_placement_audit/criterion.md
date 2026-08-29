# The eligibility criterion, in prose

Enough to rebuild the admitted set from `OK7a_M2R_data.v4.5.1.csv` **without our code**. Written
as a rule, not as an implementation. No threshold below can be tuned to raise the yield.

## Unit

One row of the CSV is one construct. A construct is admitted as an **intervention** when it is a
compensatory double mutant of a declared target base pair of its own wild type.

## The rule, in order, with every refusal counted

1. The row must carry an outcome and pass the corpus' own signal filter: `rescue_factor` present
   and `SN_filter == 1`. Otherwise **refused: no outcome**.
2. The row's wild type is the row of the same `(puzzle, method)` whose `mutA` and `mutB` are both
   empty. If there is none, **refused: no wild type**.
3. `mutA`, `mutB` are 1-based positions inside the design window. Convert to absolute 0-based
   indices with `a = mutA + sub_start - 2`, `b = mutB + sub_start - 2`. Unparsable positions are
   **refused**.
4. Parse `target_structure` into a pair map. `(a,b)` must be a declared pair of it. Otherwise
   **refused: not a declared pair**.
5. The unordered wild-type pair type at `(a,b)` must be one of `CG`, `AU`, `GU`. Otherwise
   **refused: non-canonical**.
6. **The intervention must be exactly two positions.** Compare the wild-type and mutant sequences
   over the design window `[sub_start-1, sub_end)`; the set of differing positions must equal
   `{a,b}` exactly. Anything else is **refused: not exactly two**. This refusal is the one that
   makes the experiment controlled, and its count is reported in the manuscript.

Ambiguity is refused rather than resolved at every step.

## Covariates carried, never used as selection

- **helix depth** `min(dep, up)`, where `dep` counts consecutive `k >= 0` with `(a-k-1, b+k+1)`
  paired and `up` counts consecutive `k >= 0` with `(a+k+1, b-k-1)` paired. Read off the
  **declared target structure**. No model, no folding, no training data.
- **crossing**: whether any declared pair `(x,y)` crosses `(a,b)`, i.e. `x < a < y < b` or
  `a < x < b < y`. The corpus is a pseudoknot round, so crossing is carried as a covariate and
  never as a selection rule.
- **design cell** `(puzzle, method)`.

## The outcome

`rescue_factor` is used **unaltered and untruncated, negative values included**. It is a released
column of the corpus; the manuscript reproduces it from the per-nucleotide reactivities rather
than quoting a formula from a paper, and states that it does not assert the normalisation is
identical. **The outcome contains no folding model.** If your reimplementation introduces one,
you have changed the claim.
