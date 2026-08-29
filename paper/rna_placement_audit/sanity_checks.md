# First-principles invariants this manuscript asserts

These are **not target values** and **not a supervision plan**. They are properties that must hold whatever the numbers turn out to be. Derive your own plan; use this list only to check that the manuscript did not violate something it itself relies on.

1. The interventional outcome must contain no folding model. Read the definition: if any term in it is a model output, the paper's central separation from profile-based scoring is void.

2. A matched stratified difference must drop strata missing an arm, never extrapolate into them. Check the weight is min(n_A, n_B) and that the dropped count is reported.

3. A permutation test after caliper matching must permute WITHIN strata. Permuting the treatment label across the matched set rebuilds a different matched set and narrows the null; the manuscript states the size of that error, so check the direction of its own correction.

4. A cluster bootstrap on 20 clusters is optimistic. Any interval whose width matters at the second decimal should be read with that in mind, and the paper should say so.

5. Skill on an exposed subset falling BELOW skill on an unexposed subset is not evidence of memorisation; it is evidence the exposed subset is harder. The control that separates them is a model that cannot have memorised anything.

6. Every exposure count computed against a subset of a training corpus is a LOWER bound. A sentence of the form 'not in the training set' is unsupportable from a subset.

7. A ratio of two contrasts is the estimator this work withdraws. Any number of that form anywhere in either manuscript is a finding, including in a table caption.
