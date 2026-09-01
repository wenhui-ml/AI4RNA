# First-principles invariants this manuscript asserts

These are **not target values** and **not a supervision plan**. They are properties that must hold whatever the numbers turn out to be. Derive your own plan; use this list only to check that the manuscript did not violate something it itself relies on.

1. The interventional outcome must contain no folding model. Read the definition: if any term in it is a model output, the paper's central separation from profile-based scoring is void.

2. A matched stratified difference must drop strata missing an arm, never extrapolate into them. Check the weight is min(n_A, n_B) and that the dropped count is reported.

3. A permutation test after caliper matching must permute WITHIN strata. Permuting the treatment label across the matched set rebuilds a different matched set and narrows the null; the manuscript states the size of that error, so check the direction of its own correction.

4. A cluster bootstrap on 20 clusters is optimistic. Any interval whose width matters at the second decimal should be read with that in mind, and the paper should say so.

5. Skill on an exposed subset falling BELOW skill on an unexposed subset is not evidence of memorisation; it is evidence the exposed subset is harder. The control that separates them is a model that cannot have memorised anything.

6. Every exposure count computed against a subset of a training corpus is a LOWER bound. A sentence of the form 'not in the training set' is unsupportable from a subset.

7. A ratio of two contrasts is the estimator this work withdraws. Any number of that form anywhere in either manuscript is a finding, including in a table caption.

8. A predictor scored as a panel member must see SEQUENCE ONLY. One program appears in this work twice, once restrained by the construct's own probing data to build a reference and once unrestrained as a scored member. If the restrained form were scored, the outcome's own experiment would sit on both sides of the comparison.

9. The deposit's own probing-derived structure column is malformed on a large minority of rows -- the strings are not balanced dot-bracket. Any quantity computed against that column is computed against a defect. The well-formed subset is not a fallback either: it is a structure-class selection, the malformed rows being the more pseudoknotted ones.

10. A dot-bracket parser for this corpus must keep ONE STACK PER BRACKET TYPE. It is a pseudoknot benchmark and the strings use several bracket families; a single-stack parser silently mis-pairs and every downstream number inherits it.

11. The agreement axis and the readout comparison are defined on cell sets that DIFFER. Scoring one on the other's set reproduces published numbers to about 1e-3 and no better, which looks like a tolerance problem and is not.

12. A number printed to four decimals carries +/-5e-5 of rounding. A reproduction tolerance of 5e-5 against a printed value is arithmetically unreachable; targets must be read from artifacts.

13. A permutation-plus-bootstrap routine that draws both from one generator does not give the bootstrap a fresh stream. Reproducing such an interval requires advancing the generator by the permutation count first.

14. An interval that covers zero after a second method, where the first excluded it, has not refuted anything. Check whether the shift came from extra width or from a bias correction; those are different findings and the manuscript distinguishes them.

15. NOTHING in this work is model-free. A structure folded from probing data is a thermodynamic model restrained by an experiment, not an observation; the outcome the predictors are scored against is arithmetic on reactivities and contains no folding model, which is a different and much narrower statement. Any sentence conflating the two is a finding -- one such sentence reached a committed draft of the abstract and was corrected only after review.

16. The declared target enters this work in four places that must be named rather than waved at: it says which pair is the target, it supplies helix depth, it supplies the crossing indicator, and it gates eligibility. What it never does is serve as the reference the measured outcome is scored against. A claim of the form 'no reference structure enters' is too strong; the supportable form is 'no inferred or model-generated structure of these molecules enters the measured outcome'.

17. An equivalence margin is a choice. Check whether the manuscript defends its delta as universal or declares it as an operational tolerance and reports the statistic that lets you substitute your own. The second is defensible; the first is not.

18. A decision utility is also a choice. Recovery under a fixed synthesis budget is not the only experimental utility, and a paper that converts a benchmark margin into ONE decision has said nothing about information gain, cost per synthesis or active-learning efficiency.
