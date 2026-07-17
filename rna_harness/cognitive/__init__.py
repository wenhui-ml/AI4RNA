"""Domain-agnostic cognitive core for economical tool orchestration.

The scarce capability is NOT a better solver (DesiRNA already solves Eterna100-V2 to 97). It is the
META-PROCESS a human expert brings: analyse a problem, recognise its difficulty/boundary, pick the
cheapest tool + budget likely to work, invoke, verify, and self-revise on failure -- to solve ANY
problem at the LOWEST cost. Domain knowledge lives entirely in the TOOLS (ViennaRNA / DesiRNA /
pKiss); this package is domain-agnostic ML engineering:

  features (mechanical, zero biology)  ->  learned difficulty/cost predictor  ->  cost-min routing
  ->  invoke tool  ->  verify (correct oracle)  ->  self-revise / escalate

Falsifiable thesis: at equal solve quality, this core reaches solutions at materially LOWER total
cost ($) than any fixed policy (always-DesiRNA, always-cheap, random), and the advantage GENERALISES
to problems it was not tuned on (RNAInvBench pseudoknots). If it degenerates to "always call the SOTA
at max budget" it is a wrapper, not a cognitive core -- hence every claim is measured in cost-at-quality
against fixed baselines.

The verifier (`tools.verify`) is the FLOOR everything else stands on: `solved` is only as trustworthy
as it. Two invariants:
  * proposer != judge. Every tool's output -- including DesiRNA's own "I solved it" -- is RE-FOLDED by
    `verify` under our declared oracle (DesiRNA aligned with -p 2004). A solver never certifies itself.
  * `solved` means "solved UNDER the declared predicate", never physical/biological truth. There is no
    absolute physical oracle here; the formal truth value is the value of a DECIDABLE PREDICATE obtained
    by evaluation (fold once, compare) -- exactly as a SAT assignment's truth is `phi(x)` evaluated. The
    model IS the specification, so within it the truth is absolute, reproducible, and search-free to check.

Open-boundary solving = navigating three verifier axes (all seated in `verify`), NOT just harder search:
    axis 1  WHICH oracle   nested -> ViennaRNA MFE ; pseudoknot -> pKiss  (wrong choice = category error)
    axis 2  HOW STRICT     'match' | 'umfe' | 'pfe' -- a declarable contract, not a hidden constant
    axis 3  NO oracle      functional/3D specs have no computable predicate -> ABSTAIN, never claim solved
Today the core learns to route SOLVERS; making verifier SELECTION a first-class routing axis (recognise
out-of-oracle -> switch/abstain, stay economical under an expensive oracle) is what Track B (pseudoknots
under pKiss) forces and is the next build.
"""
