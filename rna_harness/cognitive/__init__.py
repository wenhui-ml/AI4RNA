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
"""
