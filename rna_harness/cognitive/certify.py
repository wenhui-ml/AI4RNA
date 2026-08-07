"""The user-facing entry point: take an RNA design request, return a CERTIFICATE.

WHO THIS IS FOR. Someone who does not know RNA. They can state a target shape; they cannot make the five
decisions that stand between that shape and a trustworthy answer, and every one of those decisions, made
wrong, returns an answer that LOOKS right:

    decision                                what goes wrong if the user guesses          measured
    which energy model                      optimiser and verifier disagree              sequences come back bp 1-2 off
    does my target have a pseudoknot        ViennaRNA MFE drops '[' and ']'              bp=0 becomes a FALSE 'solved'   <- this one really is false
    what does "solved" even mean            match / umfe / pfe are different questions   survival 1.000 / 0.902 / 0.066
    how much should I spend                 flat runs pay for every target equally       72/100 for $0.21 vs 84/100 for $1.10
    can I trust this answer                 one heuristic oracle is not adjudication     43 of 43 claims went unconfirmed

The whole point of the certificate is the last row. Under the single pseudoknot oracle the standard tool
claims 43 of 60 targets and INDEPENDENT ADJUDICATION confirms none of them (runs/ceiling, exact McNemar
p=1.58e-30). So this module has a third verdict besides "here it is" and "could not find one":

    certified       a sequence that survives the declared criterion
    knot_confirmed  (pseudoknot targets only) the CROSSING STEMS -- the pairs that make the target a
                    pseudoknot at all -- are fully reproduced by at least one independent predictor. A
                    weaker claim than `certified` because it speaks about the knot, not the whole structure,
                    and a CALIBRATED one: 84.0% sensitivity on known natural structures at 0.4% false accept
                    on random sequences (runs/ceiling/calibrate.py).
    not_found       searched the declared budget, nothing survived
    cannot_certify  candidates satisfied SOME oracles and the independent panel disagreed, or a required
                    oracle could not run. Being able to say this is the feature, not a shortfall.

HOW STRONG IS `certified`, AND HOW WEAK IS `cannot_certify`. Both directions are measured, and the second one
is easy to overstate -- I did. The archived native-sequence positive control (runs/axis1_native_control, 250
Pseudobase targets) asks each predictor to fold the WILD-TYPE sequence back to its own known structure:

    exact recovery of the native structure   pKiss 32.4%   ProbKnot 5.6%   ShapeKnots 7.6%
    crossing-pair recall on the same natures pKiss 0.85    ProbKnot 0.62   ShapeKnots 0.63

So under exact match the panel rejects most of NATURE'S OWN answers. `certified` is therefore a very strong
statement -- it survives a filter that biology usually fails -- but `cannot_certify` is NOT evidence the
design is wrong. It means the panel cannot tell, which is exactly what the word says and why the verdict
exists. Anything that calls the rejected population "false" (including earlier commit messages of mine)
overclaims; the honest word is UNCONFIRMED.

The panel is also not evenly weighted: over 3000 screened candidates pKiss accepts 70.8%, ProbKnot 30.6% and
ShapeKnots 9.4%, and dropping ShapeKnots alone takes the deliverable count from 16 to 87 of 125
(runs/ceiling/loo_oracle.py). The conjunction is in large part a ShapeKnots filter, and a certificate names
its adjudicators so that a caller can judge that for themselves.

THE OPERATING POINT, measured on ground truth and on noise (runs/ceiling/calibrate.py). Treating the rule as
CONFIRMED(k, d) = "at least k of 3 oracles within bp_dist d", calibrated on 250 natural structures (the
wild-type sequence really does form the published structure) against a null of 250 random ACGU sequences:

    shipped rule, k=3 d=0      sensitivity  0.8%      random false-accept  0.0%

Both halves matter and they say opposite things about the two verdicts:
  * `certified` is a STRONG claim. Nothing random passes, and almost nothing does at all.
  * the ABSENCE of certification is nearly uninformative. A filter that recognises 2 of 250 known-correct
    structures rejects correct designs as a matter of course.

And it cannot be fixed by moving the threshold. Once the tolerance is normalised per target (r = bp_dist /
n_pairs, so it means the same thing on a 24 nt and a 100 nt target), NO (k, r) with r < 1 reaches 80%
sensitivity at <= 5% random false-accept; the best meaningful point is k=1, r=0.75 at 79.2% / 1.2%, and
r=0.75 permits three quarters of the target's pairs to be wrong. An earlier absolute-tolerance sweep appeared
to find a usable rule at (k=1, d=8); that was an artefact of one number meaning "mild" on large targets and
"vacuous" on small ones.

WHAT DOES WORK IS CHANGING THE JUDGED QUANTITY, not the threshold and not the instruments. Asking "is the
whole structure reproduced exactly, unanimously" is unanswerable with these tools. Asking "are the CROSSING
STEMS reproduced" is answerable:

    crossing-stem recall, k=1, tau=1.0    sensitivity 84.0%    random false-accept 0.4%

Sensitivity is flat from tau=1.0 to tau=0.9, i.e. these predictors get a knot entirely right or entirely
wrong; there is no "nearly". That is why the shipped threshold is the strictest one (tau=1.0).

One appealing composition was tried and refuted: judging the nested skeleton with an EXACT instrument
(ViennaRNA MFE on the pseudoknot-free reduction) AND the stems with the panel. ViennaRNA recovers the nested
reduction from only 20 of 186 native sequences (10.8%) -- a native's energetics are dominated by the knot,
which ViennaRNA cannot represent, so its MFE lands elsewhere. Combined sensitivity 8.6%. The parts do not
compose, so `knot_confirmed` is reported for what it is: a claim about the knot only.

So on pseudoknot targets this module reports verdicts it can defend and refuses to dress the negative one up
as evidence. Every pseudoknot certificate carries the operating point of whichever rule it used.

WHAT IT COMPOSES (nothing here is new; it is the four measured mechanisms behind one call):
  * the budget cascade + stop-at-first-verified-solve         cognitive/core.py
  * the ladder chosen from measured per-target cost/solves    cognitive/tools.py::default_toolbox
  * pool reading + conjunction adjudication on pseudoknots    cognitive/tools.py, oracles_pk.py
  * our own oracle always, never the solver's self-report     generators/desirna.py::_verify_best

ASYMMETRY WORTH KNOWING. On a NESTED target the oracle is exact and unique, so `cannot_certify` cannot
arise -- either the fold equals the target or it does not. On a PSEUDOKNOT target three independent
predictors disagree on more than 90% of sequences, which is the unsettled state of the field, not a defect
here. That is exactly why the two paths return different things.

RED LINE, repeated in every certificate: `certified` means correct under the stated folding model. It says
nothing about a test tube.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from typing import List, Optional

from .cost import solver_cost_usd
from ..structure_utils import has_pseudoknot

# Budget tiers in DesiRNA -t seconds. Chosen from the measured curve on Eterna100-V2 (runs/ceiling):
#   t=5s 72/100 at $0.00211   t=15s 77   t=60s 81   t=180s 84   -- and the cascade reaches 84 for $1.0041
# so "cheap" is not a crippled mode: it already answers 72% of targets in about 9 seconds.
#
# STANDARD DROPPED ITS 180s TIER, and that tier was 40.8% of all solver wall clock. Measured over the four
# arms in runs/ceiling/ladder_truncation.py and then confirmed prospectively on 131 pseudobase targets that
# played no part in choosing it (ladder_confirm.py), because the point was picked after seeing the first
# curve and ratifying it there would have been retuning a threshold after seeing the result:
#
#     arm         delivered  certified  mean grade  wall clock
#     (5,)            79        48        2.00        14.2%
#     (5,15,60)      102        71        2.74        59.2%     +0.74 grade for 45% more clock
#     (5,15,60,180)  105        74        2.84       100.0%     +0.10 grade for 41% more clock
#
# Dropping 180s costs 7 certificates and gains 4 (sign p=0.274) and moves the mean grade by +0.099 with a
# 95% interval of [-0.103, +0.302] (p=0.166) -- neither is distinguishable from nothing, while the saving
# is not close. `best_effort` keeps 180s and everything above it, so a caller who wants that spend can still
# ask for it; what changed is that they are no longer charged for it by default.
#
# The ladder is NOT worthless -- (5,) to (5,15,60) buys +0.74 grade, and over 251 targets the full ladder
# beats (5,) by +0.749 (p=1.4e-11). The value lives in the 15s and 60s tiers. Of 61 targets the ladder
# improves, 47 go from having no certificate at all to having one; only 14 are k improving inside
# knot_confirmed. Budget mostly buys DELIVERY, not a better grade on something already delivered.
BUDGETS = {
    "cheap": (5,),
    "standard": (5, 15, 60),
    "best_effort": (5, 15, 60, 180, 600, 1800),
}
DISCLAIMER = ("`certified` means the sequence folds to the target under the stated computational model. "
              "It is not a claim about behaviour in vitro or in vivo.")
# Measured operating point of the pseudoknot panel (runs/ceiling/calibrate.py, 250 natives vs 250 randoms).
PANEL_OPERATING_POINT = {
    "rule": "at least 3 of 3 oracles exact (k=3, d=0)",
    "sensitivity_on_known_structures": 0.008,
    "false_accept_on_random_sequences": 0.0,
    "note": ("certification is a strong claim; NON-certification is nearly uninformative, because this rule "
             "recognises only 2 of 250 natural structures whose sequences are known to form them."),
}
KNOT_OPERATING_POINT = {
    "rule": "crossing stems fully reproduced by at least 1 of 3 oracles (k=1, tau=1.0)",
    # HELD OUT, not in-sample. Threshold scanned on one half of the natives, every rate below reported on the
    # other half, with exact Clopper-Pearson intervals (runs/ceiling/calibrate_strict.py).
    "sensitivity_on_known_structures": 0.785,
    "sensitivity_ci95": [0.688, 0.863],
    "sensitivity_n": 93,
    # THREE NULLS, weakest to hardest. The one that matters is `nested`: sequences designed for the target's
    # PSEUDOKNOT-FREE skeleton and then judged against the full target -- they get the easy half right by
    # construction and have no reason to form the crossing stems. That is the population this rule actually
    # faces, and across all 183 of them not one was accepted; no single oracle even reached full crossing
    # recall (pKiss tops out at 0.80, ProbKnot 0.75, ShapeKnots 0.29).
    "false_accept_nested_decoys": 0.000,
    "false_accept_nested_ci95": [0.000, 0.039],
    "false_accept_nested_n": 92,
    "false_accept_dinucleotide_shuffle": 0.007,
    "false_accept_on_random_sequences": 0.000,
    "scope": "the CROSSING STEMS only -- this says nothing about the rest of the structure",
    "note": ("sensitivity is unchanged from tau=1.0 down to 0.9: these predictors get a knot entirely right "
             "or entirely wrong. Composing this with an exact nested-skeleton check was tried and refuted "
             "(ViennaRNA recovers the nested reduction from only 20 of 186 natives). An earlier 84.0% was "
             "quoted from a 250-target in-sample set while the hard-null work runs on the 186 of those whose "
             "sequence can be recovered; 78.5% is the held-out figure on that set, and the fit and test "
             "halves agree exactly, so the difference is the denominator rather than overfitting."),
}

# HOW MANY OF THE THREE ACTUALLY AGREED -- reported per certificate, because k=1 and k=3 are not the same
# evidence even though the shipped threshold is k>=1.
#
# WHAT THE GRADE DOES NOT BUY, stated first because the obvious reading is wrong. It does NOT buy specificity
# against random sequences: measured on the same 250-structure calibration, the false-accept rate is 0.004 at
# k=1, k=2 AND k=3 alike (runs/ceiling/calibrate.json). A random sequence essentially never fools even one
# predictor, so demanding three does not make that number smaller -- it only throws sensitivity away, 0.840
# down to 0.432.
#
# WHAT IT DOES BUY is robustness against ONE predictor's idiosyncrasy, which is a failure mode this project
# has measured directly rather than imagined: an operator that used pKiss inside its own search loop scored
# 43/60 by pKiss and 1/60 by the unanimous panel -- a 43x inflation produced entirely by optimising against
# the judge. Nothing in the random null can see that, because the null contains no sequences that were
# optimised against anything. So the grade answers "would this survive if we had picked a different
# predictor", and that is the question the random false-accept rate cannot answer.
# HOW INDEPENDENT THE THREE JUDGES ACTUALLY ARE, measured rather than asserted. Pairwise Cohen's kappa on
# "reproduces every crossing pair", over 186 natives (runs/ceiling/calibrate_strict.py):
#
#     pKiss ~ ProbKnot       0.324
#     pKiss ~ ShapeKnots     0.359
#     ProbKnot ~ ShapeKnots  0.743      <- substantial agreement
#
# So the panel is effectively TWO voices: pKiss, and a correlated ProbKnot/ShapeKnots pair. This is why k=1
# and k=3 carry the same false-accept rate, which looked like a paradox while the phrase "three independent
# predictors" was doing unexamined work. The grade is reported with that caveat attached rather than with an
# adjective the data does not support.
ORACLE_KAPPA = {"pkiss~probknot": 0.324, "pkiss~shapeknots": 0.359, "probknot~shapeknots": 0.743,
                "effective_independent_sources": 2}

# WHAT EACH GRADE IS WORTH TO AN OUTSIDER, measured rather than asserted (runs/ceiling/knotty_judge.py,
# ceiling_and_coverage.py, ipknot_judge.py; 4388 designs over 250 targets, every one of which already clears
# the shipped k>=1 floor). THREE judges that took no part in any rule adjudicate, chosen to span the field's
# algorithm classes rather than to be three binaries:
#
#     RNAPKplex   heuristic helix assembly              Turner-family energies
#     Knotty      exact sparsified DP                   HotKnots DP09 energies
#     IPknot      integer programming over base-pair    LinearPartition-C probabilities -- LEARNED, not
#                 probabilities                         thermodynamic
#
# They are pairwise NON-redundant -- Cohen's kappa 0.341, 0.109, 0.113 -- which is what makes quoting all
# three worth anything. All three run at defaults; tuning a judge while looking at its agreement with our
# designs is the error already on this project's record.
#
# A raw agreement rate is uninterpretable alone, because every predictor in this field misses most of nature.
# So each rate is quoted against that same judge's own rate on NATIVE sequences (n=186), measured in the same
# runs: RNAPKplex 66.7% [59.4, 73.4], Knotty 73.7% [66.7, 79.8], IPknot 46.2% [38.9, 53.7].
#
# WHAT REPLICATES ACROSS ALL THREE, and it is the single most useful thing a caller can be told here:
# there is a STEP at k=3. Going k=2 -> k=3 gains +64.6, +36.4 and +29.0 points respectively, while k=1 -> k=2
# gains only +0.6, +2.1 and +5.5. Most of what the panel knows is spent in the last vote.
#
# WHAT DOES NOT REPLICATE, recorded here because an earlier version of this comment asserted it. "k=3 sits
# ABOVE the judge's native ceiling" held for RNAPKplex (1.47x) and Knotty (1.14x) and FAILS for IPknot
# (0.84x) -- it was a property of the first two judges, and the pre-registration in ipknot_judge.py (P2)
# killed it. Likewise "k=2 is not measurably better than k=1" was true of the first two judges and is false
# of the third (4.6% -> 10.1%). Whether a grade clears an outside instrument's own bar is JUDGE-DEPENDENT;
# only the ordering and the size of the k=3 step survive all three.
#
# WHY THE FLOOR IS STILL k>=1. Raising it to k=3 would put 68.0% of targets (170 of 250) from "has a
# certificate" to "has none", and k=1/k=2 are not established as WRONG -- they are established as less
# recognisable to outside instruments than nature is. Withdrawing an answer is a real cost to a caller who
# has no other way to get one. So the floor holds and the grade carries the numbers instead.
_NATIVE_CEILING = {"rnapkplex": 0.667, "knotty": 0.737, "ipknot": 0.462, "both_pk_kn": 0.624, "n_native": 186,
                   "pairwise_kappa": {"rnapkplex~knotty": 0.341, "rnapkplex~ipknot": 0.109,
                                      "knotty~ipknot": 0.113},
                   "note": "each judge's own rate on NATIVE pseudoknots -- the reference every rate below "
                           "must be read against, NOT a correctness rate. The three judges are pairwise "
                           "non-redundant, so they are three views and not one repeated"}

# The first error rate this project has ever measured against PHYSICAL REALITY rather than against another
# model. Every number above it is an agreement rate between predictors; this one is not.
#
# Mutate-map-rescue asks a causal question about ONE base pair (i, j): break i and the protection at j should
# vanish; restore the pair with a compensating mutation at j and the protection should come back.
# `rescue_factor` is the fraction of the SHAPE perturbation the double mutant restores -- 0 means no evidence
# the pair exists, 1 means fully restored. On 1,412 crossing pairs from 160 designs across 20 targets
# (OpenKnot OK7a M2R, runs/ceiling/m2r_pairwise.py), graded by the same three oracles the rule uses:
#
#     0 oracles confirm   n=267   median rescue 0.233    52.4% of these pairs simply do not exist
#     1 oracle            n=236                 0.394    32.2%
#     2 oracles           n=456                 0.470    19.3%
#     3 oracles           n=453                 0.491    14.1%
#
# TWO LEVELS, AND THEY MUST NOT BE CONFLATED -- the first version of this block did conflate them.
#   per PAIR   how many oracles recover THIS pair, compared WITHIN a design because pairs inside one molecule
#              share its fold, its normalisation and its batch. Within-design Spearman of oracle count with
#              rescue: +0.275, p = 5.8e-05 over 74 designs, and it survives controlling for base-pair
#              stability (partial rho +0.240, p = 7.7e-04) -- the panel is reading structure, not GC content.
#              AN EARLIER VERSION OF THIS BLOCK QUOTED p = 2.1e-28 HERE. That came from a Mann-Whitney over
#              1,412 pooled pairs, which treats 1,412 pairs from 160 molecules as 1,412 independent
#              observations. It is pseudo-replication and the number was never real; the same comparison
#              done within design over the 15 designs carrying both strata gives +0.042 at p = 0.047.
#   per DESIGN what the certificate's k actually is: how many oracles recover ALL crossing pairs at tau=1.0.
#              Supported but far weaker, and measured AFTER the pair-level result rather than pre-registered:
#              within-target Spearman +0.396 over 17 targets, p = 0.0147. "Supported", not "established".
#
# THE UNCOMFORTABLE HALF. `false_accept_on_random_sequences` = 0.004 was measured against RANDOM SEQUENCES,
# a null so weak that passing it means very little: a random sequence has no reason to form a pseudoknot.
# Asked of pairs a real molecule declines to form, the panel is wrong far more often -- HOW often depends
# entirely on where "declines to form" is cut, so the cut travels with the number:
#
#     rescue <  0.00   k=3 pairs   2.4% absent      k=0 pairs 15.7%
#     rescue <  0.25               14.1%                      52.4%
#     rescue <  0.50               51.7%                      78.7%
#
# The 0.25 in the original pre-registration was justified as "OpenKnot's own reactivity cut for paired".
# That was a category error -- OpenKnot's 0.25 cuts normalised SHAPE REACTIVITY, while rescue_factor is a
# fraction of restored RMSD. The digits coincide; the quantities do not. The threshold-free statement is
# the one that carries weight, and it is the Mann-Whitney above.
_INTERVENTION = {
    "assay": "mutate-map-rescue (compensatory double mutants, 2A3 SHAPE)",
    "source": "eternagame/OpenKnotAIDesignData OK7a_M2R_data.v4.5.1 (Apache-2.0)",
    "n_crossing_pairs": 1412, "n_designs": 160, "n_targets": 20,
    "median_rescue_by_k": {0: 0.233, 1: 0.394, 2: 0.470, 3: 0.491},
    "per_pair_ordering_threshold_free": {
        "test": "within-design Spearman, oracles-confirming-this-pair vs rescue_factor",
        "rho": 0.275, "p": 5.83e-05, "n_designs": 74,
        "pooled_mannwhitney_RETRACTED": {"p": 2.05e-28, "why": "pseudo-replication: 1412 pairs from 160 "
                                         "molecules are not 1412 independent observations"},
        "same_comparison_within_design": {"median_diff": 0.042, "p": 0.0469, "n_designs": 15}},
    "pair_absent_rate_by_k_at_0.25": {0: 0.524, 1: 0.322, 2: 0.193, 3: 0.141},
    "pair_absent_threshold_sweep_k3": {0.0: 0.024, 0.1: 0.040, 0.2: 0.102, 0.25: 0.141,
                                       0.3: 0.192, 0.4: 0.342, 0.5: 0.517},
    "false_accept_vs_reality_at_k3": {"value": 0.141, "threshold": 0.25,
                                      "ci95_cluster_bootstrap": [0.101, 0.186],
                                      "warning": "point estimate is threshold-driven; quote the sweep"},
    "certificate_ladder_ordered_in_vitro": {"level": "per design (this is the certificate's k)",
                                            "spearman_rho": 0.396, "p": 0.0147, "n_targets": 17,
                                            "pre_registered": False,
                                            "note": "run after the per-pair result; supported, not established"},
    "not_a_stability_artifact": {"partial_rho_controlling_pair_strength": 0.240, "p": 7.71e-04},
    "negative_control": {"what": "150 tested pairs the design's own sequence cannot form",
                         "median_rescue": 0.196, "vs_complementary": 0.452, "p": 1.8e-27},
    "scope": ("20 Round-3 targets (P01-P20), designs 52-100 nt with a median of 100, 9 design methods. "
              "DesiRNA is not among them, so this validates the CRITERION, not our designs. Does not "
              "extend to the 240 nt range where our own failures live"),
}

KNOT_GRADES = {
    3: {"grade": "k=3", "agreeing_oracles": 3, "sensitivity_on_known_structures": 0.432,
        "outside_agreement": {"n": 146, "rnapkplex": 0.979, "knotty": 0.842, "ipknot": 0.390,
                              "vs_native_ceiling": {"rnapkplex": 1.47, "knotty": 1.14, "ipknot": 0.84}},
        "means": "all three predictors reproduce the crossing stems, and this is where the evidence "
                 "actually steps up -- every one of three non-redundant outside judges recognises these "
                 "designs far more often than it recognises k=2 designs (+64.6, +36.4, +29.0 points). "
                 "Whether that clears the judge's OWN rate on natural pseudoknots depends on the judge: "
                 "1.47x and 1.14x for two of them, 0.84x for the third. Note that two panel members "
                 "(ProbKnot and ShapeKnots) agree with each other at kappa=0.74, so this is closer to two "
                 "independent votes than to three"},
    2: {"grade": "k=2", "agreeing_oracles": 2, "sensitivity_on_known_structures": 0.636,
        "outside_agreement": {"n": 1759, "rnapkplex": 0.333, "knotty": 0.478, "ipknot": 0.101,
                              "vs_native_ceiling": {"rnapkplex": 0.50, "knotty": 0.65, "ipknot": 0.22}},
        "means": "two of three predictors agree; one dissents or could not run. The gain over k=1 is real "
                 "but small under every outside judge (+0.6, +2.1, +5.5 points) -- far less than the step "
                 "from k=2 to k=3. Which two agreed still matters: pKiss plus either of the others is more "
                 "independent evidence than ProbKnot plus ShapeKnots"},
    1: {"grade": "k=1", "agreeing_oracles": 1, "sensitivity_on_known_structures": 0.785,
        "outside_agreement": {"n": 2483, "rnapkplex": 0.327, "knotty": 0.458, "ipknot": 0.046,
                              "vs_native_ceiling": {"rnapkplex": 0.49, "knotty": 0.62, "ipknot": 0.10}},
        "means": "exactly one predictor reproduces the crossing stems -- this is the shipped floor, and it "
                 "is the grade most exposed to a single predictor's bias. All three outside judges "
                 "recognise these designs well below their own rate on natural pseudoknots (0.49x, 0.62x, "
                 "0.10x). That is not a demonstration that they are wrong, but it is the weakest evidence "
                 "this pipeline ships, and the caller should read it as such"},
}


def knot_grade(kr: dict) -> dict:
    """Which grade the evidence for THIS sequence actually reached, alongside the shipped floor.

    The floor decides the verdict; the grade tells the reader how much of the panel was behind it. Both are
    reported because a certificate that says only "passed" hides the difference between unanimous agreement
    and a single vote.
    """
    k = len(kr.get("confirmed_by") or [])
    g = dict(KNOT_GRADES.get(min(3, max(1, k)), KNOT_GRADES[1]))
    g["false_accept_on_random_sequences"] = 0.004      # identical at every k -- see the comment above
    # ...and the same question asked of the molecule instead of of another model. Quoted at THIS k, because
    # unlike the random-sequence figure it actually moves with the grade.
    g["intervention"] = dict(_INTERVENTION)
    g["pair_absent_rate_vs_reality"] = _INTERVENTION["pair_absent_rate_by_k_at_0.25"].get(min(3, max(1, k)))
    g["shipped_threshold"] = "k>=1"
    g["oracle_agreement_kappa"] = ORACLE_KAPPA
    g["which_agreed"] = list(kr.get("confirmed_by") or [])
    g["native_ceiling"] = dict(_NATIVE_CEILING)        # the reference `outside_agreement` must be read against
    return g


# WHERE THE CALIBRATED RULE MAY BE USED, and this is a restriction rather than a feature. The rule was built
# for, calibrated on, and validated on the topologies where the STRICT rule delivers almost nothing -- kissing
# and higher-order, where strict gives 1 of 55 and 0 of 9. A fourth predictor that never entered the pipeline
# (RNAPKplex, runs/ceiling/external_judge.py) then measured what it does on each family, against that same
# judge's own ceiling on NATIVE sequences of the same topology:
#
#     kissing / chain   ours 60.8%   its native ceiling 54.8% [38.7, 70.2]   -> inside; paired over 38 shared
#                                                                               targets it prefers OUR designs
#                                                                               8 to 2
#     H-type            ours 29.9%   its native ceiling 71.1% [62.7, 78.6]   -> far below, and paired over the
#                                                                               135 shared targets the outsider
#                                                                               backs the NATIVE 65 times to
#                                                                               our 20, p = 1.03e-06
#
# So on H-type the calibrated rule is over-permissive, and the cause is not a missing vote: raising the
# threshold to k>=2 moves external agreement from 29.9% to 26.5%, i.e. nowhere. A global energy margin was
# tried too and is refuted -- it separates H-type in the right direction (6.30 against 2.80 kcal/mol) and
# kissing in the WRONG one (2.20 against 4.50), so any cutoff that fixes one breaks the other.
#
# What does work is not a patch but a restriction: on H-type the STRICT rule still delivers, and delivers
# cleanly. Searching the same pool for a panel-unanimous sequence finds one for 36 of 187 targets, and the
# outsider backs 35 of those 36 -- 97.2% [85.5, 99.9], above its own native ceiling.
#
# The trade is stated rather than hidden. On H-type the calibrated rule issues 187 certificates of which an
# outsider backs 56; the strict rule issues 36 of which it backs 35. The calibrated rule yields MORE
# externally-backed certificates in absolute terms. We take the strict one anyway, because a certificate is a
# CLAIM: issuing 187 claims that an independent judge will not back is the exact behaviour this project
# criticises single-oracle tools for (43 of 60 claimed, none confirmed). Precision is the product.
CALIBRATED_RULE_TOPOLOGIES = ("kissing / chain", "higher-order", "crossing (non-H)")

TOPOLOGY_OPERATING_POINTS = {
    "kissing / chain": {"rule": "knot-recall (k=1, tau=1.0)",
                        "external_agreement": 0.608, "external_native_ceiling": 0.548,
                        "external_ceiling_ci95": [0.387, 0.702], "n": 51},
    "higher-order": {"rule": "knot-recall (k=1, tau=1.0)",
                     "external_agreement": 0.800, "external_native_ceiling": 0.556,
                     "external_ceiling_ci95": [0.212, 0.863], "n": 5},
    "H-type": {"rule": "panel-unanimous (the calibrated rule is NOT used here)",
               "external_agreement": 0.972, "external_agreement_ci95": [0.855, 0.999],
               "external_native_ceiling": 0.711, "external_ceiling_ci95": [0.627, 0.786], "n": 36,
               "why": ("the calibrated rule is over-permissive on this family -- an outsider backs 29.9% of "
                       "what it accepts against a 71.1% ceiling on natives of the same topology")},
}


def _topology_point(target: str) -> dict:
    """The operating point for THIS target's family, so a certificate never quotes a global average.

    The rule's error rate depends on topology by a factor of two, so one blended number would be wrong for
    every family it is shown to."""
    from ..structure_utils import classify_topology
    t = classify_topology(target)["topology"]
    return {"topology": t, **(TOPOLOGY_OPERATING_POINTS.get(t) or
                              {"note": "no external measurement for this family yet"})}


def calibrated_rule_applies(target: str) -> bool:
    """May the crossing-stem rule be used on this target, or must it meet the strict panel?"""
    from ..structure_utils import classify_topology
    return classify_topology(target)["topology"] in CALIBRATED_RULE_TOPOLOGIES


SCHEMA_VERSION = "rna-certificate/1.0"
_PROVENANCE = None


def provenance() -> dict:
    """What produced this verdict, precisely enough for a third party to reproduce it.

    A certificate that can be READ but not REPRODUCED is not an audit record. The verdict depends entirely on
    which binaries were asked -- ViennaRNA's energy model decides match/umfe/pfe, and pKiss / ProbKnot /
    ShapeKnots decide every pseudoknot verdict -- so their versions belong in the answer itself, not in a
    README that drifts away from it. Probing shells out, so it happens once per process and is cached.
    """
    global _PROVENANCE
    if _PROVENANCE is not None:
        return _PROVENANCE
    import platform
    import subprocess
    out = {"schema": SCHEMA_VERSION, "python": platform.python_version()}
    try:
        import RNA
        out["viennarna"] = getattr(RNA, "__version__", None) or str(RNA.version())
    except Exception:                                                        # noqa: BLE001
        out["viennarna"] = "unavailable"

    def _run(argv, timeout=8):
        """First line of output, or None. A tool that does not understand --version prints an error, and
        recording THAT as the version is worse than recording the path."""
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
            txt = ((r.stdout or "") + (r.stderr or "")).strip()
            if not txt:
                return None
            line = txt.splitlines()[0][:120]
            low = line.lower()
            if any(w in low for w in ("unknown option", "unrecognized", "invalid option", "usage:")):
                return None
            return line
        except Exception:                                                    # noqa: BLE001
            return None

    try:
        from ..generators.desirna import _desirna_cmd
        cmd = _desirna_cmd()
        if cmd:
            home = os.path.dirname(cmd[-1])
            sha = _run(["git", "-C", home, "rev-parse", "--short", "HEAD"])
            out["desirna"] = f"{home}@{sha}" if sha else home
        else:
            out["desirna"] = "not installed"
    except Exception:                                                        # noqa: BLE001
        out["desirna"] = "unknown"
    try:
        from ..benchmarks.rnainvbench import _pkiss_cmd
        c = _pkiss_cmd()
        out["pkiss"] = (_run(list(c) + ["--version"]) or " ".join(c)) if c else "not installed"
    except Exception:                                                        # noqa: BLE001
        out["pkiss"] = "unknown"
    try:
        from ..oracles_pk import _bin, rnastructure_available
        b = _bin("ProbKnot")
        out["rnastructure"] = (_run([b, "--version"]) or b) if (b and rnastructure_available()) else "not installed"
    except Exception:                                                        # noqa: BLE001
        out["rnastructure"] = "unknown"
    _PROVENANCE = out
    return out


@dataclass
class Certificate:
    verdict: str                      # certified | knot_confirmed | cannot_certify | not_found | invalid_input
    target: str
    n: int
    kind: str                         # nested | pseudoknot | invalid
    sequence: Optional[str] = None
    checked_by: str = ""              # which oracle(s) actually adjudicated
    strictness: Optional[str] = None  # the rung the answer passed
    rungs_passed: List[str] = field(default_factory=list)
    cost_usd: float = 0.0
    wait_s: float = 0.0
    budget: str = ""
    reason: str = ""                  # human-readable, especially for the two negative verdicts
    detail: dict = field(default_factory=dict)
    disclaimer: str = DISCLAIMER
    # --- audit block: a certificate that cannot be reproduced is not a record ---------------------
    seed: Optional[int] = None
    generated_utc: str = ""
    provenance: dict = field(default_factory=dict)
    verify_cmd: str = ""          # deterministic: re-checks THIS sequence, always the same verdict
    rerun_search: str = ""        # stochastic: re-runs the design, may return a different sequence
    reproducibility: str = ""

    def stamp(self, seed: Optional[int] = None, budget: Optional[str] = None) -> "Certificate":
        """Attach the audit block on the way out, so no exit path can forget it.

        TWO COMMANDS, because they promise different things and conflating them would be a false claim.
        MEASURED: the same seed does NOT give the same design. DesiRNA's budget is WALL-CLOCK (-t 5), not a
        number of iterations, so how far the search gets depends on machine load; two runs at seed 3 gave
        GGCCGUUGGGCC and GCUCGUAGGAGC. A single `reproduce` field would therefore have been a promise this
        cannot keep.

        What IS exactly reproducible is the thing the certificate actually claims: that THIS sequence forms
        THIS target under these oracles. Folding is deterministic, so `verify` reproduces the verdict for
        anyone, on any machine, forever. `rerun_search` re-runs the design and may legitimately return a
        different sequence.
        """
        import datetime as _dt
        self.seed = seed
        self.generated_utc = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.provenance = provenance()
        q = (self.target or "").replace("'", "")
        self.verify_cmd = (f"python -m rna_harness.cognitive.certify '{q}' --sequence {self.sequence}"
                           if self.sequence else "")
        self.rerun_search = (f"python -m rna_harness.cognitive.certify '{q}' "
                             f"--budget {budget or self.budget or 'standard'}"
                             + (f" --seed {seed}" if seed is not None else ""))
        self.reproducibility = (
            "verify_cmd reproduces this certificate's claim exactly -- folding is deterministic. "
            "rerun_search re-runs the design and may return a different sequence: the search budget is "
            "wall-clock, not iteration count, so it is not seed-reproducible."
            if self.sequence else
            "no sequence was produced, so there is nothing to re-verify; rerun_search re-runs the search.")
        return self

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- input validation
def classify(target: str):
    """(kind, error). A non-expert can and will paste something malformed; say so instead of computing."""
    t = (target or "").strip()
    if not t:
        return "invalid", "empty target"
    allowed = set("().[]{}<>")
    bad = sorted(set(t) - allowed)
    if bad:
        return "invalid", f"target contains characters that are not dot-bracket: {bad}"
    for op, cl in (("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")):
        depth = 0
        for ch in t:
            if ch == op:
                depth += 1
            elif ch == cl:
                depth -= 1
                if depth < 0:
                    return "invalid", f"unbalanced '{op}{cl}' in target"
        if depth != 0:
            return "invalid", f"unbalanced '{op}{cl}' in target"
    return ("pseudoknot" if has_pseudoknot(t) else "nested"), None


# --------------------------------------------------------------------------- verify an existing sequence
def _certify_sequence_impl(seq: str, target: str) -> Certificate:
    """"Will this sequence fold into this shape?" -- the cheapest and most common non-expert question.

    No search. On a nested target this reports every strictness rung the sequence passes, because those are
    different questions and the user has no way to know which one they meant. On a pseudoknot target it is
    adjudicated by the three-oracle panel, and a sequence that satisfies only the single reference oracle
    comes back `cannot_certify` -- the population a single-oracle tool would have handed over as solved.
    """
    kind, err = classify(target)
    t0 = time.time()
    if kind == "invalid":
        return Certificate("invalid_input", target, len(target or ""), "invalid", reason=err)
    if not seq or len(seq) != len(target):
        return Certificate("invalid_input", target, len(target), kind,
                           reason=f"sequence length {len(seq or '')} != target length {len(target)}")
    bad = sorted(set(seq.upper().replace("T", "U")) - set("ACGU"))
    if bad:
        return Certificate("invalid_input", target, len(target), kind,
                           reason=f"sequence contains non-ACGU characters: {bad}")
    seq = seq.upper().replace("T", "U")

    if kind == "nested":
        rungs = _nested_rungs(seq, target)
        w = time.time() - t0
        if "match" in rungs:
            return Certificate("certified", target, len(target), kind, sequence=seq,
                               checked_by="ViennaRNA MFE (turner2004, dangles=2)",
                               strictness=rungs[-1], rungs_passed=rungs,
                               cost_usd=solver_cost_usd(w, 1), wait_s=round(w, 2),
                               reason=f"folds exactly to the target; also passes: {', '.join(rungs[1:]) or 'none'}")
        # SAY HOW IT FAILED, not just that it did. The agent loop's whole premise is that it reads the real
        # failure and changes course; a bare "not_found" gave it nothing to change, and MEASURED consequence:
        # it climbed the budget ladder three times on one undesignable shape, paying $0.38 and 45 minutes for
        # a fact this block computes for free.
        import RNA
        got = RNA.fold(seq)[0]
        bad = [i for i in range(len(target)) if got[i] != target[i]]
        return Certificate("not_found", target, len(target), kind, sequence=None,
                           checked_by="ViennaRNA MFE (turner2004, dangles=2)",
                           cost_usd=solver_cost_usd(w, 1), wait_s=round(w, 2),
                           detail={"mfe_structure": got, "best_bp": _bp_distance(got, target),
                                   "wrong_positions": bad[:24], "n_wrong": len(bad)},
                           reason=(f"the MFE structure of this sequence is not the target: it differs at "
                                   f"{len(bad)} of {len(target)} positions"))

    v = _panel_verdict(seq, target)
    w = time.time() - t0
    if v["unanimous"]:
        return Certificate("certified", target, len(target), kind, sequence=seq,
                           checked_by="panel: pKiss AND ProbKnot AND ShapeKnots", strictness="panel-unanimous",
                           rungs_passed=["panel-unanimous"], cost_usd=solver_cost_usd(w, 1),
                           wait_s=round(w, 2), reason="all three independent predictors form the target",
                           detail={**v, "operating_point": PANEL_OPERATING_POINT})
    if v["any_failed_to_run"]:
        return Certificate("cannot_certify", target, len(target), kind, sequence=seq,
                           checked_by="panel (incomplete)", cost_usd=solver_cost_usd(w, 1),
                           wait_s=round(w, 2), detail=v,
                           reason="a required independent oracle could not be run, so this cannot be "
                                  "adjudicated. A missing oracle is not a rejection.")
    kr = _knot_recall(seq, target)
    if kr["knot_confirmed"] and calibrated_rule_applies(target):
        return Certificate("knot_confirmed", target, len(target), kind, sequence=seq,
                           checked_by="crossing stems: " + ", ".join(kr["confirmed_by"]),
                           strictness="knot-recall (k=1, tau=1.0)", rungs_passed=["knot-recall"],
                           cost_usd=solver_cost_usd(time.time() - t0, 1),
                           wait_s=round(time.time() - t0, 2),
                           detail={**v, "knot": kr, "operating_point": KNOT_OPERATING_POINT,
                                   "topology": _topology_point(target), "grade": knot_grade(kr)},
                           reason="the crossing stems that make this a pseudoknot are fully reproduced by "
                                  + ", ".join(kr["confirmed_by"]) + ". This is a claim about the KNOT, not "
                                  "the whole structure: the criterion recognises 78.5% of known natural "
                                  "structures (held out, 95% CI 68.8-86.3) and accepted 0 of 183 decoys "
                                  "designed for this target's pseudoknot-free skeleton.")
    if v["reference_only"]:
        return Certificate("cannot_certify", target, len(target), kind, sequence=seq,
                           checked_by="panel: pKiss AND ProbKnot AND ShapeKnots",
                           cost_usd=solver_cost_usd(w, 1), wait_s=round(w, 2),
                           detail={**v, "operating_point": PANEL_OPERATING_POINT},
                           reason="the reference oracle (pKiss) accepts this sequence but the independent "
                                  "predictors do not. Reported as UNCONFIRMED, not as wrong: this panel "
                                  "recognises only 2 of 250 NATURAL structures whose sequences are known to "
                                  "form them, so failing it is nearly uninformative.")
    return Certificate("not_found", target, len(target), kind, sequence=seq,
                       checked_by="panel: pKiss AND ProbKnot AND ShapeKnots",
                       cost_usd=solver_cost_usd(w, 1), wait_s=round(w, 2), detail=v,
                       reason="no independent predictor forms the target from this sequence")


def _bp_distance(a: str, b: str) -> int:
    """Base pairs in one structure and not the other, counted both ways. Symmetric, so it is a distance and
    not a recall -- an answer that invents extra pairs is as wrong as one that misses them."""
    from ..structure_utils import pair_table
    pa, pb = pair_table(a), pair_table(b)
    sa = {(i, j) for i, j in enumerate(pa) if j > i}
    sb = {(i, j) for i, j in enumerate(pb) if j > i}
    return len(sa ^ sb)


def _nested_rungs(seq: str, target: str) -> List[str]:
    """Which strictness rungs this sequence passes. Archived survival on Eterna: match 1.000, umfe 0.902,
    pfe 0.066 -- so reporting only 'match' would answer a much easier question than the user may have meant."""
    import RNA
    fc = RNA.fold_compound(seq)
    mfe_db, mfe_e = fc.mfe()
    if mfe_db != target:
        return []
    rungs = ["match"]
    # 'umfe' = the target is the UNIQUE MFE. Counted with fc.subopt(1); a callback-based count is off by one
    # because ViennaRNA fires it once more with a NULL terminator structure.
    if sum(1 for s in fc.subopt(1) if abs(s.energy - mfe_e) < 1e-6) == 1:
        rungs.append("umfe")
    fc2 = RNA.fold_compound(seq)
    fc2.pf()
    if float(fc2.pr_structure(target)) >= 0.90:
        rungs.append("pfe")
    return rungs


@lru_cache(maxsize=4096)
def _ensemble_entropy(seq: str) -> float:
    """Mean positional entropy of the sequence's nested ensemble -- how UNDECIDED its fold is.

    Only ever called to break a tie between two candidates of the same grade, so the partition function is
    not paid for on the common path. A failure returns -inf so a sequence we cannot evaluate never displaces
    an incumbent we could.
    """
    try:
        import RNA
        fc = RNA.fold_compound(seq)
        _, mfe_e = fc.mfe()
        fc.exp_params_rescale(mfe_e)
        fc.pf()
        bpp = fc.bpp()
        n, tot = len(seq), 0.0
        for i in range(1, n + 1):
            ps = [p for j in range(1, n + 1)
                  for p in ((bpp[i][j] if j > i else bpp[j][i]),) if p and p > 1e-12]
            unp = max(0.0, 1.0 - sum(ps))
            h = -sum(p * math.log(p) for p in ps)
            if unp > 1e-12:
                h -= unp * math.log(unp)
            tot += h
        return tot / n
    except Exception:                                                          # noqa: BLE001
        return float("-inf")


def _knot_recall(seq: str, target: str) -> dict:
    """Per-oracle recall of the CROSSING pairs -- the pairs that make the target a pseudoknot at all.

    This is the one quantity that calibrates. Asking whether the whole structure is reproduced exactly and
    unanimously is unanswerable with these predictors (sensitivity 0.8% on known natural structures); asking
    whether the crossing stems are reproduced by at least one of them reaches 84.0% sensitivity at 0.4% false
    accept on random sequences (runs/ceiling/calibrate.py). `None` means the oracle failed to run, which is
    never a rejection.
    """
    from ..benchmarks.rnainvbench import _round_only, pk_pairs
    from ..oracles_pk import oracle_pairs
    cross = pk_pairs(target) - pk_pairs(_round_only(target))
    out = {}
    for name, ps in oracle_pairs(seq).items():                 # one fold per sequence, shared with the
        out[name] = None if (ps is None or not cross) else len(ps & cross) / len(cross)   # panel verdict
    full = [k for k, v in out.items() if v is not None and v >= 1.0]      # tau = 1.0, the strictest point
    return {"per_oracle": out, "n_cross": len(cross), "confirmed_by": full,
            "knot_confirmed": bool(full)}


def _panel_verdict(seq: str, target: str) -> dict:
    """Adjudicate one sequence by the three independent pseudoknot predictors.

    `None` from an oracle means THE ORACLE FAILED TO RUN, not "rejected"; conflating the two made the tool
    report zero solves with no diagnostic whenever RNAstructure was absent."""
    from ..oracles_pk import forms_target, oracle_pairs
    out, failed = {}, False
    for name, ps in oracle_pairs(seq).items():                 # same folds `_knot_recall` reads
        v = forms_target(ps, target)
        out[name] = v
        failed = failed or (v is None)
    return {"per_oracle": out, "any_failed_to_run": failed,
            "unanimous": all(out.get(k) is True for k in ("pkiss", "probknot", "shapeknots")),
            "reference_only": (out.get("pkiss") is True
                               and not all(out.get(k) is True for k in ("probknot", "shapeknots")))}


# --------------------------------------------------------------------------- design + certify
def _certify_design_impl(target: str, budget: str = "standard", seed: int = 0,
                         tiers: Optional[tuple] = None, grade_max: bool = True) -> Certificate:
    """"Give me a sequence that folds into this shape", answered with a certificate.

    Runs the measured budget cascade and stops at the first sequence that survives the criterion for this
    KIND of target. Nothing is taken from the solver's self-report: every candidate is re-adjudicated here.
    """
    kind, err = classify(target)
    if kind == "invalid":
        return Certificate("invalid_input", target, len(target or ""), "invalid", reason=err, budget=budget)
    # An explicit tier list overrides the named budget. The caller that needs this is a cost cap: a cap
    # checked only BETWEEN calls cannot stop a single best_effort call, whose last tier is 1800 s x 10
    # replicas -- measured, two agent runs blew past a $0.50 cap to $0.51 and $0.44 and burned 5033 s and
    # 8948 s to return not_found. Truncating the ladder is the only place that can be enforced.
    tiers = tuple(tiers) if tiers else BUDGETS.get(budget)
    if tiers is None:
        return Certificate("invalid_input", target, len(target), kind, budget=budget,
                           reason=f"unknown budget '{budget}'; choose one of {sorted(BUDGETS)}")

    from ..generators.desirna import DesiRNAGenerator, desirna_available
    from ..task import DesignTask, OracleConfig
    if not desirna_available():
        return Certificate("cannot_certify", target, len(target), kind, budget=budget,
                           reason="the external solver (DesiRNA) is not installed, so this request cannot "
                                  "be served at the quality this certificate promises. Set DESIRNA_HOME.")
    if kind == "pseudoknot":
        from ..benchmarks.rnainvbench import pkiss_available
        from ..oracles_pk import rnastructure_available
        if not (pkiss_available() and rnastructure_available()):
            return Certificate("cannot_certify", target, len(target), kind, budget=budget,
                               reason="a pseudoknot target needs the three-oracle panel (pKiss + "
                                      "RNAstructure). At least one is unavailable, so no honest verdict "
                                      "is possible; a single-oracle answer here was 43 of 43 false positives.")

    oc = OracleConfig(param_set="turner2004", temperature=37.0, dangles=2)
    t0, cost, tried = time.time(), 0.0, []
    best_bp = None                             # closest miss, so `not_found` can say HOW close it got
    best_ref_only = None                       # a candidate the reference oracle accepts but the panel does not
    best_knot = None                           # a candidate whose CROSSING STEMS are fully reproduced
    n_screened = 0                             # unique candidates adjudicated across every tier, cumulative
    for tl in tiers:
        a0 = time.time()
        res = DesiRNAGenerator(replicas=10, timelimit_s=tl, results_number=20).design(
            DesignTask(task_id="certify", target=target, oracle=oc), seed=seed)
        cost += solver_cost_usd(time.time() - a0, 10)
        tried.append(tl)
        cands = [c for c in ([res.best_sequence] if res.best_sequence else [])
                 + list(getattr(res, "candidates", None) or []) if c]
        seen, uniq = set(), []
        for c in cands:
            if c not in seen and len(c) == len(target):
                seen.add(c)
                uniq.append(c)

        if kind == "nested":
            import RNA
            for c in uniq:
                bp = RNA.bp_distance(RNA.fold(c)[0], target)
                best_bp = bp if best_bp is None else min(best_bp, bp)
                rungs = _nested_rungs(c, target)
                if "match" in rungs:
                    w = time.time() - t0
                    return Certificate("certified", target, len(target), kind, sequence=c,
                                       checked_by="ViennaRNA MFE (turner2004, dangles=2)",
                                       strictness=rungs[-1], rungs_passed=rungs,
                                       cost_usd=cost, wait_s=round(w, 2), budget=budget,
                                       reason=f"solved at the {tl}s tier after trying {tried}",
                                       detail={"tiers_tried": tried, "n_candidates_screened": len(uniq)})
        else:
            for c in uniq:
                n_screened += 1
                v = _panel_verdict(c, target)
                if v["unanimous"]:
                    w = time.time() - t0
                    return Certificate("certified", target, len(target), kind, sequence=c,
                                       checked_by="panel: pKiss AND ProbKnot AND ShapeKnots",
                                       strictness="panel-unanimous", rungs_passed=["panel-unanimous"],
                                       cost_usd=cost, wait_s=round(w, 2), budget=budget,
                                       reason=f"all three independent predictors agree; solved at the "
                                              f"{tl}s tier after trying {tried}",
                                       detail={"tiers_tried": tried, "n_candidates_screened": len(uniq),
                                               "per_oracle": v["per_oracle"],
                                               "operating_point": PANEL_OPERATING_POINT})
                # KEEP THE BEST CANDIDATE, NOT THE FIRST ONE THAT CLEARS THE FLOOR. This used to stop at
                # the first `knot_confirmed` sequence, which threw away evidence already paid for: over 250
                # pseudobase targets, 61.6% had a strictly higher-grade candidate sitting in a pool that had
                # already been generated and already been folded, and none would have been made worse by
                # taking it (k=3 targets 8 -> 80, k=1 targets 135 -> 10). Three non-redundant outside judges
                # agree the step from k=2 to k=3 is where the panel's evidence actually lives (+64.6, +36.4,
                # +29.0 points), so shipping k=1 while a k=3 sat in the same pool was a real loss.
                #
                # It is free because oracle_pairs() already folded this candidate for the panel verdict
                # above; grading it costs set intersections, not subprocesses.
                #
                # `grade_max=False` restores the old first-clearing behaviour. It exists so the change can
                # be measured against itself on the same candidate stream rather than argued about, and it
                # is not a product option -- the shipped default is the better one.
                if best_knot is None or grade_max:
                    kr = _knot_recall(c, target)
                    if kr["knot_confirmed"]:
                        if best_knot is None or len(kr["confirmed_by"]) > len(best_knot[1]["confirmed_by"]):
                            best_knot = (c, kr)
                        elif grade_max and len(kr["confirmed_by"]) == len(best_knot[1]["confirmed_by"]):
                            # SAME GRADE: break the tie on ensemble determinedness rather than on arrival
                            # order. Our designs fold roughly three times more DEFINITELY than natural
                            # pseudoknots do (mean positional entropy 0.087 against 0.299) -- so definitely
                            # that a probability-based pseudoknot predictor has no mass left to place on the
                            # crossing pairs. Among equal-grade candidates, preferring the less over-determined
                            # one raised outside agreement on 249 targets with never-used seeds:
                            # IPknot 21.8% -> 27.0% (29 wins / 16 losses, sign p=0.0362), Knotty 62.1% ->
                            # 67.3% (p=0.077), RNAPKplex 52.8% -> 50.8% (not significant).
                            # It is a TIE-BREAKER and not a selector: picking purely on entropy scored
                            # exactly 34-34, and picking on GC was significantly WORSE (14-32), which is
                            # what rules out plain composition. Entropy is computed only when a tie actually
                            # occurs, and costs 0.6-1.0% of one candidate's panel folds.
                            if _ensemble_entropy(c) > _ensemble_entropy(best_knot[0]):
                                best_knot = (c, kr)
                if best_ref_only is None and v["reference_only"]:
                    best_ref_only = (c, v)

    w = time.time() - t0
    if kind == "pseudoknot" and best_knot is not None and calibrated_rule_applies(target):
        c, kr = best_knot
        return Certificate("knot_confirmed", target, len(target), kind, sequence=c,
                           checked_by="crossing stems: " + ", ".join(kr["confirmed_by"]),
                           strictness="knot-recall (k=1, tau=1.0)", rungs_passed=["knot-recall"],
                           cost_usd=cost, wait_s=round(w, 2), budget=budget,
                           detail={"tiers_tried": tried, "knot": kr,
                                   "operating_point": KNOT_OPERATING_POINT,
                                   "n_candidates_screened": n_screened,
                                   "topology": _topology_point(target), "grade": knot_grade(kr)},
                           reason="no candidate reproduced the whole structure under all three predictors, "
                                  "but the crossing stems that make this a pseudoknot are fully reproduced "
                                  "by " + ", ".join(kr["confirmed_by"]) + ". Claim is about the KNOT only; "
                                  "the criterion recognises 78.5% of known natural structures (held out, "
                                  "95% CI 68.8-86.3) and accepted 0 of 183 pseudoknot-free decoys.")
    if kind == "pseudoknot" and best_ref_only is not None:
        c, v = best_ref_only
        return Certificate("cannot_certify", target, len(target), kind, sequence=c,
                           checked_by="panel: pKiss AND ProbKnot AND ShapeKnots",
                           cost_usd=cost, wait_s=round(w, 2), budget=budget, detail=v,
                           reason="the search found sequences the reference oracle (pKiss) accepts, but the "
                                  "independent predictors do not confirm them. A single-oracle tool would "
                                  "have returned one of these as the answer. Reported as UNCONFIRMED rather "
                                  "than wrong: the same panel recovers only 32%/6%/8% of NATIVE structures "
                                  "exactly, so non-confirmation is weak evidence against a design.")
    nxt = {"cheap": "standard", "standard": "best_effort"}.get(budget)
    # What a larger budget is actually worth, measured (runs/ceiling/wall.py). Of the 16 Eterna100 targets
    # unsolved at t<=180s, extending to t<=1800s solved 7 and left 9 -- and four of those nine sit at bp=1-2
    # and did not move under a 10x budget, the same signature the archived E1 certified as a designability
    # wall rather than a search-budget wall. So "just pay more" is not an honest thing to tell a caller: it
    # helps SOME targets and nothing here can say which. A near miss is specifically NOT a promise.
    more = ""
    if nxt:
        more = (f" A larger budget is available (budget='{nxt}'), but it is not a promise: on the measured "
                f"Eterna100 residual, raising the budget 10x solved 7 of 16 and left 9, and this call cannot "
                f"tell you which kind yours is.")
    else:
        more = (" This is the largest budget offered. On the measured Eterna100 residual, 9 of 16 targets "
                "remain unsolved even at this tier, and four of those sit one base pair away without moving "
                "under a 10x budget -- for those, more compute is not the missing ingredient.")
    near = ""
    if best_bp is not None:
        near = (f" Closest candidate was {best_bp} base pair(s) away."
                + (" In our measurements targets stuck at 1-2 base pairs did NOT improve with 10x more"
                   " budget, so treat this as a hard case rather than an almost." if best_bp <= 2 else ""))
    return Certificate("not_found", target, len(target), kind, budget=budget,
                       checked_by=("panel: pKiss AND ProbKnot AND ShapeKnots" if kind == "pseudoknot"
                                   else "ViennaRNA MFE (turner2004, dangles=2)"),
                       cost_usd=cost, wait_s=round(w, 2),
                       detail={"tiers_tried": tried, "best_bp": best_bp},
                       reason=(f"no sequence survived the criterion within the '{budget}' budget "
                               f"(tiers {tried} s).{near}{more}"))


# --------------------------------------------------------------------------- public entry points
def certify_sequence(seq: str, target: str) -> Certificate:
    """Check an existing sequence. Thin wrapper so the audit block is attached at exactly one place."""
    return _certify_sequence_impl(seq, target).stamp(seed=None)


def certify_design(target: str, budget: str = "standard", seed: int = 0,
                   tiers: Optional[tuple] = None, grade_max: bool = True) -> Certificate:
    """Design and certify. Thin wrapper so the audit block is attached at exactly one place."""
    return _certify_design_impl(target, budget=budget, seed=seed, tiers=tiers, grade_max=grade_max).stamp(
        seed=seed, budget=(budget if not tiers else f"{budget} (capped to {list(tiers)}s)"))


# --------------------------------------------------------------------------- plain language
# The five decisions are made for the caller; reporting the RESULT in expert vocabulary undoes that. These
# tables and describe() exist so a certificate can be read by the person it was built for.
VERDICT_LABEL = {
    "certified":      ("已认证", "CERTIFIED"),
    "knot_confirmed": ("假结已确认", "KNOT CONFIRMED"),
    "cannot_certify": ("无法确认", "CANNOT CERTIFY"),
    "not_found":      ("没找到", "NOT FOUND"),
    "invalid_input":  ("输入有问题", "INVALID INPUT"),
}
STRICTNESS_DOC = {
    "match": ("折叠结果与目标完全一致",
              "the fold equals the target exactly"),
    "umfe": ("而且目标是唯一的最优折叠,不是并列多解之一",
             "and the target is the UNIQUE optimal fold, not one of several co-optimal ones"),
    "pfe": ("而且在热力学系综里目标结构占比 ≥90%,是稳健折叠而非脆弱的一次命中",
            "and the target holds >=90% of the thermodynamic ensemble -- a robust fold, not a fragile hit"),
    "panel-unanimous": ("三个相互独立的假结预测器全部认为它形成了目标结构",
                        "all three independent pseudoknot predictors agree it forms the target"),
    "knot-recall (k=1, tau=1.0)": (
        "至少一个独立预测器完整重现了使它成为假结的那些交叉碱基对(只针对假结部分,不针对整个结构)",
        "at least one independent predictor fully reproduces the crossing stems -- a claim about the KNOT only"),
}
RUNG_ORDER = ["match", "umfe", "pfe"]


def describe(target: str, lang: str = "zh") -> str:
    """Restate a dot-bracket in words. The caller was told they do not need to read brackets, so echoing the
    bracket string back as the answer would take that back."""
    from ..structure_utils import pair_table
    pt = pair_table(target)
    n = len(target)
    n_bp = sum(1 for i, j in enumerate(pt) if j > i)
    stems, run = [], 0
    for i, ch in enumerate(target):
        if ch in "([{<":
            run += 1
        elif run:
            stems.append(run)
            run = 0
    if run:
        stems.append(run)
    hairpins = []
    for i, j in enumerate(pt):
        if j > i and all(pt[k] < 0 for k in range(i + 1, j)):
            hairpins.append(j - i - 1)
    if lang == "zh":
        parts = [f"共 {n} 个碱基", f"{n_bp} 对碱基配对"]
        if hairpins:
            parts.append(f"{len(hairpins)} 个发夹环(环大小 {'、'.join(str(h) for h in hairpins)} 个未配对碱基)")
        if len(stems) > 1:
            parts.append(f"{len(stems)} 段茎")
        return ",".join(parts)
    parts = [f"{n} nucleotides", f"{n_bp} base pairs"]
    if hairpins:
        parts.append(f"{len(hairpins)} hairpin loop(s) of {', '.join(str(h) for h in hairpins)} nt")
    if len(stems) > 1:
        parts.append(f"{len(stems)} stem segments")
    return ", ".join(parts)


def _why_zh(reason: str) -> str:
    """Translate the handful of reasons this module emits. Unmatched text falls through in English rather
    than being machine-translated -- a wrong translation of WHY an answer was refused is worse than English."""
    import re as _re
    r = reason or ""
    m = _re.match(r"solved at the (\d+)s tier after trying \[([^\]]*)\]", r)
    if m:
        return f"在 {m.group(1)} 秒档解出(依次尝试了 {m.group(2)} 秒)"
    m = _re.match(r"all three independent predictors agree", r)
    if m:
        return "三个相互独立的预测器一致认为它形成了目标结构"
    if r.startswith("folds exactly to the target"):
        extra = r.split("also passes:", 1)
        more = ("，" + "、".join(STRICTNESS_DOC.get(x.strip(), (x.strip(),))[0].lstrip("而且")
                                 for x in extra[1].split(",") if x.strip() and x.strip() != "none")
                ) if len(extra) > 1 and extra[1].strip() not in ("", "none") else ""
        return "折叠结果与目标完全一致" + more
    if r.startswith("the MFE structure of this sequence is not the target"):
        return "这条序列的最低自由能结构不是目标结构"
    if r.startswith("no independent predictor forms the target"):
        return "没有任何一个独立预测器认为这条序列形成了目标结构"
    if "unbalanced" in r:
        return "括号不配对,这不是一个合法的结构式"
    if "not dot-bracket" in r:
        return "输入里有不属于点括号记法的字符"
    m = _re.match(r"sequence length (\d+) != target length (\d+)", r)
    if m:
        return (f"序列长度 {m.group(1)} 与结构长度 {m.group(2)} 不一致,无法判定。"
                f"请把两者改成同样长度,或者只给结构、让系统为你设计一条。")
    if r.startswith("no sequence survived the criterion"):
        out = ["在当前预算档内没有找到能通过判据的序列。"]
        m = _re.search(r"Closest candidate was (\d+) base pair", r)
        if m:
            out.append(f"最接近的候选差 {m.group(1)} 对碱基。")
            if int(m.group(1)) <= 2:
                out.append("注意:实测中卡在 1–2 对碱基的靶,预算加十倍也没有改善 —— 这属于硬案例,而不是「就快成了」。")
        if "larger budget is available" in r:
            out.append("可以换更大的预算档,但这不是承诺:实测 16 个这类靶里,预算加十倍解出 7 个、仍有 9 个解不出,"
                       "而且无法事先判断你的属于哪一类。")
        else:
            out.append("这已经是最大的预算档。")
        return "".join(out)
    if r.startswith("the reference oracle (pKiss) accepts"):
        return ("参考预测器认为这条序列可以,但另外两个独立预测器不确认。这里报告为「未确认」而不是「错误」:"
                "同一套判据在 250 个天然结构上也只认得出 2 个,所以不通过几乎不含信息。")
    return r


def sketch(target: str, seq: Optional[str] = None) -> Optional[str]:
    """A text drawing of the structure, or None when the shape is outside what can be drawn HONESTLY.

    A caller who was told they need not read dot-bracket should not be handed dot-bracket as the answer, and
    for a hairpin -- a shape with an obvious geometry -- a picture carries far more than a sentence. The
    scope is deliberately narrow: exactly one perfectly stacked stem closing one loop, with optional unpaired
    tails. Anything else returns None and the caller falls back to the plain-language summary, because
    guessing at a layout for a shape this cannot draw would misinform rather than help.
    """
    from ..structure_utils import pair_table
    pt = pair_table(target)
    pairs = sorted((i, j) for i, j in enumerate(pt) if j > i)
    if not pairs or len(pairs) < 2:
        return None
    i0, j0 = pairs[0]
    if any(p != (i0 + k, j0 - k) for k, p in enumerate(pairs)):
        return None                                   # not a single perfectly stacked stem
    inner_i, inner_j = pairs[-1]
    loop = list(range(inner_i + 1, inner_j))
    if any(pt[k] >= 0 for k in loop):
        return None
    if any(ch != "." for ch in target[:i0] + target[j0 + 1:]):
        return None                                   # unpaired tails only

    if seq and len(seq) != len(target):
        seq = None                                    # a mismatched sequence would draw the wrong bases

    def base(k):
        return (seq[k] if seq else "\u25cf")

    IND = 6                                           # column of the left-hand stem base
    W = max(len(loop) * 2 - 1, 1)                     # width between the two stem columns
    lines = [" " * (IND + 1) + " ".join(base(k) for k in loop),
             " " * IND + "\u256d" + "\u2500" * W + "\u256e"]
    for (a, b) in reversed(pairs):                    # innermost at the top, like a textbook hairpin
        lines.append(" " * IND + base(a) + "\u2500" * W + base(b))
    lines.append(" " * (IND - 3) + "5'\u2500\u256f" + " " * W + "\u2570\u25003'")
    n5, n3 = len(target[:i0]), len(target[j0 + 1:])
    if n5 or n3:
        lines.append(" " * (IND - 3) + f"\uff085' \u7aef\u53e6\u6709 {n5} \u4e2a\u672a\u914d\u5bf9\u78b1\u57fa"
                                       f"\uff0c3' \u7aef {n3} \u4e2a\uff09")
    return "\n".join(lines)


def to_display(c: Certificate, lang: str = "zh") -> dict:
    """A certificate as STRUCTURED fields a UI can lay out -- verdict, plain-language structure, the
    criterion with its meaning spelled out, the sequence on its own so it can be copied, and the red line."""
    zh = (lang == "zh")
    judge = c.checked_by
    if zh:
        judge = (judge.replace("ViennaRNA MFE (turner2004, dangles=2)",
                               "ViennaRNA 折叠算法（Turner 2004 能量模型）")
                      .replace("panel: pKiss AND ProbKnot AND ShapeKnots",
                               "三个相互独立的假结预测器（pKiss、ProbKnot、ShapeKnots）全部同意")
                      .replace("crossing stems: ", "重现交叉茎的预测器："))
    d = {"verdict": c.verdict, "kind": c.kind, "n": c.n,
         "headline": VERDICT_LABEL.get(c.verdict, (c.verdict, c.verdict))[0 if zh else 1],
         "sequence": c.sequence, "target": c.target,
         "plain": describe(c.target, lang) if c.kind in ("nested", "pseudoknot") else None,
         "sketch": (sketch(c.target, c.sequence) if c.kind == "nested" else None),
         "checked_by": judge, "checked_by_raw": c.checked_by,
         "cost_usd": c.cost_usd, "wait_s": c.wait_s,
         "budget": c.budget, "why": (_why_zh(c.reason) if zh else c.reason),
         "audit": {"schema": (c.provenance or {}).get("schema"), "generated_utc": c.generated_utc,
                   "seed": c.seed, "verify_cmd": c.verify_cmd, "rerun_search": c.rerun_search,
                   "reproducibility": c.reproducibility, "versions": c.provenance},
         "note": ("`已认证`的含义是:在所声明的计算模型下,这条序列折叠成目标结构。它不是关于体外或体内行为的主张。"
                  if zh else c.disclaimer)}
    if c.strictness:
        zh_t, en_t = STRICTNESS_DOC.get(c.strictness, (c.strictness, c.strictness))
        zh_t = zh_t.lstrip("而且")          # it reads as a continuation in the rung list, not as a headline
        also = [STRICTNESS_DOC.get(r, (r, r))[0 if zh else 1].lstrip("而且") if zh
                else STRICTNESS_DOC.get(r, (r, r))[1]
                for r in c.rungs_passed if r != c.strictness]
        d["criterion"] = {"code": c.strictness, "means": zh_t if zh else en_t, "also": also,
                          "is_strongest": (c.strictness == "pfe")}
    op = (c.detail or {}).get("operating_point")
    if op:
        d["operating_point"] = {
            "sens": op["sensitivity_on_known_structures"],
            "random": op["false_accept_on_random_sequences"],
            "scope": op.get("scope"),
            "text": (f"这条判据在 250 个已知天然结构上认得出 "
                     f"{op['sensitivity_on_known_structures']:.1%},在随机序列上误判 "
                     f"{op['false_accept_on_random_sequences']:.1%}"
                     if zh else
                     f"this criterion recognises {op['sensitivity_on_known_structures']:.1%} of known "
                     f"natural structures and accepts "
                     f"{op['false_accept_on_random_sequences']:.1%} of random sequences")}
    gr = (c.detail or {}).get("grade")
    if gr:
        # WHAT THE READER NEEDS TO KNOW is not "it passed" but HOW MANY of the three were behind it. The
        # random false-accept rate is identical at every grade, so it cannot express this and the text says so
        # rather than letting a reader infer a specificity difference that the calibration does not support.
        d["grade"] = {
            "code": gr["grade"], "agreeing_oracles": gr["agreeing_oracles"],
            "text": (f"三个独立预测器中有 {gr['agreeing_oracles']} 个重现了交叉茎。"
                     f"出厂门槛是 ≥1;更高的等级并不降低随机误收率(每一级都是 0.4%),"
                     f"它降低的是【结论依赖某一个预测器的偏好】这一风险。"
                     if zh else
                     f"{gr['agreeing_oracles']} of 3 independent predictors reproduced the crossing stems. "
                     f"The shipped threshold is >=1; a higher grade does NOT lower the random false-accept "
                     f"rate (it is 0.4% at every grade) -- what it lowers is the risk that the result rests "
                     f"on one predictor's idiosyncrasy.")}
    return d


# --------------------------------------------------------------------------- human-facing rendering
def render(c: Certificate) -> str:
    """A certificate as a non-expert should read it: the verdict, what was actually checked, what it cost,
    and -- for the two negative verdicts -- why, in words rather than a status code."""
    icon = {"certified": "CERTIFIED", "knot_confirmed": "KNOT CONFIRMED",
            "cannot_certify": "CANNOT CERTIFY", "not_found": "NOT FOUND",
            "invalid_input": "INVALID INPUT"}.get(c.verdict, c.verdict.upper())
    out = [f"[{icon}]  {c.kind} target, {c.n} nt"]
    if c.sequence:
        out.append(f"  sequence   {c.sequence}")
    if c.checked_by:
        out.append(f"  checked by {c.checked_by}")
    if c.strictness:
        out.append(f"  criterion  {c.strictness}"
                   + (f"   (also passes: {', '.join(c.rungs_passed[:-1])})"
                      if len(c.rungs_passed) > 1 else ""))
    op = (c.detail or {}).get("operating_point")
    if op:
        out.append(f"  this criterion recognises "
                   f"{op['sensitivity_on_known_structures']:.1%} of known natural structures and accepts "
                   f"{op['false_accept_on_random_sequences']:.1%} of random sequences")
        if op.get("scope"):
            out.append(f"  scope      {op['scope']}")
    if c.cost_usd or c.wait_s:
        out.append(f"  cost       ${c.cost_usd:.5f}   waited {c.wait_s:.1f}s"
                   + (f"   budget '{c.budget}'" if c.budget else ""))
    if c.reason:
        import textwrap
        for i, line in enumerate(textwrap.wrap(c.reason, 92)):
            out.append(("  why        " if i == 0 else "             ") + line)
    out.append(f"  note       {c.disclaimer}")
    if c.provenance:
        v = c.provenance
        out.append(f"  produced   {c.generated_utc}   seed {c.seed}   schema {v.get('schema')}")
        out.append(f"  versions   ViennaRNA {v.get('viennarna')} | DesiRNA {v.get('desirna')} | "
                   f"pKiss {v.get('pkiss')} | RNAstructure {v.get('rnastructure')}")
        if c.verify_cmd:
            out.append(f"  verify     {c.verify_cmd}")
        out.append(f"  re-search  {c.rerun_search}")
        out.append(f"  note       {c.reproducibility}")
    return "\n".join(out)


def _cli(argv=None):
    import argparse
    import json as _json
    ap = argparse.ArgumentParser(
        prog="python -m rna_harness.cognitive.certify",
        description="Design or check an RNA sequence and return a CERTIFICATE: what was verified, by which "
                    "oracles, at what strictness, at what cost -- and what it does not claim.")
    ap.add_argument("target", help="target secondary structure in dot-bracket, e.g. '((((....))))'")
    ap.add_argument("--sequence", help="check THIS sequence instead of designing one (free, no search)")
    ap.add_argument("--budget", default="standard", choices=sorted(BUDGETS),
                    help="cheap = one 5s tier (answers ~72%% of targets); standard = 5/15/60/180s; "
                         "best_effort adds 600/1800s")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    a = ap.parse_args(argv)
    c = (certify_sequence(a.sequence, a.target) if a.sequence
         else certify_design(a.target, budget=a.budget, seed=a.seed))
    print(_json.dumps(c.to_dict(), indent=1) if a.json else render(c))
    return 0 if c.verdict in ("certified", "knot_confirmed") else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
