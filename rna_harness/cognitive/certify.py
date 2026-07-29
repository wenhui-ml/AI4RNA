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

import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional

from .cost import solver_cost_usd
from ..structure_utils import has_pseudoknot

# Budget tiers in DesiRNA -t seconds. Chosen from the measured curve on Eterna100-V2 (runs/ceiling):
#   t=5s 72/100 at $0.00211   t=15s 77   t=60s 81   t=180s 84   -- and the cascade reaches 84 for $1.0041
# so "cheap" is not a crippled mode: it already answers 72% of targets in about 9 seconds.
BUDGETS = {
    "cheap": (5,),
    "standard": (5, 15, 60, 180),
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
    "sensitivity_on_known_structures": 0.840,
    "false_accept_on_random_sequences": 0.004,
    "scope": "the CROSSING STEMS only -- this says nothing about the rest of the structure",
    "note": ("sensitivity is unchanged from tau=1.0 down to 0.9: these predictors get a knot entirely right "
             "or entirely wrong. Composing this with an exact nested-skeleton check was tried and refuted "
             "(ViennaRNA recovers the nested reduction from only 20 of 186 natives)."),
}


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
def certify_sequence(seq: str, target: str) -> Certificate:
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
        return Certificate("not_found", target, len(target), kind, sequence=None,
                           checked_by="ViennaRNA MFE (turner2004, dangles=2)",
                           cost_usd=solver_cost_usd(w, 1), wait_s=round(w, 2),
                           reason="the MFE structure of this sequence is not the target")

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
    if kr["knot_confirmed"]:
        return Certificate("knot_confirmed", target, len(target), kind, sequence=seq,
                           checked_by="crossing stems: " + ", ".join(kr["confirmed_by"]),
                           strictness="knot-recall (k=1, tau=1.0)", rungs_passed=["knot-recall"],
                           cost_usd=solver_cost_usd(time.time() - t0, 1),
                           wait_s=round(time.time() - t0, 2),
                           detail={**v, "knot": kr, "operating_point": KNOT_OPERATING_POINT},
                           reason="the crossing stems that make this a pseudoknot are fully reproduced by "
                                  + ", ".join(kr["confirmed_by"]) + ". This is a claim about the KNOT, not "
                                  "the whole structure: the criterion recognises 84% of known natural "
                                  "structures at 0.4% false accept on random sequences.")
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


def _knot_recall(seq: str, target: str) -> dict:
    """Per-oracle recall of the CROSSING pairs -- the pairs that make the target a pseudoknot at all.

    This is the one quantity that calibrates. Asking whether the whole structure is reproduced exactly and
    unanimously is unanswerable with these predictors (sensitivity 0.8% on known natural structures); asking
    whether the crossing stems are reproduced by at least one of them reaches 84.0% sensitivity at 0.4% false
    accept on random sequences (runs/ceiling/calibrate.py). `None` means the oracle failed to run, which is
    never a rejection.
    """
    from ..benchmarks.rnainvbench import _round_only, pk_pairs
    from ..oracles_pk import pkiss_pairs, probknot_pairs, shapeknots_pairs
    cross = pk_pairs(target) - pk_pairs(_round_only(target))
    out = {}
    for name, fn_ in (("pkiss", pkiss_pairs), ("probknot", probknot_pairs),
                      ("shapeknots", shapeknots_pairs)):
        ps = fn_(seq)
        out[name] = None if (ps is None or not cross) else len(ps & cross) / len(cross)
    full = [k for k, v in out.items() if v is not None and v >= 1.0]      # tau = 1.0, the strictest point
    return {"per_oracle": out, "n_cross": len(cross), "confirmed_by": full,
            "knot_confirmed": bool(full)}


def _panel_verdict(seq: str, target: str) -> dict:
    """Adjudicate one sequence by the three independent pseudoknot predictors.

    `None` from an oracle means THE ORACLE FAILED TO RUN, not "rejected"; conflating the two made the tool
    report zero solves with no diagnostic whenever RNAstructure was absent."""
    from ..oracles_pk import forms_target, pkiss_pairs, probknot_pairs, shapeknots_pairs
    out, failed = {}, False
    for name, fn in (("pkiss", pkiss_pairs), ("probknot", probknot_pairs), ("shapeknots", shapeknots_pairs)):
        v = forms_target(fn(seq), target)
        out[name] = v
        failed = failed or (v is None)
    return {"per_oracle": out, "any_failed_to_run": failed,
            "unanimous": all(out.get(k) is True for k in ("pkiss", "probknot", "shapeknots")),
            "reference_only": (out.get("pkiss") is True
                               and not all(out.get(k) is True for k in ("probknot", "shapeknots")))}


# --------------------------------------------------------------------------- design + certify
def certify_design(target: str, budget: str = "standard", seed: int = 0) -> Certificate:
    """"Give me a sequence that folds into this shape", answered with a certificate.

    Runs the measured budget cascade and stops at the first sequence that survives the criterion for this
    KIND of target. Nothing is taken from the solver's self-report: every candidate is re-adjudicated here.
    """
    kind, err = classify(target)
    if kind == "invalid":
        return Certificate("invalid_input", target, len(target or ""), "invalid", reason=err, budget=budget)
    tiers = BUDGETS.get(budget)
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
                if best_knot is None:
                    kr = _knot_recall(c, target)
                    if kr["knot_confirmed"]:
                        best_knot = (c, kr)
                if best_ref_only is None and v["reference_only"]:
                    best_ref_only = (c, v)

    w = time.time() - t0
    if kind == "pseudoknot" and best_knot is not None:
        c, kr = best_knot
        return Certificate("knot_confirmed", target, len(target), kind, sequence=c,
                           checked_by="crossing stems: " + ", ".join(kr["confirmed_by"]),
                           strictness="knot-recall (k=1, tau=1.0)", rungs_passed=["knot-recall"],
                           cost_usd=cost, wait_s=round(w, 2), budget=budget,
                           detail={"tiers_tried": tried, "knot": kr,
                                   "operating_point": KNOT_OPERATING_POINT},
                           reason="no candidate reproduced the whole structure under all three predictors, "
                                  "but the crossing stems that make this a pseudoknot are fully reproduced "
                                  "by " + ", ".join(kr["confirmed_by"]) + ". Claim is about the KNOT only; "
                                  "the criterion recognises 84% of known natural structures at 0.4% false "
                                  "accept on random sequences.")
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
