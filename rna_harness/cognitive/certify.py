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


@dataclass
class Certificate:
    verdict: str                      # certified | not_found | cannot_certify | invalid_input
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
                           detail=v)
    if v["any_failed_to_run"]:
        return Certificate("cannot_certify", target, len(target), kind, sequence=seq,
                           checked_by="panel (incomplete)", cost_usd=solver_cost_usd(w, 1),
                           wait_s=round(w, 2), detail=v,
                           reason="a required independent oracle could not be run, so this cannot be "
                                  "adjudicated. A missing oracle is not a rejection.")
    if v["reference_only"]:
        return Certificate("cannot_certify", target, len(target), kind, sequence=seq,
                           checked_by="panel: pKiss AND ProbKnot AND ShapeKnots",
                           cost_usd=solver_cost_usd(w, 1), wait_s=round(w, 2), detail=v,
                           reason="the reference oracle (pKiss) accepts this sequence but the independent "
                                  "predictors do not. Reported as UNCONFIRMED, not as wrong: on native "
                                  "sequences the same panel recovers only 32%/6%/8% of known structures "
                                  "exactly, so its rejection is weak evidence about the design.")
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
                                               "per_oracle": v["per_oracle"]})
                if best_ref_only is None and v["reference_only"]:
                    best_ref_only = (c, v)

    w = time.time() - t0
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
