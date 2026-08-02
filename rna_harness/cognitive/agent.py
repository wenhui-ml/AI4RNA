"""An agent loop over exact primitives: observe a real failure, decide the next action, act, verify.

WHY A LOOP AND NOT A PIPELINE. The first design of this layer was Plan -> Gate -> Tools -> Verdict, one shot:
on failure it said "try a bigger budget" and stopped. That borrowed Claude Code's vocabulary and dropped its
mechanism. Claude Code's value is not generating the right code first time -- it is running the thing, reading
the real compiler error, and changing what it does. Our certify() already computes rich failure detail (how
many base pairs away, which constraint conflicts, which oracle disagreed) and, before this module, nothing
consumed it: the feedback existed and the loop did not.

WHAT THE EVIDENCE ACTUALLY RULED OUT. This branch measured, and I have been over-applying, two nulls:

    LLM authoring search operators        0/2   and both reward-hacked their own in-loop scorer
    LLM predicting routing from features  0     zero-loss abstention caught 0 of 16

Both are "decide WITHOUT observing". The loop below is "decide AFTER observing", which is Claude Code's
actual mechanism and which this project has never tested. Treating the first two results as evidence against
the third was a mistake, and it is why the pipeline version looked acceptable.

WHERE WE DEPART FROM CLAUDE CODE, on purpose. Claude Code can let a model pick tools freely because compiler
errors are LOUD -- pick wrong and it explodes. RNA oracles are silently wrong: hand a pseudoknot target to the
ViennaRNA MFE oracle and it returns bp=0, a confident false success, because MFE cannot represent '[' at all.
So every claim of "solved" still goes through the calibrated verdict in certify(); the agent is free in what
it TRIES and not free in what counts as done.

THE DIVISION OF LABOUR, unchanged from the rest of this codebase:
    the model    chooses the next ACTION and its declarative arguments
    the code     does every count, assembly and comparison
    the oracles  decide truth
No primitive below accepts a dot-bracket string written by the model, and none returns a verdict the model
can overrule.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

# MEASURED, not guessed: a restructure costs propose + re-check + design, and the three items that still
# failed at 10 had all FOUND a conflict-free designable shape and then ran out before they could pay for it.
MAX_STEPS = 14   # the real bound is max_cost_usd; this one only stops a loop that is going nowhere

# The output cap covers reasoning AND content, so it must clear the hardest shape's reasoning with room left
# for the json. MEASURED on the benchmark's own items: a hairpin decision costs ~300 tokens, a junction or an
# H-type knot 2900-4300 tokens of reasoning alone. At 1200 the junction family scored 0/6 with EMPTY replies.
# The shape probe is only free on short targets, and "free" is the entire reason it is run before every
# design. MEASURED on bpRNA structures, 6 per band, 20 s cap: <=110 nt answers 6/6 at a 1.1 s median; 111-150
# answers 5/6 at 14.4 s; 151-200 answers 1/6; above 200 it answers 0/6 and always burns the full cap. So
# above the cutoff it stops being either free or informative, and it declines by length instead of making a
# user wait 20 s to be told nothing.
PROBE_MAX_N = 110

THINK_TOKENS = 6000
THINK_TOKENS_MAX = 16000   # the one retry after a truncated reply; beyond this it is a real failure
WC = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")}


@dataclass
class Step:
    n: int
    action: str
    args: dict
    observation: dict
    cost_usd: float = 0.0
    wall_s: float = 0.0
    thought: str = ""


@dataclass
class Session:
    request: str
    steps: List[Step] = field(default_factory=list)
    target: Optional[str] = None
    restraint: Optional[str] = None
    gc: Optional[dict] = None
    tm: Optional[dict] = None                        # melting-temperature requirement, checked not steered
    kin: Optional[dict] = None                       # co-transcriptional requirement -- the second measured gap
    switch: Optional[dict] = None                    # two-state requirement -- the third measured gap
    sequences: Optional[List[str]] = None            # set-level answers (design_set), when one was asked for
    pins: List[dict] = field(default_factory=list)   # (motif, at) as REQUESTED, so a reshape can re-apply them
    requested_target: Optional[str] = None           # the FIRST shape proposed -- what the user asked for
    requested_by_user: bool = False                  # True when it came VERBATIM from the request text
    substituted: Optional[dict] = None               # set when the delivered shape is not the requested one
    unverified_claim: Optional[dict] = None          # a functional claim we design AROUND but never check
    certificate: Optional[dict] = None
    done: bool = False
    outcome: str = ""
    total_cost_usd: float = 0.0     # solver + the agent's own thinking; see llm_cost_usd for the split
    llm_cost_usd: float = 0.0       # what the reasoning cost, as billed -- excluded, this number flatters us

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- primitives
# Each one is exact, each returns a STRUCTURED observation the agent can act on, and none of them is allowed
# to be the thing that says "solved".
def propose_structure(elements: List[Dict]) -> dict:
    """Assemble a dot-bracket from element types and integer lengths. The model never writes brackets."""
    from .intent import Spec, assemble
    db, err = assemble(Spec(elements=elements))
    if err:
        return {"ok": False, "error": err}
    from .certify import classify, describe
    kind, cerr = classify(db)
    if cerr:
        return {"ok": False, "error": cerr}
    return {"ok": True, "target": db, "n": len(db), "kind": kind, "plain": describe(db)}


def structures_in(request: str) -> List[str]:
    """Every dot-bracket string that is LITERALLY PRESENT in the request, in order of appearance.

    This is the whole mechanism behind use_target. Most people who already work with RNA arrive holding a
    structure, and until now there was no way for one to enter this loop at all: the only route to a target
    was assembling it from a closed grammar of five element types, which covers almost none of what nature
    actually makes. A real bpRNA structure could not be worked on here.

    The obvious fix -- let the model type the structure -- would break the rule the rest of this module is
    built on: the model never writes dot-bracket, because counting parentheses is precise computation and
    that is what the code is for. So the code extracts the candidates and the model only chooses WHICH one.
    It can never introduce a bracket that was not in front of the user.
    """
    import re as _re
    from .certify import classify
    out, seen = [], set()
    for m in _re.finditer(r"[().\[\]{}<>]{4,}", request or ""):
        cand = m.group(0)
        if cand in seen or not any(c in cand for c in "([{<"):
            continue
        if classify(cand)[1] is None:                  # balanced and well-formed, else it is not a structure
            seen.add(cand)
            out.append(cand)
    return out


def use_target(request: str, which: int = 0) -> dict:
    """Adopt a structure the REQUEST already contains. `which` indexes the list; no string is accepted."""
    from .certify import classify, describe
    cands = structures_in(request)
    if not cands:
        return {"ok": False, "error": ("the request contains no dot-bracket structure, so there is nothing "
                                       "to adopt -- build one with propose_structure instead")}
    if not isinstance(which, int) or which < 0 or which >= len(cands):
        return {"ok": False, "error": (f"the request contains {len(cands)} structure(s), indexed 0.."
                                       f"{len(cands) - 1}; `which`={which!r} is not one of them"),
                "available": [{"which": i, "n": len(c)} for i, c in enumerate(cands)]}
    db = cands[which]
    kind, _ = classify(db)
    return {"ok": True, "target": db, "n": len(db), "kind": kind, "plain": describe(db),
            "note": "adopted verbatim from the request -- it is the user's structure, not a proposed one"}


def place_constraint(target: str, motif: str, at: int, existing: Optional[str] = None) -> dict:
    """Pin `motif` at 0-based position `at`, ADDING to whatever is already pinned.

    Additive because real requests have more than one fixed region, and because a caller who pins a second
    motif expects the first to survive. Two placements that disagree at the same position are reported as a
    conflict rather than silently letting the later one win -- a silent overwrite would answer a question
    nobody asked.
    """
    motif = (motif or "").upper().replace("T", "U")
    if not target:
        return {"ok": False, "error": "no target yet -- propose_structure first"}
    if not motif:
        return {"ok": False, "error": "empty motif"}
    if set(motif) - set("ACGU"):
        return {"ok": False, "error": f"motif contains non-ACGU characters: {sorted(set(motif) - set('ACGU'))}"}
    if at < 0 or at + len(motif) > len(target):
        return {"ok": False, "error": (f"a {len(motif)}-nt motif at position {at} does not fit in a "
                                       f"{len(target)}-nt target (valid range 0..{len(target) - len(motif)})")}
    if existing and len(existing) != len(target):
        return {"ok": False, "error": (f"the existing restraint is {len(existing)} long but the target is "
                                       f"{len(target)} -- it belongs to a different shape. Silently starting "
                                       f"over would drop pins the caller still expects.")}
    r = list(existing) if existing else ["N"] * len(target)
    clash = []
    for k, ch in enumerate(motif):
        cur = r[at + k]
        if cur != "N" and cur != ch:
            clash.append({"position": at + k, "already": cur, "now": ch})
        r[at + k] = ch
    if clash:
        return {"ok": False, "error": "this overlaps something already pinned to different bases",
                "clashes": clash}
    out = "".join(r)
    return {"ok": True, "restraint": out, "pinned_now": len(motif), "at": at,
            "pinned_total": sum(1 for c in out if c != "N")}


def designable_probe(target: str, timeout_s: float = 20.0) -> dict:
    """FREE, BEFORE SPENDING ANYTHING: can ViennaRNA's own inverse fold reproduce this SHAPE at all?

    The pin check below answers "do the pinned bases conflict". It says nothing about whether the shape is
    reachable, and that gap cost real money: the agent proposed a junction whose helices were 2 bp, got a
    clean feasibility pass, and then climbed cheap -> standard -> best_effort on it -- $0.38 and 45 minutes
    for one target -- because "not_found" is indistinguishable from "not yet found".

    MEASURED against what the solver actually did, on the 9 nested targets the benchmark asked it to solve:

        probe says YES and DesiRNA solved it      4        probe says NO  and DesiRNA solved it      0
        probe says NO  and DesiRNA failed         5        probe says YES and DesiRNA failed         0

    9/9, in 0.03-0.3 s against 18 s / 282 s / 2727 s of paid search. Small n and one population, so this is
    reported as EVIDENCE and not as a verdict -- inverse_fold is a heuristic local search, so a NO is not a
    proof of undesignability, and the agent stays free to spend anyway.

    TWO HARD CONSTRAINTS, both learned the expensive way:
      * it SEGFAULTS on pseudoknots ("Junk in structure at aux_structure"), so knots are skipped by name;
      * a crash here would take the whole agent down with it, so it runs in an isolated child process. `-I`
        also keeps the current directory off sys.path, which is the /tmp-shadowing failure class.
    """
    import subprocess
    import sys as _sys
    from .certify import classify
    kind, err = classify(target)
    if err:
        return {"ok": False, "error": err}
    if kind != "nested":
        return {"ok": True, "designable": None,
                "note": "skipped: ViennaRNA's inverse fold cannot represent crossing pairs (it crashes on them)"}
    if len(target) > PROBE_MAX_N:
        return {"ok": True, "designable": None, "min_helix": _min_helix(target),
                "note": (f"skipped: at {len(target)} nt this check answers less than half the time and costs "
                         f"the full {timeout_s:.0f}s when it fails (measured); it is only worth running at "
                         f"or below {PROBE_MAX_N} nt")}
    code = ("import sys, RNA\n"
            "t = sys.argv[1]\n"
            "for _ in range(3):\n"
            "    s, _d = RNA.inverse_fold(None, t)\n"
            "    if RNA.fold(s)[0] == t:\n"
            "        print('YES'); break\n"
            "else:\n"
            "    print('NO')\n")
    try:
        r = subprocess.run([_sys.executable, "-I", "-c", code, target],
                           capture_output=True, text=True, timeout=timeout_s)
        out = (r.stdout or "").strip()
    except subprocess.TimeoutExpired:
        return {"ok": True, "designable": None, "note": f"the probe did not finish in {timeout_s}s"}
    if out not in ("YES", "NO"):
        return {"ok": True, "designable": None,
                "note": f"the probe did not answer (exit {r.returncode}); treat as unknown"}
    yes = out == "YES"
    return {"ok": True, "designable": yes, "min_helix": _min_helix(target),
            "note": ("ViennaRNA's own inverse fold reproduces this shape, so a sequence for it exists"
                     if yes else
                     f"ViennaRNA's own inverse fold could NOT reproduce this shape in 3 tries (shortest "
                     f"helix {_min_helix(target)} bp). A bigger search budget is unlikely to help; "
                     f"lengthening the short helices is what changes the answer.")}


def _min_helix(target: str) -> int:
    """Length of the shortest run of stacked pairs. Short helices are the usual reason a shape is unreachable:
    on a 2-branch junction, 2 bp and 3 bp helices were both undesignable here and 4 bp was not."""
    from ..structure_utils import pair_table
    pt = pair_table(target)
    lens = []
    for i, j in enumerate(pt):
        if j > i and not (i > 0 and pt[i - 1] == j + 1):        # this pair starts a helix
            k = 0
            while i + k < len(pt) and pt[i + k] == j - k and j - k > i + k:
                k += 1
            lens.append(k)
    return min(lens) if lens else 0


def check_feasible(target: str, restraint: Optional[str]) -> dict:
    """EXACT, BEFORE SPENDING ANYTHING: are the pinned bases compatible with the target's own pairing?

    If positions i and j must pair and both are pinned to bases that cannot form a pair, no search will ever
    succeed -- and a search would burn its whole budget discovering that. Reporting the offending positions is
    what lets the agent do something useful (move the motif, move the structure) instead of buying more time.

    It also runs the SHAPE probe, in the same step: the agent already calls this before every paid design
    (18/18 on the benchmark), so folding the second free check in here costs nothing and closes the gap where
    "pins are fine" was read as "this will work".
    """
    from ..structure_utils import pair_table
    shape = designable_probe(target)
    if restraint and len(restraint) != len(target):
        # A stale restraint from a previous, differently sized shape used to index past the end and raise.
        # A primitive that crashes teaches the agent nothing; one that says what is wrong lets it recover.
        return {"ok": False, "shape": shape,
                "error": (f"the restraint is {len(restraint)} long but the target is "
                          f"{len(target)} -- it belongs to a different shape; re-pin it")}
    if not restraint or set(restraint) == {"N"}:
        # An observation that does not say what to do next wastes a step. It cost one here.
        return {"ok": True, "conflicts": [], "shape": shape,
                "note": ("nothing is pinned yet, so the pins have nothing to check -- but the SHAPE check "
                         "above did run, and it is the one that decides whether search can work at all. "
                         "Next: place_constraint if the request fixes any bases, otherwise design.")}
    pt = pair_table(target)
    conflicts = []
    for i, j in enumerate(pt):
        if j > i:
            a, b = restraint[i], restraint[j]
            if a in "ACGU" and b in "ACGU" and (a, b) not in WC:
                conflicts.append({"i": i, "j": j, "bases": f"{a}-{b}",
                                  "why": f"positions {i} and {j} must pair, but {a} and {b} cannot"})
    pinned_in_stem = sum(1 for i, j in enumerate(pt) if j >= 0 and restraint[i] != "N")
    pinned_total = sum(1 for c in restraint if c != "N")
    return {"ok": not conflicts, "conflicts": conflicts, "shape": shape,
            "pinned_total": pinned_total, "pinned_in_paired_positions": pinned_in_stem,
            "note": ("the pinned bases are compatible with this shape" if not conflicts else
                     f"{len(conflicts)} pinned pair(s) cannot form; no amount of search budget will fix this")}


def recommend_loop_cap(target: str) -> dict:
    """Ask the motif knowledge base what caps a loop of this size stably. KNOWLEDGE IN, HARD PIN OUT.

    DesiRNA's own -motifs flag applies a soft bonus, which cannot be verified afterwards -- a design either
    got the bonus or did not and there is no exact predicate for it. So this primitive only SUGGESTS: the
    agent pins the cap with place_constraint, and the pin is then checked exactly like any other. The
    knowledge base never decides anything.
    """
    from ..structure_utils import pair_table
    from ..mcp.motif_kb import recommend_hairpin
    pt = pair_table(target)
    loops = []
    for i, j in enumerate(pt):
        if j > i and all(pt[k] < 0 for k in range(i + 1, j)):
            loops.append({"start": i + 1, "len": j - i - 1, "closing_pair_at": [i, j]})
    if not loops:
        return {"ok": False, "error": "this target has no hairpin loop to cap"}
    out = []
    for lp in loops:
        r = recommend_hairpin(lp["len"]) or {}
        out.append({**lp, "motifs": r.get("motifs", []), "closing_pair": r.get("closing_pair"),
                    "note": (r.get("note") or "")[:160]})
    return {"ok": True, "loops": out,
            "how": "pin one with place_constraint(motif=<one of motifs>, at=<start>); it is then verified exactly"}


def set_gc_content(percent: float, tolerance: float = 5.0) -> dict:
    """Ask the solver for a GC target, and declare the tolerance the RESULT will be checked against.

    Two separate things, deliberately: -acgu_content STEERS the search, and the check after the fact DECIDES.
    A steer that is not checked is a wish.
    """
    try:
        pct = float(percent)
    except (TypeError, ValueError):
        return {"ok": False, "error": f"percent must be a number, got {percent!r}"}
    if not 0 <= pct <= 100:
        return {"ok": False, "error": f"GC content must be between 0 and 100, got {pct}"}
    tol = max(0.0, float(tolerance))
    gc_half, at_half = pct / 2.0, (100.0 - pct) / 2.0
    # DesiRNA rejects -acgu_content unless -acgu is on ("error: The -acgu_content option is only applicable
    # when -acgu is set to 'on'"), and argparse exits before any search -- which surfaced as an instant
    # not_found rather than as an error.
    return {"ok": True, "gc_percent": pct, "tolerance": tol,
            "acgu_content": f"{at_half:g},{gc_half:g},{gc_half:g},{at_half:g}",
            "note": f"the solver is steered toward {pct:g}% GC; the result is checked against {pct:g} +/- {tol:g}"}


def set_melting_temperature(celsius: float, tolerance: float = 5.0) -> dict:
    """Declare a melting-temperature requirement. There is no steer for this one -- only a check.

    MEASURED, and this is why the primitive exists: the external solver has no folding-temperature option at
    all; it designs at 37 C, and -tmin/-tmax are replica-exchange temperatures while -tm is a mutation
    heuristic. Asked for a Tm BELOW what it naturally produces (median 83.1 C), its own answer landed in the
    window 12 times out of 60 while reading its pool landed 54 (McNemar 42-0, p=4.6e-13). Asked for one AT
    its natural value the gap nearly closed -- so this is not "the tool is bad", it is "the request costs
    stability and the tool is not optimising for that".
    """
    try:
        c = float(celsius)
    except (TypeError, ValueError):
        return {"ok": False, "error": f"celsius must be a number, got {celsius!r}"}
    if not 0 < c < 120:
        return {"ok": False, "error": f"a melting temperature of {c} C is outside anything this can check"}
    tol = max(0.0, float(tolerance))
    return {"ok": True, "celsius": c, "tolerance": tol,
            "definition": ("the temperature at which the TARGET structure's probability in the Boltzmann "
                           "ensemble falls to 0.5 -- computed by bisection on the partition function, not "
                           "by a nearest-neighbour formula for a generic duplex"),
            "note": (f"the solver cannot be steered toward this at all; candidates are checked against "
                     f"{c:g} +/- {tol:g} C after the fact")}


def melting_point(seq: str, target: str, lo: float = 5.0, hi: float = 120.0, iters: int = 12) -> float:
    """The temperature where P(target) crosses 0.5. Exact given the model; returns the bound when it is not
    crossed inside [lo, hi]."""
    import RNA

    def p_at(T):
        md = RNA.md()
        md.temperature = float(T)
        fc = RNA.fold_compound(seq, md)
        fc.pf()
        return fc.pr_structure(target)

    if p_at(lo) < 0.5:
        return lo
    if p_at(hi) >= 0.5:
        return hi
    for _ in range(iters):
        mid = (lo + hi) / 2
        if p_at(mid) >= 0.5:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 1)


def _tm_check(seq: Optional[str], target: str, tm: Optional[dict]):
    """Exact, and reported separately from folding -- a design that folds but melts at the wrong temperature
    is not an answer to what was asked."""
    if not tm:
        return None
    if not seq:
        return {"held": None, "measured": None, "target": tm["celsius"], "tolerance": tm["tolerance"]}
    got = melting_point(seq, target)
    return {"held": abs(got - tm["celsius"]) <= tm["tolerance"], "measured": got,
            "target": tm["celsius"], "tolerance": tm["tolerance"]}


def design_set(target: str, count: int = 8, max_identity: float = 60.0,
               max_duplex_kcal: float = -12.0, budget: str = "cheap", runs: int = 4,
               restraint: Optional[str] = None) -> dict:
    """Design a SET: N sequences that all fold to the target, are pairwise dissimilar, and do not hybridise.

    WHY THIS IS A SEPARATE PRIMITIVE. Every other requirement here is a property of ONE sequence, so `design`
    can screen candidates one at a time. This one is a property of the SET, and no per-sequence filter can
    express it. It is also the requirement the external solver has no formulation for at all: its input file
    is one structure and one sequence restraint, and "these N must differ from each other" cannot be written
    down in it.

    MEASURED, asking for 8 sequences at <=60% identity and no duplex below -12 kcal/mol: running the solver 8
    times and taking each run's answer reached the full set on 0 of 5 shapes (1-2 usable of 8); reading the
    pools those same runs already produced reached it on 4 of 5. The mechanism is not diversity of search --
    it is that the solver's answers are GC-rich (73.1%) and therefore hybridise with each other, 27 of 28
    pairs violating the threshold, while the qualifying sets sit at 50-57% GC.

    Selection is greedy in the solver's OWN ranking order, so the set is drawn from what it considers best
    rather than reordered by us.
    """
    from .certify import certify_sequence
    from .cost import solver_cost_usd
    import RNA
    from ..experiments.rssd_constrained import ConstrainedDesiRNA
    from ..generators.desirna import desirna_available
    from ..task import DesignTask, OracleConfig
    if count < 2:
        return {"ok": False, "error": "a set of fewer than 2 is not a set -- use design"}
    if not desirna_available():
        return {"ok": False, "error": "the external solver is not installed"}
    from .certify import BUDGETS
    tl = BUDGETS.get(budget, BUDGETS["cheap"])[0]
    oc = OracleConfig(param_set="turner2004", temperature=37.0, dangles=2)
    t0, cost, pool, seen = time.time(), 0.0, [], set()
    for seed in range(max(1, runs)):
        a0 = time.time()
        res = ConstrainedDesiRNA(replicas=10, timelimit_s=tl, results_number=50).design(
            DesignTask(task_id="set", target=target, oracle=oc,
                       meta=({"seq_restr": restraint} if restraint else {})), seed=seed)
        cost += solver_cost_usd(time.time() - a0, 10)
        for c in ([res.best_sequence] if res.best_sequence else []) + list(res.candidates or []):
            if c and len(c) == len(target) and c not in seen:
                seen.add(c)
                pool.append(c)

    def ident(a, b):
        return 100.0 * sum(1 for x, y in zip(a, b) if x == y) / len(a)

    chosen, certs, rejected = [], [], {"folds": 0, "identity": 0, "duplex": 0}
    for c in pool:
        if len(chosen) >= count:
            break
        cert = certify_sequence(c, target)
        if cert.verdict not in ("certified", "knot_confirmed"):
            rejected["folds"] += 1
            continue
        if any(ident(c, t) > max_identity for t in chosen):
            rejected["identity"] += 1
            continue
        if any(RNA.duplexfold(c, t).energy < max_duplex_kcal for t in chosen):
            rejected["duplex"] += 1
            continue
        chosen.append(c)
        certs.append(cert.to_dict())

    worst_id = max((ident(chosen[i], chosen[j]) for i in range(len(chosen))
                    for j in range(i + 1, len(chosen))), default=None)
    worst_dup = min((RNA.duplexfold(chosen[i], chosen[j]).energy for i in range(len(chosen))
                     for j in range(i + 1, len(chosen))), default=None)
    ok = len(chosen) >= count
    return {"ok": True, "verdict": "certified" if ok else "not_found",
            "sequences": chosen, "n_delivered": len(chosen), "n_requested": count,
            "cost_usd": cost, "wall_s": round(time.time() - t0, 1), "n_screened": len(pool),
            "set_check": {"held": ok, "worst_pairwise_identity": round(worst_id, 1) if worst_id else None,
                          "strongest_duplex_kcal": round(worst_dup, 1) if worst_dup else None,
                          "max_identity": max_identity, "max_duplex_kcal": max_duplex_kcal},
            "rejected": rejected, "certificates": certs,
            "note": (f"delivered {len(chosen)} of {count}; every member is certified individually and the "
                     f"pairwise checks are exact"
                     + ("" if ok else f". The pool held {len(pool)} candidates and ran out -- ask for a "
                                      f"bigger budget or relax max_identity / max_duplex_kcal"))}


def _gc_of(seq: Optional[str]):
    if not seq:
        return None
    return 100.0 * sum(1 for c in seq if c in "GC") / len(seq)


# A tier of t seconds bills for MORE than t: process start, input writing and the final verification are all
# inside the wall clock the cost model charges for. Measured on the 64-target end-to-end run, the 5 s tier
# came to $0.00121 per target against a predicted $0.000694 -- a factor of 1.74. Rounded up, and used only to
# decide what we can AFFORD, so erring high is the safe direction.
TIER_OVERHEAD = 2.0


def affordable_tiers(budget: str, max_spend_usd: Optional[float]):
    """The prefix of a budget's ladder that fits in what is left, and never fewer than one tier.

    WHY THIS EXISTS. The agent's cost cap was checked BETWEEN steps, which cannot stop a single call: one
    best_effort design ends with a 1800 s x 10-replica tier, so a run sitting at $0.49 under a $0.50 cap could
    still commit half an hour. It did. Two runs on the 64-target end-to-end went to $0.5111 / 5033 s and
    $0.4354 / 8948 s and both returned not_found -- together 93% of that arm's entire solver spend, on two
    targets it never solved. Enforcement has to happen where the ladder is chosen.

    One tier always survives. Refusing to run at all would turn a cost cap into a silent no-answer, which is
    the failure class this project keeps finding; the honest behaviour is a cheap attempt plus a note.
    """
    from .certify import BUDGETS
    from .cost import solver_cost_usd
    tiers = list(BUDGETS.get(budget) or BUDGETS["cheap"])
    if max_spend_usd is None:
        return tiers, None
    keep, spent = [], 0.0
    for tl in tiers:
        cost = solver_cost_usd(tl, 10) * TIER_OVERHEAD
        if keep and spent + cost > max_spend_usd:
            break
        keep.append(tl)
        spent += cost
    if len(keep) == len(tiers):
        return keep, None
    return keep, (f"budget ladder truncated from {tiers} to {keep} seconds: ${max_spend_usd:.4f} left of the "
                  f"cost cap, and the next tier alone would cost about "
                  f"${solver_cost_usd(tiers[len(keep)], 10) * TIER_OVERHEAD:.4f}")


def design(target: str, restraint: Optional[str] = None, budget: str = "cheap", seed: int = 0,
           gc: Optional[dict] = None, tm: Optional[dict] = None,
           kin: Optional[dict] = None, sw: Optional[dict] = None,
           max_spend_usd: Optional[float] = None) -> dict:
    """Run the solver and report what came back, including HOW it failed."""
    from .certify import BUDGETS, certify_design
    from .cost import solver_cost_usd
    if budget not in BUDGETS:
        budget = "cheap"
    tiers, capped = affordable_tiers(budget, max_spend_usd)
    t0 = time.time()
    # STEER OR SCREEN. DesiRNA's -acgu_content steers the search toward a composition; measured on a 6-bp
    # hairpin it produced 81.2% GC against a 50% request -- WORSE than the 62.5% it reaches with no request
    # at all. Screening its candidate POOL instead found 66 of 200 candidates that both fold correctly and
    # sit within 50 +/- 10. So the requirement is enforced by reading the pool, which is exact, and not by
    # asking the solver nicely, which is not. Same mechanism that recovered 14 pseudoknot targets the tool
    # had thrown away.
    if (restraint and set(restraint) != {"N"}) or gc or tm or kin or sw:
        from ..experiments.rssd_constrained import ConstrainedDesiRNA
        from ..generators.desirna import desirna_available
        from ..task import DesignTask, OracleConfig
        from .certify import certify_sequence
        if not desirna_available():
            return {"ok": False, "error": "the external solver is not installed"}
        oc = OracleConfig(param_set="turner2004", temperature=37.0, dangles=2)
        cost, err, screened, best_partial = 0.0, None, 0, None
        for tl in tiers:
            a0 = time.time()
            res = ConstrainedDesiRNA(replicas=10, timelimit_s=tl, results_number=50).design(
                DesignTask(task_id="agent", target=target, oracle=oc,
                           meta=({"seq_restr": restraint} if restraint else {})), seed=seed)
            cost += solver_cost_usd(time.time() - a0, 10)
            err = (res.meta or {}).get("error") or err
            pool, seen = [], set()
            for c in ([res.best_sequence] if res.best_sequence else []) + list(res.candidates or []):
                if c and len(c) == len(target) and c not in seen:
                    seen.add(c)
                    pool.append(c)
            for c in pool:
                screened += 1
                held = _constraint_held(c, restraint) if restraint else None
                if held is not None and not held["held"]:
                    continue
                g = _gc_check(c, gc)
                if g is not None and not g["held"]:
                    continue
                # THE FOLD CHECK COMES BEFORE THE MELTING CHECK ON PURPOSE. A melting point costs about a
                # dozen partition-function evaluations; a fold costs one. Ordering it after the fold means
                # it is only ever paid for candidates that are already answers to the shape question.
                cert = certify_sequence(c, target)
                if cert.verdict not in ("certified", "knot_confirmed"):
                    best_partial = best_partial or c        # the closest miss still gets reported
                    continue
                # CHEAPEST DISCRIMINATOR FIRST, and every one of these only runs on candidates that already
                # answer the shape question. A switch test is one cofold; a melting point is about a dozen
                # partition functions; a co-transcriptional scan is one fold PER PREFIX, so it goes last.
                w = _switch_check(c, target, sw)
                if w is not None and not w.get("held"):
                    best_partial = best_partial or c
                    continue
                m = _tm_check(c, target, tm)
                if m is not None and not m["held"]:
                    best_partial = best_partial or c        # it folds; it melts at the wrong temperature
                    continue
                k = _kin_check(c, target, kin)
                if k is not None and not k["held"]:
                    best_partial = best_partial or c        # it folds; it traps on the way there
                    continue
                err = None
                return {"ok": True, "sequence": c, "cost_usd": cost,
                        "wall_s": round(time.time() - t0, 1), "verdict": cert.verdict,
                        "constraint_held": held, "gc": g, "tm": m, "kinetics": k, "switch": w,
                        "n_screened": screened, "budget_capped": capped,
                        "bp_away": 0, "certificate": cert.to_dict()}
        if err:
            # A TOOL FAILURE IS NOT A SEARCH FAILURE. Reporting it as not_found sent the agent up the budget
            # ladder three times, paying each time, for a request that was failing for an unrelated reason.
            return {"ok": False, "error": f"the solver did not run: {err}", "cost_usd": cost,
                    "wall_s": round(time.time() - t0, 1),
                    "hint": "this is a tool failure, not a search failure -- a bigger budget will not help"}
        cert = certify_sequence(best_partial, target) if best_partial else None
        return {"ok": True, "sequence": best_partial, "cost_usd": cost,
                "wall_s": round(time.time() - t0, 1), "verdict": "not_found", "n_screened": screened,
                "constraint_held": (_constraint_held(best_partial, restraint)
                                    if (best_partial and restraint) else None),
                "gc": _gc_check(best_partial, gc),
                "tm": (_tm_check(best_partial, target, tm) if best_partial else None),
                "kinetics": (_kin_check(best_partial, target, kin) if best_partial else None),
                "switch": (_switch_check(best_partial, target, sw) if best_partial else None),
                "bp_away": (cert.detail or {}).get("best_bp") if cert else None,
                "budget_capped": capped,
                "note": (f"screened {screened} candidates; none satisfied every requirement at once"
                         + (f". {capped}" if capped else ""))}
    c = certify_design(target, budget=budget, seed=seed, tiers=tuple(tiers))
    return {"ok": True, "sequence": c.sequence, "cost_usd": c.cost_usd, "wall_s": c.wait_s,
            "verdict": c.verdict, "constraint_held": None, "gc": _gc_check(c.sequence, gc), "tm": None,
            "bp_away": (c.detail or {}).get("best_bp"), "budget_capped": capped,
            "certificate": c.to_dict()}


def _gc_check(seq: Optional[str], gc: Optional[dict]):
    """Exact, and reported separately from folding: a design that misses the GC target is not an answer."""
    if not gc:
        return None
    got = _gc_of(seq)
    if got is None:
        return {"held": None, "measured": None, "target": gc["gc_percent"],
                "tolerance": gc["tolerance"]}
    return {"held": abs(got - gc["gc_percent"]) <= gc["tolerance"], "measured": round(got, 1),
            "target": gc["gc_percent"], "tolerance": gc["tolerance"]}


def _constraint_held(seq: Optional[str], restraint: str):
    """Exact check, reported separately from folding: a design that breaks the pin is not an answer."""
    if not seq or len(seq) != len(restraint):
        return None
    bad = [i for i, (s, r) in enumerate(zip(seq, restraint)) if r != "N" and s != r]
    return {"held": not bad, "violated_positions": bad}



def set_cotranscriptional(max_trap_fraction: float = 0.10) -> dict:
    """Require the design to fold cleanly AS IT IS SYNTHESISED, 5' to 3', not only once it is complete.

    WHY THIS IS A PRIMITIVE AND NOT A NOTE IN THE DOCS. It is a requirement the external solver has no term
    for -- its whole objective is equilibrium (MFE, ensemble defect, suboptimal gap, oligomerisation,
    composition), and its source contains no cotranscriptional/kinetic/barrier/findpath anywhere. Measured on
    24 natural structures off the SAME solver runs, both arms required to fold correctly AND pass:

        its own answer 8/24        screening its own pool 19/24        rescue 11 of 16 failures

    and it is not the composition gap in disguise: restricted to the targets whose own answer already sits in
    45-58% GC, 3/11 against 9/11 at 52.4% GC on both sides.

    A prefix is a TRAP when its MFE contains a pair the target does not have -- that pair has to be broken
    later. `max_trap_fraction` is the share of prefixes allowed to be traps; 0.10 is the benchmark's declared
    value. This is ViennaRNA MFE per prefix, deterministic and exact under the stated model. It is NOT a
    kinetic simulation and says nothing about polymerase speed or pausing, so it is a lower bound on trouble.
    """
    try:
        f = float(max_trap_fraction)
    except (TypeError, ValueError):
        return {"ok": False, "error": "max_trap_fraction must be a number between 0 and 1"}
    if not 0.0 <= f <= 1.0:
        return {"ok": False, "error": f"max_trap_fraction {f} is outside 0..1"}
    return {"ok": True, "max_trap_fraction": f,
            "means": (f"at most {f:.0%} of 5'->3' prefixes may misfold into a pair the target does not have"),
            "note": "checked by screening, never steered -- the solver has no term for this"}


def set_switch(enabled: bool = True) -> dict:
    """Require a TWO-STATE design: folds to the target alone, and opens when its trigger strand is present.

    The window is found BY CODE from the target -- an unpaired toehold immediately 5' of a stem, plus the
    stem pairs the trigger must displace. The model never picks positions, the same rule as everywhere else
    here. A target without that motif gets an honest refusal rather than a silently different requirement.

    MEASURED, and this is the strongest composition control the project has run. 72 natural structures, same
    solver runs, both arms needing both states:

        its own answer 35/72        screening its own pool 58/72

    and with our arm restricted PER TARGET to candidates whose GC EQUALS its own answer's, exactly:

        35/58 against 54/58, ours-only 19 / its-only 0, p=3.8e-06

    The mechanism is measured too, not assumed. Splitting its own answers by whether they switch: monomer MFE
    median -25.2 kcal/mol for the ones that fail against -20.2 for the ones that work, at the same lengths
    (60 vs 61) and the same complex energy (-25.9 vs -25.7). The solver optimises toward a deeper well, and a
    deeper well is one the trigger cannot displace. The requirement is directly opposed to the objective.
    """
    return {"ok": True, "switch": bool(enabled),
            "means": ("the sequence must fold to the target on its own AND be fully opened at the toehold "
                      "stem by its own complementary trigger strand"),
            "note": "the toehold window is located by code from the target; the solver cannot express this"}


def _kin_check(seq: Optional[str], target: str, kin: Optional[dict]):
    """Exact, reported separately: a design that folds but traps during synthesis is not an answer."""
    if not kin or not seq:
        return None
    from ..benchmarks.kinetic_suite import cotranscriptional
    k = cotranscriptional(seq, target)
    return {"held": k["trap_fraction"] <= kin["max_trap_fraction"],
            "trap_fraction": k["trap_fraction"], "limit": kin["max_trap_fraction"],
            "first_trap_at": k["first_trap_k"]}


def _switch_check(seq: Optional[str], target: str, sw: Optional[dict]):
    """Exact, reported separately: both states or it is not a switch."""
    if not sw or not sw.get("switch") or not seq:
        return None
    from ..benchmarks.switch_suite import INVADE, switch_window, switches
    at = switch_window(target)
    if at is None:
        return {"held": None, "error": "this target has no toehold + stem motif, so it cannot be a switch"}
    r = switches(seq, target, at)
    return {"held": r["held"], "invasion": r["invasion"], "needed": INVADE,
            "window_at": at, "trigger": r["trigger"], "complex_dg": r["complex_dg"]}


PRIMITIVES = {"propose_structure": propose_structure, "place_constraint": place_constraint,
              "use_target": use_target,
              "check_feasible": check_feasible, "design": design, "design_set": design_set,
              "recommend_loop_cap": recommend_loop_cap, "set_gc_content": set_gc_content,
              "set_melting_temperature": set_melting_temperature,
              "set_cotranscriptional": set_cotranscriptional, "set_switch": set_switch}


# --------------------------------------------------------------------------- the loop
SYSTEM = """You are solving an RNA secondary-structure design request by CALLING TOOLS and READING WHAT THEY
RETURN. You do not design RNA yourself and you do not decide whether an answer is correct -- exact tools do
both. Your job is to choose the next action given what actually happened.

Reply with ONE json object and nothing else:
{"thought":"<one short sentence of reasoning>","action":"<name>","args":{...}}

Actions:
  propose_structure {"elements":[...]}      build the target shape. Elements are
        {"type":"hairpin","stem":<bp>,"loop":<nt>}   {"type":"spacer","len":<nt>}
        {"type":"internal_loop","outer_stem":<bp>,"left":<nt>,"right":<nt>,"inner_stem":<bp>,"loop":<nt>}
        {"type":"multiloop","closing_stem":<bp>,"spacer":<nt>,"branches":[{"stem":<bp>,"loop":<nt>}, ...]}
              a multiway junction -- one closing stem enclosing two or more hairpins. Still NESTED, so it is
              judged by the exact oracle.
        {"type":"pseudoknot","stem1":<bp>,"stem2":<bp>,"loop1":<nt>,"loop2":<nt>,"loop3":<nt>}
              an H-type pseudoknot. Judged by a weaker, calibrated criterion -- only choose it when the
              request actually asks for crossing stems.
        You never write dot-bracket. You give element types and integer lengths; the code assembles and counts.
  use_target        {"which":0}              adopt a dot-bracket structure that the REQUEST already
        contains, instead of building one. Use this whenever the user supplied a structure -- it is theirs,
        and rebuilding it from a description would change it. You pass an INDEX, never a structure string.
        Once adopted, that structure is FIXED: designing for a different one is refused, because a
        certificate for a structure the user did not ask for answers nobody's question. If the requirements
        cannot all hold on it, say so.
  place_constraint  {"motif":"GGGG","at":<0-based position>}    pin a fixed sub-sequence. ADDITIVE:
        call it again to pin a second region; the first stays. Overlapping placements that disagree are
        rejected rather than silently overwritten.
  check_feasible    {}                      FREE. Run it before every `design`, always -- it answers TWO
        separate questions and either one can make search hopeless:
          conflicts  : pinned bases sitting on positions that must pair with each other and cannot. EXACT.
          shape      : whether ViennaRNA's own inverse fold can reproduce this shape at all. On the targets
                       measured so far it agreed with the solver 9 times out of 9, in under a second, where
                       the solver needed up to 45 minutes to reach the same answer. If designable is false,
                       the shape is the problem -- short helices are the usual cause -- and a bigger budget
                       is the wrong move. It is evidence, not a verdict: you may still spend if you disagree.
  recommend_loop_cap {}                     ask the motif knowledge base which caps stabilise a loop of this
        size. It only SUGGESTS; pin one with place_constraint so the result can be checked exactly.
  set_gc_content    {"percent":50,"tolerance":5}   steer the solver toward a GC fraction. The steer is not
        the check: the result is verified against percent +/- tolerance afterwards.
  set_melting_temperature {"celsius":55,"tolerance":5}   require a melting temperature -- the point where
        the TARGET structure stops being more likely than not. There is NO steer for this; candidates are
        checked after the fact. Requests BELOW what the solver naturally produces (around 83 C) are the ones
        that need this; a request at or above it usually comes for free.
  set_cotranscriptional {"max_trap_fraction":0.10}   require the design to fold cleanly AS IT IS MADE,
        5' to 3', not only once finished. Use it when the request mentions transcription, folding while being
        synthesised, kinetic traps, or "must fold correctly in the cell/in vivo" rather than just at
        equilibrium. Checked, never steered -- the solver has no term for it. Costs one fold per prefix, so
        ask for it when the request calls for it, not by default.
  set_switch        {"enabled":true}   require a TWO-STATE design: folds to the target alone, and opens at
        its toehold stem when the complementary trigger strand is present. Use it for a riboswitch, a
        toehold switch, a sensor, or anything described as changing shape when something binds. The window is
        located by code from the target; if the target has no unpaired toehold beside a stem this is refused
        outright rather than searched for.
  design_set        {"count":8,"max_identity":60,"max_duplex_kcal":-12}
        design a SET of sequences that all fold to the target, are pairwise at most max_identity per cent
        identical, and form no duplex stronger than max_duplex_kcal. Use this only when the request asks for
        several sequences that must not interfere with each other -- a library, barcodes, orthogonal parts.
        It returns a certificate per member plus the exact pairwise checks.
  design            {"budget":"cheap|standard|best_effort"}     run the solver and verify the result
  done              {"reason":"..."}        stop: either the goal is met, or you have concluded it cannot be.
        REFUSED until something has actually been observed -- you must have run a design or found a conflict.
        Saying "it worked" is not a result; only the verifier's certificate is.

Rules:
  * read the observation before choosing. A conflict at positions i and j is not fixed by a bigger budget --
    move the motif, or change the shape so those positions are not paired with each other.
  * a hairpin loop needs >= 3 unpaired nucleotides; a stem needs >= 2 base pairs.
  * prefer "cheap" first. Escalate the budget only when the failure looks like a search failure (a small
    bp_away with no constraint conflict), never when it looks structural. "best_effort" can run for 30
    minutes; it is worth that only when you already have evidence the shape is designable.
  * a failed `design` reports mfe_structure, best_bp and wrong_positions. Read them: they say WHERE the fold
    went wrong, which tells you which part of the shape to change.
  * you may pay for the same target/restraint/GC at most TWICE. A third attempt on an unchanged problem is
    refused. Change the problem instead -- lengthen short helices, move a pin into a loop, relax a
    requirement -- and re-run the free check.
  * you cannot mark anything solved. `design` returns a verdict from an exact verifier; that is the answer.
  * ONLY AN EXPLICIT TOTAL-LENGTH LOCK closes the door. If the request says the length must be exactly N and
    not one base more, then a certified sequence for a different shape does not answer it, and `done` naming
    the conflict is correct. Otherwise you have room and you are EXPECTED to use it: describing the stems and
    loops is not the same as forbidding a longer version of them. Lengthen a loop, add a spacer, move the
    pinned region out of a paired position -- then re-check, which is free. A pin conflict is a reason to
    reshape, not a reason to stop. Any reshape is reported to the user alongside the answer.
"""


def _reapply(pins: List[dict], target: str):
    """Re-place the requested pins on a new shape. Returns (restraint, kept, dropped).

    Position-based and therefore honest about what a reshape can break: a pin past the end of the new shape
    cannot be carried, and is REPORTED rather than silently forgotten. Whether the carried pins conflict with
    the new pairing is a separate question, and check_feasible is the thing that answers it.
    """
    kept, dropped, r = [], [], ["N"] * len(target)
    for p in pins:
        at, motif = int(p["at"]), str(p["motif"])
        if at < 0 or at + len(motif) > len(target):
            dropped.append({"motif": motif, "at": at, "why": f"does not fit in {len(target)} nt"})
            continue
        for k, ch in enumerate(motif):
            r[at + k] = ch
        kept.append({"motif": motif, "at": at})
    out = "".join(r)
    return (out if set(out) != {"N"} else None), kept, dropped


def _prime(sess: "Session") -> None:
    """Do the part that is not a decision, before the model is asked anything.

    MEASURED: on eight "here is my structure, give me a sequence" requests the agent produced eight
    IDENTICAL trajectories -- use_target -> check_feasible -> design -- at three model calls each, and the
    whole loop cost 6x the money and 8x the wall clock of calling the ladder directly for exactly the same
    8/8 answers. Two of those three calls were spent on steps with nothing to decide:

      use_target      the request contains exactly ONE structure, so there is no other thing to adopt
      check_feasible  free, side-effect-free, and the rules already say to run it before spending

    So the code does them. This is the same division of labour as everywhere else here -- the model decides
    what needs deciding -- and it is deliberately conservative: with zero structures or more than one,
    choosing is a real decision and the model makes it.
    """
    cands = structures_in(sess.request)
    if len(cands) != 1:
        return
    obs = use_target(sess.request, 0)
    if not obs.get("ok"):
        return
    obs["automatic"] = ("adopted by the harness: the request contains exactly one structure, so there was "
                        "nothing to choose")
    sess.target = sess.requested_target = obs["target"]
    sess.requested_by_user = True
    sess.steps.append(Step(len(sess.steps) + 1, "use_target", {"which": 0}, obs))
    chk = check_feasible(sess.target, sess.restraint)
    chk["automatic"] = "run by the harness: it is free and it is required before spending"
    sess.steps.append(Step(len(sess.steps) + 1, "check_feasible", {}, chk))


def _observations(sess: "Session") -> str:
    out = [f"REQUEST: {sess.request}"]
    if sess.target:
        out.append(f"current target: {sess.target}  ({len(sess.target)} nt)")
    if sess.restraint:
        out.append(f"current restraint: {sess.restraint}")
    for s in sess.steps:
        out.append(f"[step {s.n}] {s.action}({json.dumps(s.args, ensure_ascii=False)}) -> "
                   f"{json.dumps(s.observation, ensure_ascii=False)[:1400]}")
    return "\n".join(out)


def run(request: str, client=None, model: str = "z-ai/glm-5.2", max_steps: int = MAX_STEPS,
        max_cost_usd: float = 0.5, workspace: Optional[str] = None) -> Session:
    """Observe -> decide -> act -> verify, until certified, refused, or out of steps/budget.

    Every step is appended to the session and, if `workspace` is given, written to disk as it happens, so a
    caller can see what was tried rather than only what came out.
    """
    from .intent import functional_request
    sess = Session(request=request)
    hit = functional_request(request)
    if hit and not structures_in(request):
        sess.done, sess.outcome = True, "refused"
        sess.steps.append(Step(0, "refuse", {"word": hit}, {
            "reason": ("the request describes a biological FUNCTION, not a SHAPE. Nothing here can verify "
                       "that a sequence DOES anything, and inventing a shape would produce a certificate "
                       "that is true about the wrong target.")}))
        return _persist(sess, workspace)
    if hit:
        # THE GUARD'S PREMISE IS THAT WE WOULD HAVE TO INVENT A SHAPE. When the request already carries an
        # explicit dot-bracket, there is nothing to invent and the premise does not hold.
        #
        # MEASURED, and this is why the exception exists rather than being a matter of taste: on 96 agent
        # runs the guard refused two request phrasings outright, 12 times out of 12 each, BEFORE the model
        # was called even once -- "like a riboswitch" tripped on the word switch and "expressed in vivo"
        # tripped on expressed. Both requests supplied the structure. The switch requirement is the
        # most-named of nine across 1676 RNA-design papers and no external tool offers it, so this keyword
        # list was the entire reason our one piece of unique supply could not be reached.
        #
        # PROCEEDING IS NOT THE SAME AS COVERING IT. The functional claim stays unverified, and saying so is
        # the whole point: refusing loses the user an answer they can have, and staying silent would let a
        # certificate about the SHAPE be read as a claim about the FUNCTION -- which is the confusion the
        # guard was built to prevent in the first place.
        sess.unverified_claim = {
            "word": hit,
            "means": ("this request also describes a biological FUNCTION. The structure you gave is "
                      "designed and certified; the function is NOT checked by anything here."),
        }
        sess.steps.append(Step(0, "note_unverified", {"word": hit}, {
            "ok": True, "automatic": "recorded by the harness",
            **sess.unverified_claim}))
    if client is None:
        from ..llm.openrouter import OpenRouterClient
        client = OpenRouterClient(model=model)

    def _ask(extra: str = "", max_tokens: int = THINK_TOKENS):
        """Every model call goes through here, so its billed cost is counted exactly once -- including the
        retries, which is precisely where an uncounted call would hide."""
        msg = _observations(sess) + (("\n\n" + extra) if extra else "")
        r = client.chat([{"role": "user", "content": msg}], system=SYSTEM,
                        temperature=0.0, max_tokens=max_tokens)
        c = float(getattr(r, "cost_usd", 0.0) or 0.0)
        sess.llm_cost_usd += c
        sess.total_cost_usd += c
        return r

    def _truncated(r) -> bool:
        """Did the reply run out of output budget before it could emit the action?

        MEASURED, and the reason this check exists: `max_tokens` is the cap on reasoning PLUS content, so a
        model that thinks hard about a junction can spend the whole cap thinking and return content="" with
        finish_reason="length". At a 1200-token cap that was every junction item and half the pseudoknot
        items -- 9 of 18 -- and the loop reported it as "the agent did not return an action", which reads as
        a model that cannot do the task rather than a cap that was too small. Same silent-failure class as
        the /tmp shadowing and the argparse exit: a tool failure wearing the costume of a negative result.
        """
        try:
            return (r.raw["choices"][0].get("finish_reason") == "length"
                    and not (getattr(r, "text", "") or "").strip())
        except Exception:                                                     # noqa: BLE001
            return False

    def _parse(text: str):
        """Prefer a fenced block, else the LAST balanced object. A long `thought` followed by the action is
        the common shape, and a greedy match from the first brace swallows braces inside the reasoning."""
        import re as _re
        blocks = _re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text or "", _re.DOTALL)
        cands = list(blocks)
        depth, start = 0, None
        for k, ch in enumerate(text or ""):
            if ch == "{":
                if depth == 0:
                    start = k
                depth += 1
            elif ch == "}" and depth:
                depth -= 1
                if depth == 0 and start is not None:
                    cands.append(text[start:k + 1])
        for c in reversed(cands):
            try:
                d = json.loads(c)
                if isinstance(d, dict) and d.get("action"):
                    return d
            except json.JSONDecodeError:
                continue
        return None

    paid: Dict[tuple, int] = {}          # how many times each exact problem has been PAID for
    _prime(sess)                         # the deterministic opening, if there is one; costs no model call
    _persist(sess, workspace)
    n0 = len(sess.steps)                 # primed steps are free, so they do not eat the model's step budget
    for n in range(n0 + 1, n0 + max_steps + 1):
        r = _ask()
        if getattr(r, "error", None):
            sess.outcome = f"agent unavailable: {r.error}"
            break
        act = _parse(getattr(r, "text", "") or "")
        if act is None:                       # one retry, told plainly what went wrong
            if _truncated(r):                 # it never got to the answer -- give it room, do not scold it
                r = _ask('Your last reply ran out of room while thinking. Decide now and reply with ONLY '
                         'the json object; keep "thought" under 20 words.', max_tokens=THINK_TOKENS_MAX)
            else:
                r = _ask('Your last reply contained no usable action. Reply with ONLY the json object, '
                         'nothing before or after it, and keep "thought" under 25 words.')
            act = _parse(getattr(r, "text", "") or "")
        if act is None:
            sess.outcome = ("the agent's reply was cut off before it produced an action, twice, even at "
                            f"{THINK_TOKENS_MAX} tokens -- this is a budget failure, not a refusal"
                            if _truncated(r) else "the agent did not return an action")
            break
        name, args = str(act.get("action", "")), dict(act.get("args") or {})
        thought = str(act.get("thought", ""))[:200]

        if name == "done":
            # A CONCLUSION NEEDS AN OBSERVATION. Measured: on one item the agent's FIRST action was `done`
            # with the reason "Design succeeded: MFE structure matches the target" -- it had proposed
            # nothing, run nothing and verified nothing. No false certificate was issued, because only the
            # oracle can issue one, but the session still ended carrying a sentence that reads like success.
            # For someone who cannot check, that sentence IS the answer, so it does not get to exist.
            seen = (sess.certificate is not None
                    or any(st.action == "design" for st in sess.steps)
                    or any(st.action == "check_feasible" and st.observation.get("conflicts")
                           for st in sess.steps))
            if not seen:
                sess.steps.append(Step(n, "done", args, {"ok": False, "error": (
                    "nothing has been observed yet: no design was run and no conflict was found, so there is "
                    "nothing to conclude from. Propose a structure, check it, and design -- or find the "
                    "conflict that makes it impossible.")}, thought=thought))
                continue
            sess.done = True
            sess.outcome = (sess.outcome if sess.outcome == "certified"
                            else "the agent stopped: " + str(args.get("reason", "no reason given")))
            sess.steps.append(Step(n, "done", args, {"reason": sess.outcome}, thought=thought))
            break
        fn = PRIMITIVES.get(name)
        if fn is None:
            sess.steps.append(Step(n, name, args, {"ok": False, "error": f"unknown action {name!r}"},
                                   thought=thought))
            continue

        t0 = time.time()
        if name == "use_target":
            obs = fn(sess.request, args.get("which", 0))
            if obs.get("ok"):
                if sess.requested_target is None:
                    sess.requested_target = obs["target"]      # the user's own structure IS the request
                    sess.requested_by_user = True
                sess.target = obs["target"]
                sess.restraint, sess.pins, dropped = _reapply(sess.pins, sess.target)
                if dropped:
                    obs["pins_dropped"] = dropped
        elif name == "propose_structure":
            obs = fn(args.get("elements") or [])
            if obs.get("ok"):
                if sess.requested_target is None:
                    sess.requested_target = obs["target"]      # the shape the REQUEST asked for
                # CARRY THE PINS ACROSS A RESHAPE. They come from the request, not from the agent, so a new
                # shape does not retract them -- and clearing them made the agent spend two steps re-issuing
                # what it had never chosen to drop. Three benchmark items reached a conflict-free designable
                # shape and then ran out of steps on that bookkeeping, without ever paying for a design.
                sess.target = obs["target"]
                sess.restraint, kept, dropped = _reapply(sess.pins, sess.target)
                sess.pins = kept
                if sess.pins or dropped:
                    obs["pins_carried_over"] = [{"motif": p["motif"], "at": p["at"]} for p in sess.pins]
                    if dropped:
                        obs["pins_dropped"] = dropped
                        obs["note"] = (f"{len(dropped)} pinned region(s) no longer fit in a "
                                       f"{len(sess.target)}-nt shape and were dropped; re-place them if the "
                                       f"request still needs them")
        elif name == "place_constraint":
            obs = fn(sess.target or "", args.get("motif", ""), int(args.get("at", 0)),
                     existing=sess.restraint)
            if obs.get("ok"):
                sess.restraint = obs["restraint"]
                sess.pins = [p for p in sess.pins if p["at"] != int(args.get("at", 0))]
                sess.pins.append({"motif": str(args.get("motif", "")).upper().replace("T", "U"),
                                  "at": int(args.get("at", 0))})
        elif name == "set_gc_content":
            obs = fn(args.get("percent"), float(args.get("tolerance", 5.0)))
            if obs.get("ok"):
                sess.gc = obs
        elif name == "set_melting_temperature":
            obs = fn(args.get("celsius"), float(args.get("tolerance", 5.0)))
            if obs.get("ok"):
                sess.tm = obs
        elif name == "set_cotranscriptional":
            obs = fn(float(args.get("max_trap_fraction", 0.10)))
            if obs.get("ok"):
                sess.kin = obs
        elif name == "set_switch":
            obs = fn(bool(args.get("enabled", True)))
            if obs.get("ok"):
                sess.switch = obs
                # AN IMPOSSIBLE REQUIREMENT SHOULD BE REFUSED NOW, not paid for and then reported as
                # not_found -- that confusion is what the whole feasibility step exists to prevent.
                if sess.target:
                    from ..benchmarks.switch_suite import switch_window
                    if switch_window(sess.target) is None:
                        sess.switch = None
                        obs = {"ok": False, "error": ("this target has no unpaired toehold next to a stem, "
                                                      "so no trigger can open it"),
                               "hint": "a switch needs a single-stranded run immediately 5' of a helix"}
        elif name == "design_set":
            if not sess.target:
                obs = {"ok": False, "error": "no target yet -- propose_structure or use_target first"}
            elif sess.total_cost_usd >= max_cost_usd:
                obs = {"ok": False, "error": f"cost cap ${max_cost_usd} reached"}
            else:
                obs = fn(sess.target, int(args.get("count", 8)),
                         float(args.get("max_identity", 60.0)),
                         float(args.get("max_duplex_kcal", -12.0)),
                         str(args.get("budget", "cheap")), int(args.get("runs", 4)), sess.restraint)
                sess.total_cost_usd += float(obs.get("cost_usd") or 0.0)
                if obs.get("verdict") == "certified":
                    sess.sequences = obs["sequences"]
                    sess.certificate = (obs.get("certificates") or [None])[0]
                    sess.done, sess.outcome = True, "certified"
        elif name == "recommend_loop_cap":
            obs = fn(sess.target) if sess.target else \
                {"ok": False, "error": "no target yet -- propose_structure first"}
        elif name == "check_feasible":
            obs = fn(sess.target or "", sess.restraint) if sess.target else \
                {"ok": False, "error": "no target yet -- propose_structure first"}
        else:  # design
            problem = (sess.target, sess.restraint, json.dumps(sess.gc, sort_keys=True),
                       json.dumps(sess.tm, sort_keys=True), json.dumps(sess.kin, sort_keys=True),
                       json.dumps(sess.switch, sort_keys=True))
            if not sess.target:
                obs = {"ok": False, "error": "no target yet -- propose_structure first"}
            elif sess.total_cost_usd >= max_cost_usd:
                obs = {"ok": False, "error": f"cost cap ${max_cost_usd} reached"}
            elif sess.requested_by_user and sess.target != sess.requested_target:
                # THE USER'S STRUCTURE IS DATA, NOT A PROPOSAL. Measured on natural bpRNA shapes: given a
                # structure plus two pins that must pair and cannot, plus an explicit length lock, the agent
                # SHIFTED the whole structure a few positions so the pins fell off the stem, and returned a
                # genuinely certified sequence -- for a structure the user never wrote. Every oracle was
                # satisfied. 2 of 8 items, and it had already found and named the conflict in all 8.
                # Reshaping is legitimate when the agent built the shape itself from a description; it is
                # never legitimate when the shape came verbatim from the person asking.
                obs = {"ok": False,
                       "error": ("this structure came from the request itself, so it cannot be replaced. "
                                 "You are about to design for a different one."),
                       "requested": sess.requested_target, "about_to_design_for": sess.target,
                       "hint": ("use_target to go back to what was given. If the requirements cannot all "
                                "hold on it, that is the answer -- say so with `done` and name the "
                                "conflicting positions.")}
            elif paid.get(problem, 0) >= 2:
                # BUYING MORE TIME IS NOT A NEW IDEA. Measured: on one undesignable junction the agent went
                # cheap -> standard -> best_effort without changing anything, and the third call alone cost
                # $0.38 and 45 minutes to return the same not_found. Two paid attempts on an unchanged
                # problem is enough evidence; the third has to be a different problem.
                obs = {"ok": False, "error": ("you have already paid twice for this exact target, restraint "
                                              "and GC requirement, and it did not work"),
                       "hint": ("a larger budget searches the same space harder. Change the PROBLEM: "
                                "lengthen the short helices, move a pinned motif into a loop, or relax a "
                                "requirement -- then check_feasible again, which is free.")}
            else:
                paid[problem] = paid.get(problem, 0) + 1
                obs = fn(sess.target, sess.restraint, str(args.get("budget", "cheap")),
                         gc=sess.gc, tm=sess.tm, kin=sess.kin, sw=sess.switch,
                         max_spend_usd=max(0.0, max_cost_usd - sess.total_cost_usd))
                sess.total_cost_usd += float(obs.get("cost_usd") or 0.0)
                if obs.get("certificate"):
                    sess.certificate = obs["certificate"]
                held, gcc, tmc = obs.get("constraint_held"), obs.get("gc"), obs.get("tm")
                kinc, swc = obs.get("kinetics"), obs.get("switch")
                # every declared requirement must hold, not just the folding one -- a design that folds but
                # breaks the pin or misses the GC target is not the answer to what was asked
                if (obs.get("verdict") in ("certified", "knot_confirmed")
                        and (held is None or held["held"])
                        and (gcc is None or gcc.get("held"))
                        and (tmc is None or tmc.get("held"))
                        and (kinc is None or kinc.get("held"))
                        and (swc is None or swc.get("held"))):
                    sess.done, sess.outcome = True, "certified"
        sess.steps.append(Step(n, name, args, obs, cost_usd=float(obs.get("cost_usd") or 0.0),
                               wall_s=round(time.time() - t0, 1), thought=thought))
        _persist(sess, workspace)
        if sess.done:
            break
    if not sess.outcome:
        sess.outcome = "out of steps"
    # AN UNVERIFIED FUNCTIONAL CLAIM MUST REACH THE OUTCOME LINE, not sit in a step nobody reads. The whole
    # reason the guard was allowed to pass is that we say what the certificate does not cover.
    if sess.unverified_claim and sess.outcome == "certified":
        sess.outcome = ("certified (shape only -- the functional claim in the request "
                        f"[\"{sess.unverified_claim['word']}\"] is NOT verified by anything here)")
    # DID WE ANSWER THE QUESTION THAT WAS ASKED? The certificate is honest about "this sequence folds to THIS
    # structure" -- and says nothing about whether THIS structure is the one the user wanted. Measured: on a
    # request that fixed the shape completely (closing stem 5, two 4-bp branches, 2-nt spacers, exactly 40
    # nt) the agent delivered a genuinely certified sequence for a DIFFERENT 40-nt shape (closing stem 6, no
    # spacers). Every oracle was satisfied and the user's question was not. Substituting the shape can be the
    # right move -- it is exactly how a conflicting pin gets resolved -- so this states it instead of
    # forbidding it. What must never happen is that it goes unsaid.
    if sess.requested_target and sess.target and sess.target != sess.requested_target:
        sess.substituted = {
            "requested": sess.requested_target, "delivered": sess.target,
            "same_length": len(sess.target) == len(sess.requested_target),
            "note": ("the answer is for a DIFFERENT shape than the one first built from the request. If the "
                     "request fixed the shape, this answer does not satisfy it.")}
    return _persist(sess, workspace)


def _persist(sess: "Session", workspace: Optional[str]):
    if workspace:
        os.makedirs(os.path.dirname(workspace) or ".", exist_ok=True)
        with open(workspace, "w") as f:
            json.dump(sess.to_dict(), f, ensure_ascii=False, indent=1)
    return sess


def render(sess: "Session", lang: str = "zh") -> str:
    """The trace, so a caller can see what was TRIED, not only what came out."""
    zh = lang == "zh"
    out = [("请求：" if zh else "request: ") + sess.request]
    for s in sess.steps:
        head = f"  [{s.n}] {s.action}"
        if s.thought:
            head += f"   ← {s.thought}"
        out.append(head)
        o = s.observation
        if o.get("error"):
            out.append(f"        {'失败' if zh else 'failed'}: {o['error']}")
        elif s.action == "propose_structure":
            out.append(f"        {o['target']}   {o.get('plain', '')}")
        elif s.action == "recommend_loop_cap":
            for lp in (o.get("loops") or [])[:2]:
                out.append(f"        loop@{lp['start']} len={lp['len']} -> {lp.get('motifs')}")
        elif s.action == "check_feasible":
            out.append("        " + (o.get("note") or ""))
            for c in o.get("conflicts", [])[:4]:
                out.append(f"          - {c['why']}")
        elif s.action == "design":
            out.append(f"        {o.get('verdict')}   bp_away={o.get('bp_away')}   "
                       f"${o.get('cost_usd', 0):.5f}   {o.get('wall_s')}s")
            if o.get("gc") is not None:
                g = o["gc"]
                out.append(f"        GC: {'达标' if zh else 'on target'}={g.get('held')}  "
                           f"{'实测' if zh else 'measured'}={g.get('measured')}%  "
                           f"{'目标' if zh else 'target'}={g.get('target')}±{g.get('tolerance')}")
            if o.get("tm") is not None:
                m = o["tm"]
                out.append(f"        {'熔点' if zh else 'melting point'}: {m.get('held')}  "
                           f"{'实测' if zh else 'measured'}={m.get('measured')}C  "
                           f"{'目标' if zh else 'target'}={m.get('target')}±{m.get('tolerance')}C")
            if o.get("set_check") is not None:
                k = o["set_check"]
                out.append(f"        {'集合' if zh else 'set'}: {o.get('n_delivered')}/{o.get('n_requested')}  "
                           f"{'最大两两相同度' if zh else 'worst identity'}={k.get('worst_pairwise_identity')}%  "
                           f"{'最强二聚体' if zh else 'strongest duplex'}={k.get('strongest_duplex_kcal')} kcal/mol")
            if o.get("constraint_held") is not None:
                h = o["constraint_held"]
                out.append(f"        {'固定区保持' if zh else 'constraint held'}: {h['held']}"
                           + (f"  {'违反位置' if zh else 'violated'}: {h['violated_positions']}"
                              if not h["held"] else ""))
    if sess.substituted:
        out.append(("⚠ 交付的形状与最初按请求搭出来的不同：" if zh else
                    "! the delivered shape is not the one first built from the request:"))
        out.append(("   请求：" if zh else "   requested: ") + sess.substituted["requested"])
        out.append(("   交付：" if zh else "   delivered: ") + sess.substituted["delivered"])
        out.append(("   若请求把形状写死了，这个答案就不满足它。" if zh else
                    "   if the request fixed the shape, this answer does not satisfy it."))
    out.append(("结果：" if zh else "outcome: ") + sess.outcome
               + f"   {'总花费' if zh else 'total'} ${sess.total_cost_usd:.5f}"
               + (f"  ({'其中思考' if zh else 'of which thinking'} ${sess.llm_cost_usd:.5f})"
                  if sess.llm_cost_usd else ""))
    return "\n".join(out)
