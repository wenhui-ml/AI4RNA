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

MAX_STEPS = 10   # a request needing a restructure takes ~6; the real bound is max_cost_usd
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
    certificate: Optional[dict] = None
    done: bool = False
    outcome: str = ""
    total_cost_usd: float = 0.0

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


def check_feasible(target: str, restraint: Optional[str]) -> dict:
    """EXACT, BEFORE SPENDING ANYTHING: are the pinned bases compatible with the target's own pairing?

    If positions i and j must pair and both are pinned to bases that cannot form a pair, no search will ever
    succeed -- and a search would burn its whole budget discovering that. Reporting the offending positions is
    what lets the agent do something useful (move the motif, move the structure) instead of buying more time.
    """
    from ..structure_utils import pair_table
    if restraint and len(restraint) != len(target):
        # A stale restraint from a previous, differently sized shape used to index past the end and raise.
        # A primitive that crashes teaches the agent nothing; one that says what is wrong lets it recover.
        return {"ok": False, "error": (f"the restraint is {len(restraint)} long but the target is "
                                       f"{len(target)} -- it belongs to a different shape; re-pin it")}
    if not restraint or set(restraint) == {"N"}:
        # An observation that does not say what to do next wastes a step. It cost one here.
        return {"ok": True, "conflicts": [], "note": ("nothing is pinned yet, so there is nothing to check -- "
                                                      "either place_constraint first, or go straight to design")}
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
    return {"ok": not conflicts, "conflicts": conflicts,
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


def _gc_of(seq: Optional[str]):
    if not seq:
        return None
    return 100.0 * sum(1 for c in seq if c in "GC") / len(seq)


def design(target: str, restraint: Optional[str] = None, budget: str = "cheap", seed: int = 0,
           gc: Optional[dict] = None) -> dict:
    """Run the solver and report what came back, including HOW it failed."""
    from .certify import BUDGETS, certify_design
    from .cost import solver_cost_usd
    if budget not in BUDGETS:
        budget = "cheap"
    t0 = time.time()
    # STEER OR SCREEN. DesiRNA's -acgu_content steers the search toward a composition; measured on a 6-bp
    # hairpin it produced 81.2% GC against a 50% request -- WORSE than the 62.5% it reaches with no request
    # at all. Screening its candidate POOL instead found 66 of 200 candidates that both fold correctly and
    # sit within 50 +/- 10. So the requirement is enforced by reading the pool, which is exact, and not by
    # asking the solver nicely, which is not. Same mechanism that recovered 14 pseudoknot targets the tool
    # had thrown away.
    if (restraint and set(restraint) != {"N"}) or gc:
        from ..experiments.rssd_constrained import ConstrainedDesiRNA
        from ..generators.desirna import desirna_available
        from ..task import DesignTask, OracleConfig
        from .certify import certify_sequence
        if not desirna_available():
            return {"ok": False, "error": "the external solver is not installed"}
        oc = OracleConfig(param_set="turner2004", temperature=37.0, dangles=2)
        cost, err, screened, best_partial = 0.0, None, 0, None
        for tl in BUDGETS[budget]:
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
                cert = certify_sequence(c, target)
                if cert.verdict in ("certified", "knot_confirmed"):
                    err = None
                    return {"ok": True, "sequence": c, "cost_usd": cost,
                            "wall_s": round(time.time() - t0, 1), "verdict": cert.verdict,
                            "constraint_held": held, "gc": g, "n_screened": screened,
                            "bp_away": 0, "certificate": cert.to_dict()}
                best_partial = best_partial or c
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
                "bp_away": (cert.detail or {}).get("best_bp") if cert else None,
                "note": (f"screened {screened} candidates; none satisfied every requirement at once")}
    c = certify_design(target, budget=budget, seed=seed)
    return {"ok": True, "sequence": c.sequence, "cost_usd": c.cost_usd, "wall_s": c.wait_s,
            "verdict": c.verdict, "constraint_held": None, "gc": _gc_check(c.sequence, gc),
            "bp_away": (c.detail or {}).get("best_bp"),
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


PRIMITIVES = {"propose_structure": propose_structure, "place_constraint": place_constraint,
              "check_feasible": check_feasible, "design": design,
              "recommend_loop_cap": recommend_loop_cap, "set_gc_content": set_gc_content}


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
        You never write dot-bracket. You give element types and integer lengths; the code assembles and counts.
  place_constraint  {"motif":"GGGG","at":<0-based position>}    pin a fixed sub-sequence. ADDITIVE:
        call it again to pin a second region; the first stays. Overlapping placements that disagree are
        rejected rather than silently overwritten.
  check_feasible    {}                      FREE and EXACT. If pinned bases sit on positions that must pair
        with each other and cannot, no search will ever succeed. Always run this before spending on `design`
        when anything is pinned.
  recommend_loop_cap {}                     ask the motif knowledge base which caps stabilise a loop of this
        size. It only SUGGESTS; pin one with place_constraint so the result can be checked exactly.
  set_gc_content    {"percent":50,"tolerance":5}   steer the solver toward a GC fraction. The steer is not
        the check: the result is verified against percent +/- tolerance afterwards.
  design            {"budget":"cheap|standard|best_effort"}     run the solver and verify the result
  done              {"reason":"..."}        stop: either the goal is met, or you have concluded it cannot be

Rules:
  * read the observation before choosing. A conflict at positions i and j is not fixed by a bigger budget --
    move the motif, or change the shape so those positions are not paired with each other.
  * a hairpin loop needs >= 3 unpaired nucleotides; a stem needs >= 2 base pairs.
  * prefer "cheap" first. Escalate the budget only when the failure looks like a search failure (a small
    bp_away with no constraint conflict), never when it looks structural.
  * you cannot mark anything solved. `design` returns a verdict from an exact verifier; that is the answer.
"""


def _observations(sess: "Session") -> str:
    out = [f"REQUEST: {sess.request}"]
    if sess.target:
        out.append(f"current target: {sess.target}  ({len(sess.target)} nt)")
    if sess.restraint:
        out.append(f"current restraint: {sess.restraint}")
    for s in sess.steps:
        out.append(f"[step {s.n}] {s.action}({json.dumps(s.args, ensure_ascii=False)}) -> "
                   f"{json.dumps(s.observation, ensure_ascii=False)[:600]}")
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
    if hit:
        sess.done, sess.outcome = True, "refused"
        sess.steps.append(Step(0, "refuse", {"word": hit}, {
            "reason": ("the request describes a biological FUNCTION, not a SHAPE. Nothing here can verify "
                       "that a sequence DOES anything, and inventing a shape would produce a certificate "
                       "that is true about the wrong target.")}))
        return _persist(sess, workspace)
    if client is None:
        from ..llm.openrouter import OpenRouterClient
        client = OpenRouterClient(model=model)

    def _ask(extra: str = ""):
        msg = _observations(sess) + (("\n\n" + extra) if extra else "")
        return client.chat([{"role": "user", "content": msg}], system=SYSTEM,
                           temperature=0.0, max_tokens=1200)

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

    for n in range(1, max_steps + 1):
        r = _ask()
        if getattr(r, "error", None):
            sess.outcome = f"agent unavailable: {r.error}"
            break
        act = _parse(getattr(r, "text", "") or "")
        if act is None:                       # one retry, told plainly what went wrong
            r = _ask('Your last reply contained no usable action. Reply with ONLY the json object, '
                     'nothing before or after it, and keep "thought" under 25 words.')
            act = _parse(getattr(r, "text", "") or "")
        if act is None:
            sess.outcome = "the agent did not return an action"
            break
        name, args = str(act.get("action", "")), dict(act.get("args") or {})
        thought = str(act.get("thought", ""))[:200]

        if name == "done":
            sess.done, sess.outcome = True, args.get("reason", "the agent stopped")
            sess.steps.append(Step(n, "done", args, {"reason": sess.outcome}, thought=thought))
            break
        fn = PRIMITIVES.get(name)
        if fn is None:
            sess.steps.append(Step(n, name, args, {"ok": False, "error": f"unknown action {name!r}"},
                                   thought=thought))
            continue

        t0 = time.time()
        if name == "propose_structure":
            obs = fn(args.get("elements") or [])
            if obs.get("ok"):
                sess.target, sess.restraint = obs["target"], None
        elif name == "place_constraint":
            obs = fn(sess.target or "", args.get("motif", ""), int(args.get("at", 0)),
                     existing=sess.restraint)
            if obs.get("ok"):
                sess.restraint = obs["restraint"]
        elif name == "set_gc_content":
            obs = fn(args.get("percent"), float(args.get("tolerance", 5.0)))
            if obs.get("ok"):
                sess.gc = obs
        elif name == "recommend_loop_cap":
            obs = fn(sess.target) if sess.target else \
                {"ok": False, "error": "no target yet -- propose_structure first"}
        elif name == "check_feasible":
            obs = fn(sess.target or "", sess.restraint) if sess.target else \
                {"ok": False, "error": "no target yet -- propose_structure first"}
        else:  # design
            if not sess.target:
                obs = {"ok": False, "error": "no target yet -- propose_structure first"}
            elif sess.total_cost_usd >= max_cost_usd:
                obs = {"ok": False, "error": f"cost cap ${max_cost_usd} reached"}
            else:
                obs = fn(sess.target, sess.restraint, str(args.get("budget", "cheap")), gc=sess.gc)
                sess.total_cost_usd += float(obs.get("cost_usd") or 0.0)
                if obs.get("certificate"):
                    sess.certificate = obs["certificate"]
                held, gcc = obs.get("constraint_held"), obs.get("gc")
                # every declared requirement must hold, not just the folding one -- a design that folds but
                # breaks the pin or misses the GC target is not the answer to what was asked
                if (obs.get("verdict") in ("certified", "knot_confirmed")
                        and (held is None or held["held"])
                        and (gcc is None or gcc.get("held"))):
                    sess.done, sess.outcome = True, "certified"
        sess.steps.append(Step(n, name, args, obs, cost_usd=float(obs.get("cost_usd") or 0.0),
                               wall_s=round(time.time() - t0, 1), thought=thought))
        _persist(sess, workspace)
        if sess.done:
            break
    if not sess.outcome:
        sess.outcome = "out of steps"
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
            if o.get("constraint_held") is not None:
                h = o["constraint_held"]
                out.append(f"        {'固定区保持' if zh else 'constraint held'}: {h['held']}"
                           + (f"  {'违反位置' if zh else 'violated'}: {h['violated_positions']}"
                              if not h["held"] else ""))
    out.append(("结果：" if zh else "outcome: ") + sess.outcome
               + f"   {'总花费' if zh else 'total'} ${sess.total_cost_usd:.5f}")
    return "\n".join(out)
