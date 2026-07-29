"""Turn what a person says into a target the exact verifier can judge. The LLM does semantics; it never counts.

THE GAP THIS CLOSES. certify() makes the five expert decisions for a caller, but its input is still a
dot-bracket string. Anyone who can write `((((....[[[[))))....]]]]` is not the non-expert the product is for,
so in practice "does not know RNA" had been narrowed to "knows dot-bracket but not oracles and budgets".
This is the layer that removes that requirement.

THE ONE DESIGN RULE, and it is the same argument that keeps an LLM out of the oracle: a next-token predictor
must not be asked to do precise computation. Writing dot-bracket IS precise computation -- it is counting
parentheses -- and miscounting long paren/dot runs is a known failure mode this repo already guards against
elsewhere (mcp/motif_server.py::_guard_target exists precisely because agents retype targets wrong). So the
model never emits brackets. It emits a STRUCTURED SPEC of element types and integer lengths, and
deterministic code assembles the string:

    model  ->  {"elements": [{"type": "hairpin", "stem": 6, "loop": 4}]}      semantics
    code   ->  "((((((....))))))"                                            counting
    code   ->  classify() validates, certify() adjudicates                   truth

Every arithmetic and every truth claim stays with code that cannot hallucinate. The model contributes the one
thing it is good at and nothing else.

DOMAIN CONSTRAINTS THE CALLER DOES NOT KNOW are enforced here rather than discovered as a mysterious failure:
a hairpin loop needs at least 3 unpaired nucleotides to close at all, and a stem of 1 base pair is a lonely
pair that will not hold. Rejecting those with an explanation is more useful than designing against them.

SCOPE. Hairpins, chains of hairpins joined by spacers, internal loops/bulges, MULTIWAY JUNCTIONS and H-type
PSEUDOKNOTS.

The last two were refused in v1 for the wrong reason. They were out of scope because this assembler was
simple, not because anything downstream could not judge them: a multiway junction is NESTED, so the exact
ViennaRNA oracle handles it with no new work at all, and a pseudoknot is adjudicated by the already-calibrated
knot_confirmed rule (84.0% sensitivity on known natural structures, 0.4% false accept on random). Refusing
them confused the author's implementation boundary with the system's capability boundary, which is a mistake
worth naming: it made the product look narrower than the machinery underneath it.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

MIN_LOOP = 3          # a hairpin loop shorter than this cannot close
MIN_STEM = 2          # a 1-bp "stem" is a lonely pair

# A FUNCTION is not a SHAPE, and this system can only judge shapes. Asked for "an RNA that binds protein X"
# a language model will happily invent a plausible structure, the pipeline will design a sequence for it and
# the verifier will certify that sequence -- correctly, against a target nobody has any reason to believe is
# the right one. That is a misleading success: every downstream number is true and the answer is worthless.
# The oracle's silence about function is the project's red line, so the refusal is deterministic rather than
# left to the model's judgement, which can be talked around.
FUNCTION_WORDS = (
    "结合", "绑定", "靶向", "适配体", "适体", "核酶", "核开关", "riboswitch", "开关", "调控", "沉默",
    "干扰", "敲低", "抑制", "表达", "翻译", "剪接", "催化", "药物", "治疗", "疫苗", "蛋白", "基因",
    "受体", "配体", "免疫",
    "bind", "binding", "binds", "aptamer", "ribozyme", "riboswitch", "silence", "knockdown", "knock down",
    "inhibit", "express", "translation", "splic", "cataly", "therapeut", "drug", "vaccine", "protein",
    "gene ", "receptor", "ligand", "immun",
)


def functional_request(text: str):
    """The word that makes this a function request, or None. Deterministic on purpose."""
    low = (text or "").lower()
    for w in FUNCTION_WORDS:
        if w in low:
            return w
    return None


@dataclass
class Spec:
    elements: List[Dict] = field(default_factory=list)
    budget: str = "standard"
    interpretation: str = ""          # what the model understood, in words, for the caller to check
    raw_target: Optional[str] = None  # set when the caller supplied dot-bracket directly
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- deterministic assembly
def assemble(spec: Spec):
    """(dot_bracket, error). Pure arithmetic -- this is the part the model is not allowed to do."""
    if spec.raw_target:
        return spec.raw_target, None
    if not spec.elements:
        return None, "no structural elements were identified in the request"
    out = []
    for i, e in enumerate(spec.elements):
        t = str(e.get("type", "")).lower()
        try:
            if t == "hairpin":
                stem, loop = int(e["stem"]), int(e["loop"])
                if stem < MIN_STEM:
                    return None, (f"element {i} asks for a {stem}-bp stem; a stem needs at least "
                                  f"{MIN_STEM} base pairs or it is a lonely pair that will not hold")
                if loop < MIN_LOOP:
                    return None, (f"element {i} asks for a {loop}-nt hairpin loop; a hairpin loop needs at "
                                  f"least {MIN_LOOP} unpaired nucleotides to close at all")
                out.append("(" * stem + "." * loop + ")" * stem)
            elif t == "spacer":
                n = int(e.get("len", e.get("length", 0)))
                if n < 0:
                    return None, f"element {i} asks for a negative spacer"
                out.append("." * n)
            elif t == "internal_loop":
                so, si = int(e["outer_stem"]), int(e["inner_stem"])
                lft, rgt, loop = int(e.get("left", 0)), int(e.get("right", 0)), int(e["loop"])
                if min(so, si) < MIN_STEM:
                    return None, f"element {i} has a stem below {MIN_STEM} bp"
                if loop < MIN_LOOP:
                    return None, f"element {i} has a {loop}-nt hairpin loop; the minimum is {MIN_LOOP}"
                out.append("(" * so + "." * lft + "(" * si + "." * loop + ")" * si + "." * rgt + ")" * so)
            elif t in ("multiloop", "junction"):
                # A multiway junction: one closing stem, then `branches` hairpins separated by spacers, all
                # nested -- so the exact MFE oracle judges it exactly as it judges a hairpin.
                cl = int(e["closing_stem"])
                br = list(e.get("branches") or [])
                sp = int(e.get("spacer", 1))
                if cl < MIN_STEM:
                    return None, f"element {i} has a {cl}-bp closing stem; the minimum is {MIN_STEM}"
                if len(br) < 2:
                    return None, (f"element {i} is a junction with {len(br)} branch(es); a junction needs at "
                                  f"least 2 -- one branch inside a stem is just an internal loop")
                inner = []
                for b in br:
                    bs, bl = int(b["stem"]), int(b["loop"])
                    if bs < MIN_STEM:
                        return None, f"element {i} has a branch stem of {bs} bp; the minimum is {MIN_STEM}"
                    if bl < MIN_LOOP:
                        return None, f"element {i} has a branch loop of {bl} nt; the minimum is {MIN_LOOP}"
                    inner.append("(" * bs + "." * bl + ")" * bs)
                out.append("(" * cl + ("." * sp) + ("." * sp).join(inner) + ("." * sp) + ")" * cl)
            elif t in ("pseudoknot", "h_pseudoknot"):
                # H-type pseudoknot: stem1 opens, stem2 opens inside loop1, stem1 closes, stem2 closes in the
                # tail. The crossing is what makes it a pseudoknot, and knot_confirmed already judges it.
                s1, s2 = int(e["stem1"]), int(e["stem2"])
                l1 = int(e.get("loop1", MIN_LOOP))
                l2 = int(e.get("loop2", MIN_LOOP))
                l3 = int(e.get("loop3", MIN_LOOP))
                if min(s1, s2) < MIN_STEM:
                    return None, f"element {i} has a stem below {MIN_STEM} bp"
                if min(l1, l2, l3) < 1:
                    return None, f"element {i} needs at least 1 unpaired nucleotide in each connecting loop"
                out.append("(" * s1 + "." * l1 + "[" * s2 + "." * l2 + ")" * s1 + "." * l3 + "]" * s2)
            elif t in ("kissing", "kissing_loop"):
                # A kissing complex is two hairpin loops pairing with EACH OTHER -- a different topology from
                # an H-type knot, and assembling it as one would silently answer a different question.
                return None, (f"element {i} is a kissing complex, which this assembler does not build; it is "
                              f"not an H-type pseudoknot. Supply the dot-bracket directly for this one.")
            else:
                return None, f"element {i} has unknown type {t!r}"
        except (KeyError, TypeError, ValueError) as ex:
            return None, f"element {i} is missing or malformed: {ex}"
    return "".join(out), None


# --------------------------------------------------------------------------- the model's half
SYSTEM = f"""You translate a person's description of an RNA secondary structure into a STRUCTURED SPEC.

You must NOT write dot-bracket notation. You emit element types and integer lengths only; the caller's code
assembles the string. This is deliberate: writing brackets is counting, and you are not the right tool for
counting.

Reply with ONE json object and nothing else:
{{"elements": [...], "budget": "cheap|standard|best_effort", "interpretation": "one sentence, plain words"}}

Element types:
  {{"type":"hairpin","stem":<bp>,"loop":<nt>}}                a stem of `stem` base pairs closing a loop of
                                                              `loop` unpaired nucleotides
  {{"type":"spacer","len":<nt>}}                              unpaired nucleotides between elements
  {{"type":"internal_loop","outer_stem":<bp>,"left":<nt>,
    "right":<nt>,"inner_stem":<bp>,"loop":<nt>}}              a stem interrupted by an asymmetric bulge
  {{"type":"multiloop","closing_stem":<bp>,"spacer":<nt>,
    "branches":[{{"stem":<bp>,"loop":<nt>}}, ...]}}           a multiway junction: one closing stem enclosing
                                                              two or more hairpin branches
  {{"type":"pseudoknot","stem1":<bp>,"stem2":<bp>,
    "loop1":<nt>,"loop2":<nt>,"loop3":<nt>}}                  an H-type pseudoknot (two crossing stems)

Rules:
  * a hairpin loop must be at least {MIN_LOOP} nt and a stem at least {MIN_STEM} bp; if the request asks for
    less, still emit what was asked and let the caller's validator explain the problem.
  * choose "cheap" unless the request implies a hard or long structure.
  * `interpretation` is what the caller will be shown to confirm you understood; write it for someone who
    does not know RNA, and WRITE IT IN THE SAME LANGUAGE THE REQUEST WAS WRITTEN IN. A Chinese question
    answered in English defeats the whole point of this layer.
  * a pseudoknot is judged by a different, weaker criterion than a nested shape -- do not choose one unless
    the request actually asks for crossing stems.
  * if the request asks for a biological FUNCTION rather than a SHAPE -- binding something, catalysing
    something, switching, silencing a gene, a therapeutic -- reply
    {{"out_of_scope":"function","interpretation":"<say what they asked for, in their language>"}}
    and NOTHING else. Do not invent a structure. The system downstream can verify that a sequence folds into
    a given shape; it cannot verify that it does anything. Inventing a shape here would produce a certificate
    that is true about the wrong target.
"""


def translate(request: str, model: str = "z-ai/glm-5.2", temperature: float = 0.0,
              client=None) -> Spec:
    """Natural language -> Spec. Never returns a structure; returns a description the caller's code builds."""
    text = (request or "").strip()
    if not text:
        return Spec(error="empty request")
    hit = functional_request(text)
    if hit:
        return Spec(error=(
            f"这个请求描述的是 RNA 的「功能」（出现了「{hit}」），而不是它的「形状」。"
            f"本系统的判定器只能验证「一条序列是否折叠成给定的二级结构」，无法验证它是否具有某种生物功能——"
            f"如果在这里凭空编一个结构，后面每一个数字都会是真的，而答案是错的。\n\n"
            f"可以这样用：把功能需求先转成一个具体形状（例如「茎 6 对、环 4 nt 的发夹」），"
            f"或者直接给出 dot-bracket 结构式。"
            if any("\u4e00" <= ch <= "\u9fff" for ch in text) else
            f"This asks for a biological FUNCTION (the word {hit!r}), not a SHAPE. The verifier here can "
            f"check whether a sequence folds into a given secondary structure; it cannot check that a "
            f"sequence DOES anything. Inventing a structure at this point would produce a certificate that "
            f"is entirely true about the wrong target.\n\n"
            f"What works: state a shape (\"a hairpin with a 6 bp stem and a 4 nt loop\"), or pass a "
            f"dot-bracket structure directly."))
    if set(text) <= set("().[]{}<> \n") and any(c in text for c in "()[]"):
        return Spec(raw_target=text.replace(" ", "").replace("\n", ""),
                    interpretation="the request already is a dot-bracket structure; passed through unchanged")
    if client is None:
        from ..llm.openrouter import OpenRouterClient
        client = OpenRouterClient(model=model)
    r = client.chat([{"role": "user", "content": text}], system=SYSTEM,
                    temperature=temperature, max_tokens=600)
    if getattr(r, "error", None):
        return Spec(error=f"translation unavailable: {r.error}")
    blob = getattr(r, "text", "") or ""
    m = re.search(r"\{.*\}", blob, re.DOTALL)
    if not m:
        return Spec(error=f"the model did not return a spec: {blob[:160]!r}")
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError as ex:
        return Spec(error=f"the model's spec was not valid json: {ex}")
    if d.get("out_of_scope"):
        return Spec(interpretation=str(d.get("interpretation") or ""),
                    error=(f"the request is out of scope ({d['out_of_scope']}): this system verifies that a "
                           f"sequence folds into a given SHAPE, and cannot verify function. State a shape, "
                           f"or pass a dot-bracket structure directly."))
    return Spec(elements=list(d.get("elements") or []),
                budget=str(d.get("budget") or "standard"),
                interpretation=str(d.get("interpretation") or ""))


# --------------------------------------------------------------------------- the whole flow
def design_from_request(request: str, budget: Optional[str] = None, dry_run: bool = False,
                        model: str = "z-ai/glm-5.2", client=None):
    """(spec, dot_bracket, certificate). `dry_run` stops after assembly so a caller can check the
    interpretation before spending anything -- the model can misread intent, and that is not something the
    verifier can catch: an exactly-folded sequence for the wrong structure is still the wrong answer."""
    spec = translate(request, model=model, client=client)
    if spec.error:
        return spec, None, None
    if budget:
        spec.budget = budget
    target, err = assemble(spec)
    if err:
        spec.error = err
        return spec, None, None
    from .certify import BUDGETS, certify_design, classify
    kind, cerr = classify(target)
    if cerr:
        spec.error = f"the assembled structure is not valid: {cerr}"
        return spec, target, None
    if spec.budget not in BUDGETS:
        spec.budget = "standard"
    if dry_run:
        return spec, target, None
    return spec, target, certify_design(target, budget=spec.budget)


def render(spec: Spec, target: Optional[str], cert) -> str:
    from .certify import render as _render
    out = [f"understood as: {spec.interpretation or '(no interpretation given)'}"]
    if target:
        out.append(f"structure    : {target}")
        out.append(f"             : {len(target)} nt"
                   + (f", budget '{spec.budget}'" if not spec.raw_target else ""))
    if spec.error:
        out.append(f"PROBLEM      : {spec.error}")
    out.append("")
    out.append(_render(cert) if cert is not None else "(no certificate: dry run or the request was refused)")
    return "\n".join(out)


def _cli(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        prog="python -m rna_harness.cognitive.intent",
        description="Describe an RNA structure in words; get a certificate. The model translates intent into "
                    "a structured spec, code assembles and counts, and the exact verifier judges.")
    ap.add_argument("request", help='e.g. "a hairpin with a 6 bp stem and a 4 nt loop"')
    ap.add_argument("--budget", default=None, help="override the budget the model chose")
    ap.add_argument("--dry-run", action="store_true", help="show the interpretation without spending")
    ap.add_argument("--model", default="z-ai/glm-5.2")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    spec, target, cert = design_from_request(a.request, budget=a.budget, dry_run=a.dry_run, model=a.model)
    if a.json:
        print(json.dumps({"spec": spec.to_dict(), "target": target,
                          "certificate": (cert.to_dict() if cert else None)}, indent=1))
    else:
        print(render(spec, target, cert))
    return 0 if (cert and cert.verdict in ("certified", "knot_confirmed")) else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
