#!/usr/bin/env python3
"""(3) Autonomous operator INVENTION for chomsky_climb: does an LLM agent, given only low-level primitives
and the folding oracles (NOT the decompose->compose->repair recipe, NOT any external pseudoknot solver),
write a reusable pseudoknot-design operator that then clears the certificate on held-out targets?

Loop: the agent writes `design(target, seed) -> seq` using an injected primitive API; we sandbox-scan and
execute it on a TRAINING pool (Pseudobase[110:130], disjoint from the held-out), score under pKiss AND
ProbKnot, feed the score + failures back, and let it revise for K rounds. We then FREEZE the best operator
and certify it on the held-out [0:60] under the full panel (pKiss/ProbKnot/ShapeKnots). Autonomy is
structural: no external solver is exposed. Everything (every round's code + scores) is logged.

Honest by construction: if the agent fails to invent a panel-robust operator, that is the reported result
-- the certificate then audits an agent-written operator that games or underperforms, which is on-thesis.
"""
import json, os, re, signal, sys, time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import RNA
from rna_harness.oracles_pk import (pkiss_pairs, probknot_pairs, shapeknots_pairs,
                                     forms_target, pk_pairs, rnastructure_available)
from rna_harness.benchmarks.rnainvbench import load_pseudobase
from rna_harness.llm.openrouter import OpenRouterClient

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "invent_operator"); os.makedirs(OUT, exist_ok=True)
COMP = {"A": "U", "U": "A", "G": "C", "C": "G"}

# ------------------------------------------------------------------ primitives exposed to the agent
def fold_cf(seq):
    """ViennaRNA MFE structure (pseudoknot-FREE, nested only)."""
    st, _ = RNA.fold(seq); return st

def inverse_fold_cf(nested_target, seed=0):
    """ViennaRNA inverse fold for a NESTED target -> a sequence (best-effort)."""
    import RNA as _R
    seq, _ = _R.inverse_fold(_rand_seq(len(nested_target), seed), nested_target)
    return seq

def fold_pkiss_pairs(seq):
    """Crossing-aware fold (pKiss): returns the set of base pairs (i,j)."""
    return set(pkiss_pairs(seq))

def fold_probknot_pairs(seq):
    """Crossing-aware fold (ProbKnot): returns the set of base pairs (i,j)."""
    return set(probknot_pairs(seq))

def target_pairs(dotbracket):
    """Parse a (possibly crossing) dot-bracket target into a set of base pairs (i,j)."""
    return set(pk_pairs(dotbracket))

def _rand_seq(n, seed=0):
    import random as _r; g = _r.Random(seed)
    return "".join(g.choice("ACGU") for _ in range(n))

PRIMS = {"fold_cf": fold_cf, "inverse_fold_cf": inverse_fold_cf, "fold_pkiss_pairs": fold_pkiss_pairs,
         "fold_probknot_pairs": fold_probknot_pairs, "target_pairs": target_pairs, "rand_seq": _rand_seq,
         "COMP": COMP}

API_DOC = """Available primitives (call them directly; do NOT import anything else):
  fold_cf(seq) -> str                : ViennaRNA MFE dot-bracket, PSEUDOKNOT-FREE (nested only).
  inverse_fold_cf(nested_target, seed=0) -> str : a sequence whose nested MFE approximates nested_target.
  fold_pkiss_pairs(seq) -> set[(i,j)]: crossing-aware pKiss fold as a base-pair set.
  fold_probknot_pairs(seq) -> set[(i,j)]: crossing-aware ProbKnot fold as a base-pair set.
  target_pairs(dotbracket) -> set[(i,j)]: parse a crossing dot-bracket into base pairs.
  rand_seq(n, seed=0) -> str         : random ACGU string of length n.
  COMP                               : dict {A:U,U:A,G:C,C:G}.
Modules available in scope: random, math, itertools. NOTHING else (no import, no os/sys/file/network).
Write EXACTLY one function `design(target, seed=0) -> str` returning an ACGU string of len==len(target).
A target is 'solved' iff fold_pkiss_pairs(seq)==target_pairs(target) AND fold_probknot_pairs(seq)==target_pairs(target).
You MAY call the oracles inside design to guide a search. You may NOT call any external pseudoknot solver."""

FORBIDDEN = re.compile(r"\b(import|__import__|open|exec|eval|compile|os\.|sys\.|subprocess|socket|"
                       r"globals|locals|getattr|setattr|__\w+__|input|file)\b")


class _TO(Exception):
    pass


def _alarm(signum, frame):
    raise _TO()


def make_design(code, per_call_s=8):
    """Sandbox-exec the agent's code; return a design(target,seed) callable or raise."""
    if FORBIDDEN.search(code):
        raise ValueError("forbidden token in code: " + (FORBIDDEN.search(code).group(0)))
    import random, math, itertools
    import builtins as _b
    safe_builtins = {k: getattr(_b, k) for k in (
        "range", "len", "list", "dict", "set", "frozenset", "tuple", "str", "int", "float", "bool",
        "min", "max", "sum", "sorted", "enumerate", "zip", "abs", "map", "filter", "any", "all",
        "reversed", "round", "print", "isinstance", "chr", "ord", "Exception", "ValueError", "KeyError",
        "IndexError", "TypeError", "ZeroDivisionError", "StopIteration", "True", "False", "None") if hasattr(_b, k)}
    ns = {"__builtins__": safe_builtins, "random": random, "math": math, "itertools": itertools, **PRIMS}
    exec(code, ns)
    fn = ns.get("design")
    if not callable(fn):
        raise ValueError("no design() defined")

    def wrapped(target, seed=0):
        signal.signal(signal.SIGALRM, _alarm); signal.alarm(per_call_s)
        try:
            return fn(target, seed)
        finally:
            signal.alarm(0)
    return wrapped


def solved_pk_pb(seq, target):
    if not seq or len(seq) != len(target):
        return False
    tp = set(pk_pairs(target))
    return set(pkiss_pairs(seq)) == tp and set(probknot_pairs(seq)) == tp


def score_on(design, targets):
    """Return (n_solved, per-target ok, errors)."""
    ok, errs = [], []
    for t in targets:
        try:
            s = design(t, 0)
            ok.append(bool(solved_pk_pb(s, t)))
        except Exception as e:
            ok.append(False); errs.append(repr(e)[:120])
    return sum(ok), ok, errs[:5]


def extract_code(text):
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def main(rounds=6, n_train=16, n_heldout=60, model=None):
    model = model or os.environ.get("OPENAI_MODEL", "anthropic/claude-opus-4.8")
    train = [t.target for t in load_pseudobase(limit=130)[110:126][:n_train]]
    held = [t.target for t in load_pseudobase(limit=n_heldout)]
    client = OpenRouterClient(model=model)
    rnastr = rnastructure_available()
    print(f"INVENT: model={model} train={len(train)} held-out={len(held)} rnastr={rnastr}", flush=True)

    system = ("You are an expert RNA structural bioinformatics researcher. You invent a REUSABLE operator "
              "for designing sequences that fold into PSEUDOKNOTTED (crossing) RNA secondary structures. "
              "You are given only low-level primitives; you must compose the whole strategy yourself. "
              "Return ONLY a Python code block defining `design(target, seed=0)`.")
    hist, best = [], {"code": None, "train_solved": -1, "round": -1}
    # carry the full conversation so the LLM sees its own prior code + the feedback and can truly revise
    messages = [{"role": "user", "content":
                 f"{API_DOC}\n\nExample training targets (crossing dot-bracket):\n"
                 + "\n".join(train[:4]) + "\n\nWrite design(target, seed=0). Aim to maximize train solves."}]
    for rd in range(rounds):
        # max_tokens=None: a reasoning model must not have content starved by a small cap
        r = client.chat(messages, system=system, temperature=0.4, max_tokens=None)
        if r.error:
            print(f"  round {rd}: LLM error {r.error}", flush=True); break
        raw = r.text or ""
        if "def design" not in raw and r.reasoning and "def design" in r.reasoning:
            raw = r.reasoning  # some reasoning models emit the code block only in the reasoning field
        messages.append({"role": "assistant", "content": (r.text or raw)[:8000]})  # keep its code in-context
        code = extract_code(raw)
        rec = {"round": rd, "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens}
        try:
            design = make_design(code, per_call_s=12)
            nsolved, okv, errs = score_on(design, train)
            rec.update({"train_solved": nsolved, "n_train": len(train), "errors": errs})
            print(f"  round {rd}: train solved {nsolved}/{len(train)}  errs={errs[:2]}", flush=True)
            if nsolved > best["train_solved"]:
                best = {"code": code, "train_solved": nsolved, "round": rd}
            fails = [train[i] for i, o in enumerate(okv) if not o][:3]
            fb = (f"Your operator solved {nsolved}/{len(train)} training targets (pKiss AND ProbKnot exact)."
                  + (f" Runtime errors: {errs}." if errs else "")
                  + f" Failing examples: {fails}. Revise your design() to solve more; return only the code block.")
        except Exception as e:
            rec.update({"train_solved": None, "compile_error": repr(e)[:200]})
            print(f"  round {rd}: compile/sandbox error {repr(e)[:120]}", flush=True)
            fb = (f"Your code failed to load: {repr(e)[:200]}. Fix it. "
                  f"No imports; only the listed primitives + random/math/itertools. Return only the code block.")
        messages.append({"role": "user", "content": fb})
        hist.append(rec)
        json.dump({"model": model, "rounds": hist, "best_round": best["round"],
                   "best_train_solved": best["train_solved"]}, open(os.path.join(OUT, "invent_log.json"), "w"), indent=1)

    # ---------- freeze best and CERTIFY on held-out under the full panel ----------
    result = {"model": model, "best_train_solved": best["train_solved"], "best_round": best["round"],
              "rounds": hist, "best_code": best["code"], "heldout": []}
    if best["code"]:
        design = make_design(best["code"], per_call_s=20)
        npk = npb = nunan = 0
        for k, t in enumerate(held):
            try:
                s = design(t, 0)
                pk = bool(forms_target(pkiss_pairs(s), t)) if s else False
                pb = bool(forms_target(probknot_pairs(s), t)) if s else False
                sk = bool(forms_target(shapeknots_pairs(s), t)) if (s and rnastr) else None
                unan = (pk and pb and bool(sk)) if rnastr else None
                npk += int(pk); npb += int(pb); nunan += int(bool(unan))
                result["heldout"].append({"id": f"pseudobase:{k}", "pkiss": pk, "probknot": pb,
                                          "shapeknots": sk, "unanimous": unan})
            except Exception as e:
                result["heldout"].append({"id": f"pseudobase:{k}", "error": repr(e)[:120]})
            if k % 10 == 9:
                print(f"  held-out {k+1}/{len(held)}: pKiss={npk} pk&pb={npb} unanimous={nunan}", flush=True)
        result.update({"heldout_pkiss": npk, "heldout_pk_and_pb": npb, "heldout_panel_unanimous": nunan})
        print(f"\nINVENTED operator on held-out {len(held)}: pKiss={npk}, pKiss&ProbKnot={npb}, "
              f"panel-unanimous={nunan}. (hand-built panel_auto: 3-8/60; floor: 0/60.)", flush=True)
    else:
        print("\nNo compilable operator was invented.", flush=True)
    json.dump(result, open(os.path.join(OUT, "result.json"), "w"), indent=1)
    print("-> invent_operator/result.json", flush=True)


if __name__ == "__main__":
    main(rounds=int(sys.argv[1]) if len(sys.argv) > 1 else 6)
