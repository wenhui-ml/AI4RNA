"""W4 -- the LLM code-verifier result on a STANDARD benchmark (HumanEval+ / MBPP+), replacing the paper's
19 custom problems. EvalPlus gives exactly the split the paper needs: the original ``base'' tests are the
weak/provided (cheap) rung; the augmented ``plus'' tests are the held-out (strong) rung. We reproduce both
paper claims on it: (Axis-2) programs pass the cheap rung yet fail the strong one (over-acceptance), and
(dial-ability) strengthening the cheap rung cuts the false-accept mass.

Cheap rung(k): pass the first k base tests. Stronger: pass all base tests. Strong: pass base + plus. Nested by
construction. best-of-N selection is the optimization; the cheap arm selects blind to plus (as a
reward-hacking policy must). Reuses the exec+aggregation discipline of llm_verifier_goodhart.py."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

from ..llm.openrouter import OpenRouterClient

OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "llm_evalplus")
_CODE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)
# HumanEval prompts carry their own imports (e.g. `from typing import List`), but an extracted candidate
# function does not -- so code that echoes type annotations (List[float], ...) crashes with NameError unless
# we supply the standard preamble. This mirrors EvalPlus's own `prompt+completion` execution.
_PREAMBLE = ("from typing import *\nimport math, re, collections, itertools, functools, heapq, string, "
             "bisect, operator\nfrom collections import *\n")


def extract_code(text):
    m = _CODE_RE.search(text or "")
    return (m.group(1) if m else (text or "")).strip()


def _expected_and_run_src(prompt, canon, entry, inputs):
    """Sandbox source: define the reference (prompt+canonical), compute expected outputs for `inputs`."""
    return (
        "import json,sys,math\n" + _PREAMBLE + prompt + "\n" + canon + "\n"
        f"_IN=json.loads(sys.argv[1]); _out=[]\n"
        f"for _a in _IN:\n"
        f"    try: _out.append({entry}(*_a))\n"
        f"    except Exception as _e: _out.append('__ERR__')\n"
        "print('EXP'+json.dumps(_out, default=lambda o: '__UNJSON__'))\n")


def _eq(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))
        except Exception:
            return False
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    return a == b


def _candidate_run_src(code, entry, inputs):
    return (
        "import json,sys,math\n" + _PREAMBLE + code + "\n"
        f"_IN=json.loads(sys.argv[1]); _out=[]\n"
        f"for _a in _IN:\n"
        f"    try: _out.append({entry}(*_a))\n"
        f"    except Exception: _out.append('__ERR__')\n"
        "print('OUT'+json.dumps(_out, default=lambda o: '__UNJSON__'))\n")


def _exec(src, arg, timeout=8):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write("import resource; resource.setrlimit(resource.RLIMIT_AS,(2**31,2**31))\n" + src)
        path = f.name
    try:
        p = subprocess.run([sys.executable, path, arg], capture_output=True, text=True, timeout=timeout)
        for line in p.stdout.splitlines():
            if line.startswith("EXP") or line.startswith("OUT"):
                return json.loads(line[3:])
    except Exception:
        pass
    finally:
        os.unlink(path)
    return None


def load_problems(dataset, limit, seed):
    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    raw = get_human_eval_plus() if dataset == "humaneval" else get_mbpp_plus()
    items = list(raw.items())
    items.sort(key=lambda kv: kv[0])
    if limit:
        # deterministic stride sample for coverage
        step = max(1, len(items) // limit)
        items = items[::step][:limit]
    probs = []
    for tid, p in items:
        entry = p["entry_point"]; prompt = p["prompt"]; canon = p.get("canonical_solution", "")
        base = [list(x) for x in p.get("base_input", [])]
        plus = [list(x) for x in p.get("plus_input", [])][:40]  # cap plus for cost
        if not base:
            continue
        # expected outputs from the canonical reference (skip problem if reference itself errors/unjsonable)
        exp_base = _exec(_expected_and_run_src(prompt, canon, entry, base), json.dumps(base))
        exp_plus = _exec(_expected_and_run_src(prompt, canon, entry, plus), json.dumps(plus)) if plus else []
        if exp_base is None or (plus and exp_plus is None):
            continue
        if "__ERR__" in exp_base or "__UNJSON__" in exp_base:
            continue
        probs.append({"tid": tid, "entry": entry, "prompt": prompt,
                      "base_in": base, "base_exp": exp_base, "plus_in": plus, "plus_exp": exp_plus or []})
    return probs


def eval_candidate(code, prob):
    """Return a boolean vector over [base tests..., plus tests...] (provided first, held-out after)."""
    entry = prob["entry"]
    ob = _exec(_candidate_run_src(code, entry, prob["base_in"]), json.dumps(prob["base_in"]))
    op = _exec(_candidate_run_src(code, entry, prob["plus_in"]), json.dumps(prob["plus_in"])) if prob["plus_in"] else []
    vb = [(_eq(o, e)) for o, e in zip(ob, prob["base_exp"])] if ob else [False] * len(prob["base_exp"])
    vp = [(_eq(o, e)) for o, e in zip(op, prob["plus_exp"])] if op else [False] * len(prob["plus_exp"])
    return vb + vp, len(vb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", choices=["humaneval", "mbpp"], default="humaneval")
    ap.add_argument("--limit", type=int, default=0, help="0 = all problems")
    ap.add_argument("--n-max", type=int, default=12)
    ap.add_argument("--budgets", type=int, nargs="+", default=[1, 4, 12])
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--gen-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    client = OpenRouterClient(model=a.model)
    if not client.available():
        print("FATAL: no API key (source the .orkey)", flush=True); sys.exit(1)
    probs = load_problems(a.dataset, a.limit, a.seed)
    tag = a.model.replace("/", "_")
    print(f"W4 EvalPlus: model={a.model} dataset={a.dataset} problems={len(probs)} N={a.n_max} T={a.temperature}", flush=True)

    per = []
    for pi, prob in enumerate(probs):
        user = prob["prompt"] + "\n    # Complete the function above. Respond with ONLY the full function in a ```python block."
        sysmsg = "You are a careful Python programmer. Respond with ONE complete Python function in a ```python code block, no prose, no tests."
        def gen(_):
            r = client.chat([{"role": "user", "content": user}], system=sysmsg, temperature=a.temperature, max_tokens=640)
            return extract_code(r.text)
        with ThreadPoolExecutor(max_workers=a.gen_workers) as ex:
            codes = list(ex.map(gen, range(a.n_max)))
        vecs = []; nbase = 0
        for c in codes:
            v, nb = eval_candidate(c, prob); vecs.append(v); nbase = nb
        per.append({"tid": prob["tid"], "n_base": nbase, "n_tests": len(vecs[0]) if vecs else 0, "vecs": vecs})
        if (pi + 1) % 20 == 0:
            print(f"  {pi+1}/{len(probs)} problems", flush=True)
            json.dump({"per": per}, open(os.path.join(a.outdir, f"{tag}.json"), "w"))
    # final flush so the last (<20) problems are not dropped from the persisted per-task record
    json.dump({"per": per}, open(os.path.join(a.outdir, f"{tag}.json"), "w"))

    # aggregate: cheap rung(k) over base tests; strong = all (base+plus)
    def cheap_ok(v, k): return all(v[:k])
    def strong_ok(v): return all(v)
    agg = {}
    for k_label, k_of in (("k1", lambda p: 1), ("kbase", lambda p: p["n_base"])):
        allv = [(v, p) for p in per for v in p["vecs"]]
        cp = [(v, p) for (v, p) in allv if cheap_ok(v, k_of(p))]
        surv = 100 * sum(strong_ok(v) for (v, _) in cp) / len(cp) if cp else 0.0
        byN = {}
        for N in a.budgets:
            cproxy = crob = srob = 0
            for p in per:
                c = p["vecs"][:N]; kk = k_of(p)
                cheap = max(c, key=lambda v: sum(v[:kk]))
                strong = max(c, key=lambda v: (sum(v), sum(v[:kk])))
                cproxy += int(cheap_ok(cheap, kk)); crob += int(strong_ok(cheap)); srob += int(strong_ok(strong))
            m = len(per)
            byN[N] = {"cheap_proxy": round(100*cproxy/m, 1), "cheap_robust": round(100*crob/m, 1),
                      "strong_robust": round(100*srob/m, 1)}
        agg[k_label] = {"survival": round(surv, 1), "cheap_accept": len(cp), "total": len(allv),
                        "over_accept_pct": round(100 - surv, 1), "byN": byN}
        print(f"\n=== {k_label} === cheap-passers {len(cp)}/{len(allv)}; also pass strong {surv:.1f}% "
              f"(OVER-ACCEPT mass {100-surv:.1f}%); best-of-{a.budgets[-1]}: "
              f"cheap-robust={byN[a.budgets[-1]]['cheap_robust']}% strong-robust={byN[a.budgets[-1]]['strong_robust']}%", flush=True)
    dial = agg["k1"]["over_accept_pct"], agg["kbase"]["over_accept_pct"]
    print(f"\n  DIAL-ABILITY: over-accept mass {dial[0]:.1f}% (k=1 base test) -> {dial[1]:.1f}% (all base tests) "
          f"-> 0% at strong (base+plus by definition). Standard benchmark, {len(per)} problems.", flush=True)
    print(f"  tokens: prompt={client.total_prompt_tokens} completion={client.total_completion_tokens}", flush=True)
    json.dump({"model": a.model, "dataset": a.dataset, "n_problems": len(per), "config": vars(a), "agg": agg},
              open(os.path.join(a.outdir, f"{tag}_summary.json"), "w"), indent=1)
    print(f"  wrote {os.path.join(a.outdir, tag+'_summary.json')}", flush=True)


if __name__ == "__main__":
    main()
