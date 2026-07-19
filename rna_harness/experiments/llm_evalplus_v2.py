"""R3-3.5: REPRODUCIBLE HumanEval+ rerun -- fixed open-weights models, ALL artifacts saved.

Fixes the round-2 reproducibility gaps the reviewer flagged: (i) fixed open-weights models (a local pinned
vllm checkpoint + open-weights OpenRouter models), (ii) SAVE every raw completion + prompt + system + params
+ seed, (iii) SAVE per-candidate sandbox diagnostics (extraction ok?, exec status, timeout, exception),
(iv) keep ALL 164 tasks and record any drop with its reason, (v) final flush so no task is lost. The
downstream analyzer (llm_evalplus_v2_analyze.py) then uses a FIXED pre-registered split (not 100 averaged
random splits) and reports over-acceptance, a proportion-rung dial curve, Goodhart-under-selection, and a
risk--coverage--cost curve, all from the saved artifacts (which are re-scorable and inspectable)."""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import openai

from .llm_evalplus import (extract_code, _exec, _eq, _expected_and_run_src,
                           _candidate_run_src)

OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "llm_evalplus_v2")
SYS = "You are a careful Python programmer. Respond with ONE complete Python function in a ```python code block, no prose, no tests."

MODELS = {
    "llama-3.1-8b":    {"kind": "openrouter", "model": "meta-llama/llama-3.1-8b-instruct"},
    "qwen-coder-32b":  {"kind": "openrouter", "model": "qwen/qwen-2.5-coder-32b-instruct", "no_seed": True},
    "qwen36-local":    {"kind": "local", "base_url": "http://127.0.0.1:8001/v1", "model": "qwen36", "no_think": True},
}
# the local qwen36 vllm server is a Qwen3 reasoning model; for code completion we disable thinking (else the
# reasoning trace consumes the token budget and content comes back empty). Round-robin across its 8 replicas.
_QWEN36_PORTS = list(range(8001, 8009))


def _client(spec):
    if spec["kind"] == "local":
        return openai.OpenAI(base_url=spec["base_url"], api_key="none"), spec["model"]
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    return openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key), spec["model"]


def load_all_problems(plus_cap):
    """All 164 HumanEval+ tasks; records drop reason per task (never silently drops)."""
    from evalplus.data import get_human_eval_plus
    raw = get_human_eval_plus()
    items = sorted(raw.items(), key=lambda kv: kv[0])
    probs, dropped = [], []
    for tid, p in items:
        entry, prompt, canon = p["entry_point"], p["prompt"], p.get("canonical_solution", "")
        base = [list(x) for x in p.get("base_input", [])]
        plus = [list(x) for x in p.get("plus_input", [])][:plus_cap]
        if not base:
            dropped.append({"tid": tid, "reason": "no_base_input"}); continue
        exp_base = _exec(_expected_and_run_src(prompt, canon, entry, base), json.dumps(base))
        exp_plus = _exec(_expected_and_run_src(prompt, canon, entry, plus), json.dumps(plus)) if plus else []
        if exp_base is None or (plus and exp_plus is None):
            dropped.append({"tid": tid, "reason": "canonical_reference_no_output"}); continue
        if "__ERR__" in exp_base or "__UNJSON__" in exp_base:
            dropped.append({"tid": tid, "reason": "canonical_reference_error_or_unjson"}); continue
        probs.append({"tid": tid, "entry": entry, "prompt": prompt,
                      "base_in": base, "base_exp": exp_base, "plus_in": plus, "plus_exp": exp_plus or []})
    return probs, dropped


def _exec_diag(src, arg, timeout=8):
    """Like _exec but also returns a status string for sandbox-diagnostic logging."""
    import subprocess, sys, tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write("import resource; resource.setrlimit(resource.RLIMIT_AS,(2**31,2**31))\n" + src)
        path = f.name
    status = "no_output"
    try:
        p = subprocess.run([__import__("sys").executable, path, arg], capture_output=True, text=True, timeout=timeout)
        for line in p.stdout.splitlines():
            if line.startswith("EXP") or line.startswith("OUT"):
                os.unlink(path); return json.loads(line[3:]), "ok"
        status = "nonzero_exit" if p.returncode != 0 else "no_marker"
    except subprocess.TimeoutExpired:
        status = "timeout"
    except Exception:
        status = "exec_exception"
    try:
        os.unlink(path)
    except OSError:
        pass
    return None, status


def eval_candidate_diag(code, prob):
    """Boolean vector over [base..., plus...] plus sandbox diagnostics."""
    entry = prob["entry"]
    extraction_ok = bool(code.strip())
    ob, sb = _exec_diag(_candidate_run_src(code, entry, prob["base_in"]), json.dumps(prob["base_in"]))
    op, sp = (_exec_diag(_candidate_run_src(code, entry, prob["plus_in"]), json.dumps(prob["plus_in"]))
              if prob["plus_in"] else ([], "empty"))
    vb = [_eq(o, e) for o, e in zip(ob, prob["base_exp"])] if ob else [False] * len(prob["base_exp"])
    vp = [_eq(o, e) for o, e in zip(op, prob["plus_exp"])] if op else [False] * len(prob["plus_exp"])
    return vb + vp, len(vb), {"extraction_ok": extraction_ok, "base_status": sb, "plus_status": sp}


def generate(client, model, prompt, temperature, max_tokens, seed, no_think=False, no_seed=False):
    user = prompt + "\n    # Complete the function above. Respond with ONLY the full function in a ```python block."
    kw = {}
    if no_think:                                           # Qwen3 reasoning model: disable thinking for code
        kw["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    if not no_seed:                                        # some OpenRouter providers reject `seed` (400)
        kw["seed"] = seed
    r = client.chat.completions.create(
        model=model, messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user}],
        temperature=temperature, max_tokens=max_tokens, **kw)
    return r.choices[0].message.content or ""


def run(model_key, n_max, temperature, max_tokens, plus_cap, gen_workers, seed, outdir):
    os.makedirs(outdir, exist_ok=True)
    spec = MODELS[model_key]
    client, model = _client(spec)
    probs, dropped = load_all_problems(plus_cap)
    print(f"[{model_key}] {len(probs)} tasks kept, {len(dropped)} dropped; model={model} kind={spec['kind']}", flush=True)
    tag = model_key
    per = []
    for pi, prob in enumerate(probs):
        def gen_one(j):
            for attempt in range(3):
                try:
                    return generate(client, model, prob["prompt"], temperature, max_tokens, seed + j,
                                    no_think=spec.get("no_think", False), no_seed=spec.get("no_seed", False))
                except Exception as e:
                    if attempt == 2:
                        return f"__GEN_ERR__ {type(e).__name__}"
                    time.sleep(2 * (attempt + 1))
        with ThreadPoolExecutor(max_workers=gen_workers) as ex:
            completions = list(ex.map(gen_one, range(n_max)))
        cand = []
        for j, comp in enumerate(completions):
            code = extract_code(comp)
            vec, nb, diag = eval_candidate_diag(code, prob)
            cand.append({"seed": seed + j, "completion": comp, "code": code, "vec": vec, "diag": diag})
        per.append({"tid": prob["tid"], "entry": prob["entry"], "n_base": len(prob["base_exp"]),
                    "n_plus": len(prob["plus_exp"]), "candidates": cand})
        if (pi + 1) % 10 == 0 or pi + 1 == len(probs):
            print(f"  [{model_key}] {pi+1}/{len(probs)}", flush=True)
            json.dump({"model_key": model_key, "model": model, "kind": spec["kind"], "temperature": temperature,
                       "max_tokens": max_tokens, "n_max": n_max, "seed": seed, "plus_cap": plus_cap,
                       "system": SYS, "dropped": dropped, "per": per},
                      open(os.path.join(outdir, f"{tag}.json"), "w"))
    # final flush (never drop the tail)
    json.dump({"model_key": model_key, "model": model, "kind": spec["kind"], "temperature": temperature,
               "max_tokens": max_tokens, "n_max": n_max, "seed": seed, "plus_cap": plus_cap,
               "system": SYS, "dropped": dropped, "per": per},
              open(os.path.join(outdir, f"{tag}.json"), "w"))
    print(f"[{model_key}] DONE -> {os.path.join(outdir, tag+'.json')} ({len(per)} tasks, all completions saved)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--n-max", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--plus-cap", type=int, default=40)
    ap.add_argument("--gen-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=OUT)
    a = ap.parse_args()
    run(a.model, a.n_max, a.temperature, a.max_tokens, a.plus_cap, a.gen_workers, a.seed, a.outdir)
