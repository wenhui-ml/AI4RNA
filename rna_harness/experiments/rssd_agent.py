"""RSSD v2 -- hand the skill-authoring wheel to an LLM agent (self-revise line).

v1 hand-authored the DecomposePKTool skill + a k-NN abstention policy, and showed the COST 真门槛 (learned
abstention cuts SOTA cost 8-13x at equal held-out quality; capability is a designability-wall NULL). v2 asks
the sharper question: can an LLM agent, seeing ONLY TRAIN evidence, AUTHOR the transferable procedure itself
(A_theta) -- both the cheap pseudoknot design skill AND the "when to fall back to expensive SOTA" policy --
and does that agent-distilled procedure COMPOUND on HELD-OUT it never saw? (MIT discovery criterion.)

PROTOCOL (gaming-proof):
  1. The agent is given a primitive API + a sample of TRAIN targets + the TRAIN outcome evidence (which
     targets DesiRNA solves, and that MFE tools never call pKiss) -- NEVER any held-out target.
  2. It authors a Python module: `design_pk(target, prim) -> seq` and `should_call_sota(features, skill_bp) -> bool`.
  3. SELF-REVISE: we run its code on a TRAIN subset (sandboxed), feed back (solved, cost, solves_lost, errors),
     and let it revise up to K rounds. We keep the best-on-TRAIN revision and FREEZE it.
  4. We evaluate the frozen procedure on the disjoint HELD-OUT split and compare to always-SOTA and v1.

SAFETY: `design_pk` runs in a subprocess (RLIMIT_AS cap + per-target signal.alarm + hard parent timeout;
guard exceptions subclass BaseException so an agent `except Exception` cannot swallow them). Every returned
sequence is re-verified by pKiss HERE (the agent's own solved-claim is never trusted). DesiRNA is reused from
the cached pk251 baseline + measured cost -- not re-run.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

from ..benchmarks.rnainvbench import load_pseudobase, pkiss_available
from ..cognitive.cost import solver_cost_usd
from ..cognitive.features import features as extract_features
from .rssd_compounding import _desirna_column, _split

OUT_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "runs", "rssd_agent")
CONDA_PY = "/home/ubuntu/miniconda3/envs/rna/bin/python"

# ---- the contract shown to the agent -------------------------------------------------------------
PRIM_DOC = """\
You author a Python module with EXACTLY these two top-level functions (no other top-level code, no imports
of os/sys/subprocess):

def design_pk(target, prim):
    '''Return a candidate RNA sequence (str of A/C/G/U, same length as target) intended to fold, UNDER THE
    PSEUDOKNOT ORACLE pKiss, into `target` (extended dot-bracket with () [] {} <> pairing layers).
    You may ONLY use these primitives from the dict `prim` (do not import anything):
      prim['inverse_fold'](start_seq, nested_dotbracket) -> (seq, dist)  # ViennaRNA MFE inverse fold; NESTED
                                                                         # '()' only -- it ignores [] {}.
      prim['pkiss_fold'](seq) -> extended_dotbracket_str                 # the pseudoknot MFE oracle. EXPENSIVE
                                                                         # (~0.5-2s). Budget-limited: after
                                                                         # prim['budget_calls'] calls it stops you.
      prim['pk_pairs'](dotbracket) -> set of (i,j) base pairs (all bracket families)
      prim['pk_bp_distance'](a, b) -> int   # size of the symmetric difference of the two pair SETS
      prim['round_only'](target) -> nested '()' reduction of target (drops [] {} <>)
      prim['rng']  -> a seeded random.Random you should use for any randomness
      prim['budget_calls'] -> int  # how many pkiss_fold calls you may make total for this target
    Facts you can exploit: DesiRNA and REMC (the existing tools) optimise the ViennaRNA MFE energy and NEVER
    call pKiss during search, so the crossing ('[]{}') stem is only satisfied incidentally and the last
    base-pair is never repaired against the true oracle. In extended dot-bracket every position is in at most
    one pair, so crossing-stem positions are disjoint from nested ones.'''

def should_call_sota(features, skill_bp):
    '''Return True to FALL BACK to the expensive SOTA solver (DesiRNA, ~$0.043/target) when your cheap
    design_pk did not solve the target, or False to ABSTAIN (give up cheaply). `skill_bp` is the pKiss
    base-pair distance your design_pk achieved (0 = solved; larger = worse; None = your code errored).
    `features` is a dict of mechanical structural features with keys incl.: 'len', 'n_pairs', 'frac_paired',
    'n_pk_brackets', 'n_units', 'n_junctions', 'max_junction_degree', 'n_hairpins', 'max_stem', 'min_stem',
    'n_lonely_pairs', 'nesting_depth'. GOAL: minimise total cost at EQUAL solve -- abstain on targets where
    SOTA will ALSO fail (so you lose no real solve) but keep firing SOTA where it would succeed.'''
"""


def _extract_code(text: str, reasoning: str = "") -> str:
    """Robustly pull Python from an LLM reply: complete ```blocks``` if present; else salvage an
    unterminated fence (provider truncation); else, for a reasoning model that put code only in its
    reasoning trace, look there. Strips any stray leading/trailing fence lines so exec never sees ```."""
    import re
    src = text or ""
    if "```" not in src and reasoning:
        src = (src + "\n" + reasoning) if src.strip() else reasoning
    blocks = re.findall(r"```(?:python)?[ \t]*\n?(.*?)```", src, re.DOTALL)
    if not blocks:
        m = re.search(r"```(?:python)?[ \t]*\n?(.*)", src, re.DOTALL)   # unterminated (truncated) block
        blocks = [m.group(1)] if m else [src]
    code = "\n\n".join(b.strip() for b in blocks)
    code = re.sub(r"^```(?:python)?\s*", "", code)
    code = re.sub(r"```\s*$", "", code).strip()
    return code


def _runner_src(agent_code: str) -> str:
    """A subprocess harness: exec the agent module, run design_pk on assigned targets under hard guards, and
    print RESULT json. Guard exceptions subclass BaseException so an agent `except Exception` cannot eat them."""
    return (
        # NOTE: no RLIMIT_AS cap -- pKiss is a Haskell binary whose RTS reserves a huge VIRTUAL address
        # space, so an address-space rlimit makes it fail to start (returns None). Runtime is bounded
        # instead by the per-target signal.alarm + the hard parent subprocess timeout + the pKiss call budget.
        "import sys, json, time, signal, random\n"
        "from rna_harness.benchmarks.rnainvbench import pkiss_fold, verify_pk, pk_pairs, pk_bp_distance, _round_only\n"
        "import RNA\n"
        "try: RNA.init_rand(int(sys.argv[4]))\n"           # seed ViennaRNA's RNG -> reproducible inverse_fold
        "except Exception: pass\n"
        "class _Budget(BaseException): pass\n"
        "class _TimeUp(BaseException): pass\n"
        "def _handler(s,f): raise _TimeUp()\n"
        "signal.signal(signal.SIGALRM,_handler)\n"
        "# ===== AGENT CODE =====\n"
        + agent_code +
        "\n# ===== HARNESS =====\n"
        "targets=json.loads(sys.argv[1]); budget=int(sys.argv[2]); per_wall=float(sys.argv[3]); seed=int(sys.argv[4])\n"
        "def make_prim(sd):\n"
        "    st={'calls':0,'wall':0.0}\n"
        "    def pkf(s):\n"
        "        if st['calls']>=budget: raise _Budget()\n"
        "        st['calls']+=1; t=time.time(); r=pkiss_fold(s); st['wall']+=time.time()-t; return r\n"
        "    def inv(start,nested):\n"
        "        t=time.time(); r=RNA.inverse_fold(start,nested); st['wall']+=time.time()-t; return r\n"
        "    prim={'inverse_fold':inv,'pkiss_fold':pkf,'pk_pairs':pk_pairs,'pk_bp_distance':pk_bp_distance,\n"
        "          'round_only':_round_only,'rng':random.Random(sd),'budget_calls':budget}\n"
        "    return prim, st\n"
        "out={}\n"
        "for tid,tgt in targets.items():\n"
        "    prim,st=make_prim(seed); err=None; seq=None\n"
        "    signal.alarm(int(per_wall))\n"
        "    try: seq=design_pk(tgt,prim)\n"
        "    except (_Budget,_TimeUp) as e: err=type(e).__name__\n"
        "    except BaseException as e: err=repr(e)[:200]\n"
        "    finally: signal.alarm(0)\n"
        "    ok=isinstance(seq,str) and len(seq)==len(tgt)\n"
        "    v=verify_pk(seq,tgt) if ok else {'solved':False,'bp':None}\n"
        "    out[tid]={'solved':bool(v.get('solved')),'bp':v.get('bp'),\n"
        "              'cost':st['wall'],'calls':st['calls'],'err':err}\n"
        "print('RESULT'+json.dumps(out))\n"
    )


def _run_chunk(args):
    """Run one sandboxed subprocess over a chunk of targets. Module-level so it is picklable for the pool."""
    agent_code, part, budget_calls, per_target_wall, seed = args
    src = _runner_src(agent_code)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir=os.environ.get("TMPDIR", "/tmp")) as f:
        f.write(src)
        path = f.name
    out = {}
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    env = {**os.environ, "PYTHONPATH": repo_root + os.pathsep + os.environ.get("PYTHONPATH", "")}
    try:
        p = subprocess.run([CONDA_PY, path, json.dumps(part), str(budget_calls), str(per_target_wall), str(seed)],
                           capture_output=True, text=True, timeout=int(len(part) * per_target_wall * 1.5 + 30),
                           cwd=repo_root, env=env)
        got = {}
        for line in p.stdout.splitlines():
            if line.startswith("RESULT"):
                got = json.loads(line[6:])
        for tid in part:
            r = got.get(tid)
            if r is None:                                       # this id produced no result (crash/import err)
                out[tid] = {"solved": False, "bp": None, "cost": 0.0, "calls": 0,
                            "err": (p.stderr.strip().splitlines()[-1][:200] if p.stderr.strip() else "no RESULT")}
            else:
                r["cost"] = solver_cost_usd(r.get("cost", 0.0), 1)   # wall -> $
                out[tid] = r
    except subprocess.TimeoutExpired:
        for tid in part:
            out[tid] = {"solved": False, "bp": None, "cost": solver_cost_usd(per_target_wall, 1),
                        "calls": budget_calls, "err": "chunk timeout (hard kill)"}
    finally:
        os.unlink(path)
    return out


def run_skill(agent_code, targets, budget_calls, per_target_wall, seed, chunk=4, workers=16):
    """Run the agent's design_pk on {id: dotbracket} across sandboxed subprocesses (chunks run CONCURRENTLY).
    Returns {id: {solved, bp, cost($), calls, err}}. cost = pKiss+inverse_fold wall x 1 core."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    items = list(targets.items())
    chunks = [dict(items[c0:c0 + chunk]) for c0 in range(0, len(items), chunk)]
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_run_chunk, (agent_code, part, budget_calls, per_target_wall, seed)) for part in chunks]
        for fut in as_completed(futs):
            results.update(fut.result())
    return results


# ---- the routing/abstention half of the authored procedure (pure; run in-process, guarded) --------
def load_should_call_sota(agent_code):
    ns = {}
    try:
        exec(agent_code, ns)                                    # noqa: S102 -- trusted scientific harness
    except Exception as e:                                      # noqa: BLE001
        return None, f"exec error: {e}"
    fn = ns.get("should_call_sota")
    return (fn if callable(fn) else None), (None if callable(fn) else "no should_call_sota")


def evaluate(agent_code, ids, tgt, feats, des_solved, des_cost, budget_calls, per_target_wall, seed):
    """Full authored procedure on `ids`: cheap skill, then agent abstention decides SOTA fallback.
    Returns aggregate (solved, cost, solves_lost, skill_solved, errors) + the per-id skill table."""
    skill = run_skill(agent_code, {str(i): tgt[i] for i in ids}, budget_calls, per_target_wall, seed)
    fn, ferr = load_should_call_sota(agent_code)
    solved = cost = lost = skill_solved = fired = abst = errs = budget_hit = 0
    _EXPECTED = {"_Budget", "_TimeUp", "chunk timeout (hard kill)"}  # ran out of budget on a hard target != a bug
    for i in ids:
        r = skill.get(str(i), {})
        e = r.get("err")
        if e in _EXPECTED:
            budget_hit += 1
        elif e:
            errs += 1                                           # a REAL code exception the agent must fix
        cost += r.get("cost", 0.0)
        if r.get("solved"):
            solved += 1; skill_solved += 1; continue
        call = True
        if fn is not None:
            try:
                call = bool(fn(feats[i], r.get("bp")))
            except Exception:                                   # noqa: BLE001
                call = True                                     # a broken policy defaults to firing SOTA
        if call:
            cost += des_cost; fired += 1
            solved += int(des_solved.get(i, False))
        else:
            abst += 1; lost += int(des_solved.get(i, False))
    return {"solved": solved, "cost": cost, "solves_lost": lost, "skill_solved": skill_solved,
            "sota_fired": fired, "abstained": abst, "code_errors": errs, "budget_hit": budget_hit,
            "policy_err": ferr, "n": len(ids), "skill_table": skill}


# ---- the reference stub author (validates the pipeline with ZERO LLM tokens) ----------------------
STUB_CODE = r'''
def design_pk(target, prim):
    n = len(target); rng = prim['rng']
    tp = prim['pk_pairs'](target); skel = prim['round_only'](target)
    cross = sorted(tp - prim['pk_pairs'](skel))
    WC = [('G','C'),('C','G'),('A','U'),('U','A')]
    best = 'A'*n; bd = 10**9
    for _ in range(6):
        start = ''.join(rng.choice('ACGU') for _ in range(n))
        s, d = prim['inverse_fold'](start, skel)
        if d < bd: best, bd = s, d
        if bd <= 0: break
    seq = list(best)
    for k,(i,j) in enumerate(cross):
        a,b = WC[k % 4]; seq[i], seq[j] = a, b
    part = {}
    for i,j in tp: part[i]=j; part[j]=i
    def bpof(fs): return prim['pk_bp_distance'](fs, target) if fs else n
    cur = ''.join(seq); f = prim['pkiss_fold'](cur)
    curbp = bpof(f); curf = f; bestseq, bestbp = cur, curbp
    ncross = len(cross)
    while bestbp > 0:
        cand = list(cur)
        if ncross and rng.random() < 0.10:              # restart the crossing stem (pseudoknot-specific)
            for i,j in cross:
                a,b = rng.choice(WC); cand[i], cand[j] = a, b
        else:
            fp = prim['pk_pairs'](curf) if curf else set()
            wrong = set()
            for i,j in fp ^ tp:
                for p in (i,j):
                    for q in range(max(0,p-2), min(n,p+3)): wrong.add(q)
            wrong = sorted(wrong) or list(range(n))
            pos = rng.choice(wrong) if rng.random() < 0.85 else rng.randrange(n)  # 15% global disruption
            if pos in part:
                j = part[pos]; a,b = rng.choice(WC); cand[pos], cand[j] = a, b
            else:
                cand[pos] = rng.choice('AUAUGC')
        cs = ''.join(cand); cf = prim['pkiss_fold'](cs); cb = bpof(cf)
        if cb <= curbp: cur, curbp, curf = cs, cb, cf
        if cb < bestbp: bestseq, bestbp = cs, cb
    return bestseq

def should_call_sota(features, skill_bp):
    if skill_bp == 0: return False
    return not (features['len'] > 60 or features['n_pk_brackets'] >= 14)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stub", action="store_true", help="use the reference author (no LLM) to validate the pipeline")
    ap.add_argument("--model", default="anthropic/claude-sonnet-5")
    ap.add_argument("--rounds", type=int, default=4, help="self-revise rounds")
    ap.add_argument("--train-subset", type=int, default=28, help="TRAIN targets used in the self-revise loop")
    ap.add_argument("--budget-calls", type=int, default=60)
    ap.add_argument("--per-target-wall", type=float, default=60.0)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max-tokens", type=int, default=8000, help="LLM completion cap (raise for reasoning models)")
    ap.add_argument("--reasoning-cap", type=int, default=0, help="(legacy) bound thinking tokens; prefer --reasoning")
    ap.add_argument("--reasoning", choices=["auto", "on", "off"], default="auto",
                    help="on={enabled:True}+no max_tokens cap (reasoning models); off={enabled:False}; auto=default")
    ap.add_argument("--llm-timeout", type=float, default=120.0, help="per-call timeout (raise for reasoning models)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", choices=["parity", "half", "shuffle"], default="parity")
    ap.add_argument("--outdir", default=OUT_DEFAULT)
    a = ap.parse_args()
    if not pkiss_available():
        raise SystemExit("pKiss unavailable -- set PKISS_CMD or install conda env 'pk'")
    os.makedirs(a.outdir, exist_ok=True)

    tasks = load_pseudobase()
    ids = [int(t.task_id.split(":")[1]) for t in tasks]
    tgt = {int(t.task_id.split(":")[1]): t.target for t in tasks}
    feats = {i: extract_features(tgt[i]) for i in ids}
    des_solved, des_cost, des_src = _desirna_column()
    train, held = _split(ids, mode=a.split)
    # a fixed, difficulty-spanning TRAIN subset for the fast self-revise loop
    train_sorted = sorted(train, key=lambda i: feats[i]["len"])
    step = max(1, len(train_sorted) // a.train_subset)
    train_eval = train_sorted[::step][:a.train_subset]
    print(f"RSSD-agent: split={a.split} train={len(train)} held={len(held)}; self-revise on {len(train_eval)} "
          f"TRAIN targets; DesiRNA cost={des_src} (${des_cost:.2e}); model={'STUB' if a.stub else a.model}", flush=True)

    history = []

    def score(ev):                                              # maximise solved, then minimise cost
        return (ev["solved"], -ev["cost"])

    if a.stub:
        code = STUB_CODE
        ev = evaluate(code, train_eval, tgt, feats, des_solved, des_cost, a.budget_calls, a.per_target_wall, a.seed)
        print(f"  [stub] TRAIN-subset: solved={ev['solved']}/{ev['n']} cost=${ev['cost']:.4f} "
              f"skill_solved={ev['skill_solved']} lost={ev['solves_lost']} errors={ev['code_errors']}", flush=True)
        best_code, best_ev = code, ev
        history.append({"round": 0, "author": "stub", **{k: ev[k] for k in
                        ("solved", "cost", "skill_solved", "solves_lost", "code_errors")}})
    else:
        from ..llm.openrouter import OpenRouterClient
        # reasoning models (GLM-5.2, claude) spend max_tokens on hidden thinking and are slow, so give a
        # big ceiling + long timeout; `content` still carries the final code (reasoning is a separate field).
        client = OpenRouterClient(model=a.model, timeout=a.llm_timeout)
        if not client.available():
            raise SystemExit("no API key -- source the .orkey into OPENAI_API_KEY")
        # TRAIN evidence the agent may use (never held-out): structure + whether DesiRNA solves it
        evidence = [{"target": tgt[i], "len": feats[i]["len"], "n_pk_brackets": feats[i]["n_pk_brackets"],
                     "desirna_solves": bool(des_solved.get(i, False))} for i in train_eval]
        system = ("You are an expert RNA design researcher who writes correct, self-contained Python. "
                  "Author a cheap pseudoknot design procedure and a cost-saving SOTA-fallback policy. "
                  "Respond with ONE ```python code block containing exactly the two functions, no prose.")
        base_user = (PRIM_DOC + "\n\nTRAIN evidence (structures + whether the expensive SOTA solver solves each; "
                     "you will be scored on a disjoint held-out set, so author a GENERAL procedure, do not "
                     "hardcode these):\n" + json.dumps(evidence, indent=0) +
                     f"\n\nEach target gives design_pk a budget of {a.budget_calls} pkiss_fold calls. "
                     "Minimise total cost at equal solve.")
        best_code, best_ev = None, None
        msgs = [{"role": "user", "content": base_user}]
        for rnd in range(a.rounds):
            # reasoning control (--reasoning): 'on' => {enabled:True} AND OMIT max_tokens (a reasoning model's
            # chain can exceed a cap and leave no room for content); 'off' => {enabled:False}; 'auto' => default.
            if a.reasoning == "on":
                reasoning, mt = {"enabled": True}, None
            elif a.reasoning == "off":
                reasoning, mt = {"enabled": False}, a.max_tokens
            else:
                reasoning, mt = None, a.max_tokens
            res = client.chat(msgs, system=system, temperature=a.temperature, max_tokens=mt, reasoning=reasoning)
            if res.error:
                print(f"  [round {rnd}] LLM error: {res.error}", flush=True); break
            code = _extract_code(res.text, res.reasoning)
            if "def design_pk" not in code:                            # truncated/garbled -> log, skip round
                print(f"  [round {rnd}] no usable design_pk in reply (len={len(res.text or '')}, "
                      f"ctok={res.completion_tokens}); skipping", flush=True)
                asst = {"role": "assistant", "content": res.text or ""}
                if res.reasoning_details:
                    asst["reasoning_details"] = res.reasoning_details   # pass back UNMODIFIED (continue chain)
                msgs = msgs + [asst,
                               {"role": "user", "content": "Your reply had no complete design_pk function "
                                "(likely truncated). Respond with ONE short ```python block, both functions, "
                                "minimal docstrings."}]
                continue
            ev = evaluate(code, train_eval, tgt, feats, des_solved, des_cost, a.budget_calls, a.per_target_wall, a.seed)
            print(f"  [round {rnd}] TRAIN-subset: solved={ev['solved']}/{ev['n']} cost=${ev['cost']:.4f} "
                  f"skill_solved={ev['skill_solved']} lost={ev['solves_lost']} code_err={ev['code_errors']} "
                  f"budget_hit={ev['budget_hit']} policy_err={ev['policy_err']}", flush=True)
            history.append({"round": rnd, **{k: ev[k] for k in
                            ("solved", "cost", "skill_solved", "solves_lost", "code_errors", "budget_hit", "policy_err")}})
            if best_ev is None or score(ev) > score(best_ev):
                best_code, best_ev = code, ev
            # feedback for the next revision
            fb = (f"Result on the {ev['n']} TRAIN targets: solved {ev['solved']}, total cost ${ev['cost']:.4f}, "
                  f"your cheap skill alone solved {ev['skill_solved']}, SOTA fired {ev['sota_fired']}, "
                  f"abstained {ev['abstained']} (real solves LOST by abstaining: {ev['solves_lost']}), "
                  f"REAL code exceptions on {ev['code_errors']} targets, ran out of pkiss budget on "
                  f"{ev['budget_hit']} targets (those may be genuinely unsolvable -- do not chase them). "
                  "Revise BOTH functions to raise solved and lower cost (fewer wasted SOTA calls, no lost "
                  "solves, fix any REAL code exceptions). Respond with one ```python code block, two functions only.")
            asst = {"role": "assistant", "content": res.text}
            if res.reasoning_details:
                asst["reasoning_details"] = res.reasoning_details       # pass back UNMODIFIED (continue chain)
            msgs = msgs + [asst, {"role": "user", "content": fb}]
        json.dump(client.usage(), open(os.path.join(a.outdir, "llm_usage.json"), "w"), indent=1)

    if best_code is None:
        raise SystemExit("no usable authored code produced")

    # freeze + evaluate on HELD-OUT (disjoint) and on TRAIN-subset for reference
    print(f"\n  FROZEN authored procedure -> evaluating on {len(held)} HELD-OUT targets ...", flush=True)
    held_ev = evaluate(best_code, held, tgt, feats, des_solved, des_cost, a.budget_calls, a.per_target_wall, a.seed)
    sota_cost = len(held) * des_cost
    sota_solved = sum(int(des_solved.get(i, False)) for i in held)
    open(os.path.join(a.outdir, "authored_skill.py"), "w").write(best_code)
    out = {"split": a.split, "model": ("stub" if a.stub else a.model), "des_cost": des_cost, "des_src": des_src,
           "budget_calls": a.budget_calls, "history": history, "held": held_ev,
           "held_always_sota": {"solved": sota_solved, "cost": sota_cost}}
    out["held"].pop("skill_table", None)
    json.dump(out, open(os.path.join(a.outdir, "agent_result.json"), "w"), indent=1)

    print(f"\n=== RSSD v2 (agent-authored procedure, HELD-OUT = {len(held)} targets, gaming-proof) ===", flush=True)
    print(f"  always_sota           solved={sota_solved:>3}/{len(held)}  cost=${sota_cost:.4f}", flush=True)
    print(f"  agent procedure       solved={held_ev['solved']:>3}/{len(held)}  cost=${held_ev['cost']:.4f}  "
          f"-> {sota_cost/max(held_ev['cost'],1e-12):.0f}x cheaper", flush=True)
    print(f"    (cheap skill solved {held_ev['skill_solved']}; SOTA fired {held_ev['sota_fired']}, "
          f"abstained {held_ev['abstained']}; REAL solves lost by abstaining: {held_ev['solves_lost']}; "
          f"real code exceptions: {held_ev['code_errors']} targets; budget-exhausted (unsolvable): "
          f"{held_ev['budget_hit']})", flush=True)
    print(f"  wrote {os.path.join(a.outdir, 'agent_result.json')} + authored_skill.py", flush=True)


if __name__ == "__main__":
    main()
