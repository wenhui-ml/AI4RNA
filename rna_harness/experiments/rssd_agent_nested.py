"""ROUTE A -- does agent method-discovery GENERALISE to a second problem type?

On pseudoknots, a GLM-5.2 agent (rssd_agent.py) independently discovered the crossing-complementarity
procedure. Route A tests whether that is a fluke or a general capability: same self-revise harness, but the
NESTED (Eterna100 / ViennaRNA MFE) domain -- different primitives, different oracle (MFE not pKiss). The agent
authors a cheap nested design procedure + a SOTA-fallback policy, self-revises on TRAIN only, and is evaluated
frozen on a disjoint HELD-OUT set. If the agent authors a WORKING nested skill (solves cheaply) it shows
method-discovery is not PK-specific -- the first step toward a self-expanding, cross-problem skill library.

Reuses rssd_agent's proven pieces (_extract_code, load_should_call_sota, sandboxed subprocess exec, verify-
honest evaluation). DesiRNA (the expensive SOTA) outcomes+cost are read from the learning_vs_cold Eterna table
(snapshotted to avoid a concurrent-writer race with Route B's table build)."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..benchmarks.eterna100 import load_eterna100
from ..cognitive.cost import solver_cost_usd
from ..cognitive.features import FEATURE_NAMES, features as extract_features
from ..cognitive.router import Router
from ..cognitive.tools import DesirnaTool
from .rssd_agent import _extract_code, load_should_call_sota

OUT_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "runs", "rssd_agent_nested")
CONDA_PY = "/home/ubuntu/miniconda3/envs/rna/bin/python"
TABLE = os.path.join(os.path.dirname(__file__), "..", "runs", "learning_vs_cold", "table.json")
SOTA = "desirna@180s"

PRIM_DOC = """\
You author a Python module with EXACTLY these two top-level functions (no imports of os/sys/subprocess):

def design_pk(target, prim):
    '''Return an RNA sequence (str of A/C/G/U, len == len(target)) intended to fold, under the ViennaRNA MFE
    model, EXACTLY into the NESTED dot-bracket `target` (only '(' ')' '.'; no pseudoknots here). Primitives:
      prim['inverse_fold'](start_seq, target) -> (seq, dist)  # ViennaRNA inverse fold; dist = #positions off
      prim['fold'](seq) -> mfe_structure_str                  # ViennaRNA MFE fold (CHEAP, microseconds)
      prim['bp_distance'](structA, structB) -> int            # base-pair distance between two structures
      prim['rng']          -> seeded random.Random
      prim['budget_calls'] -> int  # max fold()+inverse_fold() calls (MFE is cheap, so this is generous)
    The function name is design_pk for interface compatibility; the task here is NESTED design.'''

def should_call_sota(features, skill_bp):
    '''Return True to fall back to the expensive SOTA solver (DesiRNA) when design_pk did not solve, else
    False to ABSTAIN. skill_bp = the MFE bp-distance your design_pk reached (0 = solved; None = errored).
    features keys incl: 'len','n_pairs','frac_paired','n_units','n_junctions','max_junction_degree',
    'n_hairpins','max_stem','min_stem','n_lonely_pairs','n_triloops','nesting_depth'. Minimise total cost at
    equal solve: abstain where SOTA will ALSO fail; keep firing SOTA where it would succeed.'''
"""


def _runner_src(agent_code: str) -> str:
    return (
        "import sys, json, time, signal, random\n"
        "import RNA\n"
        "try: RNA.init_rand(int(sys.argv[4]))\n"
        "except Exception: pass\n"
        "class _Budget(BaseException): pass\n"
        "class _TimeUp(BaseException): pass\n"
        "def _handler(s,f): raise _TimeUp()\n"
        "signal.signal(signal.SIGALRM,_handler)\n"
        "# ===== AGENT CODE =====\n" + agent_code + "\n# ===== HARNESS =====\n"
        "targets=json.loads(sys.argv[1]); budget=int(sys.argv[2]); per_wall=float(sys.argv[3]); seed=int(sys.argv[4])\n"
        "def make_prim(sd):\n"
        "    st={'calls':0,'wall':0.0}\n"
        "    def fold(s):\n"
        "        if st['calls']>=budget: raise _Budget()\n"
        "        st['calls']+=1; t=time.time(); r=RNA.fold(s)[0]; st['wall']+=time.time()-t; return r\n"
        "    def inv(start,tgt):\n"
        "        if st['calls']>=budget: raise _Budget()\n"
        "        st['calls']+=1; t=time.time(); r=RNA.inverse_fold(start,tgt); st['wall']+=time.time()-t; return r\n"
        "    prim={'inverse_fold':inv,'fold':fold,'bp_distance':RNA.bp_distance,\n"
        "          'rng':random.Random(sd),'budget_calls':budget}\n"
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
        "    if ok:\n"
        "        st2=RNA.fold(seq)[0]; solved=(st2==tgt); bp=RNA.bp_distance(st2,tgt)\n"
        "    else: solved=False; bp=None\n"
        "    out[tid]={'solved':bool(solved),'bp':bp,'cost':st['wall'],'calls':st['calls'],'err':err}\n"
        "print('RESULT'+json.dumps(out))\n"
    )


def _run_chunk(args):
    agent_code, part, budget, wall, seed = args
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir=os.environ.get("TMPDIR", "/tmp")) as f:
        f.write(_runner_src(agent_code)); path = f.name
    out = {}
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    env = {**os.environ, "PYTHONPATH": repo + os.pathsep + os.environ.get("PYTHONPATH", "")}
    try:
        p = subprocess.run([CONDA_PY, path, json.dumps(part), str(budget), str(wall), str(seed)],
                           capture_output=True, text=True, timeout=int(len(part) * wall * 1.5 + 30), cwd=repo, env=env)
        got = {}
        for line in p.stdout.splitlines():
            if line.startswith("RESULT"):
                got = json.loads(line[6:])
        for tid in part:
            r = got.get(tid)
            out[tid] = ({"solved": False, "bp": None, "cost": 0.0, "calls": 0,
                         "err": (p.stderr.strip().splitlines()[-1][:200] if p.stderr.strip() else "no RESULT")}
                        if r is None else {**r, "cost": solver_cost_usd(r.get("cost", 0.0), 1)})
    except subprocess.TimeoutExpired:
        for tid in part:
            out[tid] = {"solved": False, "bp": None, "cost": solver_cost_usd(wall, 1), "calls": budget,
                        "err": "chunk timeout"}
    finally:
        os.unlink(path)
    return out


def run_skill(agent_code, targets, budget, wall, seed, chunk=6, workers=16):
    items = list(targets.items())
    chunks = [dict(items[c:c + chunk]) for c in range(0, len(items), chunk)]
    res = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(_run_chunk, (agent_code, c, budget, wall, seed)) for c in chunks]):
            res.update(fut.result())
    return res


def evaluate(code, ids, tgt, feats, des_solved, des_cost, budget, wall, seed):
    skill = run_skill(code, {str(i): tgt[i] for i in ids}, budget, wall, seed)
    fn, ferr = load_should_call_sota(code)
    solved = cost = lost = skill_solved = fired = abst = errs = bhit = 0
    sample_err = None
    _EXP = {"_Budget", "_TimeUp", "chunk timeout"}
    for i in ids:
        r = skill.get(str(i), {})
        e = r.get("err")
        if e in _EXP: bhit += 1
        elif e:
            errs += 1
            if sample_err is None: sample_err = e          # surface a real exception to the agent
        cost += r.get("cost", 0.0)
        if r.get("solved"):
            solved += 1; skill_solved += 1; continue
        call = True
        if fn is not None:
            try: call = bool(fn(feats[i], r.get("bp")))
            except Exception: call = True
        if call:
            cost += des_cost; fired += 1; solved += int(des_solved.get(i, False))
        else:
            abst += 1; lost += int(des_solved.get(i, False))
    return {"solved": solved, "cost": cost, "solves_lost": lost, "skill_solved": skill_solved,
            "sota_fired": fired, "abstained": abst, "code_errors": errs, "budget_hit": bhit,
            "policy_err": ferr, "sample_err": sample_err, "n": len(ids)}


def _eterna_desirna_column():
    """Snapshot the (concurrently-written by Route B) Eterna table, return {pid: solved} + measured
    desirna@180 cost. Retry the snapshot because Route B rewrites the whole file per entry (a copy caught
    mid-write can be truncated/invalid JSON)."""
    import time as _t
    table = None
    for _ in range(20):
        tmp = TABLE + f".snap.{os.getpid()}"
        try:
            shutil.copy(TABLE, tmp)
            table = json.load(open(tmp))
            os.unlink(tmp)
            break
        except Exception:
            try: os.unlink(tmp)
            except Exception: pass
            _t.sleep(0.5)
    if table is None:
        raise SystemExit("could not read a consistent Eterna table snapshot")
    solved, costs = {}, []
    for k, v in table.items():
        if v.get("tool") == SOTA:
            pid = int(k.split(":")[0]); solved[pid] = bool(v["solved"]); costs.append(float(v["cost"]))
    med = sorted(costs)[len(costs) // 2] if costs else solver_cost_usd(180, 10)
    return solved, med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="z-ai/glm-5.2")
    ap.add_argument("--reasoning", choices=["auto", "on", "off"], default="off")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--train-subset", type=int, default=24)
    ap.add_argument("--budget-calls", type=int, default=4000)   # MFE folds are cheap -> generous
    ap.add_argument("--per-target-wall", type=float, default=60.0)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--llm-timeout", type=float, default=360.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=OUT_DEFAULT)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    des_solved, des_cost = _eterna_desirna_column()
    puz = {p["id"]: p["target"] for p in load_eterna100(version="V2")}
    ids = sorted([i for i in des_solved if i in puz])          # only puzzles with a DesiRNA outcome in the table
    tgt = {i: puz[i] for i in ids}
    feats = {i: extract_features(tgt[i]) for i in ids}
    train = [i for i in ids if i % 2 == 0]; held = [i for i in ids if i % 2 == 1]
    tr_sorted = sorted(train, key=lambda i: feats[i]["len"])
    step = max(1, len(tr_sorted) // a.train_subset)
    train_eval = tr_sorted[::step][:a.train_subset]
    print(f"Route A (nested/Eterna agent-authoring): {len(ids)} puzzles with DesiRNA outcome "
          f"(train {len(train)}/held {len(held)}); self-revise on {len(train_eval)}; model={a.model} "
          f"reasoning={a.reasoning}; DesiRNA cost=${des_cost:.2e}", flush=True)

    from ..llm.openrouter import OpenRouterClient
    client = OpenRouterClient(model=a.model, timeout=a.llm_timeout)
    if not client.available():
        raise SystemExit("no API key -- source the .orkey")
    evidence = [{"target": tgt[i], "len": feats[i]["len"], "desirna_solves": bool(des_solved.get(i))}
                for i in train_eval]
    system = ("You are an expert RNA design researcher who writes correct, self-contained Python. Author a "
              "cheap NESTED (MFE) design procedure + a cost-saving SOTA-fallback policy. Respond with ONE "
              "```python code block, exactly the two functions, no prose.")
    base_user = (PRIM_DOC + "\n\nTRAIN evidence (structures + whether expensive SOTA solves each; author a "
                 "GENERAL procedure, do not hardcode):\n" + json.dumps(evidence, indent=0) +
                 f"\n\nBudget {a.budget_calls} fold-calls/target. Minimise total cost at equal solve.")
    best_code, best_ev, history = None, None, []
    msgs = [{"role": "user", "content": base_user}]

    def score(ev): return (ev["solved"], -ev["cost"])

    for rnd in range(a.rounds):
        reasoning = ({"enabled": True} if a.reasoning == "on" else
                     {"enabled": False} if a.reasoning == "off" else None)
        mt = None if a.reasoning == "on" else a.max_tokens
        res = client.chat(msgs, system=system, temperature=a.temperature, max_tokens=mt, reasoning=reasoning)
        if res.error:
            print(f"  [round {rnd}] LLM error: {res.error}", flush=True); break
        code = _extract_code(res.text, res.reasoning)
        if "def design_pk" not in code:
            print(f"  [round {rnd}] no usable code; skipping", flush=True)
            msgs += [{"role": "assistant", "content": res.text or ""},
                     {"role": "user", "content": "No complete design_pk. Respond with ONE short ```python block."}]
            continue
        ev = evaluate(code, train_eval, tgt, feats, des_solved, des_cost, a.budget_calls, a.per_target_wall, a.seed)
        print(f"  [round {rnd}] TRAIN: solved={ev['solved']}/{ev['n']} cost=${ev['cost']:.4f} "
              f"skill_solved={ev['skill_solved']} lost={ev['solves_lost']} code_err={ev['code_errors']}", flush=True)
        history.append({"round": rnd, **{k: ev[k] for k in ("solved", "cost", "skill_solved", "solves_lost", "code_errors")}})
        if best_ev is None or score(ev) > score(best_ev):
            best_code, best_ev = code, ev
        errhint = (f" EXAMPLE ERROR to fix: {ev['sample_err']} (note: prim['rng'] is a random.Random INSTANCE "
                   "and prim['budget_calls'] is an INT -- use them directly, do NOT call them)."
                   if ev.get("sample_err") else "")
        fb = (f"On {ev['n']} TRAIN nested targets: solved {ev['solved']}, cost ${ev['cost']:.4f}, your cheap "
              f"skill solved {ev['skill_solved']}, SOTA fired {ev['sota_fired']}, abstained {ev['abstained']} "
              f"(real lost {ev['solves_lost']}), code errors on {ev['code_errors']} targets.{errhint} Revise BOTH "
              "functions to raise solved + lower cost. ONE ```python block, two functions.")
        asst = {"role": "assistant", "content": res.text}
        if res.reasoning_details: asst["reasoning_details"] = res.reasoning_details
        msgs += [asst, {"role": "user", "content": fb}]
    json.dump(client.usage(), open(os.path.join(a.outdir, "llm_usage.json"), "w"), indent=1)
    if best_code is None:
        raise SystemExit("no usable authored code")

    print(f"\n  FROZEN -> HELD-OUT ({len(held)} nested puzzles) ...", flush=True)
    hv = evaluate(best_code, held, tgt, feats, des_solved, des_cost, a.budget_calls, a.per_target_wall, a.seed)
    sota_cost = len(held) * des_cost; sota_solved = sum(int(des_solved.get(i, False)) for i in held)
    open(os.path.join(a.outdir, "authored_skill.py"), "w").write(best_code)
    json.dump({"model": a.model, "reasoning": a.reasoning, "des_cost": des_cost, "history": history, "held": hv,
               "held_always_sota": {"solved": sota_solved, "cost": sota_cost}},
              open(os.path.join(a.outdir, "agent_result.json"), "w"), indent=1)
    print(f"\n=== ROUTE A (nested agent-authoring, HELD-OUT={len(held)}) ===", flush=True)
    print(f"  always_sota      solved={sota_solved}/{len(held)}  cost=${sota_cost:.4f}", flush=True)
    print(f"  agent procedure  solved={hv['solved']}/{len(held)}  cost=${hv['cost']:.4f}  "
          f"-> {sota_cost/max(hv['cost'],1e-12):.0f}x cheaper", flush=True)
    print(f"    (cheap nested skill solved {hv['skill_solved']}; SOTA fired {hv['sota_fired']}, abstained "
          f"{hv['abstained']}; real solves lost {hv['solves_lost']}; code exceptions {hv['code_errors']})", flush=True)
    print(f"  => agent method-discovery {'GENERALISES to nested' if hv['skill_solved']>0.3*len(held) else 'did NOT generalise'} "
          f"(skill alone solved {hv['skill_solved']}/{len(held)})", flush=True)
    print(f"  wrote {a.outdir}/agent_result.json + authored_skill.py", flush=True)


if __name__ == "__main__":
    main()
