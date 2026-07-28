#!/usr/bin/env python3
"""Experiment A: is GLM-5.2's authored crossing operator's "91/125" real capability, or pKiss false accepts?

THE CLAIM UNDER TEST. runs/rssd_agent_z-ai_glm-5.2/agent_result.json:

    always_sota (DesiRNA)   solved 96/125   $5.4199
    agent procedure         solved 91/125   $0.0870      -> 62x cheaper, 5 solves lost
      (the cheap authored skill solved 91 of those itself; SOTA fired once)

That "91" is adjudicated by pKiss ALONE -- the same oracle the operator calls INSIDE its own search loop.
Today's sota_gate experiment showed what that can be worth: plain DesiRNA claims 43/60 crossing targets under
pKiss, and under three-oracle adjudication 43 of 43 are false accepts. The chomsky-ascent MCF operator
reproduced it: 43/60 under pKiss, 1/60 panel-unanimous. An operator that both SEARCHES and is SCORED by the
same heuristic has an obvious way to win that has nothing to do with designing RNA.

This operator has a specific reason to be suspect. `fix_crossing` in the authored code slams every crossing
pair to G-C:

    for i, j in crossing_pairs: s_list[i] = 'G'; s_list[j] = 'C'

A GC-saturated crossing stem is exactly the kind of sequence where pKiss's shape-model and RNAstructure's
partition-function/MFE models are most likely to part ways.

FALSIFIER, STATED FIRST. If the panel-unanimous count is >= ~70% of 91, the 91 is real capability, the 62x
cost advantage stands on a real solve count, and "LLM-authored operators lose on accuracy" is wrong on this
benchmark. If it collapses toward the 1/60 of the ascent artefact, the 62x is an artefact of scoring an
operator with its own in-loop oracle, and the accuracy question is settled against the operator.

WHAT IS NOT RECOMPUTED. The DesiRNA comparator columns already exist per-target and are read from the
archive, not re-run:
  * agent/desirna_panel_pool.json  (runs/axis1_oracle_panel)     -- best-of-candidate-pool, 251 targets
  * agent/desirna_panel_fixed.json (runs/axis1_desirna_fixedcand) -- ONE fixed candidate, 250 targets
The fixed-candidate column is the like-for-like comparator, because this operator returns ONE sequence.

REPRODUCTION FIDELITY. The runner below is copied verbatim from the archived rssd_agent._runner_src; the ONLY
change is that the per-target record now carries 'seq'. Not storing the sequence is precisely why the original
run could never be re-adjudicated. Two details must be preserved or the sequences change:
  * chunk = 4. RNA.init_rand(seed) is called ONCE per subprocess, so the 2nd/3rd/4th target in a chunk sees an
    advanced ViennaRNA global RNG. chunk=1 would give different (equally valid, but not comparable) designs.
  * target order = sorted odd ids, grouped consecutively in fours, exactly as run_skill built them.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)

CONDA_PY = "/home/ubuntu/miniconda3/envs/rna/bin/python"
AGENT = os.path.join(HERE, "agent")
SKILL = os.path.join(AGENT, "crossing_skill_glm52.py")
OUT = os.path.join(HERE, "agent_panel")
# The generated runner must NOT be written into /tmp. Python puts a script's own directory at sys.path[0],
# and this machine's /tmp holds 1937 stray .py files from past sessions, two of which shadow stdlib modules
# (/tmp/inspect.py, /tmp/turtle.py). With the archived harness's `dir=TMPDIR or /tmp`, every target failed
# with "module 'inspect' has no attribute 'signature'" -- a silent, total failure that looks like the agent's
# code erroring. A dedicated empty directory keeps sys.path[0] clean.
RUNDIR = os.path.join(OUT, "_runner")


# --------------------------------------------------------------------- the archived runner, +1 field
def _runner_src(agent_code: str) -> str:
    """Verbatim from rna-specialization:rna_harness/experiments/rssd_agent.py::_runner_src, except that the
    output record gains 'seq'. Guard exceptions subclass BaseException so an agent `except Exception` cannot
    eat them; no RLIMIT_AS cap because pKiss is a Haskell binary that reserves a huge virtual address space."""
    return (
        "import sys, json, time, signal, random\n"
        "from rna_harness.benchmarks.rnainvbench import pkiss_fold, verify_pk, pk_pairs, pk_bp_distance, _round_only\n"
        "import RNA\n"
        "try: RNA.init_rand(int(sys.argv[4]))\n"
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
        "              'cost':st['wall'],'calls':st['calls'],'err':err,\n"
        "              'seq':(seq if isinstance(seq,str) else None)}\n"   # <-- the only addition
        "print('RESULT'+json.dumps(out))\n"
    )


def _design_chunk(args):
    """One sandboxed subprocess over a chunk of targets, then panel-adjudicate its sequences in-process.
    Writes ONE json per chunk so a kill cannot lose the run (agent_curve.py lost 72 minutes that way)."""
    agent_code, part, budget_calls, per_wall, seed, cidx = args
    dest = os.path.join(OUT, f"c{cidx:03d}.json")
    if os.path.exists(dest):
        return dest
    from rna_harness.cognitive.cost import solver_cost_usd
    from rna_harness.oracles_pk import forms_target, probknot_pairs, shapeknots_pairs

    os.makedirs(RUNDIR, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir=RUNDIR) as f:
        f.write(_runner_src(agent_code))
        path = f.name
    env = {**os.environ, "PYTHONPATH": REPO + os.pathsep + os.environ.get("PYTHONPATH", "")}
    got = {}
    try:
        p = subprocess.run([CONDA_PY, path, json.dumps(part), str(budget_calls), str(per_wall), str(seed)],
                           capture_output=True, text=True,
                           timeout=int(len(part) * per_wall * 1.5 + 30), cwd=REPO, env=env)
        for line in p.stdout.splitlines():
            if line.startswith("RESULT"):
                got = json.loads(line[6:])
        stderr_tail = (p.stderr.strip().splitlines()[-1][:200] if p.stderr.strip() else "no RESULT")
    except subprocess.TimeoutExpired:
        stderr_tail = "chunk timeout (hard kill)"
    finally:
        os.unlink(path)

    recs = {}
    for tid, tgt in part.items():
        r = got.get(tid)
        if r is None:
            recs[tid] = {"solved": False, "bp": None, "cost": 0.0, "calls": 0,
                         "err": stderr_tail, "seq": None, "n": len(tgt)}
            continue
        r["cost"] = solver_cost_usd(r.get("cost", 0.0), 1)        # wall -> $, as the archive did
        r["n"] = len(tgt)
        seq = r.get("seq")
        if seq:
            # pKiss verdict already computed by verify_pk inside the runner (the ORIGINAL criterion).
            # Add the two independent oracles. None = that oracle failed to run -> recorded, never guessed.
            t0 = time.time()
            r["probknot"] = forms_target(probknot_pairs(seq), tgt)
            r["shapeknots"] = forms_target(shapeknots_pairs(seq), tgt)
            r["panel_s"] = round(time.time() - t0, 2)
        else:
            r["probknot"] = r["shapeknots"] = None
            r["panel_s"] = 0.0
        recs[tid] = r
    with open(dest, "w") as f:
        json.dump(recs, f)
    return dest


# --------------------------------------------------------------------------------- comparator columns
def _desirna_columns(held):
    """Read the ARCHIVED DesiRNA panel columns; subset to the held-out ids. Nothing is re-folded."""
    cols = {}
    pool_p = os.path.join(AGENT, "desirna_panel_pool.json")
    fixed_p = os.path.join(AGENT, "desirna_panel_fixed.json")
    if os.path.exists(pool_p):
        rows = json.load(open(pool_p))["rows"]
        d = {int(r["task_id"].split(":")[1]): r for r in rows}
        cols["desirna_pool"] = {i: {"pkiss": d[i].get("forms_T_pkiss"),
                                    "probknot": d[i].get("forms_T_probknot"),
                                    "shapeknots": d[i].get("forms_T_shapeknots")}
                                for i in held if i in d}
    if os.path.exists(fixed_p):
        rows = json.load(open(fixed_p))["rows"]
        d = {int(r["id"].split(":")[1]): r for r in rows}
        cols["desirna_fixed"] = {i: {k: (d[i][k].get("exact") if isinstance(d[i].get(k), dict) else None)
                                     for k in ("pkiss", "probknot", "shapeknots")}
                                 for i in held if i in d}
    return cols


def _tally(name, col, held):
    """(#pkiss, #probknot, #shapeknots, #unanimous, #scored) over the held ids present in `col`."""
    ids = [i for i in held if i in col]
    g = lambda i, k: col[i].get(k) is True
    pk = sum(g(i, "pkiss") for i in ids)
    pb = sum(g(i, "probknot") for i in ids)
    sk = sum(g(i, "shapeknots") for i in ids)
    un = sum(g(i, "pkiss") and g(i, "probknot") and g(i, "shapeknots") for i in ids)
    return name, pk, pb, sk, un, len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-calls", type=int, default=120)      # from the archived agent_result.json
    ap.add_argument("--per-target-wall", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=4)               # MUST stay 4: see module docstring
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only the first N held-out targets")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    from rna_harness.benchmarks.rnainvbench import load_pseudobase, pkiss_available
    from rna_harness.oracles_pk import rnastructure_available
    if not (pkiss_available() and rnastructure_available()):
        raise SystemExit("need pKiss AND RNAstructure; refusing to report a partial panel")

    tasks = load_pseudobase()
    tgt = {int(t.task_id.split(":")[1]): t.target for t in tasks}
    ids = sorted(tgt)
    held = [i for i in ids if i % 2 == 1]                          # _split(..., 'parity') -> odd = HELD-OUT
    if a.limit:
        # Smoke test only. It truncates the LAST chunks, never re-groups, so the surviving chunks are
        # byte-identical to the full run's and their cached json is reusable by the full run.
        held = held[:a.limit]
    print(f"Experiment A: re-adjudicate GLM-5.2's authored crossing operator under the 3-oracle panel")
    print(f"  pseudobase {len(ids)} targets, parity split -> HELD-OUT {len(held)} (odd ids)")
    print(f"  archived claim on this exact split: 91/125 under pKiss, $0.0870, vs DesiRNA 96/125 at $5.4199")
    print(f"  operator code: {os.path.relpath(SKILL, REPO)} (byte-identical to the archive, never edited)\n",
          flush=True)

    code = open(SKILL).read()
    items = [(str(i), tgt[i]) for i in held]                       # same ordering run_skill used
    chunks = [dict(items[c:c + a.chunk]) for c in range(0, len(items), a.chunk)]

    if not a.analyse_only:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        jobs = [(code, part, a.budget_calls, a.per_target_wall, a.seed, k) for k, part in enumerate(chunks)]
        jobs = [j for j in jobs if not os.path.exists(os.path.join(OUT, f"c{j[5]:03d}.json"))]
        print(f"  {len(chunks)} chunks of {a.chunk}; {len(jobs)} still to run, {a.workers} workers",
              flush=True)
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(_design_chunk, j) for j in jobs]
            for n, _fu in enumerate(as_completed(futs), 1):
                el = time.time() - t0
                print(f"    {n}/{len(jobs)} chunks  {el/60:.1f} min  "
                      f"eta {el/n*(len(jobs)-n)/60:.1f} min", flush=True)

    rec = {}
    for f in sorted(os.listdir(OUT)):
        if f.endswith(".json"):
            rec.update({int(k): v for k, v in json.load(open(os.path.join(OUT, f))).items()})
    if len(rec) < len(held):
        print(f"  WARNING: {len(rec)}/{len(held)} targets present; the table below is partial\n")

    agent_col = {i: {"pkiss": (r.get("solved") is True), "probknot": r.get("probknot"),
                     "shapeknots": r.get("shapeknots")} for i, r in rec.items()}
    cols = {"glm52_operator": agent_col, **_desirna_columns(held)}

    print(f"\n=== forms the TARGET, per oracle, on the SAME {len(held)} held-out crossing targets ===")
    print(f"  {'system':<22} {'pKiss':>7} {'ProbKnot':>9} {'ShapeKnots':>11} {'unanimous':>10} {'n':>5}")
    tallies = {}
    for name, col in cols.items():
        t = _tally(name, col, held)
        tallies[name] = t
        print(f"  {t[0]:<22} {t[1]:>7} {t[2]:>9} {t[3]:>11} {t[4]:>10} {t[5]:>5}")

    ag = tallies["glm52_operator"]
    print(f"\n=== the falsifier ===")
    print(f"  archived pKiss claim            : 91/125")
    print(f"  reproduced pKiss claim          : {ag[1]}/{ag[5]}"
          f"   {'(reproduced)' if abs(ag[1] - 91) <= 3 else '(DOES NOT REPRODUCE -- see note)'}")
    if ag[1]:
        print(f"  panel-unanimous / pKiss claim   : {ag[4]}/{ag[1]} = {100.0*ag[4]/ag[1]:.1f}%")
        print(f"  pre-declared threshold          : >= 70% -> real capability;  "
              f"collapse -> in-loop-oracle artefact")
        print(f"  VERDICT                         : "
              f"{'REAL' if ag[4] >= 0.70 * ag[1] else 'ARTEFACT of the in-loop oracle'}")
    errs = sum(1 for r in rec.values() if r.get("err") not in (None, "_Budget", "_TimeUp"))
    bud = sum(1 for r in rec.values() if r.get("err") in ("_Budget", "_TimeUp"))
    cost = sum(r.get("cost", 0.0) for r in rec.values())
    ok_pb = sum(1 for r in rec.values() if r.get("probknot") is not None)
    ok_sk = sum(1 for r in rec.values() if r.get("shapeknots") is not None)
    print(f"\n  operator CPU cost this run: ${cost:.4f}  (archived: $0.0870)")
    print(f"  budget-exhausted {bud}, real code exceptions {errs}")
    print(f"  oracle coverage: ProbKnot ran on {ok_pb}/{len(rec)}, ShapeKnots on {ok_sk}/{len(rec)}")

    json.dump({"held": held, "n_held": len(held), "budget_calls": a.budget_calls, "seed": a.seed,
               "cost_usd": cost, "budget_hit": bud, "code_errors": errs,
               "tallies": {k: {"pkiss": v[1], "probknot": v[2], "shapeknots": v[3],
                               "unanimous": v[4], "n": v[5]} for k, v in tallies.items()},
               "per_target": rec},
              open(os.path.join(HERE, "agent_panel.json"), "w"), indent=1)
    print(f"\n  wrote {os.path.relpath(os.path.join(HERE, 'agent_panel.json'), REPO)}")


if __name__ == "__main__":
    main()
