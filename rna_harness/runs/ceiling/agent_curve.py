#!/usr/bin/env python3
"""Measure GLM-5.2's already-authored operator as a CURVE, not a point. No LLM call, so this is free.

What exists. rssd_agent_nested.py had GLM-5.2 author a nested-design procedure, self-revise on TRAIN, and
be scored frozen on a disjoint HELD-OUT set of 37 Eterna targets. The archived result:

    held: solved 24/37, cost $0.038, solves_lost 13, skill_solved 23, abstained 13, budget_hit 13
    always_sota (DesiRNA@180s): 37/37, cost $0.955

so the reported boundary is "24/37 at $0.038 against SOTA's 37/37 at $0.955" -- 25x cheaper, 13 solves
lost. That single number is what the agent was asked for: its prompt says "Budget 4000 fold-calls/target.
Minimise total cost at equal solve." A fixed budget and a cost objective produce a POINT.

Why the point is misleading. budget_hit = 13. Thirteen held-out targets ran OUT OF BUDGET rather than
proving unsolvable, and 13 is exactly the number of solves lost. So 24/37 may be a budget artefact, and
the operator's real capability ceiling -- S_inf, the coordinate today's surface separates from cost -- is
unmeasured.

This re-runs the SAME authored code, unchanged, across a budget sweep, and reports (S_inf, B_half) instead
of one number. Nothing about the agent changes; only the measurement does. If S_inf > 24 the boundary we
published was ours, not the agent's.

Sandboxing and primitives are copied from the original harness so the operator sees an identical interface.
"""
import argparse, json, os, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
CONDA_PY = "/home/ubuntu/miniconda3/envs/rna/bin/python"
SKILL = os.path.join(HERE, "agent", "authored_skill.py")


def _runner_src(code: str) -> str:
    """Subprocess runner: identical primitive dict to rssd_agent_nested's, plus a call counter."""
    return (
        "import json,random,sys,time\n"
        "import RNA\n"
        + code + "\n"
        "def main():\n"
        "    part=json.loads(sys.argv[1]); budget=int(sys.argv[2]); wall=float(sys.argv[3])\n"
        "    sd=int(sys.argv[4])\n"
        "    out=[]\n"
        "    for tid,target in part:\n"
        "        cnt=[0]\n"
        "        def fold(s):\n"
        "            cnt[0]+=1\n"
        "            return RNA.fold(s)[0]\n"
        "        def inv(start,tg):\n"
        "            cnt[0]+=1\n"
        "            return RNA.inverse_fold(start,tg)\n"
        "        prim={'inverse_fold':inv,'fold':fold,'bp_distance':RNA.bp_distance,\n"
        "              'rng':random.Random(sd),'budget_calls':budget}\n"
        "        t0=time.time(); err=None; seq=None\n"
        "        try:\n"
        "            seq=design_pk(target,prim)\n"
        "        except Exception as e:\n"
        "            err=repr(e)[:150]\n"
        "        el=time.time()-t0\n"
        "        ok=False\n"
        "        if isinstance(seq,str) and len(seq)==len(target) and not (set(seq)-set('ACGU')):\n"
        "            ok=(RNA.fold(seq)[0]==target)\n"
        "        out.append({'id':tid,'solved':bool(ok),'calls':cnt[0],'s':round(el,2),\n"
        "                    'hit':cnt[0]>=budget,'err':err})\n"
        "    print('RESULT'+json.dumps(out))\n"
        "main()\n")


def run(code, part, budget, wall, seed):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(_runner_src(code)); path = f.name
    try:
        p = subprocess.run([CONDA_PY, path, json.dumps(part), str(budget), str(wall), str(seed)],
                           capture_output=True, text=True, timeout=max(600, wall * len(part) * 2))
        for ln in (p.stdout or "").splitlines():
            if ln.startswith("RESULT"):
                return json.loads(ln[6:])
        return [{"id": t, "solved": False, "calls": 0, "s": 0, "hit": False,
                 "err": (p.stderr or "")[-160:]} for t, _ in part]
    finally:
        os.unlink(path)


def _one(code, budget, seed, tid, target, wall):
    """One (budget, seed, target) cell in its own subprocess."""
    r = run(code, [(tid, target)], budget, wall, 1000 + seed)
    return r[0] if r else {"solved": False, "calls": 0, "hit": False, "err": "no result"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="50,125,250,500,1000,2000,4000,8000,16000,32000")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--wall", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--split", default=os.path.join(HERE, "agent", "held.json"))
    a = ap.parse_args()

    from rna_harness.benchmarks.eterna100 import load_eterna100
    T = {f"eterna100:{t['id']}:{t['name']}": t["target"] for t in load_eterna100()}
    if os.path.exists(a.split):
        held = json.load(open(a.split))
    else:
        # The original split is not on this branch. Use a DECLARED deterministic held-out set of the same
        # size (37) and say so: every number below is on this split, not on the archived one, so it is not
        # comparable to "24/37" as a like-for-like -- what IS comparable is the SHAPE of the curve.
        ids = sorted(T, key=lambda k: int(k.split(":")[1]))
        held = ids[2::3][:37]
        json.dump(held, open(a.split, "w"), indent=1)
        print(f"  NOTE: archived split unavailable on this branch; using a declared stride-3 split of "
              f"{len(held)} targets, written to {os.path.basename(a.split)}")
    code = open(SKILL).read()
    part = [(k, T[k]) for k in held if k in T]
    print(f"GLM-5.2's authored nested operator, {len(part)} held-out targets, "
          f"{a.seeds} seeds, budgets {a.budgets}")
    print("  (the operator code is UNCHANGED from the archive; only the budget axis is swept)\n")

    # One subprocess PER (budget, seed, target). The original harness ran a whole split sequentially in one
    # process with a 60 s per-target wall, so a single (budget, seed) cell could take 37 minutes and the
    # full sweep would take days. Fanning out is the only way to sweep an axis at all.
    from concurrent.futures import ProcessPoolExecutor, as_completed
    budgets = [int(x) for x in a.budgets.split(",")]
    jobs = [(b, s, k, T[k]) for b in budgets for s in range(a.seeds) for k in held if k in T]
    print(f"  {len(jobs)} cells = {len(budgets)} budgets x {a.seeds} seeds x {len(part)} targets, "
          f"{a.workers} workers", flush=True)
    got = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_one, code, b, s, k, tg, a.wall): (b, s, k) for b, s, k, tg in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            b, s, k = futs[fu]
            try:
                got[(b, s, k)] = fu.result()
            except Exception as e:
                got[(b, s, k)] = {"solved": False, "calls": 0, "hit": False, "err": repr(e)[:120]}
            if i % 100 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"    {i}/{len(jobs)}  {el/60:.1f} min  eta {el/i*(len(jobs)-i)/60:.1f} min",
                      flush=True)

    rows = []
    print(f"\n  {'budget':>7} {'solved/seed':>28} {'mean':>6} {'S_inf(union)':>13} "
          f"{'hit cap':>8} {'calls/target':>13}")
    for b in budgets:
        per, union, hits, calls = [], set(), 0, []
        for s in range(a.seeds):
            sv = [k for k in held if got.get((b, s, k), {}).get("solved")]
            per.append(len(sv)); union |= set(sv)
            hits += sum(1 for k in held if got.get((b, s, k), {}).get("hit"))
            calls += [got.get((b, s, k), {}).get("calls", 0) for k in held]
        rows.append({"budget": b, "per_seed": per, "union": len(union), "hit": hits,
                     "mean_calls": sum(calls) / max(len(calls), 1)})
        print(f"  {b:7d} {str(per):>28} {sum(per)/len(per):6.2f} {len(union):13d} "
              f"{hits:8d} {rows[-1]['mean_calls']:13.0f}", flush=True)
    json.dump({"skill": "rssd_agent_nested/authored_skill.py", "n_held": len(part),
               "seeds": a.seeds, "rows": rows, "held": held},
              open(os.path.join(HERE, "agent_curve.json"), "w"), indent=1)

    print("\n=== the two coordinates the single number hid ===")
    best = max(r["union"] for r in rows)
    sat = [r for r in rows if r["union"] >= best]
    print(f"  S_inf (union over seeds, best observed): {best}/{len(part)}   "
          f"first reached at budget {sat[0]['budget'] if sat else '-'}")
    half = best / 2.0
    hb = next((r["budget"] for r in rows if r["union"] >= half), None)
    print(f"  B_half (budget for half the ceiling): {hb} fold-calls/target")
    print(f"  attempts that hit the cap at the archived budget 4000: "
          f"{next((r['hit'] for r in rows if r['budget'] == 4000), '-')}")
    print("\n  What this does and does not show. It does NOT show the agent improved -- the code is byte")
    print("  identical. It shows whether the boundary we reported was the OPERATOR's or the BUDGET's.")


if __name__ == "__main__":
    main()
