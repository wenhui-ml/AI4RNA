#!/usr/bin/env python3
"""Put the surface into the AGENT's objective: score S_inf, not solves-at-one-budget.

What changed and why. rssd_agent_nested.py had GLM-5.2 author a nested design procedure against this
objective, stated verbatim in its prompt:

    "Budget 4000 fold-calls/target. Minimise total cost at equal solve."

A fixed budget plus a cost objective asks for a POINT, and the archived held-out result reads like one:
24/37 solved, $0.038, solves_lost 13 -- with budget_hit 13. Thirteen targets ran out of budget and
thirteen solves were lost. The reported boundary may be the budget we chose, not the operator the agent
wrote. agent_curve.py measures that directly by re-running the same code across a budget sweep.

This file changes the other half: the agent is now scored on the surface it is actually being asked to
push, so the objective and the measurement finally agree.

    old   score(ev) = (solved_at_4000, -cost)
    new   score(ev) = (S_inf, U, -cost)      S_inf = targets solved at ANY budget in the sweep
                                             U     = mean solves over log-budget

ONE DESIGN DECISION MATTERS MORE THAN THE REST. The original composite includes a should_call_sota policy
that can fire DesiRNA, which solves 37/37. With a SOTA escape hatch in the objective, "raise S_inf" has a
trivial gamed optimum: always call SOTA. So the objective here scores THE AGENT'S OWN SKILL ONLY. The
SOTA-fallback composite is still computed and reported, but it cannot influence selection. Without this,
the experiment measures nothing.

Everything else is held fixed against the archive so the comparison is like-for-like: same benchmark, same
even/odd train/held split over the same puzzle set, same primitives, same model (GLM-5.2), same sandbox.
Invention capex is recorded from the client's token usage and reported through cost(N) = c_cpu + C_invent/N,
the amortisation form from cognitive/budget.py.
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

from rna_harness.benchmarks.eterna100 import load_eterna100                       # noqa: E402
from rna_harness.cognitive.cost import llm_cost_usd                                # noqa: E402
from rna_harness.cognitive.features import features as extract_features            # noqa: E402
from rna_harness.experiments.rssd_agent import _extract_code, load_should_call_sota  # noqa: E402
from rna_harness.experiments.rssd_agent_nested import PRIM_DOC, _run_chunk         # noqa: E402

BUDGETS = [125, 500, 2000, 8000]          # declared before the run; the agent is told all four


def run_at(code, targets, budget, wall, seed, chunk=6, workers=16):
    items = list(targets.items())
    chunks = [dict(items[c:c + chunk]) for c in range(0, len(items), chunk)]
    res = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(_run_chunk, (code, c, budget, wall, seed)) for c in chunks]):
            res.update(fut.result())
    return res


def evaluate_surface(code, ids, tgt, feats, des_solved, des_cost, budgets, wall, seed, workers=16):
    """The agent's own skill across a budget sweep. S_inf and U come from the skill ALONE."""
    rows, ever, errs, sample_err = [], set(), 0, None
    _EXP = {"_Budget", "_TimeUp", "chunk timeout"}
    for b in budgets:
        r = run_at(code, {str(i): tgt[i] for i in ids}, b, wall, seed, workers=workers)
        sv, cost, hit = 0, 0.0, 0
        for i in ids:
            x = r.get(str(i), {})
            e = x.get("err")
            if e in _EXP:
                hit += 1
            elif e:
                errs += 1
                if sample_err is None:
                    sample_err = e
            cost += x.get("cost", 0.0)
            if x.get("solved"):
                sv += 1; ever.add(i)
        rows.append({"budget": b, "solved": sv, "cost": cost, "budget_hit": hit})
    lb = [math.log(r["budget"]) for r in rows]
    U = (sum((rows[k]["solved"] + rows[k + 1]["solved"]) / 2 * (lb[k + 1] - lb[k])
             for k in range(len(rows) - 1)) / (lb[-1] - lb[0])) if len(rows) > 1 else rows[0]["solved"]
    out = {"rows": rows, "s_inf": len(ever), "U": U, "n": len(ids),
           "cost_max": rows[-1]["cost"], "code_errors": errs, "sample_err": sample_err,
           "ever_ids": sorted(ever)}
    # the SOTA-fallback composite, REPORTED but never scored -- see the module docstring
    fn, ferr = load_should_call_sota(code)
    top = run_at(code, {str(i): tgt[i] for i in ids}, budgets[-1], wall, seed, workers=workers)
    comp_solved = comp_cost = fired = abst = lost = 0
    for i in ids:
        x = top.get(str(i), {})
        comp_cost += x.get("cost", 0.0)
        if x.get("solved"):
            comp_solved += 1; continue
        call = True
        if fn is not None:
            try:
                call = bool(fn(feats[i], x.get("bp")))
            except Exception:
                call = True
        if call:
            comp_cost += des_cost; fired += 1; comp_solved += int(des_solved.get(i, False))
        else:
            abst += 1; lost += int(des_solved.get(i, False))
    out["composite"] = {"solved": comp_solved, "cost": comp_cost, "sota_fired": fired,
                        "abstained": abst, "solves_lost": lost, "policy_err": ferr}
    return out


def curve_str(ev):
    return ", ".join(f"{r['budget']}->{r['solved']}" for r in ev["rows"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="z-ai/glm-5.2")
    ap.add_argument("--reasoning", choices=["auto", "on", "off"], default="off")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--train-subset", type=int, default=16)
    ap.add_argument("--per-target-wall", type=float, default=45.0)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--llm-timeout", type=float, default=360.0)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=os.path.join(HERE, "agent_surface"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    puz = {p["id"]: p["target"] for p in load_eterna100(version="V2")}
    ids = sorted(puz)
    tgt = {i: puz[i] for i in ids}
    feats = {i: extract_features(tgt[i]) for i in ids}
    # SAME even/odd split as the archive so the held-out number is comparable
    train = [i for i in ids if i % 2 == 0]
    held = [i for i in ids if i % 2 == 1]
    tr = sorted(train, key=lambda i: feats[i]["len"])
    step = max(1, len(tr) // a.train_subset)
    train_eval = tr[::step][:a.train_subset]
    # DesiRNA reference is not on this branch (runs/learning_vs_cold absent), and it is only needed for the
    # REPORTED composite, never for scoring -- so the missing table degrades the report, not the objective.
    des_solved, des_cost = {}, 0.0
    tbl = os.path.join(HERE, "..", "learning_vs_cold", "table.json")
    if os.path.exists(tbl):
        try:
            from rna_harness.experiments.rssd_agent_nested import _eterna_desirna_column
            des_solved, des_cost = _eterna_desirna_column()
        except Exception as e:
            print(f"  (DesiRNA column unavailable: {repr(e)[:80]})")
    else:
        print("  NOTE: no DesiRNA reference table on this branch -> the composite is reported without a"
              " SOTA baseline. The OBJECTIVE never used it, so scoring is unaffected.")

    print(f"agent-surface (Eterna/nested, GLM-5.2): {len(ids)} puzzles, train {len(train)} / held {len(held)}; "
          f"self-revise on {len(train_eval)}")
    print(f"  objective = (S_inf, U, -cost) over budgets {BUDGETS}, SKILL ONLY (no SOTA escape hatch)")
    print(f"  archive baseline for reference: held 24/37 at a single budget 4000, budget_hit 13\n", flush=True)

    from rna_harness.llm.openrouter import OpenRouterClient
    client = OpenRouterClient(model=a.model, timeout=a.llm_timeout)
    if not client.available():
        raise SystemExit("no API key -- source the .orkey")

    evidence = [{"target": tgt[i], "len": feats[i]["len"]} for i in train_eval]
    system = ("You are an expert RNA design researcher who writes correct, self-contained Python. Author a "
              "NESTED (ViennaRNA MFE) design procedure + a should_call_sota policy. Respond with ONE "
              "```python code block, exactly the two functions, no prose.")
    base_user = (
        PRIM_DOC + "\n\nTRAIN evidence (structures only; author a GENERAL procedure, do not hardcode):\n"
        + json.dumps(evidence, indent=0) +
        f"\n\nHOW YOU ARE SCORED -- this is NOT a fixed-budget cost problem.\n"
        f"Your procedure is run at SEVERAL budgets: prim['budget_calls'] will be one of {BUDGETS} "
        f"fold-calls/target, and you do not know which.\n"
        f"  PRIMARY: S_inf = how many targets you solve at ANY of those budgets. Maximise it.\n"
        f"  THEN:    U     = average solves across the budgets (log-spaced). Maximise it.\n"
        f"  THEN:    cost. Lower is better, but never trade a solved target for cost.\n"
        f"So: do NOT tune to one budget. At a small budget, return your best attempt rather than giving up; "
        f"at a large budget, keep searching instead of stopping early. Respect prim['budget_calls'] as a hard "
        f"cap -- exceeding it aborts the target and scores 0.")
    msgs = [{"role": "user", "content": base_user}]
    best_code, best_ev, history = None, None, []

    def score(ev):
        return (ev["s_inf"], ev["U"], -ev["cost_max"])

    for rnd in range(a.rounds):
        reasoning = ({"enabled": True} if a.reasoning == "on" else
                     {"enabled": False} if a.reasoning == "off" else None)
        mt = None if a.reasoning == "on" else a.max_tokens
        res = client.chat(msgs, system=system, temperature=a.temperature, max_tokens=mt,
                          reasoning=reasoning)
        if res.error:
            print(f"  [round {rnd}] LLM error: {res.error}", flush=True); break
        code = _extract_code(res.text, res.reasoning)
        if "def design_pk" not in code:
            print(f"  [round {rnd}] no usable code; retrying", flush=True)
            msgs += [{"role": "assistant", "content": res.text or ""},
                     {"role": "user", "content": "No complete design_pk. ONE short ```python block."}]
            continue
        t0 = time.time()
        ev = evaluate_surface(code, train_eval, tgt, feats, des_solved, des_cost, BUDGETS,
                             a.per_target_wall, a.seed, a.workers)
        print(f"  [round {rnd}] TRAIN curve {curve_str(ev)}  S_inf={ev['s_inf']}/{ev['n']} "
              f"U={ev['U']:.2f} cost=${ev['cost_max']:.4f} err={ev['code_errors']} "
              f"({time.time()-t0:.0f}s)", flush=True)
        history.append({"round": rnd, "curve": ev["rows"], "s_inf": ev["s_inf"], "U": ev["U"],
                        "cost_max": ev["cost_max"], "code_errors": ev["code_errors"]})
        if best_ev is None or score(ev) > score(best_ev):
            best_code, best_ev = code, ev
        errhint = (f" EXAMPLE ERROR to fix: {ev['sample_err']} (prim['rng'] is a random.Random INSTANCE and "
                   "prim['budget_calls'] is an INT -- use them directly, do NOT call them)."
                   if ev.get("sample_err") else "")
        hits = ", ".join(f"{r['budget']}->{r['budget_hit']}" for r in ev["rows"])
        fb = (f"On {ev['n']} TRAIN targets your solve curve was: {curve_str(ev)} "
              f"(budget -> solved). S_inf = {ev['s_inf']}/{ev['n']}, U = {ev['U']:.2f}, "
              f"cost ${ev['cost_max']:.4f}. Targets that hit the budget cap: {hits}. "
              f"Code errors on {ev['code_errors']} targets.{errhint} "
              f"Your score is S_inf first, then U. Raise the WHOLE curve: the low-budget end needs a good "
              f"answer without many calls, the high-budget end needs you to keep searching. "
              f"ONE ```python block, two functions.")
        asst = {"role": "assistant", "content": res.text}
        if res.reasoning_details:
            asst["reasoning_details"] = res.reasoning_details
        msgs += [asst, {"role": "user", "content": fb}]

    usage = client.usage()
    json.dump(usage, open(os.path.join(a.outdir, "llm_usage.json"), "w"), indent=1)
    if best_code is None:
        raise SystemExit("no usable authored code")
    capex = llm_cost_usd(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), a.model)
    print(f"\n  invention capex: {usage.get('n_calls')} calls, "
          f"{usage.get('prompt_tokens')} in / {usage.get('completion_tokens')} out = ${capex:.4f}")

    print(f"  FROZEN -> HELD-OUT ({len(held)} puzzles), same surface ...", flush=True)
    hv = evaluate_surface(best_code, held, tgt, feats, des_solved, des_cost, BUDGETS,
                          a.per_target_wall, a.seed, a.workers)
    print(f"  HELD curve {curve_str(hv)}  S_inf={hv['s_inf']}/{hv['n']} U={hv['U']:.2f} "
          f"cost=${hv['cost_max']:.4f}")
    print(f"  composite (reported, never scored): {hv['composite']}")
    for N in (1, 37, 370):
        print(f"  cost per target at N={N:4d} reuses: "
              f"${hv['cost_max']/max(hv['n'],1) + capex/N:.5f}")
    open(os.path.join(a.outdir, "authored_skill.py"), "w").write(best_code)
    json.dump({"model": a.model, "reasoning": a.reasoning, "budgets": BUDGETS,
               "objective": "(S_inf, U, -cost) on the SKILL only; SOTA fallback reported not scored",
               "train_eval": train_eval, "held": held, "history": history, "held_surface": hv,
               "capex_usd": capex, "llm_usage": usage},
              open(os.path.join(a.outdir, "agent_surface.json"), "w"), indent=1)
    print(f"  wrote {a.outdir}/agent_surface.json + authored_skill.py")


if __name__ == "__main__":
    main()
