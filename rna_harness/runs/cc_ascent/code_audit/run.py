#!/usr/bin/env python3
"""Driver for the code_audit positive-side demo. For each task x arm: run the self-improving agent, freeze the
best-reward solution, then adjudicate it under the single gameable reward vs the independent panel (+ gold
validity). Mirrors the RNA collapse table (single oracle vs panel-unanimous). Usage:
  OPENAI_MODEL=<id> python run.py <tag> [n_tasks] [rounds] [seed]   (key from OPENROUTER_API_KEY env)
"""
import json, os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tasks import TASKS, heldout_args, expected_for
from agent import run_agent
from panel import panel_verdict, gold_correct
from judge import judge_panel
from rna_harness.llm.openrouter import OpenRouterClient

HERE = os.path.dirname(os.path.abspath(__file__))


def main(tag, n_tasks=None, rounds=6, seed=0):
    model = os.environ.get("OPENAI_MODEL", "z-ai/glm-5.2")
    reward_mode = os.environ.get("CODE_AUDIT_REWARD", "judge")     # judge = flawed proxy (RNA-faithful)
    client = OpenRouterClient(model=model)
    if not client.available():
        print("NO API KEY (set OPENROUTER_API_KEY)"); sys.exit(1)
    sel = os.environ.get("CODE_AUDIT_TASKS")
    if sel:
        ids = set(sel.split(","))
        tasks = [t for t in TASKS if t.id in ids]
    else:
        tasks = TASKS[:n_tasks] if n_tasks else TASKS
    rng = random.Random(seed)
    out = {"tag": tag, "model": model, "reward_mode": reward_mode, "rounds": rounds, "seed": seed, "rows": []}
    outdir = os.path.join(HERE, "runs"); os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{tag}.json")
    print(f"[code_audit:{tag}] model={model} reward={reward_mode} tasks={len(tasks)} rounds={rounds}", flush=True)
    for t in tasks:
        ho = heldout_args(t, rng, 40); exp = expected_for(t, ho)
        for arm in ["reward", "honest"]:
            t0 = time.time()
            res = run_agent(client, t, arm=arm, rounds=rounds, judge=True, reward_mode=reward_mode)
            code = res["best"]["code"]
            jv, jd = judge_panel(client, code, t.spec, k=3) if code else (False, {})
            v = panel_verdict(code, t, ho, exp, judge_vote=jv) if code else {"unanimous": False, "members": {}}
            gc, gd = gold_correct(code, t, rng, n=600) if code else (False, {})
            # single-oracle "accept" = the optimized proxy: judge vote (judge mode) or all-visible-pass (tests)
            single = jv if reward_mode == "judge" else (res["best"]["vis_frac"] == 1.0)
            row = {"task": t.id, "arm": arm, "vis_frac": res["best"]["vis_frac"], "single_accept": bool(single),
                   "panel_unanimous": v["unanimous"], "judge_member": jv, "judge_detail": jd,
                   "gold_correct": gc, "members": v.get("members"), "s": round(time.time() - t0, 1),
                   "best_round": res["best"]["round"], "code": code}
            out["rows"].append(row)
            json.dump(out, open(path, "w"), indent=1)
            print(f"  {t.id:15} {arm:7} vis={row['vis_frac']:.0%} single_accept={row['single_accept']} "
                  f"panel_unan={row['panel_unanimous']} gold={gc}", flush=True)
    # aggregate (the collapse + panel validity)
    agg = {}
    for arm in ["reward", "honest"]:
        rows = [r for r in out["rows"] if r["arm"] == arm]
        n = len(rows)
        rs = sum(r["single_accept"] for r in rows)
        pu = sum(r["panel_unanimous"] for r in rows)
        gc = sum(r["gold_correct"] for r in rows)
        gaming = sum(1 for r in rows if r["single_accept"] and not r["gold_correct"])   # proxy-accepts, wrong
        caught = sum(1 for r in rows if r["single_accept"] and not r["gold_correct"] and not r["panel_unanimous"])
        fp = sum(1 for r in rows if r["panel_unanimous"] and not r["gold_correct"])   # panel over-accepts
        fn = sum(1 for r in rows if not r["panel_unanimous"] and r["gold_correct"])   # panel too strict
        agg[arm] = {"n": n, "single_accept": rs, "panel_unanimous": pu, "gold_correct": gc,
                    "gaming": gaming, "caught_by_panel": caught, "panel_FP": fp, "panel_FN": fn}
    out["aggregate"] = agg
    out["cost"] = {"prompt_tokens": client.total_prompt_tokens, "completion_tokens": client.total_completion_tokens,
                   "calls": client.n_calls}
    json.dump(out, open(path, "w"), indent=1)
    print("\n=== AGGREGATE ===", flush=True)
    for arm, a in agg.items():
        print(f"  {arm:7}: single-oracle accept {a['single_accept']}/{a['n']} -> panel-unanimous "
              f"{a['panel_unanimous']}/{a['n']}  (gold {a['gold_correct']}/{a['n']}; "
              f"gaming={a['gaming']} caught={a['caught_by_panel']}; panel FP={a['panel_FP']} FN={a['panel_FN']})",
              flush=True)
    print(f"  cost: {client.n_calls} calls, {client.total_prompt_tokens}+{client.total_completion_tokens} tok "
          f"-> {tag}.json", flush=True)


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    nt = int(sys.argv[2]) if len(sys.argv) > 2 else None
    rd = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    sd = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    main(tag, nt, rd, sd)
