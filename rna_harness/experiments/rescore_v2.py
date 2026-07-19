"""Re-score saved HumanEval+ v2 completions with the FIXED sandbox preamble (typing imports), WITHOUT
regenerating -- demonstrating exactly the auditability the saved completions were meant to provide. Re-runs
each saved candidate's code through eval_candidate_diag (now with _PREAMBLE) and rewrites the vec/diag."""
from __future__ import annotations

import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from .llm_evalplus_v2 import load_all_problems, eval_candidate_diag, OUT


def _rescore_task(args):
    tid, cand_codes, prob = args
    out = []
    for code in cand_codes:
        if isinstance(code, str) and code.startswith("__GEN_ERR__"):
            out.append((None, None, {"gen_err": True}))
            continue
        vec, nb, diag = eval_candidate_diag(code, prob)
        out.append((vec, nb, diag))
    return tid, out


def run(models, workers):
    probs, dropped = load_all_problems(plus_cap=40)
    pmap = {p["tid"]: p for p in probs}
    for mk in models:
        path = os.path.join(OUT, f"{mk}.json")
        if not os.path.exists(path):
            print(f"  {mk}: no file, skip"); continue
        rec = json.load(open(path))
        tasks = [(p["tid"], [c["code"] for c in p["candidates"]], pmap[p["tid"]])
                 for p in rec["per"] if p["tid"] in pmap]
        results = {}
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_rescore_task, t) for t in tasks]
            for i, f in enumerate(as_completed(futs)):
                tid, out = f.result(); results[tid] = out
                if (i + 1) % 40 == 0:
                    print(f"  [{mk}] rescored {i+1}/{len(tasks)}", flush=True)
        changed = 0
        for p in rec["per"]:
            out = results.get(p["tid"])
            if not out:
                continue
            for c, (vec, nb, diag) in zip(p["candidates"], out):
                if vec is not None:
                    if c["vec"] != vec:
                        changed += 1
                    c["vec"] = vec; c["diag"] = diag
        rec["rescored_preamble"] = True
        json.dump(rec, open(path, "w"))
        print(f"[{mk}] rescored; {changed} candidate vectors changed by the preamble fix -> {path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["llama-3.1-8b", "qwen36-local"])
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    run(a.models, a.workers)
