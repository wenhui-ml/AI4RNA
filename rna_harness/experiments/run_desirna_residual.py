"""Plan-A measurement: run DesiRNA (via the oracle-verified adapter) across the 42 model-unsolved
Eterna100-V2 targets to get the ORCHESTRATED solve total ("if the agent calls the SOTA tool, how many
of the puzzles the base model missed can we recover"). Per-size timelimit; every returned sequence is
re-verified by our own primary oracle. Requires DESIRNA_CMD set (see reference-desirna-integration).
"""
from __future__ import annotations

import json
import os
import time

from ..benchmarks.eterna100 import load_eterna100
from ..generators.desirna import DesiRNAGenerator, desirna_available
from ..task import DesignTask

OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "desirna_residual")
BASE_SUMMARY = os.path.join(os.path.dirname(__file__), "..", "runs", "glm5.2_eterna100", "summary.json")
MODEL_SOLVED = 58  # base GLM-5.2 solve count (glm5.2_eterna100)


def main():
    os.makedirs(OUT, exist_ok=True)
    if not desirna_available():
        raise SystemExit("DESIRNA_CMD not set / DesiRNA not found -- cannot run.")
    puz = {p["id"]: p for p in load_eterna100(version="V2")}
    summ = json.load(open(BASE_SUMMARY))
    unsolved = sorted([int(r["task_id"].split(":")[1]) for r in summ["records"] if not r["solved"]],
                      key=lambda i: len(puz[i]["target"]))
    print(f"DesiRNA on {len(unsolved)} model-unsolved targets; per-size timelimit = min(180, 30+0.5*n)s; "
          f"R=10; every candidate re-verified by our primary oracle.", flush=True)
    rows, solved = [], 0
    for pid in unsolved:
        t = puz[pid]["target"]
        n = len(t)
        tl = min(180, int(30 + 0.5 * n))
        t0 = time.time()
        r = DesiRNAGenerator(replicas=10, timelimit_s=tl, results_number=20).design(
            DesignTask(task_id=str(pid), target=t))
        solved += int(r.solved)
        rows.append({"pid": pid, "name": puz[pid]["name"], "len": n, "timelimit_s": tl,
                     "bp": r.best_bp_distance, "solved": r.solved, "candidates": r.n_evals,
                     "available": r.meta.get("available")})
        print(f"  pid {pid:>3} n={n:>4} tl={tl:>3}s | DesiRNA bp={r.best_bp_distance:>3} "
              f"solved={r.solved} cand={r.n_evals} ({time.time()-t0:.0f}s)", flush=True)
        json.dump({"model_solved": MODEL_SOLVED, "n_desirna_solved_of_unsolved": solved,
                   "orchestrated_total": MODEL_SOLVED + solved, "rows": rows},
                  open(os.path.join(OUT, "desirna_residual.json"), "w"), indent=1)
    resid = [r["pid"] for r in rows if not r["solved"]]
    print(f"\nDesiRNA solved {solved}/{len(unsolved)} of the model-unsolved set.", flush=True)
    print(f"ORCHESTRATED TOTAL = {MODEL_SOLVED} (model) + {solved} (DesiRNA) = "
          f"{MODEL_SOLVED + solved}/100", flush=True)
    print(f"still unsolved within budget ({len(resid)}): {resid}", flush=True)


if __name__ == "__main__":
    main()
