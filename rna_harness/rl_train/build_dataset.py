"""Build an online-GRPO prompts dataset from benchmark tasks.

Each row = {prompt, target, constraints, oracle, task_id, task_family}. The reward function
(rna_harness.rl.reward_fn.rna_reward) reads `target`/`constraints`/`oracle` from the row, so
GRPO can score generated sequences against the exact oracle.
"""
from __future__ import annotations

import argparse
import json
import os

from ..benchmarks import eterna100_tasks, synthesize_hybrid_tasks
from ..rl.prompts import render_task_prompt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, default=80)
    ap.add_argument("--ids", type=int, nargs="*", default=None)
    ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("--out", default="runs/grpo_prompts.jsonl")
    args = ap.parse_args(argv)

    tasks = eterna100_tasks(ids=args.ids, max_len=args.max_len)
    if args.hybrid:
        extra = []
        for t in tasks:
            extra += synthesize_hybrid_tasks(t)
        tasks += extra

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n = 0
    with open(args.out, "w") as f:
        for t in tasks:
            f.write(json.dumps({
                "prompt": render_task_prompt(t),
                "target": t.target,
                "constraints": t.constraints.to_dict(),
                "oracle": t.oracle.to_dict(),
                "task_id": t.task_id,
                "task_family": t.family,
            }) + "\n")
            n += 1
    print(f"wrote {n} prompts -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
