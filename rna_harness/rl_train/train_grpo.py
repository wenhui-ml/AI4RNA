"""Reference GRPO trainer (TRL) using the ViennaRNA verifier as the online reward.

The reward is `rna_harness.rl.reward_fn.rna_reward` (deterministic ViennaRNA). No reward
model. Requires `trl` + `transformers` + GPU. Deferred by default; run when GPU is free.
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="grpo_prompts.jsonl (fields: prompt, target, constraints, oracle)")
    ap.add_argument("--model", default="google/gemma-4-31b", help="base or RFT checkpoint")
    ap.add_argument("--out", default="ckpts/grpo")
    ap.add_argument("--num-generations", type=int, default=8, help="GRPO group size")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-completion-len", type=int, default=512)
    ap.add_argument("--steps", type=int, default=500)
    args = ap.parse_args(argv)

    try:
        from datasets import load_dataset
        from trl import GRPOConfig, GRPOTrainer
    except Exception as e:  # noqa: BLE001
        print(f"[train_grpo] missing deps ({e}). Install into a GPU env: "
              f"pip install 'trl>=0.12' transformers accelerate datasets", file=sys.stderr)
        return 2

    from ..rl.reward_fn import rna_reward, rna_solved_reward

    ds = load_dataset("json", data_files=args.data, split="train")
    cfg = GRPOConfig(
        output_dir=args.out,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_len,
        max_steps=args.steps,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=100,
    )
    # reward_funcs receive (prompts, completions, **columns) -- our fns read `target` etc.
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[rna_reward, rna_solved_reward],
        args=cfg,
        train_dataset=ds,
    )
    trainer.train()
    trainer.save_model(args.out)
    print(f"[train_grpo] saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
