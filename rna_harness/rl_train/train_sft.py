"""Reference RFT/SFT trainer (TRL). Consumes rft.jsonl (prompt/completion or messages).

Reference implementation: kept small and dependency-lazy. Requires `trl` + `transformers` +
a GPU. Deferred by default; run when GPU is free. Equivalently, point the user's existing
Qwen2.5 SFT stack at rft.jsonl -- no conversion needed.
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="rft.jsonl (fields: prompt, completion)")
    ap.add_argument("--model", default="google/gemma-4-31b")
    ap.add_argument("--out", default="ckpts/rft")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    args = ap.parse_args(argv)

    try:
        from datasets import load_dataset
        from trl import SFTConfig, SFTTrainer
    except Exception as e:  # noqa: BLE001
        print(f"[train_sft] missing deps ({e}). Install into a GPU env: "
              f"pip install 'trl>=0.12' transformers accelerate datasets", file=sys.stderr)
        return 2

    ds = load_dataset("json", data_files=args.data, split="train")
    # TRL SFT accepts prompt+completion columns directly.
    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        max_seq_length=args.max_seq_len,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
    )
    trainer = SFTTrainer(model=args.model, args=cfg, train_dataset=ds)
    trainer.train()
    trainer.save_model(args.out)
    print(f"[train_sft] saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
