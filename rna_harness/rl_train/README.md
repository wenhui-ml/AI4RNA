# RL training scaffold (RFT → GRPO)

The harness collects **model-agnostic, verifiable-reward** RL data (see `rna_harness/rl/`).
Training itself is deferred by design — this directory provides *data-format-ready* reference
scripts so the collected data plugs straight into a training stack (TRL/verl, or the user's
existing Qwen2.5 SFT/RL pipeline). Target open policy: **Gemma4-31b** (any HF causal LM works).

## Why this works here (and not for catalysis)
The reward is the ViennaRNA oracle: exact, non-drifting, self-checkable (R_det ≈ 1). There is
no error-bearing surrogate in the reward path, so RL genuinely internalizes search into
weights ("capability eats knowledge") instead of learning to fool a surrogate. RNA is the
high-α_f positive control.

## Two-stage recipe (as chosen: RFT first, then GRPO)

### Stage 1 — RFT / Expert-Iteration (cheapest, most robust)
Rejection-sampling fine-tuning: keep only high-reward / solved samples and SFT on them.
```bash
# 1. Collect a dataset (zero-credit classical policy, or agentic Opus traces):
python -m rna_harness.experiments.run --experiment collect --max-len 80 --k 16 \
    --collect-budget 30 --out runs/rft_data          # writes rft.jsonl (solved-only)
# 2. SFT on the kept (prompt, completion) pairs:
python -m rna_harness.rl_train.train_sft --data runs/rft_data/rft.jsonl \
    --model google/gemma-4-31b --out ckpts/rft_gemma
```

### Stage 2 — GRPO (online, group-relative, verifiable reward)
Group-relative policy optimization using the ViennaRNA verifier as the reward function —
no reward model.
```bash
python -m rna_harness.rl_train.build_dataset --max-len 80 --out runs/grpo_prompts.jsonl
python -m rna_harness.rl_train.train_grpo --data runs/grpo_prompts.jsonl \
    --model ckpts/rft_gemma --out ckpts/grpo_gemma --num-generations 8
```
The reward is `rna_harness.rl.reward_fn.rna_reward` (dense) + optional `rna_solved_reward`
(sparse). Both are deterministic ViennaRNA calls.

## Data formats
- `rft.jsonl`   : `{prompt, completion, messages, reward, solved, task_id, task_family}`
- `grpo.jsonl`  : `{prompt, response, reward, advantage, group_id, ...}` (offline groups)
- `grpo_prompts.jsonl` (for online GRPO): `{prompt, target, constraints, oracle, task_id}`
- `agentic_rlvr.jsonl` (from `agent_runner`): `{prompt, final_sequence, reward, solved,
  trace_stream_json, tool_calls, ...}` — full agentic trajectories for distilling tool use.

## Notes / knobs
- Hardware: RL training needs free GPU memory on the B300s (folding reward is CPU, cheap).
- Model portability: the SAME data trains any base model; swap `--model`.
- To reuse the existing Qwen2.5 SFT stack instead of TRL: point it at `rft.jsonl`
  (prompt/completion or messages format) — no conversion needed.
- The scripts import `trl`/`transformers` lazily and print an install hint if missing; they
  are reference implementations kept intentionally small.
