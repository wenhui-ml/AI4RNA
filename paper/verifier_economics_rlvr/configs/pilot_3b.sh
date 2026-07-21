#!/usr/bin/env bash
# GRPO pilot: Qwen2.5-3B-Instruct, one verifier arm per run (HANDOFF §5/§7).
# Usage:  pilot_3b.sh <arm: cheap|strong|cascade|dial> <gpus e.g. 0,1> [seed=0] [max_steps=250] [tag=pilot3b]
# For arm=dial, set env VE_DIAL_P in [0,1] (escalation probability; 0==cheap, 1==cascade).
# Env overrides: VLLM_UTIL (default 0.55), DATASET (default data/train.jsonl), SAVE_STEPS (default 25),
#                MAX_COMPLETION (default 2048), NUM_GEN (default 8), PER_DEV_BS (default 8)
set -euo pipefail
ARM=${1:?arm: cheap|strong|cascade}
GPUS=${2:?comma-separated gpu ids}
SEED=${3:-0}
MAX_STEPS=${4:-250}
TAG=${5:-pilot3b}

BASE=/data/wenhui/openclaude_rna/paper/verifier_economics_rlvr
PY_ENV=$HOME/miniconda3/envs/cua
MODEL=${MODEL:-/data/wenhui/LowRankClone_improve/models/Qwen2.5-3B-Instruct}
RUN=$BASE/runs/${TAG}_${ARM}_s${SEED}
mkdir -p "$RUN"
NPROC=$(awk -F, '{print NF}' <<< "$GPUS")
export PATH="$PY_ENV/bin:$PATH"   # subprocesses (ninja, etc.) must resolve from the env
# unique rendezvous port per arm — concurrent same-host torchrun jobs collide on default 29500
case "$ARM" in cheap) MPORT=29611;; strong) MPORT=29612;; cascade) MPORT=29613;; dial) MPORT=29614;; *) MPORT=29610;; esac
export MASTER_PORT=${MASTER_PORT:-$MPORT}

USE_HF=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
VE_RUN_DIR=$RUN \
VE_DIAL_P="${VE_DIAL_P:-}" \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=$GPUS \
NPROC_PER_NODE=$NPROC \
"$PY_ENV/bin/swift" rlhf \
    --rlhf_type grpo \
    --model "$MODEL" \
    ${MODEL_TYPE:+--model_type "$MODEL_TYPE"} \
    ${TEMPLATE:+--template "$TEMPLATE"} \
    --external_plugins "$BASE/verifiers/plugin.py" \
    --reward_funcs "ve_${ARM}" \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization "${VLLM_UTIL:-0.55}" \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_model_len 4096 \
    --max_length 1024 \
    --max_completion_length "${MAX_COMPLETION:-2048}" \
    --dataset "${DATASET:-$BASE/data/train.jsonl}" \
    --system "$BASE/configs/system_prompt.txt" \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --seed "$SEED" \
    --max_steps "$MAX_STEPS" \
    --per_device_train_batch_size "${PER_DEV_BS:-8}" \
    --num_generations "${NUM_GEN:-8}" \
    --steps_per_generation 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate "${LR:-1e-6}" \
    --temperature 1.0 \
    --beta 0.001 \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --importance_sampling_level token \
    --loss_type grpo \
    --advantage_estimator grpo \
    --num_iterations 1 \
    --async_generate false \
    --sleep_level 1 \
    --offload_model true \
    --offload_optimizer true \
    --save_steps "${SAVE_STEPS:-25}" \
    --save_total_limit 100 \
    --save_only_model true \
    --eval_steps 1000000 \
    --logging_steps 1 \
    --log_completions true \
    --report_to tensorboard \
    --dataloader_num_workers 2 \
    --output_dir "$RUN/ckpt" \
    2>&1 | tee -a "$RUN/train.log"
