#!/usr/bin/env bash
# 7B MAIN experiment, fully unattended (run inside tmux):
#   phase 1: difficulty-filter 24k train prompts with base-7B pass-rate, 8 GPUs in parallel (~40 min)
#   phase 2: launch 3 arms (cheap/strong/cascade) x 800 steps x lr 2e-6 on filtered data, 2 GPUs each,
#            via per-arm tmux sessions that auto-chain train -> held-out eval -> aggregate.
# Usage: tmux new -d -s ve_main7b "bash scripts/main7b_pipeline.sh"
set -uo pipefail
BASE=/data/wenhui/openclaude_rna/paper/verifier_economics_rlvr
PY=$HOME/miniconda3/envs/cua/bin/python
MODEL=${MODEL:-/data/wenhui/LowRankClone_improve/models/Qwen2.5-7B-Instruct}
TAG=${TAG:-main7b}
FILTERED=${FILTERED:-$BASE/data/train_7b_filtered.jsonl}
FILTER_DIR=${FILTER_DIR:-$BASE/data/filter7b}
export PATH=$HOME/miniconda3/envs/cua/bin:$PATH
export USE_HF=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
LOG=$BASE/runs/${TAG}_pipeline.log
STEPS=${STEPS:-800}
SEED=${SEED:-0}
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

if [ ! -s "$FILTERED" ]; then
    log "phase 1: difficulty filtering on 8 GPUs (model=$MODEL)"
    for i in 0 1 2 3 4 5 6 7; do
        CUDA_VISIBLE_DEVICES=$i $PY $BASE/scripts/filter_difficulty.py --shard $i --num-shards 8 \
            --model "$MODEL" --out-dir "$FILTER_DIR" > "${FILTER_DIR}_shard$i.log" 2>&1 &
    done
    wait
    $PY $BASE/scripts/filter_difficulty.py --merge --out-dir "$FILTER_DIR" --train-out "$FILTERED" 2>&1 | tee -a "$LOG"
    if [ ! -s "$FILTERED" ]; then log "FILTER FAILED — aborting (see ${FILTER_DIR}_shard*.log)"; exit 1; fi
else
    log "phase 1 skipped: $FILTERED already exists"
fi

log "phase 2: launching 3 arms (${STEPS} steps, lr 2e-6, seed $SEED)"
i=0
for ARM in cheap strong cascade; do
    G1=$((i)); G2=$((i + 1)); i=$((i + 2))
    RUN=$BASE/runs/${TAG}_${ARM}_s${SEED}
    SESSION=ve_${TAG}_${ARM}_s${SEED}
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    tmux new-session -d -s "$SESSION" \
        "export PATH=$HOME/miniconda3/envs/cua/bin:\$PATH; \
         LR=2e-6 MODEL=$MODEL DATASET=$FILTERED SAVE_STEPS=100 \
         bash $BASE/configs/pilot_3b.sh $ARM $G1,$G2 $SEED $STEPS $TAG; \
         echo \"train exit=\$? — running held-out eval\"; \
         CUDA_VISIBLE_DEVICES=$G1 USE_HF=1 HF_HUB_OFFLINE=1 $PY $BASE/scripts/eval_checkpoints.py $RUN --base $MODEL --util 0.4 2>&1 | tee -a $RUN/eval.log; \
         $PY $BASE/scripts/aggregate.py $RUN 2>&1 | tee -a $RUN/eval.log; \
         echo ALL_DONE_${TAG}_${ARM}_s${SEED}; exec bash"
    log "  arm=$ARM gpus=$G1,$G2 tmux=$SESSION run=$RUN"
done
log "all arms launched; pipeline orchestrator exiting"
