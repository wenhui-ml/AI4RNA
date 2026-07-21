#!/usr/bin/env bash
# Experiment queue, TRAINING-FIRST allocation (user directive 2026-07-20: all GPUs train; evals run
# on each arm's own GPUs after its training, via the per-arm chains).
# Already running outside this script: main7b seed-0 arms (GPUs 0-5), main7b strong seed-1 (GPUs 6,7).
# Phase B (when seed-0 curves done): seed-1 cheap (0,1) + cascade (4,5); Llama filter on GPUs 2,3.
# Phase C (when all seed-1 curves + llama filter done): llama8b s0 3 arms (0-5) + llama8b strong s1 (6,7).
set -uo pipefail
BASE=/data/wenhui/openclaude_rna/paper/verifier_economics_rlvr
PY=$HOME/miniconda3/envs/cua/bin/python
M7B=/data/wenhui/LowRankClone_improve/models/Qwen2.5-7B-Instruct
ML8=/data/wenhui/LowRankClone_improve/models/Llama-3.1-8B-Instruct
Q7B=$BASE/data/train_7b_filtered.jsonl
QL8=$BASE/data/train_llama8b_filtered.jsonl
LOG=$BASE/runs/queue.log
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

wait_runs() {
    for r in "$@"; do
        until [ -f "$BASE/runs/$r/curve.csv" ]; do sleep 300; done
        log "complete: $r"
    done
}

# launch_arm <tag> <arm> <gpus> <seed> <port> <model> <dataset>
launch_arm() {
    local TAG=$1 ARM=$2 GPUS=$3 SEED=$4 PORT=$5 MODEL=$6 DS=$7
    local RUN=$BASE/runs/${TAG}_${ARM}_s${SEED} G1=${GPUS%%,*} S=ve_${TAG}_${ARM}_s${SEED}
    tmux kill-session -t "$S" 2>/dev/null || true
    tmux new-session -d -s "$S" \
        "export PATH=$HOME/miniconda3/envs/cua/bin:\$PATH; \
         MASTER_PORT=$PORT LR=2e-6 MODEL=$MODEL DATASET=$DS SAVE_STEPS=100 \
         bash $BASE/configs/pilot_3b.sh $ARM $GPUS $SEED 800 $TAG; \
         echo \"train exit=\$? — running held-out eval\"; \
         CUDA_VISIBLE_DEVICES=$G1 USE_HF=1 HF_HUB_OFFLINE=1 $PY $BASE/scripts/eval_checkpoints.py $RUN --base $MODEL --util 0.4 2>&1 | tee -a $RUN/eval.log; \
         $PY $BASE/scripts/aggregate.py $RUN 2>&1 | tee -a $RUN/eval.log; \
         echo ALL_DONE_${TAG}_${ARM}_s${SEED}; exec bash"
    log "  launched ${TAG}/${ARM}/s${SEED} gpus=$GPUS port=$PORT tmux=$S"
}

log "QUEUE(v2) start: waiting for main7b seed 0 (strong s1 already training on 6,7)"
wait_runs main7b_cheap_s0 main7b_strong_s0 main7b_cascade_s0

log "phase B: seed-1 cheap+cascade; llama filter on GPUs 2,3"
launch_arm main7b cheap   0,1 1 29611 "$M7B" "$Q7B"
launch_arm main7b cascade 4,5 1 29613 "$M7B" "$Q7B"
if [ ! -s "$QL8" ]; then
    tmux kill-session -t ve_filter_llama 2>/dev/null || true
    tmux new-session -d -s ve_filter_llama \
        "export PATH=$HOME/miniconda3/envs/cua/bin:\$PATH; \
         for i in 0 2 4 6; do CUDA_VISIBLE_DEVICES=2 USE_HF=1 HF_HUB_OFFLINE=1 $PY $BASE/scripts/filter_difficulty.py --shard \$i --num-shards 8 --model $ML8 --out-dir $BASE/data/filter_llama8b > $BASE/data/filter_llama8b_shard\$i.log 2>&1 & \
         CUDA_VISIBLE_DEVICES=3 USE_HF=1 HF_HUB_OFFLINE=1 $PY $BASE/scripts/filter_difficulty.py --shard \$((i+1)) --num-shards 8 --model $ML8 --out-dir $BASE/data/filter_llama8b > $BASE/data/filter_llama8b_shard\$((i+1)).log 2>&1 & \
         wait; done; \
         $PY $BASE/scripts/filter_difficulty.py --merge --out-dir $BASE/data/filter_llama8b --train-out $QL8 2>&1 | tee $BASE/data/filter_llama8b_merge.log; \
         echo LLAMA_FILTER_DONE; exec bash"
    log "  llama filter running on GPUs 2,3 (4 rounds x 2 shards)"
fi
wait_runs main7b_strong_s1 main7b_cheap_s1 main7b_cascade_s1
until [ -s "$QL8" ]; do log "waiting for llama filter"; sleep 300; done

log "phase C: llama8b s0 x 3 arms + llama8b strong s1 (8 GPUs full)"
launch_arm llama8b cheap   0,1 0 29611 "$ML8" "$QL8"
launch_arm llama8b strong  2,3 0 29612 "$ML8" "$QL8"
launch_arm llama8b cascade 4,5 0 29613 "$ML8" "$QL8"
launch_arm llama8b strong  6,7 1 29622 "$ML8" "$QL8"
wait_runs llama8b_cheap_s0 llama8b_strong_s0 llama8b_cascade_s0 llama8b_strong_s1

log "QUEUE COMPLETE: main7b s0+s1(strong,cheap,cascade) + llama8b s0(3 arms)+strong s1"
