#!/usr/bin/env bash
# Launch all 4 Llama-3.1-8B runs, packing free GPUs first. Llama needs explicit model_type/template
# (swift can't auto-detect from the local dir name): MODEL_TYPE=llama, TEMPLATE=llama3_2 (verified
# against swift/model/models/llama.py). Produces exactly the run dirs ve_queue_dial waits for.
# GPU plan: 2,3 / 4,5 / 6,7 are free now -> launch strong_s0 / strong_s1 / cascade_s0 immediately;
# 0,1 still busy with main7b_cheap_s1 eval -> launch cheap_s0 there once it frees (curve.csv).
set -uo pipefail
BASE=/data/wenhui/openclaude_rna/paper/verifier_economics_rlvr
PY=$HOME/miniconda3/envs/cua/bin/python
ML8=/data/wenhui/LowRankClone_improve/models/Llama-3.1-8B-Instruct
QL8=$BASE/data/train_llama8b_filtered.jsonl
LOG=$BASE/runs/queue.log
log() { echo "[$(date '+%F %T')] [llama-now] $*" | tee -a "$LOG"; }

launch() {  # arm seed g1 g2 port
    local ARM=$1 SD=$2 G1=$3 G2=$4 PORT=$5
    local RUN=$BASE/runs/llama8b_${ARM}_s${SD} S=ve_llama8b_${ARM}_s${SD}
    tmux kill-session -t "$S" 2>/dev/null || true
    tmux new-session -d -s "$S" \
        "export PATH=$HOME/miniconda3/envs/cua/bin:\$PATH; \
         MASTER_PORT=$PORT LR=2e-6 MODEL=$ML8 MODEL_TYPE=llama TEMPLATE=llama3_2 DATASET=$QL8 SAVE_STEPS=100 \
         bash $BASE/configs/pilot_3b.sh $ARM $G1,$G2 $SD 800 llama8b; \
         echo \"train exit=\$? — eval\"; \
         CUDA_VISIBLE_DEVICES=$G1 USE_HF=1 HF_HUB_OFFLINE=1 $PY $BASE/scripts/eval_checkpoints.py $RUN --base $ML8 --util 0.4 2>&1 | tee -a $RUN/eval.log; \
         $PY $BASE/scripts/aggregate.py $RUN 2>&1 | tee -a $RUN/eval.log; \
         echo ALL_DONE_llama8b_${ARM}_s${SD}; exec bash"
    log "  launched llama8b/${ARM}/s${SD} gpus=$G1,$G2 port=$PORT"
}

log "TAKEOVER v2: launching Llama-8B on free GPUs 2,3 / 4,5 / 6,7 now"
launch strong  0 2 3 29651
launch strong  1 4 5 29654
launch cascade 0 6 7 29653

log "waiting for GPUs 0,1 to free (main7b_cheap_s1 curve.csv) -> llama cheap_s0"
until [ -f "$BASE/runs/main7b_cheap_s1/curve.csv" ]; do sleep 60; done
launch cheap 0 0 1 29652

log "all 4 Llama runs launched; downstream ve_queue_dial takes over"
