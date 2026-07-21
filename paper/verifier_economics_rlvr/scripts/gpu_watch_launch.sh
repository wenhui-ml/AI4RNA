#!/usr/bin/env bash
# Watch for idle GPUs; when NEED_GPUS are stably idle, launch the 3-arm 3B pilot in tmux, one
# session per arm on its own GPU pair. After each arm trains, its session automatically runs
# held-out checkpoint eval (eval_checkpoints.py) and curve aggregation (aggregate.py).
# Run me inside tmux:  tmux new -d -s ve_monitor "bash scripts/gpu_watch_launch.sh"
# Env: NEED_GPUS=6 MEM_IDLE_MB=8000 UTIL_IDLE=10 STABLE_POLLS=3 POLL_S=60 SEED=0 STEPS=250 TAG=pilot3b
set -uo pipefail
BASE=/data/wenhui/openclaude_rna/paper/verifier_economics_rlvr
PY=$HOME/miniconda3/envs/cua/bin/python
NEED_GPUS=${NEED_GPUS:-6}
MEM_IDLE_MB=${MEM_IDLE_MB:-8000}
UTIL_IDLE=${UTIL_IDLE:-10}
STABLE_POLLS=${STABLE_POLLS:-3}
POLL_S=${POLL_S:-60}
SEED=${SEED:-0}
STEPS=${STEPS:-250}
TAG=${TAG:-pilot3b}
LOG=$BASE/runs/monitor.log
mkdir -p "$BASE/runs"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

idle_gpus() {  # prints comma-joined idle gpu ids
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
        | awk -F', *' -v m="$MEM_IDLE_MB" -v u="$UTIL_IDLE" '$2 < m && $3 < u {print $1}' | paste -sd,
}

log "monitor start: need $NEED_GPUS idle GPUs (<${MEM_IDLE_MB}MiB, <${UTIL_IDLE}% util) for $STABLE_POLLS consecutive polls every ${POLL_S}s"
stable=0
while true; do
    IDS=$(idle_gpus)
    N=$(awk -F, '{print NF}' <<< "${IDS:-}")
    [ -z "${IDS:-}" ] && N=0
    if [ "$N" -ge "$NEED_GPUS" ]; then
        stable=$((stable + 1))
        log "idle gpus: [$IDS] ($N >= $NEED_GPUS), stable $stable/$STABLE_POLLS"
    else
        [ "$stable" -gt 0 ] && log "idle gpus: [$IDS] — lost stability, reset"
        stable=0
    fi
    if [ "$stable" -ge "$STABLE_POLLS" ]; then
        break
    fi
    sleep "$POLL_S"
done

IFS=',' read -ra G <<< "$IDS"
log "LAUNCHING 3-arm pilot (tag=$TAG seed=$SEED steps=$STEPS) on gpus [${G[*]:0:6}]"
i=0
for ARM in cheap strong cascade; do
    PAIR="${G[$i]},${G[$((i + 1))]}"
    i=$((i + 2))
    RUN=$BASE/runs/${TAG}_${ARM}_s${SEED}
    SESSION=ve_${ARM}
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    tmux new-session -d -s "$SESSION" \
        "export PATH=$HOME/miniconda3/envs/cua/bin:\$PATH; \
         bash $BASE/configs/pilot_3b.sh $ARM $PAIR $SEED $STEPS $TAG; \
         echo \"train exit=\$? — running held-out eval\"; \
         CUDA_VISIBLE_DEVICES=${G[$((i - 2))]} USE_HF=1 HF_HUB_OFFLINE=1 $PY $BASE/scripts/eval_checkpoints.py $RUN --base /data/wenhui/LowRankClone_improve/models/Qwen2.5-3B-Instruct --util 0.4 2>&1 | tee -a $RUN/eval.log; \
         $PY $BASE/scripts/aggregate.py $RUN 2>&1 | tee -a $RUN/eval.log; \
         echo ALL_DONE_${ARM}; exec bash"
    log "  arm=$ARM gpus=$PAIR tmux=$SESSION run=$RUN"
done
log "all three arms launched; monitor exiting. Attach: tmux attach -t ve_cheap|ve_strong|ve_cascade"
