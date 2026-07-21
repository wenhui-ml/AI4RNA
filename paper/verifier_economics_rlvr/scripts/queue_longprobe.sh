#!/usr/bin/env bash
# Longer-horizon / higher-lr probe, appended AFTER the dial sweep (strictly last; zero GPU contention,
# never edits a running script). Goal: push held-out up for strong and show cheap falls behind at a
# horizon where quality actually moves. 3 arms x 2000 steps x lr 5e-6 on 7B Qwen + filtered data.
# Run: tmux new -d -s ve_queue_long "bash scripts/queue_longprobe.sh"
set -uo pipefail
BASE=/data/wenhui/openclaude_rna/paper/verifier_economics_rlvr
PY=$HOME/miniconda3/envs/cua/bin/python
M7B=/data/wenhui/LowRankClone_improve/models/Qwen2.5-7B-Instruct
Q7B=$BASE/data/train_7b_filtered.jsonl
LOG=$BASE/runs/queue.log
STEPS=2000
LR=5e-6
SEED=0
TAG=long7b
log() { echo "[$(date '+%F %T')] [long] $*" | tee -a "$LOG"; }

# prereq: wait for the dial sweep (the last thing in the chain) to finish, so GPUs are free
log "waiting for dial sweep to finish before long-horizon probe"
for r in main7bd025_dial_s0 main7bd050_dial_s0 main7bd075_dial_s0 main7bd050_dial_s1; do
    until [ -f "$BASE/runs/$r/curve.csv" ]; do sleep 300; done
    log "prereq complete: $r"
done

log "launching long probe x ${STEPS} steps x lr ${LR}: cheap/strong/cascade s0 + strong s1 -> 8 GPUs"
# spec: arm:seed:g1:g2:port  (strong seed1 on 6,7 fills the last pair with a useful second seed)
launch_long() {  # arm seed g1 g2 port
    local ARM=$1 SD=$2 G1=$3 G2=$4 PORT=$5
    local RUN=$BASE/runs/${TAG}_${ARM}_s${SD} S=ve_${TAG}_${ARM}_s${SD}
    tmux kill-session -t "$S" 2>/dev/null || true
    tmux new-session -d -s "$S" \
        "export PATH=$HOME/miniconda3/envs/cua/bin:\$PATH; \
         MASTER_PORT=$PORT LR=$LR MODEL=$M7B DATASET=$Q7B SAVE_STEPS=200 \
         bash $BASE/configs/pilot_3b.sh $ARM $G1,$G2 $SD $STEPS $TAG; \
         echo \"train exit=\$? — eval\"; \
         CUDA_VISIBLE_DEVICES=$G1 USE_HF=1 HF_HUB_OFFLINE=1 $PY $BASE/scripts/eval_checkpoints.py $RUN --base $M7B --util 0.4 2>&1 | tee -a $RUN/eval.log; \
         $PY $BASE/scripts/aggregate.py $RUN 2>&1 | tee -a $RUN/eval.log; \
         echo ALL_DONE_${TAG}_${ARM}_s${SD}; exec bash"
    log "  launched ${TAG}/${ARM}/s${SD} gpus=$G1,$G2 port=$PORT run=$RUN"
}
for spec in cheap:0:0:1:29641 strong:0:2:3:29642 cascade:0:4:5:29643 strong:1:6:7:29644; do
    IFS=':' read -r ARM SD G1 G2 PORT <<< "$spec"
    launch_long "$ARM" "$SD" "$G1" "$G2" "$PORT"
    sleep 20
done
log "long probe dispatched (cheap/strong/cascade s0 + strong s1, 2000 steps, lr 5e-6)"
