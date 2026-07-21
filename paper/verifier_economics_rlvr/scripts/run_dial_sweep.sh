#!/usr/bin/env bash
# Risk-calibrated dial sweep: one GRPO run per (escalation prob p, seed), tracing the continuous
# cost x robustness frontier between cheap (p=0) and cascade (p=1). Each run trains then auto-evals.
# Explicit spec list (full control over GPU packing) — each spec is  p:seed:g1:g2:port
# Usage: run_dial_sweep.sh <model> <dataset> <tag> <steps> <spec> [spec ...]
set -uo pipefail
MODEL=$1; DATASET=$2; TAG=$3; STEPS=$4; shift 4
SPECS=("$@")
BASE=/data/wenhui/openclaude_rna/paper/verifier_economics_rlvr
PY=$HOME/miniconda3/envs/cua/bin/python
LOG=$BASE/runs/dial_sweep.log
export PATH=$HOME/miniconda3/envs/cua/bin:$PATH
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

run_dir_for() {  # p seed -> run dir path
    local PP; PP=$(printf '%03d' "$(awk "BEGIN{printf \"%d\", $1*100}")")
    echo "$BASE/runs/${TAG}d${PP}_dial_s${2}"
}

launch() {  # p seed g1 g2 port
    local P=$1 SEED=$2 G1=$3 G2=$4 PORT=$5
    local PP; PP=$(printf '%03d' "$(awk "BEGIN{printf \"%d\", $P*100}")")
    local PTAG=${TAG}d${PP}                       # run dir = ${PTAG}_dial_s${SEED}
    local RUN=$BASE/runs/${PTAG}_dial_s${SEED} S=ve_${PTAG}_dial_s${SEED}
    tmux kill-session -t "$S" 2>/dev/null || true
    tmux new-session -d -s "$S" \
        "export PATH=$HOME/miniconda3/envs/cua/bin:\$PATH; \
         VE_DIAL_P=$P MASTER_PORT=$PORT LR=2e-6 MODEL=$MODEL DATASET=$DATASET SAVE_STEPS=100 \
         bash $BASE/configs/pilot_3b.sh dial $G1,$G2 $SEED $STEPS $PTAG; \
         echo \"train exit=\$? — eval\"; \
         CUDA_VISIBLE_DEVICES=$G1 USE_HF=1 HF_HUB_OFFLINE=1 $PY $BASE/scripts/eval_checkpoints.py $RUN --base $MODEL --util 0.4 2>&1 | tee -a $RUN/eval.log; \
         $PY $BASE/scripts/aggregate.py $RUN 2>&1 | tee -a $RUN/eval.log; \
         echo ALL_DONE_${PTAG}_s${SEED}; exec bash"
    log "  launched dial p=$P seed=$SEED gpus=$G1,$G2 port=$PORT run=$RUN"
}

log "DIAL SWEEP start: tag=$TAG steps=$STEPS specs=[${SPECS[*]}] model=$MODEL"
for spec in "${SPECS[@]}"; do
    IFS=':' read -r P SEED G1 G2 PORT <<< "$spec"
    launch "$P" "$SEED" "$G1" "$G2" "$PORT"
    sleep 20   # stagger vLLM inits
done

for spec in "${SPECS[@]}"; do
    IFS=':' read -r P SEED _ _ _ <<< "$spec"
    until [ -f "$(run_dir_for "$P" "$SEED")/curve.csv" ]; do sleep 120; done
    log "  complete: p=$P seed=$SEED"
done
log "DIAL SWEEP COMPLETE: tag=$TAG specs=[${SPECS[*]}]"
