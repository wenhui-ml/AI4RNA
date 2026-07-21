#!/usr/bin/env bash
# Work-stealing dispatcher (model-general): keeps all 4 GPU pairs busy with NO phase barriers.
# Pool of pending training jobs; when a pair's job fully completes (curve.csv), immediately launches
# the next pending job there. Each job self-describes its model, so Qwen and Llama jobs coexist.
# CRITICAL disk hygiene: each job DELETES its own ckpt/ right after eval writes held-out (the paper
# needs the numbers in heldout.jsonl/curve.csv, not the weights) -> disk stays bounded.
#   Dry run: DISPATCH_DRYRUN=1 bash dispatcher.sh
#   Real:    tmux new -d -s ve_dispatch "bash dispatcher.sh"
set -uo pipefail
BASE=/data/wenhui/openclaude_rna/paper/verifier_economics_rlvr
PY=$HOME/miniconda3/envs/cua/bin/python
M7B=/data/wenhui/LowRankClone_improve/models/Qwen2.5-7B-Instruct
ML8=/data/wenhui/LowRankClone_improve/models/Llama-3.1-8B-Instruct
Q7B=$BASE/data/train_7b_filtered.jsonl
QL8=$BASE/data/train_llama8b_filtered.jsonl
LOG=$BASE/runs/dispatch.log
DRY=${DISPATCH_DRYRUN:-0}
log() { echo "[$(date '+%F %T')] [dispatch] $*" | tee -a "$LOG"; }

# Job spec = arm|dial_p|seed|steps|lr|save|modelkey   (modelkey: q7b=Qwen2.5-7B, l8b=Llama-3.1-8B)
# long: 2000 steps, lr 5e-6, save 400 (5 ckpts) -> held-out-repair probe. dial: 800, lr 2e-6, save 200.
# llama cheap: 800, lr 2e-6, save 200 (the one Llama arm still owed). Reduced save freq keeps disk low;
# ckpt is deleted after eval regardless.
declare -A SPEC=(
  [long7b_strong_s0]="strong||0|2000|5e-6|400|q7b"
  [long7b_cascade_s0]="cascade||0|2000|5e-6|400|q7b"
  [long7b_cheap_s0]="cheap||0|2000|5e-6|400|q7b"
  [long7b_strong_s1]="strong||1|2000|5e-6|400|q7b"
  [llama8b_cheap_s0]="cheap||0|800|2e-6|200|l8b"
  [main7bd025_dial_s0]="dial|0.25|0|800|2e-6|200|q7b"
  [main7bd050_dial_s0]="dial|0.5|0|800|2e-6|200|q7b"
  [main7bd050_dial_s1]="dial|0.5|1|800|2e-6|200|q7b"
  [main7bd075_dial_s0]="dial|0.75|0|800|2e-6|200|q7b"
)
# LPT: longest (2000-step) first; then the 800-step jobs.
PENDING=(long7b_strong_s0 long7b_cascade_s0 long7b_cheap_s0 long7b_strong_s1 \
         llama8b_cheap_s0 main7bd025_dial_s0 main7bd050_dial_s0 main7bd050_dial_s1 main7bd075_dial_s0)

PAIRS=("0,1" "2,3" "4,5" "6,7")
declare -A PORT=([0,1]=29661 [2,3]=29662 [4,5]=29663 [6,7]=29664)
declare -A CUR=([0,1]="" [2,3]="" [4,5]="" [6,7]="")   # all pairs free at start

launch() {  # name pair
    local NAME=$1 PAIR=$2 G1=${2%%,*} P=${PORT[$2]}
    IFS='|' read -r ARM DIALP SEED STEPS JLR JSAVE MKEY <<< "${SPEC[$NAME]}"
    local MODEL MT TMPL DS
    if [ "$MKEY" = l8b ]; then MODEL=$ML8; MT=llama; TMPL=llama3_2; DS=$QL8
    else MODEL=$M7B; MT=""; TMPL=""; DS=$Q7B; fi
    local RUN=$BASE/runs/$NAME S=ve_$NAME TAGROOT=${NAME%_${ARM}_s${SEED}}
    if [ "$DRY" = 1 ]; then
        log "DRYRUN $NAME on $PAIR (arm=$ARM p=${DIALP:-–} seed=$SEED steps=$STEPS lr=$JLR save=$JSAVE model=$MKEY port=$P tag=$TAGROOT)"
        return
    fi
    tmux kill-session -t "$S" 2>/dev/null || true
    tmux new-session -d -s "$S" \
        "export PATH=$HOME/miniconda3/envs/cua/bin:\$PATH; \
         ${DIALP:+VE_DIAL_P=$DIALP} ${MT:+MODEL_TYPE=$MT} ${TMPL:+TEMPLATE=$TMPL} \
         MASTER_PORT=$P LR=$JLR MODEL=$MODEL DATASET=$DS SAVE_STEPS=$JSAVE \
         bash $BASE/configs/pilot_3b.sh $ARM $PAIR $SEED $STEPS $TAGROOT; \
         echo train_exit=\$?; \
         CUDA_VISIBLE_DEVICES=$G1 USE_HF=1 HF_HUB_OFFLINE=1 $PY $BASE/scripts/eval_checkpoints.py $RUN --base $MODEL --util 0.4 2>&1 | tee -a $RUN/eval.log; \
         $PY $BASE/scripts/aggregate.py $RUN 2>&1 | tee -a $RUN/eval.log; \
         if [ -f $RUN/curve.csv ]; then rm -rf $RUN/ckpt && echo CKPT_CLEANED_$NAME; fi; \
         echo ALL_DONE_$NAME; exec bash"
    log "launched $NAME on $PAIR (arm=$ARM p=${DIALP:-–} seed=$SEED steps=$STEPS lr=$JLR save=$JSAVE model=$MKEY port=$P)"
}

done_run() { [ -f "$BASE/runs/$1/curve.csv" ]; }

log "dispatcher start (DRY=$DRY). pending=${#PENDING[@]} pairs=${#PAIRS[@]}"
while :; do
    for PAIR in "${PAIRS[@]}"; do
        cur=${CUR[$PAIR]:-}
        if [ -n "$cur" ] && ! done_run "$cur"; then continue; fi
        if [ ${#PENDING[@]} -eq 0 ]; then CUR[$PAIR]=""; continue; fi
        next=${PENDING[0]}; PENDING=("${PENDING[@]:1}")
        [ -n "$cur" ] && log "pair $PAIR freed (‘$cur’ done)"
        launch "$next" "$PAIR"; CUR[$PAIR]=$next
        [ "$DRY" = 1 ] && continue
        sleep 25
    done
    busy=0; for PAIR in "${PAIRS[@]}"; do c=${CUR[$PAIR]:-}; [ -n "$c" ] && ! done_run "$c" && busy=1; done
    [ ${#PENDING[@]} -eq 0 ] && [ $busy -eq 0 ] && break
    [ "$DRY" = 1 ] && { log "DRYRUN done"; break; }
    sleep 60
done
log "DISPATCH COMPLETE: all jobs done"
