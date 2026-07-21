#!/usr/bin/env bash
# Waiter that appends the dial sweep AFTER the existing ve_queue finishes (its final artifacts are
# the llama8b runs), so there is zero GPU contention and the running queue script is never touched.
# Dial frontier is run on the 7B Qwen main policy + its filtered data at p in {0.25,0.5,0.75}
# (cheap=p0 and cascade=p1 already exist from main7b, so these 3 interior points complete the curve).
# Run: tmux new -d -s ve_queue_dial "bash scripts/queue_dial.sh"
set -uo pipefail
BASE=/data/wenhui/openclaude_rna/paper/verifier_economics_rlvr
M7B=/data/wenhui/LowRankClone_improve/models/Qwen2.5-7B-Instruct
Q7B=$BASE/data/train_7b_filtered.jsonl
LOG=$BASE/runs/queue.log
log() { echo "[$(date '+%F %T')] [dial] $*" | tee -a "$LOG"; }

log "waiting for existing queue to finish (llama8b runs) before dial sweep"
for r in llama8b_cheap_s0 llama8b_strong_s0 llama8b_cascade_s0 llama8b_strong_s1; do
    until [ -f "$BASE/runs/$r/curve.csv" ]; do sleep 300; done
    log "prereq complete: $r"
done

log "launching dial sweep on 7B Qwen: p=0.25/0.5/0.75 (seed0) + p=0.5 seed1 -> full 8 GPUs"
bash "$BASE/scripts/run_dial_sweep.sh" "$M7B" "$Q7B" main7b 800 \
    0.25:0:0:1:29631 \
    0.5:0:2:3:29632 \
    0.75:0:4:5:29633 \
    0.5:1:6:7:29634
log "dial sweep dispatched; queue_dial done"
