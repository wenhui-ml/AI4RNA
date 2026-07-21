#!/usr/bin/env bash
# Backfill the checkpoint-800 held-out point for the 3 Llama runs that were evaluated BEFORE
# sentencepiece/tiktoken was installed (their eval failed only on the final checkpoint's tokenizer
# conversion). Waits for the dispatcher batch to finish (GPUs free) so it never competes with or
# risks OOM-ing a training run. eval_checkpoints.py is idempotent -> only the missing 800 is scored.
set -uo pipefail
BASE=/data/wenhui/openclaude_rna/paper/verifier_economics_rlvr
PY=$HOME/miniconda3/envs/cua/bin/python
ML8=/data/wenhui/LowRankClone_improve/models/Llama-3.1-8B-Instruct
LOG=$BASE/runs/queue.log
export PATH=$HOME/miniconda3/envs/cua/bin:$PATH
log() { echo "[$(date '+%F %T')] [reeval] $*" | tee -a "$LOG"; }

log "waiting for dispatcher batch to complete before backfilling Llama step-800 evals"
until grep -q "DISPATCH COMPLETE" "$BASE/runs/dispatch.log" 2>/dev/null; do sleep 300; done
log "dispatcher done; GPUs free -> backfilling"
for r in llama8b_strong_s0 llama8b_cascade_s0 llama8b_strong_s1; do
    n=$(wc -l < "$BASE/runs/$r/heldout.jsonl" 2>/dev/null || echo 0)
    if [ "$n" -ge 9 ]; then log "$r already complete ($n pts), skip"; continue; fi
    CUDA_VISIBLE_DEVICES=0 USE_HF=1 HF_HUB_OFFLINE=1 "$PY" "$BASE/scripts/eval_checkpoints.py" \
        "$BASE/runs/$r" --base "$ML8" --util 0.4 2>&1 | tee -a "$BASE/runs/$r/eval.log"
    "$PY" "$BASE/scripts/aggregate.py" "$BASE/runs/$r" 2>&1 | tail -1 | tee -a "$BASE/runs/$r/eval.log"
    log "$r backfilled -> $(wc -l < $BASE/runs/$r/heldout.jsonl) pts"
    [ -f "$BASE/runs/$r/curve.csv" ] && rm -rf "$BASE/runs/$r/ckpt" && log "$r ckpt cleaned (disk hygiene)"
done
log "Llama step-800 backfill complete"
