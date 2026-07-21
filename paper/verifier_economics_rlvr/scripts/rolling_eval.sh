#!/usr/bin/env bash
# Rolling checkpoint evaluator: repeatedly eval new checkpoints of the given runs on one GPU
# until each run's chain has written curve.csv (training + final eval complete).
# Usage: rolling_eval.sh <gpu> <base_model_path> <run_name> [run_name ...]
set -uo pipefail
GPU=$1; BASEM=$2; shift 2
BASE=/data/wenhui/openclaude_rna/paper/verifier_economics_rlvr
PY=$HOME/miniconda3/envs/cua/bin/python
export PATH=$HOME/miniconda3/envs/cua/bin:$PATH
while true; do
    alldone=1
    for r in "$@"; do
        [ -f "$BASE/runs/$r/curve.csv" ] && continue
        alldone=0
        [ -d "$BASE/runs/$r" ] || continue
        CUDA_VISIBLE_DEVICES=$GPU USE_HF=1 HF_HUB_OFFLINE=1 "$PY" "$BASE/scripts/eval_checkpoints.py" \
            "$BASE/runs/$r" --base "$BASEM" --util 0.5 --min-age-s 180 2>&1 | tail -3 \
            | tee -a "$BASE/runs/$r/eval_rolling.log"
    done
    [ "$alldone" -eq 1 ] && break
    sleep 180
done
echo "ROLLING_EVAL_DONE gpu$GPU: $*"
