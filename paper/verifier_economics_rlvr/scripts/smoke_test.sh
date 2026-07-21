#!/usr/bin/env bash
# Full-loop smoke test on SHARED gpus (others' jobs present): 2 optimizer steps, cascade arm,
# small completions, modest vLLM memory slice. Validates swift+vllm+plugin+logging end to end.
# Usage: smoke_test.sh <gpus e.g. 4,7>
set -euo pipefail
GPUS=${1:?gpus}
BASE=/data/wenhui/openclaude_rna/paper/verifier_economics_rlvr
rm -rf "$BASE/runs/smoke_cascade_s0"
VLLM_UTIL=0.45 DATASET=$BASE/data/train_smoke.jsonl SAVE_STEPS=1000 \
MAX_COMPLETION=1024 NUM_GEN=4 PER_DEV_BS=4 \
bash "$BASE/configs/pilot_3b.sh" cascade "$GPUS" 0 2 smoke
echo '=== smoke: aggregate ==='
~/miniconda3/envs/cua/bin/python "$BASE/scripts/aggregate.py" "$BASE/runs/smoke_cascade_s0"
