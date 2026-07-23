#!/bin/bash
# 3-way pure-base-model Eterna100, CONCURRENT (all 3 models at once). Leak now fixed, so the only
# risk is CPU oversubscription: 3 runs x 3 workers = 9 episodes, each parallel_search capped at
# RNA_MAX_CORES=20 -> worst case 9x20=180 cores (< 256, leaves headroom for vllm/lrc users).
set -u
export OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY (OpenRouter) in your environment}"
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
export RNA_MAX_CORES=20
export PYTHONPATH=/data/wenhui/openclaude_rna
PY=/home/ubuntu/miniconda3/envs/rna/bin/python
cd /data/wenhui/openclaude_rna || exit 1
COMMON="--max-len 400 --n-tasks 100 --max-budget-usd 100 --timeout-s 1500 --workers 3"
MASTER=rna_harness/runs/three_way_master.log
rm -f rna_harness/runs/.three_way_done
echo "=== 3-WAY CONCURRENT START $(date) (RNA_MAX_CORES=$RNA_MAX_CORES, 3 runs x 3 workers) ===" > "$MASTER"

run() {  # model, out-dir
  echo "=== START $1 -> $2 @ $(date) ===" >> "$MASTER"
  $PY -m rna_harness.experiments.agent_runner --model "$1" $COMMON \
      --out "rna_harness/runs/$2" > "rna_harness/runs/${2}_run.log" 2>&1
  echo "=== DONE $1 (exit $?) @ $(date) ===" >> "$MASTER"
}

run "z-ai/glm-5.2"              "glm5.2_eterna100" &
run "deepseek/deepseek-v4-pro"  "deepseek_v4pro_eterna100" &
run "anthropic/claude-sonnet-5" "sonnet5_eterna100" &
wait
echo "=== ALL 3 COMPLETE $(date) ===" >> "$MASTER"
touch rna_harness/runs/.three_way_done
