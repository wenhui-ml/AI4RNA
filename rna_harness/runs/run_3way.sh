#!/bin/bash
# 3-way pure-base-model Eterna100, SEQUENTIAL (one model at a time) so the shared 256-core box
# never oversubscribes. Root-fixed harness: exact-length + emit-best-so-far prompt, best-effort
# capture, leak-proof compute pools. RNA_MAX_CORES=48 -> worst case 3 workers x 48 = 144 cores.
set -u
export OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY (OpenRouter) in your environment}"
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
export RNA_MAX_CORES=48
export PYTHONPATH=/data/wenhui/openclaude_rna
PY=/home/ubuntu/miniconda3/envs/rna/bin/python
cd /data/wenhui/openclaude_rna || exit 1
COMMON="--max-len 400 --n-tasks 100 --max-budget-usd 100 --timeout-s 1500 --workers 3"
MASTER=rna_harness/runs/three_way_master.log
rm -f rna_harness/runs/.three_way_done
echo "=== 3-WAY SEQUENTIAL START $(date) (RNA_MAX_CORES=$RNA_MAX_CORES, workers=3, timeout 1500s) ===" > "$MASTER"

run() {  # model, out-dir
  echo "=== START $1 -> $2 @ $(date) ===" >> "$MASTER"
  $PY -m rna_harness.experiments.agent_runner --model "$1" $COMMON \
      --out "rna_harness/runs/$2" > "rna_harness/runs/${2}_run.log" 2>&1
  echo "=== DONE $1 (exit $?) @ $(date) ===" >> "$MASTER"
}

run "z-ai/glm-5.2"              "glm5.2_eterna100"
run "deepseek/deepseek-v4-pro"  "deepseek_v4pro_eterna100"
run "anthropic/claude-sonnet-5" "sonnet5_eterna100"

echo "=== ALL 3 COMPLETE $(date) ===" >> "$MASTER"
touch rna_harness/runs/.three_way_done
