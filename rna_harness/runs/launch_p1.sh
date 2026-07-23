#!/usr/bin/env bash
# P1 launcher (Paper C) — de-scaffolding FLOOR vs CEILING, matched single-shot, 3 independent repeats.
# Launch inside tmux so it survives disconnect:
#     tmux new -d -s p1 'bash /data/wenhui/openclaude_rna/rna_harness/runs/launch_p1.sh'
#     tmux attach -t p1        # watch;  Ctrl-b d to detach
# Polite default: SEQUENTIAL processes, WORKERS=3 -> at most 3 concurrent episodes on the shared box.
# To ~2x throughput at the cost of 6 concurrent episodes: set P1_CONCURRENT=1 (runs each ceiling+floor
# tail35 repeat as a concurrent pair). Leave off if other users' big jobs are running.
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rna
cd /data/wenhui/openclaude_rna
# OpenRouter auth: openclaude CLI runs in OpenAI-compat mode (CLAUDE_CODE_USE_OPENAI=1) and reads
# OPENAI_API_KEY. Source it from the 0600 .orkey (never echoed). Exported once here -> inherited by
# every spawned episode subprocess for the whole run, even if scratchpad is later cleaned.
ORK=$(ls /tmp/claude-1000/-data-wenhui-openclaude-rna/*/scratchpad/.orkey 2>/dev/null | head -1)
[ -z "$ORK" ] && { echo "FATAL: no .orkey found — cannot authenticate to OpenRouter"; exit 1; }
export OPENAI_API_KEY="$(cat "$ORK")"
[ -z "$OPENAI_API_KEY" ] && { echo "FATAL: .orkey empty"; exit 1; }
echo "[auth] OPENAI_API_KEY loaded (${#OPENAI_API_KEY} chars)"
export M2_MODEL="${M2_MODEL:-z-ai/glm-5.2}"
export TIMEOUT="${TIMEOUT:-1500}"
export WORKERS="${WORKERS:-3}"
RUNS=/data/wenhui/openclaude_rna/rna_harness/runs
LOGS=$RUNS/p1_logs; mkdir -p "$LOGS"
PY=/home/ubuntu/miniconda3/envs/rna/bin/python

run() {  # COND TASKSET REPEAT
  local cond=$1 ts=$2 r=$3
  local tag="p1_${cond}_${ts}_r${r}"
  echo "=== [$(date -u +%H:%M:%S)] START $tag ===" | tee -a "$LOGS/master.log"
  COND=$cond TASKSET=$ts REPEAT=$r OUT="$RUNS/$tag" \
    "$PY" "$RUNS/p1_runner.py" 2>&1 | tee "$LOGS/$tag.log"
  echo "=== [$(date -u +%H:%M:%S)] END   $tag ===" | tee -a "$LOGS/master.log"
}

# order: first ceiling+floor tail35 r1 (first delta ~9h), then remaining tail35 repeats, then full100 last
if [ "${P1_CONCURRENT:-0}" = "1" ]; then
  for r in 1 2 3; do run ceiling tail35 $r & run floor tail35 $r & wait; done
else
  for r in 1 2 3; do run ceiling tail35 $r; run floor tail35 $r; done
fi
run floor full100 1     # the clean autonomous floor (the P0 methodological fix) — longest (~24h)

echo "===== P1 ALL DONE $(date -u) =====" | tee -a "$LOGS/master.log"
