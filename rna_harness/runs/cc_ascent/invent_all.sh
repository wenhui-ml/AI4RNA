#!/bin/bash
# Full open-vs-closed invention sweep, HARDENED loop (fork+killpg per design -> no orphaned pKiss).
# Runs in a detached tmux session so it survives laptop/SSH disconnect. Sequential (certify is CPU).
# glm-5.2 (open) already done separately. Closed: opus-4.8, gpt-5.6-sol, gemini-3.1-pro.
# Open (new): deepseek-v4-pro, kimi-k3.  OPENROUTER_API_KEY read from env (never stored in this file).
set +e
cd /data/wenhui/openclaude_rna/rna_harness/runs/cc_ascent
export PYTHONPATH=/data/wenhui/openclaude_rna
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY before launching}"
PY=/home/ubuntu/miniconda3/envs/rna/bin/python
LOG=invent_all.master.log
echo "=== INVENT ALL (tmux, hardened) start uptime=$(cut -d' ' -f1 /proc/uptime)s ===" > $LOG
for M in anthropic/claude-opus-4.8 openai/gpt-5.6-sol deepseek/deepseek-v4-pro moonshotai/kimi-k3 google/gemini-3.1-pro-preview; do
  slug=$(echo "$M" | tr '/.' '__' | tr -d '-')
  if [ -f "invent_operator/$slug/result.json" ]; then
    echo "[$M] already has result.json, skipping" | tee -a $LOG; continue
  fi
  echo "[$M] starting" | tee -a $LOG
  OPENAI_MODEL="$M" $PY invent_operator.py 12 > "invent_${slug}.log" 2>&1
  echo "  [$M] rc=$? -> invent_operator/${slug}/result.json" | tee -a $LOG
done
echo "=== INVENT ALL DONE ===" | tee -a $LOG
