#!/bin/bash
# Remaining models for the open-vs-closed invention distribution, with the HARDENED loop (fork+killpg
# per design call -> no orphaned pKiss). glm-5.2 (open) already done; gemini dropped (tool-hang, per user).
# Closed: claude-opus-4.8, gpt-5.6-sol.  Open: deepseek-v4-pro, kimi-k3.  Sequential (certify CPU).
set +e
cd /data/wenhui/openclaude_rna/rna_harness/runs/cc_ascent
export PYTHONPATH=/data/wenhui/openclaude_rna
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY in the environment before launching}"
PY=/home/ubuntu/miniconda3/envs/rna/bin/python
LOG=invent_models2.master.log
echo "=== INVENT (hardened) remaining models: start ===" > $LOG
for M in anthropic/claude-opus-4.8 openai/gpt-5.6-sol deepseek/deepseek-v4-pro moonshotai/kimi-k3; do
  slug=$(echo "$M" | tr '/.' '__' | tr -d '-')
  echo "[$M]" | tee -a $LOG
  OPENAI_MODEL="$M" $PY invent_operator.py 12 > "invent_${slug}.log" 2>&1
  echo "  rc=$? -> invent_operator/${slug}/result.json" | tee -a $LOG
done
echo "=== DONE ===" | tee -a $LOG
