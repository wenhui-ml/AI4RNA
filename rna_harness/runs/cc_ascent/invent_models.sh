#!/bin/bash
# Run the autonomous operator-invention loop (12 rounds + patience) across three frontier models,
# each writing to invent_operator/<model-slug>/. OPENROUTER_API_KEY is read from the environment
# (never stored here). Sequential so the per-model certify (60 held-out designs) does not oversubscribe CPU.
set +e
cd /data/wenhui/openclaude_rna/rna_harness/runs/cc_ascent
export PYTHONPATH=/data/wenhui/openclaude_rna
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY in the environment before launching}"
PY=/home/ubuntu/miniconda3/envs/rna/bin/python
LOG=invent_models.master.log
echo "=== INVENT across models: $(date -u 2>/dev/null || echo start) ===" > $LOG
for M in google/gemini-3.1-pro-preview anthropic/claude-opus-4.8 openai/gpt-5.6-sol; do
  slug=$(echo "$M" | tr '/.' '__' | tr -d '-')
  echo "[$M]" | tee -a $LOG
  OPENAI_MODEL="$M" $PY invent_operator.py 12 > "invent_${slug}.log" 2>&1
  echo "  rc=$? -> invent_operator/${slug}/result.json" | tee -a $LOG
done
echo "=== DONE ===" | tee -a $LOG
