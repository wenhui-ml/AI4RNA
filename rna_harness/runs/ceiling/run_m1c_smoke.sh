#!/bin/bash
cd /data/wenhui/openclaude_rna/rna_harness/runs/ceiling
/home/ubuntu/miniconda3/envs/rna/bin/python m1c_skeleton_bookkeeping.py --splits 70 --workers 14 \
  --out /tmp/claude-1000/-data-wenhui-openclaude-rna/6e34da6a-b0bb-4401-a731-1ad021750f1c/scratchpad/m1c_smoke.json \
  > /tmp/claude-1000/-data-wenhui-openclaude-rna/6e34da6a-b0bb-4401-a731-1ad021750f1c/scratchpad/m1c_smoke.log 2>&1
echo "rc=$?" >> /tmp/claude-1000/-data-wenhui-openclaude-rna/6e34da6a-b0bb-4401-a731-1ad021750f1c/scratchpad/m1c_smoke.log
