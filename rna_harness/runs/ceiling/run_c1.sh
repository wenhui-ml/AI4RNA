#!/bin/bash
cd /data/wenhui/openclaude_rna/rna_harness/runs/ceiling
L=c1_cost_tiers.log
{ echo "=== C1 start $(date -Is) ==="
  /home/ubuntu/miniconda3/envs/rna/bin/python c1_cost_tiers.py --splits 200 --workers 14 --seed 0
  echo "=== C1 done $(date -Is) rc=$? ==="; } 2>&1 | tee "$L"
