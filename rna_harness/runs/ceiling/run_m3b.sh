#!/bin/bash
cd /data/wenhui/openclaude_rna/rna_harness/runs/ceiling
{ echo "=== M3b start $(date -Is) ==="
  /home/ubuntu/miniconda3/envs/rna/bin/python m3b_placebo_on_matched.py --splits 200 --workers 14 --seed 0
  echo "=== M3b done $(date -Is) rc=$? ==="; } 2>&1 | tee m3b_placebo_on_matched.log
