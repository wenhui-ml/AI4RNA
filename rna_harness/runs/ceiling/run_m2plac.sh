#!/bin/bash
cd /data/wenhui/openclaude_rna/rna_harness/runs/ceiling
{ echo "=== R6 start $(date -Is) ==="
  /home/ubuntu/miniconda3/envs/rna/bin/python m2b_review_placebo.py --splits 200 --workers 14 --seed 0
  echo "=== R6 done $(date -Is) rc=$? ==="; } 2>&1 | tee m2b_review_placebo.log
