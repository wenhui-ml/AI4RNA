#!/bin/bash
cd /data/wenhui/openclaude_rna/rna_harness/runs/ceiling
{ echo "=== R6b (12 noise draws) start $(date -Is) ==="
  R_NOISE=12 /home/ubuntu/miniconda3/envs/rna/bin/python m2b_review_placebo.py --splits 200 --workers 14 --seed 0 \
      --out /data/wenhui/openclaude_rna/rna_harness/runs/ceiling/m2b_review_placebo12.json
  echo "=== R6b done $(date -Is) rc=$? ==="; } 2>&1 | tee m2b_review_placebo12.log
