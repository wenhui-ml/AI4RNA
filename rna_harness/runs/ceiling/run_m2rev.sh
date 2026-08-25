#!/bin/bash
cd /data/wenhui/openclaude_rna/rna_harness/runs/ceiling
{ echo "=== M2 REVIEW start $(date -Is) ==="
  /home/ubuntu/miniconda3/envs/rna/bin/python m2b_review_audit.py --splits 200 --workers 14 --seed 0
  echo "=== M2 REVIEW done $(date -Is) rc=$? ==="; } 2>&1 | tee m2b_review_audit.log
