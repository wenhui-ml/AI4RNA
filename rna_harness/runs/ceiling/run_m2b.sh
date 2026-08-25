#!/bin/bash
cd /data/wenhui/openclaude_rna/rna_harness/runs/ceiling
{ echo "=== M2b start $(date -Is) ==="
  /home/ubuntu/miniconda3/envs/rna/bin/python m2b_gt_target_property.py --splits 200 --workers 14 --seed 0
  echo "=== M2b done $(date -Is) rc=$? ==="; } 2>&1 | tee m2b_gt_target_property.log
