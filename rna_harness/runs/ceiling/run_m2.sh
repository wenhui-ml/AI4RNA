#!/bin/bash
cd /data/wenhui/openclaude_rna/rna_harness/runs/ceiling
{ echo "=== M2 start $(date -Is) ==="
  /home/ubuntu/miniconda3/envs/rna/bin/python m2_invariant_split.py --splits 200 --workers 14 --seed 0
  echo "=== M2 done $(date -Is) rc=$? ==="; } 2>&1 | tee m2_invariant_split.log
