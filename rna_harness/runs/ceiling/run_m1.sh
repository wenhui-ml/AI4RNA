#!/bin/bash
cd /data/wenhui/openclaude_rna/rna_harness/runs/ceiling
{ echo "=== M1 start $(date -Is) ==="
  /home/ubuntu/miniconda3/envs/rna/bin/python m1_indexset_bookkeeping.py --splits 200 --workers 14 --seed 0
  echo "=== M1 done $(date -Is) rc=$? ==="; } 2>&1 | tee m1_indexset_bookkeeping.log
