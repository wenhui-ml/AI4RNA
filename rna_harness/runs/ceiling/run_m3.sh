#!/bin/bash
cd /data/wenhui/openclaude_rna/rna_harness/runs/ceiling
{ echo "=== M3 start $(date -Is) ==="
  /home/ubuntu/miniconda3/envs/rna/bin/python m3_gc_cancelled.py --workers 14 --seed 0
  echo "=== M3 done $(date -Is) rc=$? ==="; } 2>&1 | tee m3_gc_cancelled.log
