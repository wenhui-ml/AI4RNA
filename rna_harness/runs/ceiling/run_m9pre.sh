#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd /data/wenhui/openclaude_rna/rna_harness/runs/ceiling
{ echo "=== M9-pre start $(date -Is) ==="
  /home/ubuntu/miniconda3/envs/rna/bin/python -u m9pre_heavytail_vs_concentration.py --workers 14 --seed 0
  echo "=== M9-pre done $(date -Is) rc=$? ==="; } 2>&1 | tee m9pre_heavytail_vs_concentration.log
