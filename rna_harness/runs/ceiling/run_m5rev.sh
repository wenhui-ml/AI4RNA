#!/bin/bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
cd /data/wenhui/openclaude_rna/rna_harness/runs/ceiling
{ echo "=== M5 review start $(date -Is) ==="
  /home/ubuntu/miniconda3/envs/rna/bin/python -u m5_review_interval.py --splits 200 --workers 12 --seed 0
  echo "=== done $(date -Is) rc=$? ==="; } 2>&1 | tee m5_review_interval.log
