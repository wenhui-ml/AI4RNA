#!/bin/bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
cd /data/wenhui/openclaude_rna/rna_harness/runs/ceiling
{ echo "=== M1e start $(date -Is) (OMP=$OMP_NUM_THREADS) ==="
  /home/ubuntu/miniconda3/envs/rna/bin/python -u m1e_meanp_placebo.py --splits 200 --workers 12 --seed 0
  echo "=== M1e done $(date -Is) rc=$? ==="; } 2>&1 | tee m1e_meanp_placebo.log
