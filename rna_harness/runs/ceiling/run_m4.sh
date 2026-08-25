#!/bin/bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
cd /data/wenhui/openclaude_rna/rna_harness/runs/ceiling
{ echo "=== M4 start $(date -Is) (OMP=$OMP_NUM_THREADS) ==="
  /home/ubuntu/miniconda3/envs/rna/bin/python -u m4_selection_currency.py --splits 200 --workers 12 --seed 0
  echo "=== M4 done $(date -Is) rc=$? ==="; } 2>&1 | tee m4_selection_currency.log
