#!/bin/bash
# BLAS thread cap: lstsq is called ~20 blocks x 200 splits x 9 runs. With OMP unset OpenBLAS grabs all 256
# cores, which blows the 160-core house limit -- two runs were SIGKILLed at 189MB RSS, so it was never memory.
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
cd /data/wenhui/openclaude_rna/rna_harness/runs/ceiling
{ echo "=== M1c start $(date -Is) (OMP_NUM_THREADS=$OMP_NUM_THREADS) ==="
  /home/ubuntu/miniconda3/envs/rna/bin/python -u m1c_skeleton_bookkeeping.py --splits 200 --workers 14 --seed 0
  echo "=== M1c done $(date -Is) rc=$? ==="; } 2>&1 | tee m1c_skeleton_bookkeeping.log
