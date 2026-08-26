#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd /data/wenhui/openclaude_rna/rna_harness/runs/ceiling
{ echo "=== M11 start $(date -Is) ==="
  /home/ubuntu/miniconda3/envs/rna/bin/python -u m11_method_routing.py --seed 0
  echo "=== M11 done $(date -Is) rc=$? ==="; } 2>&1 | tee m11_method_routing.log
