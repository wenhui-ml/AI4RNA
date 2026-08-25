#!/bin/bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
cd /data/wenhui/openclaude_rna/rna_harness/runs/ceiling
{ echo "=== R1b start $(date -Is) ==="
  /home/ubuntu/miniconda3/envs/rna/bin/python -u r1b_ceiling_interval.py --seed 0
  echo "=== R1b done $(date -Is) rc=$? ==="; } 2>&1 | tee r1b_ceiling_interval.log
