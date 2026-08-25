#!/bin/bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
cd /data/wenhui/openclaude_rna/rna_harness/runs/ceiling
/home/ubuntu/miniconda3/envs/rna/bin/python -u m3c_noise_calibrated_replication.py --splits 80 --workers 12 --seed 0 \
  --out /tmp/claude-1000/-data-wenhui-openclaude-rna/6e34da6a-b0bb-4401-a731-1ad021750f1c/scratchpad/m3c_smoke.json
