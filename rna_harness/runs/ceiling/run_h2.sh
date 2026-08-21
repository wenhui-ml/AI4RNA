#!/bin/bash
cd /data/wenhui/openclaude_rna/rna_harness/runs/ceiling
L=h2_heldout_cluster_ci.log
{ echo "=== H2 start $(date -Is) ==="
  /home/ubuntu/miniconda3/envs/rna/bin/python h2_heldout_cluster_ci.py --splits 200 --workers 14 --seed 0
  echo "=== H2 done $(date -Is) rc=$? ==="; } 2>&1 | tee "$L"
