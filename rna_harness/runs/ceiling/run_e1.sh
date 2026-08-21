#!/bin/bash
set -u
cd /data/wenhui/openclaude_rna/rna_harness/runs/ceiling
echo "=== E1 start $(date -Is) ==="
/home/ubuntu/miniconda3/envs/rna/bin/python -u e1_ensemble_marginals.py --workers 10 --nperm 200 --draws 400
echo "=== E1 done $(date -Is) rc=$? ==="
