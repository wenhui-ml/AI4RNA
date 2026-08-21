#!/bin/bash
set -u
cd /data/wenhui/openclaude_rna/rna_harness/runs/ceiling
echo "=== H1 start $(date -Is) ==="
/home/ubuntu/miniconda3/envs/rna/bin/python -u h1_heldout_ensemble_defect.py --splits 200 --workers 10
echo "=== H1 done $(date -Is) rc=$? ==="
