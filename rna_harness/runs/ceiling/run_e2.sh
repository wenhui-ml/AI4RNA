#!/bin/bash
set -u
cd /data/wenhui/openclaude_rna/rna_harness/runs/ceiling
echo "=== E2 start $(date -Is) ==="
/home/ubuntu/miniconda3/envs/rna/bin/python -u e2_placebo_positions.py --workers 10 --nperm 200 --draws 400
echo "=== E2 done $(date -Is) rc=$? ==="
