#!/bin/bash
set -u
cd /data/wenhui/openclaude_rna/rna_harness/runs/ceiling
echo "=== E3 start $(date -Is) ==="
/home/ubuntu/miniconda3/envs/rna/bin/python -u e3_scrambled_reference.py --workers 10 --nperm 200 --draws 400
echo "=== E3 done $(date -Is) rc=$? ==="
