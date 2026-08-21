#!/bin/bash
set -u
cd /data/wenhui/openclaude_rna/rna_harness/runs/ceiling
P=/home/ubuntu/miniconda3/envs/rna/bin/python
echo "=== G4 start $(date -Is) ==="
$P g4_signed_vs_spread.py --draws 600 --nperm 200 2>&1
echo "=== G4 done $(date -Is) rc=$? ==="
