#!/bin/bash
set -u
cd /data/wenhui/openclaude_rna/rna_harness/runs/ceiling
echo "=== T1 start $(date -Is) ==="
/home/ubuntu/miniconda3/envs/rna/bin/python -u t1_topology_strata.py --nperm 200 --draws 400
echo "=== T1 done $(date -Is) rc=$? ==="
