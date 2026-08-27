#!/usr/bin/env bash
# M26c -> designaudit selftest, run in sequence. Logs land next to the artefacts.
set -u
CEIL=/data/wenhui/openclaude_rna/rna_harness/runs/ceiling
GEMMA=/home/ubuntu/miniconda3/envs/gemma/bin/python
export CUDA_VISIBLE_DEVICES=4
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8

echo "### $(date -Is) M26c start"
cd "$CEIL" && $GEMMA m26c_neural_scored.py --seed 0 2>&1 | tee "$CEIL/m26c_run.out"
echo "### $(date -Is) M26c exit=$?"
