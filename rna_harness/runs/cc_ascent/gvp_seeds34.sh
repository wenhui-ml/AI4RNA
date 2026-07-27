#!/usr/bin/env bash
# R6 self-review gap: the execution-replicated Delta_gamer covers 3 nominal seeds, while the ORIGINAL
# single-execution estimate covered 5. That makes the better-designed estimate the one with LESS seed
# coverage, and forces the paper to carry both. Extending to seeds 3-4 x 3 executions gives
# 5 seeds x 3 executions = 15 paired cells, so the replicated estimate strictly dominates the old one
# and the single-execution band can be retired. Search-only: no LLM, no $ cost.
set -uo pipefail
cd "$(dirname "$0")"
ARMS=autonomous,panel_auto
export NWORKERS=${NWORKERS:-60}
for rep in 0 1 2; do
  for s in 3 4; do
    echo "=== gamer-vs-panel execution $rep, nominal seed $s ==="
    /home/ubuntu/miniconda3/envs/rna/bin/python sweep.py "gvp_e${rep}_s$s" 0 60 "$s" "$ARMS"
  done
done
echo GVP_SEEDS34_DONE
