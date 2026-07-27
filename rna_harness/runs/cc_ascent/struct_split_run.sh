#!/usr/bin/env bash
# R6 owed experiment #1: gamer vs panel operator on the STRUCTURALLY DISJOINT pool built by
# build_struct_split.py (27 targets in 24 clean clusters, every one >15% dot-bracket distance from every
# target that entered any stage of development). 5 nominal seeds x 3 executions, both arms in the same
# worker call so pairing holds at (target, seed, execution). Search-only: no LLM, no $ cost.
set -uo pipefail
cd "$(dirname "$0")"
export TARGETS_JSON="$PWD/struct_split.json"
export NWORKERS=${NWORKERS:-27}
for rep in 0 1 2; do
  for s in 0 1 2 3 4; do
    echo "=== struct-split execution $rep, nominal seed $s ==="
    /home/ubuntu/miniconda3/envs/rna/bin/python sweep.py "sdis_e${rep}_s$s" 0 0 "$s" autonomous,panel_auto
  done
done
echo STRUCT_SPLIT_DONE
