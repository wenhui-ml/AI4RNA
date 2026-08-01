#!/usr/bin/env bash
# R10: the unbounded family at scale -- 100 targets x 5 seeds x 3 arms, deterministic.
set -uo pipefail
cd "$(dirname "$0")"
export PKX_DETERMINISTIC=1
export PKD_ITERS=800
for s in 0 1 2 3 4; do
  echo "=== family grid, seed $s ==="
  /home/ubuntu/miniconda3/envs/rna/bin/python family_grid.py "$s"
done
echo FAMILY_GRID_DONE
