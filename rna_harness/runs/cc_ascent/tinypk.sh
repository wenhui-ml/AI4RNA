#!/usr/bin/env bash
# R9: the compute control run the OTHER way. agent_budget.py measured the frozen agent operators at ~28 and
# ~13 realized pKiss calls per target against the hand-built operator's 128 in the same regime, and kimi-k3's
# source writes `budget = [30]`. So the surviving hypothesis for the carry-over gap is not "more search" but
# "less search": an operator forced to commit early may accept candidates from a different region of sequence
# space. This arm sets the pKiss allowance to 30 and changes nothing else, deterministically, 5 seeds.
# A mechanism counts only if carry-over rises AND r_in does not collapse -- read both columns.
set -uo pipefail
cd "$(dirname "$0")"
ARMS=panel_auto,pv_tinypk,pv_bigpk
export NWORKERS=${NWORKERS:-60}
export PKX_DETERMINISTIC=1
export PKX_MAXPKISS=${PKX_MAXPKISS:-150}
export PKD_ITERS=800
for s in 0 1 2 3 4; do
  echo "=== tinypk sweep, nominal seed $s ==="
  /home/ubuntu/miniconda3/envs/rna/bin/python sweep.py "tiny_s$s" 0 60 "$s" "$ARMS"
done
echo TINYPK_DONE
