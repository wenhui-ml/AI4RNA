#!/usr/bin/env bash
# R9, D3 hypothesis (b), STAGE 2 -- the test that decides.
#
# Stage 1 (d3_structural.py) pre-stated six structural features of accepted designs and required each to pass
# TWO gates, both adjusted for overall GC after the negative control showed the unadjusted gate admits
# anything GC-correlated: (a) it differs between the agent and hand-built arms, and (b) it predicts held-out
# carry-over WITHIN the hand-built arm. Exactly one passed: `decoy`, the count of off-target windows
# complementary to the crossing stem's 5' arm. Agent designs carry 0.003 of them per design; ours carry 0.060.
#
# pv_nodecoy adds that count as a TIEBREAK in the acceptance rule and changes nothing else. It is oracle-free,
# so no compute confound is possible. Judged by the rule every transplant in this paper has faced, stated
# before the run: carry-over must RISE while r_in does NOT collapse. Passing an observational gate is not a
# mechanism; only reproducing the gain is D3.
set -uo pipefail
cd "$(dirname "$0")"
ARMS=panel_auto,pv_nodecoy
export NWORKERS=${NWORKERS:-60}
export PKX_DETERMINISTIC=1
export PKX_MAXPKISS=${PKX_MAXPKISS:-150}
export PKD_ITERS=800
for s in 0 1 2 3 4; do
  echo "=== nodecoy sweep, nominal seed $s ==="
  /home/ubuntu/miniconda3/envs/rna/bin/python sweep.py "ndc2_s$s" 0 60 "$s" "$ARMS"
done
echo NODECOY_DONE
