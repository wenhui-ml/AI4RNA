#!/usr/bin/env bash
# Overnight batch. Everything here is unattended, deterministic in its protocol, and lands its own json.
#
# WHY THESE FOUR. This project already paid for the lesson once: 51/55 was quoted as the delivery rate and
# turned out to be the TOP of a five-seed distribution whose mean was 87.6%. The gap numbers for the OTHER
# two capabilities -- co-transcriptional folding and the two-state switch -- are each still a SINGLE draw
# from a nondeterministic solver. Replicating them is not polish; it is applying a correction we already know
# we needed to the two places we have not yet applied it.
#
# The third job generalises the position-specificity null: the register-shift decoy passed 0/79, but only on
# kissing topologies. If `knot_confirmed` is position-specific it should be so everywhere it is used.
#
# The fourth is the honest failure mode of the third: H-type now routes to the strict panel, so the shifted
# decoy there tests a DIFFERENT rule and the result must be read as such.
set -u
cd "$(dirname "$0")"
PY=/home/ubuntu/miniconda3/envs/rna/bin/python
mkdir -p seeds
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== 1/4  co-transcriptional gap, 10 independent solver seeds ==="
for s in 0 1 2 3 4 5 6 7 8 9; do
  log "  kinetic seed=$s"
  $PY kinetic_bench.py --n 24 --timelimit 15 --workers 6 --seed "$s" \
      --out "seeds/kinetic_s${s}.json" >"seeds/kinetic_s${s}.log" 2>&1
done

log "=== 2/4  two-state switch gap, 10 independent solver seeds ==="
for s in 0 1 2 3 4 5 6 7 8 9; do
  log "  switch seed=$s"
  $PY switch_bench.py --pool 2000 --n 72 --timelimit 15 --workers 6 --seed "$s" \
      --out "seeds/switch_s${s}.json" >"seeds/switch_s${s}.log" 2>&1
done

log "=== 3/4  register-shift decoy on higher-order (kissing already done: 0/79) ==="
$PY shift_decoy.py --group "higher-order" --timelimit 60 --workers 10 \
    --out "seeds/shift_higher.json" >"seeds/shift_higher.log" 2>&1

log "=== 4/4  register-shift decoy on H-type -- NOTE: H-type now routes to the STRICT panel, so this"
log "         measures a different rule than the kissing run and must be reported as such ==="
$PY shift_decoy.py --group "H-type" --timelimit 60 --workers 10 \
    --out "seeds/shift_htype.json" >"seeds/shift_htype.log" 2>&1

log "=== all done; summarising ==="
$PY overnight_summary.py 2>&1 | tee seeds/SUMMARY.txt
log "=== wrote seeds/SUMMARY.txt ==="
