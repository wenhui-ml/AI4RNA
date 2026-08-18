#!/bin/bash
# The precision chain: measure on OUR OWN pools, then recalibrate, then see who should choose.
#
# Runs under tmux so it survives a closed laptop. Sequential on purpose -- step 3 reads step 1's output, and
# the box is shared (160-core cap, other users' jobs at ~240% CPU each), so 16 workers is the ceiling here.
#
#   1  pool_precision.py       ~2 h   precision for all 8,730 of our own dumped candidates
#                                     C0 identity check: recomputed recall must equal what the dump recorded
#   2  calibrate_precision.py  ~30 m  where the operating point goes if the certificate reads precision:
#                                     sensitivity on 250 natives vs false-accept on 250 randoms, rule grid
#   3  selector_heldout.py     ~1 h   arms judged by IPknot / Knotty / RNAPKplex -- the three predictors that
#                                     have never been in any selection rule. Adds the panel_f1 arms now that
#                                     step 1 exists. Writes to selector_heldout_f1.json so the RNet-only run
#                                     already on disk is not clobbered.
#
# A step that fails stops the chain: every later step reads the earlier one's numbers, and running on a
# half-written file is how a null gets manufactured.
#
# COST: $0. All local folds.
set -u
cd /data/wenhui/openclaude_rna || exit 1
PY=/home/ubuntu/miniconda3/envs/rna/bin/python
D=rna_harness/runs/ceiling
export DESIRNA_CMD="/home/ubuntu/miniconda3/envs/rna/bin/python /data/wenhui/DesiRNA/DesiRNA.py"
export PYTHONHASHSEED=0

step () {                                    # step <name> <logfile> <cmd...>
  local name="$1"; local log="$2"; shift 2
  echo "=== [$name] 开始 $(date -u +%H:%M:%SZ) ===" | tee -a "$log"
  "$@" >> "$log" 2>&1
  local rc=$?
  echo "=== [$name] 结束 rc=$rc $(date -u +%H:%M:%SZ) ===" | tee -a "$log"
  if [ $rc -ne 0 ]; then
    echo "!! [$name] 非零退出 ($rc) —— 链条停在这里,不用半成品往下算。"
    exit $rc
  fi
}

step pool_precision      "$D/pool_precision.log" \
     $PY $D/pool_precision.py --workers 16

step calibrate_precision "$D/calibrate_precision.log" \
     $PY $D/calibrate_precision.py --workers 16

step selector_heldout_f1 "$D/selector_heldout_f1.log" \
     $PY $D/selector_heldout.py --workers 12 \
        --pool-precision $D/pool_precision.json \
        --out $D/selector_heldout_f1.json

echo "=== 全链完成 $(date -u +%H:%M:%SZ) ==="
echo "产物: $D/{pool_precision,calibrate_precision,selector_heldout_f1}.json"
