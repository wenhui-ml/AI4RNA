#!/bin/bash
# A8: score the native/random population with both rulers, then run the declared test.
# Two environments on purpose -- rnet_env has torch and no ViennaRNA, `rna` has ViennaRNA and no torch.
# Sequential: a step that fails stops the chain, because analysing a half-written score file manufactures
# a result. COST: $0, all local.
set -u
cd /data/wenhui/openclaude_rna/rna_harness/runs/ceiling || exit 1
export PYTHONHASHSEED=0
export DESIRNA_CMD="/home/ubuntu/miniconda3/envs/rna/bin/python /data/wenhui/DesiRNA/DesiRNA.py"
L=arbitration_absolute.log
: > "$L"
step () {
  local name="$1"; shift
  echo "=== [$name] 开始 $(date -u +%H:%M:%SZ) ===" | tee -a "$L"
  "$@" >> "$L" 2>&1
  local rc=$?
  echo "=== [$name] 结束 rc=$rc $(date -u +%H:%M:%SZ) ===" | tee -a "$L"
  [ $rc -ne 0 ] && { echo "!! [$name] 非零退出,链条停在这里" | tee -a "$L"; exit $rc; }
}
step rnet    /data/wenhui/rnet_env/bin/python        arbitration_absolute.py --stage rnet  --workers 14
step panel   /home/ubuntu/miniconda3/envs/rna/bin/python arbitration_absolute.py --stage panel --workers 24
step analyse /home/ubuntu/miniconda3/envs/rna/bin/python arbitration_absolute.py --stage analyse
echo "=== 全链完成 $(date -u +%H:%M:%SZ) ===" | tee -a "$L"
