#!/bin/bash
# A9: two hard-null populations, both rulers, then the declared test. Sequential; a failure stops the chain.
# COST: $0, all local. rnet_env has torch and no ViennaRNA; `rna` has ViennaRNA and no torch.
set -u
cd /data/wenhui/openclaude_rna/rna_harness/runs/ceiling || exit 1
export PYTHONHASHSEED=0
RNA=/home/ubuntu/miniconda3/envs/rna/bin/python
RNET=/data/wenhui/rnet_env/bin/python
L=arbitration_a9.log
: > "$L"
step () {
  local name="$1"; shift
  echo "=== [$name] 开始 $(date -u +%H:%M:%SZ) ===" | tee -a "$L"
  "$@" >> "$L" 2>&1
  local rc=$?
  echo "=== [$name] 结束 rc=$rc $(date -u +%H:%M:%SZ) ===" | tee -a "$L"
  [ $rc -ne 0 ] && { echo "!! [$name] 非零退出,链条停在这里" | tee -a "$L"; exit $rc; }
}
for POP in nested crossnative; do
  step "$POP/build"   $RNA  arbitration_a9.py --pop $POP --stage build
  step "$POP/rnet"    $RNET arbitration_a9.py --pop $POP --stage rnet  --workers 14
  step "$POP/panel"   $RNA  arbitration_a9.py --pop $POP --stage panel --workers 24
  step "$POP/analyse" $RNA  arbitration_a9.py --pop $POP --stage analyse
done
echo "=== 全链完成 $(date -u +%H:%M:%SZ) ===" | tee -a "$L"
