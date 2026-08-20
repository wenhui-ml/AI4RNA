#!/usr/bin/env bash
# A15 re-run after the bad-target-string bug. Serial; $0; offline.
set -u
P=/home/ubuntu/miniconda3/envs/rna/bin/python
cd /data/wenhui/openclaude_rna/rna_harness/runs/ceiling || exit 1
say() { echo "=== $(date -u +%H:%M:%SZ) $* ==="; }
say "A15 重跑(修 str(T[p]) -> T[p]['target'])阶段 1:IPknot + RNAPKplex"
$P a15_judge.py --judges ipknot,pkplex --workers 20 > a15_judge_fast.log 2>&1; say "rc=$?"; tail -6 a15_judge_fast.log
say "阶段 2:A15 分析(主口径)"
$P a15_analyse.py > a15.log 2>&1; say "rc=$?"; tail -26 a15.log
say "阶段 3:Knotty(慢,次口径)"
$P a15_judge.py --judges knotty --workers 12 > a15_judge_knotty.log 2>&1; say "rc=$?"; tail -6 a15_judge_knotty.log
say "阶段 4:A15 重算(含 Knotty)"
$P a15_analyse.py > a15.log 2>&1; say "rc=$?"; tail -32 a15.log
say "A15 重跑完成"
