#!/usr/bin/env bash
# Overnight chain for 2026-08-19 -> 08-20. SERIAL on purpose: DesiRNA forks 10 replicas per job, so two
# compute stages at once would blow past the 160-core ceiling and hurt other users on this box.
# Every stage is guarded: a failure logs and the chain continues, so one broken stage cannot cost the night.
# All stages are $0 and offline. No API is called anywhere in here.
set -u
P=/home/ubuntu/miniconda3/envs/rna/bin/python
cd /data/wenhui/openclaude_rna/rna_harness/runs/ceiling || exit 1
say() { echo "=== $(date -u +%H:%M:%SZ) $* ==="; }

say "阶段 0:等 A14 第四批判官跑完(它在 tmux a14 里)"
for _ in $(seq 1 240); do
  grep -q '=== rc=' panel_f1_fourth.log 2>/dev/null && break
  $P -c "import time;time.sleep(60)"
done
if ! grep -q '=== rc=' panel_f1_fourth.log 2>/dev/null; then
  say "阶段 0 超时 4 小时:第四批没跑完,跳过 A14/A15,直接进 A16"
else
  tail -3 panel_f1_fourth.log
  say "阶段 1:A14 分析(试管判,预注册)"
  $P a14_analyse.py > a14.log 2>&1; say "A14 rc=$?"; tail -22 a14.log
fi

say "阶段 2:A16 ACGU 干预实验(能力向;57 靶标 × 2 臂 × 3 种子,workers 12 ≈ 120 核)"
$P acgu_intervention.py --workers 12 > acgu_intervention.log 2>&1; say "A16 rc=$?"; tail -30 acgu_intervention.log

if [ -f panel_f1_fourth.json ]; then
  say "阶段 3:A15 第一步 —— IPknot + RNAPKplex(固定判官集,两家都快)"
  $P a15_judge.py --judges ipknot,pkplex --workers 16 > a15_judge_fast.log 2>&1
  say "A15 判官 rc=$?"; tail -8 a15_judge_fast.log
  say "阶段 4:A15 分析(主口径,Knotty 还没有 → 次口径留空)"
  $P a15_analyse.py > a15.log 2>&1; say "A15 rc=$?"; tail -26 a15.log

  say "阶段 5:Knotty(慢,只作覆盖率如实上报的次口径;workers 8)"
  $P a15_judge.py --judges knotty --workers 8 > a15_judge_knotty.log 2>&1
  say "Knotty rc=$?"; tail -6 a15_judge_knotty.log
  say "阶段 6:A15 重算(这次带 Knotty 次口径)"
  $P a15_analyse.py > a15.log 2>&1; say "A15(含 Knotty) rc=$?"; tail -30 a15.log
fi

say "全链完成"
