import json, statistics as st
C="/data/wenhui/openclaude_rna/rna_harness/runs/ceiling/"
D=json.load(open(C+"k1_diagnosis.json"))
R=D['rows']
acc=[r for r in R if r['y']]; rej=[r for r in R if not r['y']]
print(f"IPknot 接受 {len(acc)} 条 / 拒绝 {len(rej)} 条")
for f,lab in (('entropy','位点熵'),('p_mfe','MFE 系综概率'),('gc','GC%')):
    a=st.mean([r[f] for r in acc]); b=st.mean([r[f] for r in rej])
    print(f"  {lab:14s} 被接受的 {a:.4f}   被拒绝的 {b:.4f}   差 {a-b:+.4f}")
print()
print("=== 方向 ===")
ea=st.mean([r['entropy'] for r in acc]); er=st.mean([r['entropy'] for r in rej])
print(f"  IPknot 接受的设计【熵更{'高' if ea>er else '低'}】 -> "
      f"{'留出概率质量给交叉配对的序列才被接受' if ea>er else '越确定越被接受'}")
print(f"  天然序列的熵 0.2985,比【被 IPknot 接受的设计】{'还高' if 0.2985>ea else '低'}")
print()
print("=== 按面板等级看熵与接受率 ===")
for k in (1,2,3):
    g=[r for r in R if r['k']==k]
    if not g: continue
    print(f"  k={k}  n={len(g):5d}  平均熵 {st.mean([r['entropy'] for r in g]):.4f}  "
          f"IPknot 接受率 {st.mean([r['y'] for r in g]):.1%}")
