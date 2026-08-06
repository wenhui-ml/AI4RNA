"""Rival explanation: the paired gap is not 'consensus', it is 'ShapeKnots agreed'."""
import json, statistics as st
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon

D=json.load(open("/data/wenhui/openclaude_rna/rna_harness/runs/ceiling/judge_paired.json"))
by=defaultdict(list)
for r in D["rows"]:
    if r.get("error") or r.get("skipped"): continue
    for d in r.get("designs",[]):
        if d["external_ran"]: by[r["task_id"]].append(d)

def ptest(lab, hi, lo):
    P=[]
    for t,v in by.items():
        a=[d["external"] for d in v if hi(d)]; b=[d["external"] for d in v if lo(d)]
        if a and b: P.append((st.mean(a),st.mean(b)))
    if len(P)<8: print(f"  {lab:36s} 只有 {len(P)} 个配对靶标,不检验"); return None
    A=np.array([p[0] for p in P]); B=np.array([p[1] for p in P]); d=A-B
    p=float(wilcoxon(A,B,alternative='greater')[1])
    print(f"  {lab:36s} {len(P):3d} 靶标  差 {d.mean():+6.1%}  p={p:.3g}   好{int((d>0).sum())} 差{int((d<0).sum())} 平{int((d==0).sum())}")
    return d.mean()

sk=lambda d:"shapeknots" in d["confirmed_by"]
print("=== 同靶标配对,三个互相竞争的分组规则 ===")
ptest("A 面板一致(3家) vs 分裂",      lambda d:d["k_panel"]==3, lambda d:1<=d["k_panel"]<=2)
ptest("B ShapeKnots 确认 vs 未确认",   sk, lambda d:not sk(d))
ptest("C 家数 2 vs 1(都排除 SK)",     lambda d:d["k_panel"]==2 and not sk(d), lambda d:d["k_panel"]==1 and not sk(d))
ptest("D 3家 vs 2家(都含 SK)",        lambda d:d["k_panel"]==3, lambda d:d["k_panel"]==2 and sk(d))

print("\n=== 决定性检验:固定 ShapeKnots 已确认,加一致性还有增量吗 ===")
ptest("E SK+其他 vs 只有 SK",          lambda d:sk(d) and d["k_panel"]>=2, lambda d:sk(d) and d["k_panel"]==1)
print("\n=== 反向:固定 ShapeKnots 未确认,一致性还有用吗 ===")
ptest("F 无SK的2家 vs 无SK的1家",      lambda d:not sk(d) and d["k_panel"]==2, lambda d:not sk(d) and d["k_panel"]==1)

print("\n=== 边际:各规则在全体设计上的分辨力 ===")
allv=[d for v in by.values() for d in v]
for lab,f in (("面板一致(3家)",lambda d:d["k_panel"]==3),("ShapeKnots 确认",sk)):
    hi=[d["external"] for d in allv if f(d)]; lo=[d["external"] for d in allv if not f(d)]
    print(f"  {lab:20s} 命中 {len(hi):5d} 条  精确率 {st.mean(hi):6.1%} 对 {st.mean(lo):6.1%}"
          f"   召回 {sum(hi)/max(1,sum(d['external'] for d in allv)):6.1%}")
