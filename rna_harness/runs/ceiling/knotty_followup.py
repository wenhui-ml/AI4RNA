"""Follow-ups the two-judge data now permits: the conjunction label, and how much of 4.15's
'ShapeKnots dominates consensus' was RNAPKplex-specific."""
import json, statistics as st, sys
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0,"/data/wenhui/openclaude_rna"); sys.path.insert(0,"/data/wenhui/openclaude_rna/rna_harness/runs/ceiling")
from rna_harness.benchmarks.rnainvbench import _round_only, load_pseudobase, pk_pairs

C="/data/wenhui/openclaude_rna/rna_harness/runs/ceiling/"
P=json.load(open(C+"judge_paired.json")); K=json.load(open(C+"knotty_judge.json"))
kn={s:(None if v is None else {tuple(x) for x in v}) for s,v in K["knotty"].items()}
tgt={str(t.task_id):str(t.target) for t in load_pseudobase()}
by=defaultdict(list)
for r in P["rows"]:
    if r.get("error") or r.get("skipped"): continue
    tg=tgt.get(r["task_id"]);
    if not tg: continue
    cross=pk_pairs(tg)-pk_pairs(_round_only(tg))
    for d in r.get("designs",[]):
        if not d["external_ran"]: continue
        k=kn.get(d["seq"])
        if k is None: continue
        by[r["task_id"]].append({**d,"pk":bool(d["external"]),"kn":bool(k>=cross)})

def ptest(lab,hi,lo,key):
    Q=[]
    for t,v in by.items():
        A=[d[key] for d in v if hi(d)]; B=[d[key] for d in v if lo(d)]
        if A and B: Q.append((st.mean(A),st.mean(B)))
    if len(Q)<8: return print(f"  {lab:32s} 配对靶标 {len(Q)} <8,不检验")
    a=np.array([q[0] for q in Q]); b=np.array([q[1] for q in Q]); d=a-b
    print(f"  {lab:32s} {len(Q):3d} 靶标  差 {d.mean():+6.1%}  p={float(wilcoxon(a,b,alternative='greater')[1]):.3g}")

sk=lambda d:"shapeknots" in d["confirmed_by"]
for key,name in (("pk","RNAPKplex"),("kn","Knotty")):
    print(f"\n=== 标签 = {name} ===")
    ptest("ShapeKnots 确认 vs 未确认", sk, lambda d:not sk(d), key)
    ptest("对照:2家vs1家(都排除SK)", lambda d:d["k_panel"]==2 and not sk(d), lambda d:d["k_panel"]==1 and not sk(d), key)
    ptest("决定性:SK+其他 vs 只有SK", lambda d:sk(d) and d["k_panel"]>=2, lambda d:sk(d) and d["k_panel"]==1, key)

print("\n=== 严格标签:两个外部裁判【都】确认 ===")
allv=[d for v in by.values() for d in v]
for lab,f in (("面板一致(现行)",lambda d:d["k_panel"]==3),("ShapeKnots 确认",sk),
              ("SK 且 >=2 家",lambda d:sk(d) and d["k_panel"]>=2)):
    for key,name in (("pk","RNAPKplex"),("kn","Knotty"),("both","两家都认")):
        g=(lambda d:d["pk"] and d["kn"]) if key=="both" else (lambda d:d[key])
        hi=[g(d) for d in allv if f(d)]; lo=[g(d) for d in allv if not f(d)]
        pos=sum(g(d) for d in allv)
        print(f"  {lab:16s} 对 {name:10s} 命中{len(hi):5d} 精确率 {st.mean(hi):6.1%}(基线{st.mean(lo):6.1%}) 召回 {sum(hi)/max(1,pos):6.1%}")
    print()
