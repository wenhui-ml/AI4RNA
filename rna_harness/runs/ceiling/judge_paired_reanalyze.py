"""Re-analysis of judge_paired.json: does the +36.1 point paired gap survive GC control,
is it monotone in panel size, and is it driven by one oracle?"""
import json, statistics as st
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon

D = json.load(open("/data/wenhui/openclaude_rna/rna_harness/runs/ceiling/judge_paired.json"))
by = defaultdict(list)
for r in D["rows"]:
    if r.get("error") or r.get("skipped"): continue
    for d in r.get("designs", []):
        if d["external_ran"]: by[r["task_id"]].append(d)

# ---- 1. dose-response: is external agreement monotone in k_panel? -------------------
print("=== 剂量-反应:外部同意率随面板家数 ===")
for k in (1,2,3):
    v = [d["external"] for vv in by.values() for d in vv if d["k_panel"]==k]
    print(f"  {k}家确认  n={len(v):5d}  外部同意 {st.mean(v):6.1%}")

# ---- 2. GC-matched pairing: only pairs whose GC differs by <2 points -----------------
def paired(filter_gc=None):
    P=[]
    for t,v in by.items():
        un=[d for d in v if d["k_panel"]==3]; sp=[d for d in v if 1<=d["k_panel"]<=2]
        if not(un and sp): continue
        gu=st.mean([d["gc"] for d in un]); gs=st.mean([d["gc"] for d in sp])
        if filter_gc is not None and abs(gu-gs)>filter_gc: continue
        P.append((t, st.mean([d["external"] for d in un]), st.mean([d["external"] for d in sp]), gu-gs))
    return P

for lab,f in (("全部配对",None),("GC 差 <2 点",2.0),("GC 差 <1 点",1.0)):
    P=paired(f)
    if len(P)<8:
        print(f"\n  {lab}: 只剩 {len(P)} 个靶标 < 8,不检验"); continue
    A=np.array([p[1] for p in P]); B=np.array([p[2] for p in P]); d=A-B
    p=float(wilcoxon(A,B,alternative='greater')[1]) if (d!=0).sum() else float('nan')
    print(f"\n  {lab}: {len(P)} 个配对靶标  差 {d.mean():+.1%}  p={p:.3g}  "
          f"(好 {(d>0).sum()} 差 {(d<0).sum()} 平 {(d==0).sum()})  平均 GC 差 {np.mean([p_[3] for p_ in P]):+.2f}")

# ---- 3. is it one oracle doing the work? -------------------------------------------
print("\n=== 拆开:哪一家的确认在携带信息(单家确认时的外部同意率) ===")
for name in ("pkiss","probknot","shapeknots"):
    v=[d["external"] for vv in by.values() for d in vv if d["confirmed_by"]==[name]]
    if v: print(f"  只有 {name:11s} 确认  n={len(v):4d}  外部同意 {st.mean(v):6.1%}")
print("  --- 两家组合 ---")
for combo in (["pkiss","probknot"],["pkiss","shapeknots"],["probknot","shapeknots"]):
    v=[d["external"] for vv in by.values() for d in vv if d["confirmed_by"]==sorted(combo)]
    if v: print(f"  {'+'.join(combo):24s} n={len(v):4d}  外部同意 {st.mean(v):6.1%}")

# ---- 4. RNAstructure counted as ONE source: 2-source unanimity ----------------------
print("\n=== 按 v2 协议把 RNAstructure 的两个二进制算作一个来源 ===")
def src(d):
    s=set(d["confirmed_by"]); out=set()
    if "pkiss" in s: out.add("pkiss")
    if s & {"probknot","shapeknots"}: out.add("rnastructure")
    return out
for n in (1,2):
    v=[d["external"] for vv in by.values() for d in vv if len(src(d))==n]
    print(f"  {n} 个独立来源确认  n={len(v):5d}  外部同意 {st.mean(v):6.1%}")
P=[]
for t,v in by.items():
    un=[d for d in v if len(src(d))==2]; sp=[d for d in v if len(src(d))==1]
    if un and sp: P.append((st.mean([d["external"] for d in un]), st.mean([d["external"] for d in sp])))
A=np.array([p[0] for p in P]); B=np.array([p[1] for p in P]); d=A-B
print(f"  同靶标配对(2来源 vs 1来源):{len(P)} 个靶标  差 {d.mean():+.1%}  "
      f"p={float(wilcoxon(A,B,alternative='greater')[1]):.3g}")
