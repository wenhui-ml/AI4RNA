import json,statistics as st,sys
from collections import defaultdict
sys.path.insert(0,"/data/wenhui/openclaude_rna"); sys.path.insert(0,"/data/wenhui/openclaude_rna/rna_harness/runs/ceiling")
from rna_harness.benchmarks.rnainvbench import _round_only, load_pseudobase, pk_pairs
C="/data/wenhui/openclaude_rna/rna_harness/runs/ceiling/"
P=json.load(open(C+"judge_paired.json")); K=json.load(open(C+"knotty_judge.json")); S=json.load(open(C+"ceiling_and_coverage.json"))["S_ext"]
kn={s:(None if v is None else {tuple(x) for x in v}) for s,v in K["knotty"].items()}
tgt={str(t.task_id):str(t.target) for t in load_pseudobase()}
allv=[]
for r in P["rows"]:
    if r.get("error") or r.get("skipped"): continue
    tg=tgt.get(r["task_id"]);
    if not tg: continue
    cross=pk_pairs(tg)-pk_pairs(_round_only(tg))
    for d in r.get("designs",[]):
        if not d["external_ran"] or kn.get(d["seq"]) is None: continue
        allv.append({**d,"pk":bool(d["external"]),"kn":bool(kn[d["seq"]]>=cross)})
print(f"{'grade':>6s}{'n':>6s}{'PKplex':>9s}{'/ceil':>7s}{'Knotty':>9s}{'/ceil':>7s}{'both':>8s}{'/ceil':>7s}")
out={}
for k in (1,2,3):
    v=[d for d in allv if d["k_panel"]==k]
    a=st.mean([d["pk"] for d in v]); b=st.mean([d["kn"] for d in v]); c=st.mean([d["pk"] and d["kn"] for d in v])
    out[k]={"n":len(v),"pkplex":round(a,4),"knotty":round(b,4),"both":round(c,4),
            "pkplex_vs_ceiling":round(a/S["RNAPKplex"]["rate"],3),"knotty_vs_ceiling":round(b/S["Knotty"]["rate"],3),
            "both_vs_ceiling":round(c/S["both"]["rate"],3)}
    print(f"{'k='+str(k):>6s}{len(v):6d}{a:9.1%}{a/S['RNAPKplex']['rate']:7.0%}{b:9.1%}{b/S['Knotty']['rate']:7.0%}{c:8.1%}{c/S['both']['rate']:7.0%}")
print(json.dumps(out,indent=1))
