import json, statistics as st
import numpy as np
from scipy.stats import wilcoxon
C="/data/wenhui/openclaude_rna/rna_harness/runs/ceiling/"
def load(p):
    by={}
    for r in json.load(open(p))['rows']:
        if 'error' not in r: by.setdefault(r['task_id'],{})[r['arm']]=r
    return by
def od(r):
    if r['verdict']=='certified': return 4
    if r['verdict']=='knot_confirmed': return max(1,min(3,r['k']))
    return 0
new=load(C+"ladder_confirm.json"); old=load(C+"ladder_truncation.json")
pool=dict(old)
for k,v in new.items(): pool.setdefault(k,{}).update(v)

print("=== E2 的 +0.749 是【怎么】来的:拆开 61 个 full 更高的靶标 ===")
P=[(v['5'],v['full']) for v in pool.values() if '5' in v and 'full' in v]
hi=[(a,b) for a,b in P if od(b)>od(a)]
to_cert=sum(1 for a,b in hi if b['verdict']=='certified' and a['verdict']!='certified')
new_deliv=sum(1 for a,b in hi if b['delivered'] and not a['delivered'])
pure_k=sum(1 for a,b in hi if a['verdict']==b['verdict']=='knot_confirmed')
print(f"  共 {len(hi)} 个提升:")
print(f"    从【未交付】变成有证书      {new_deliv:3d}")
print(f"    从 knot_confirmed 升到 certified {to_cert - new_deliv if to_cert>=new_deliv else to_cert:3d}"
      f"  (其中含上一行以外的)")
print(f"    两臂都 knot_confirmed、纯 k 提升   {pure_k:3d}   <- 上一轮【只看这一类】,所以只看到 +0.13")
print()
d_all=np.array([od(b)-od(a) for a,b in P])
kk=[(a,b) for a,b in P if a['verdict']==b['verdict']=='knot_confirmed']
d_kk=np.array([b['k']-a['k'] for a,b in kk])
print(f"  全部 {len(d_all)} 个靶标,序数平均差 {d_all.mean():+.3f}")
print(f"  只看两臂都 knot_confirmed 的 {len(d_kk)} 个,k 平均差 {d_kk.mean():+.3f}"
      f"  (上一轮 n~30 时是 +0.13)")
print("  -> 预算主要买的是【把靶标推过 certified 那道线】,不是在 knot_confirmed 内部涨 k")

print()
print("=== E1:截断到 (5,15,60) 的 +0.10 等级代价,本身可分辨吗 ===")
F={k:v for k,v in new.items() if all(n in v for n in ('5','5+15+60','full'))}
d=np.array([od(v['full'])-od(v['5+15+60']) for v in F.values()])
nz=int((d!=0).sum())
p=float(wilcoxon([od(v['full']) for v in F.values()],[od(v['5+15+60']) for v in F.values()],alternative='greater')[1]) if nz>=8 else float('nan')
se=d.std(ddof=1)/len(d)**0.5
print(f"  n={len(d)}  平均 {d.mean():+.3f}  95% [{d.mean()-1.96*se:+.3f}, {d.mean()+1.96*se:+.3f}]")
print(f"  更高 {int((d>0).sum())}  更低 {int((d<0).sum())}  持平 {int((d==0).sum())}   Wilcoxon p={p:.3g}")
print(f"  -> {'代价【可分辨】:过了 <=0.20 的线,但不是零,报价时必须说出来' if p<0.05 else '代价与零不可分辨'}")
