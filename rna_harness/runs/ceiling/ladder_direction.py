import json, statistics as st
from math import comb
D=json.load(open('/data/wenhui/openclaude_rna/rna_harness/runs/ceiling/ladder_truncation.json'))
by={}
for r in D['rows']:
    if 'error' in r: continue
    by.setdefault(r['task_id'],{})[r['arm']]=r
names=['5','5+15','5+15+60','full']
full={k:v for k,v in by.items() if all(n in v for n in names)}
def sgn(u,d):
    n=u+d; return sum(comb(n,i) for i in range(u,n+1))/2**n if n else float('nan')
print(f"{'臂':>10s}{'仅 full 有证':>13s}{'仅该臂有证':>12s}{'符号 p':>10s}")
for n in names[:-1]:
    only_f=sum(1 for v in full.values() if v['full']['delivered'] and not v[n]['delivered'])
    only_a=sum(1 for v in full.values() if v[n]['delivered'] and not v['full']['delivered'])
    print(f"{n:>10s}{only_f:13d}{only_a:12d}{sgn(only_f,only_a):10.3g}")
print()
print("-> 若两个方向数量相当,那 7 张的净差就是噪声,不是截断造成的损失")
