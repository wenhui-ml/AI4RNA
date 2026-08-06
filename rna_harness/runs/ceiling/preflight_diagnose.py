import json, statistics as st
from collections import Counter
D=json.load(open('/data/wenhui/openclaude_rna/rna_harness/runs/ceiling/designability_preflight.json'))
R=[r for r in D['rows'] if 'error' not in r and r['rep']==0 and r['delivered']]
print('按判决看【tiers_tried 的最后一档】:')
for v in ('certified','knot_confirmed'):
    c=Counter(r['tiers'][-1] for r in R if r['verdict']==v and r.get('tiers'))
    print(f'  {v:15s} {dict(sorted(c.items()))}')
n=sum(1 for r in R if r['verdict']=='knot_confirmed')
print()
print('knot_confirmed 的 return 在 tier 循环【之外】 -> 它必然跑满全部四档。')
print(f'那 31 个不是「180s 才解出来」,是「必然跑满」。受影响 {n} 个 = 交付的 {n/len(R):.1%}')
allr=[r for r in D['rows'] if 'error' not in r and r['rep']==0]
kc=[r for r in allr if r['verdict']=='knot_confirmed']
cf=[r for r in allr if r['verdict']=='certified']
nd=[r for r in allr if not r['delivered']]
tw=sum(r['wall_s'] for r in allr)
for lab,g in (('certified',cf),('knot_confirmed',kc),('未交付',nd)):
    if g: print(f'  {lab:15s} n={len(g):3d}  中位 {st.median([r["wall_s"] for r in g]):6.0f}s  占总时间 {sum(r["wall_s"] for r in g)/tw:5.1%}')
print()
print('=== ② 真正的杠杆:knot_confirmed 与未交付【都】跑满阶梯 ===')
print(f'  两者合计占 {(sum(r["wall_s"] for r in kc)+sum(r["wall_s"] for r in nd))/tw:.1%} 的求解器时间')
