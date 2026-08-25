#!/usr/bin/env python3
"""Diagnostic on M2b, NOT a re-decision. The frozen K3 compared the SELECTED half's effect/MDE against the
max-of-two-halves under random splitting. That comparator is weak by construction: the pooled mean is
positive, so BOTH halves of a random split are positive and the null sits high (1.71). The statistic that is
zero under random splitting is the DIFFERENCE between the halves. Measured here, labelled as motivation for
the next declaration, never as a change to M2b's verdict.
"""
from __future__ import annotations
import json, math, os, random, statistics as st, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)

R = json.load(open(os.path.join(HERE, "m2b_gt_target_property.json")))
G = R["g_all"]; PRIMARY = "panel_f1_fresh.json"
SEL = [k for k in G if k != PRIMARY]
sel = {}
for t in set().union(*[set(G[k]) for k in SEL]):
    v = [G[k][t] for k in SEL if t in G[k]]
    if len(v) >= 2:
        sel[t] = st.fmean(v)
thr = R["arm2"]["threshold"]
gp = G[PRIMARY]
ts = sorted(t for t in gp if t in sel)
hi = [gp[t] for t in ts if sel[t] > thr]; lo = [gp[t] for t in ts if sel[t] <= thr]
d = st.fmean(hi) - st.fmean(lo)
print(f"冻结阈值 {thr:+.5f};主人群靶标 {len(ts)} = 高 {len(hi)} + 低 {len(lo)}")
print(f"高半 {st.fmean(hi):+.4f} (sd {st.stdev(hi):.4f})  低半 {st.fmean(lo):+.4f} (sd {st.stdev(lo):.4f})"
      f"  **半间差 {d:+.4f}**")

rng = random.Random(20260825); N = 20000; ge = 0; nulls = []
for _ in range(N):
    v = [gp[t] for t in ts]; rng.shuffle(v)
    dd = st.fmean(v[:len(hi)]) - st.fmean(v[len(hi):])
    nulls.append(dd)
    if abs(dd) >= abs(d):
        ge += 1
nulls.sort()
print(f"置换零分布({N} 次,统计量在随机切分下期望为 0):中位 {nulls[N//2]:+.5f}"
      f"  95% [{nulls[int(.025*N)]:+.4f}, {nulls[int(.975*N)]:+.4f}]")
print(f"**半间差的双侧置换 p = {(ge+1)/(N+1):.5f}**")

# cluster bootstrap on the difference (targets are the unit)
B = 5000; rb = random.Random(77); ds = []
H_ = [gp[t] for t in ts if sel[t] > thr]; L_ = [gp[t] for t in ts if sel[t] <= thr]
for _ in range(B):
    a = [H_[rb.randrange(len(H_))] for _ in range(len(H_))]
    b = [L_[rb.randrange(len(L_))] for _ in range(len(L_))]
    ds.append(st.fmean(a) - st.fmean(b))
ds.sort()
ci = [round(ds[int(.025*B)], 4), round(ds[int(.975*B)], 4)]
print(f"半间差的靶标 cluster bootstrap 95% {ci}  {'排除 0' if ci[0] > 0 or ci[1] < 0 else '含 0'}")
ex = 0
for s in range(50):
    r2 = random.Random(4000 + 13 * s); dd = []
    for _ in range(2000):
        a = [H_[r2.randrange(len(H_))] for _ in range(len(H_))]
        b = [L_[r2.randrange(len(L_))] for _ in range(len(L_))]
        dd.append(st.fmean(a) - st.fmean(b))
    dd.sort()
    c = (dd[50], dd[1949])
    ex += 1 if (c[0] > 0 or c[1] < 0) else 0
print(f"[第51条] 种子翻转 {ex}/50  {'裁定稳定' if ex in (0, 50) else '**由 RNG 决定**'}")

# what the selection would have to be worth: the high half's own interval, and the low half's
for nm, v in (("高半", H_), ("低半", L_)):
    bb = []
    for _ in range(B):
        bb.append(st.fmean([v[rb.randrange(len(v))] for _ in range(len(v))]))
    bb.sort()
    print(f"  {nm} 均值 95% [{bb[int(.025*B)]:+.4f}, {bb[int(.975*B)]:+.4f}]")
print("\n**这是诊断,不是改判。M2b 的冻结裁定仍是 `main_part_below_resolution`(K2 在 1.93 对 2.00 上触发)。**")
