#!/usr/bin/env python3
"""POST-HOC: what |D| does the M10b ladder actually inject?

The ladder builds  y_t = sgn * (g/2) * z_t + N(0,1),  z_t = normalised length rank in [-1,1],
and labels the rung  "真实 |D| ≈ g".  But D is a difference of two SPEARMAN correlations, and

    rho = (g/2)*sd(z) / sqrt((g/2)^2 sd(z)^2 + 1)   with sd(z) ~ 0.577,

so the induced |D| = |rho_a - rho_b| = 2*rho is roughly g/2, not g. This file measures it directly,
with exactly the construction and the target counts the run used. Offline, $0. Changes no verdict.
"""
import math, random, statistics as st, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.stats import spearmanr

GAPS = (0.1, 0.2, 0.4, 0.6, 0.8)
NS = (56, 56, 52, 52, 56, 52)          # the six core pairwise common-target counts used in the run
REPS = 4000


def induced(g, n, rng):
    z = [(r - n / 2) / (n / 2) for r in range(n)]
    L = list(range(n))
    ya = [+(g / 2) * z[i] + rng.gauss(0, 1) for i in range(n)]
    yb = [-(g / 2) * z[i] + rng.gauss(0, 1) for i in range(n)]
    return spearmanr(L, ya)[0] - spearmanr(L, yb)[0]


print("标称 g   注入的系统差 E[D]   实测 |D| 均值(含抽样噪声)   |D| 的 5%~95%")
print("-" * 76)
rows = []
for g in GAPS:
    rng = random.Random(20260826 + int(g * 1000))
    sg, ab = [], []
    for k in range(REPS):
        d = induced(g, NS[k % len(NS)], rng)
        sg.append(d); ab.append(abs(d))
    ab.sort()
    m, s_ = st.fmean(ab), st.fmean(sg)
    rows.append((g, s_))
    print(f"  {g:.1f}          **{s_:+.4f}**            {m:.4f}"
          f"                 [{ab[int(.05*REPS)]:.4f}, {ab[int(.95*REPS)]:.4f}]")
print("\n注入的系统差 E[D] 约等于 0.57*g,不是 g —— 因为 D 是两条【Spearman 相关】之差,")
print("而构造把幅度 g/2 的线性项加在单位方差噪声上,相关被 sd(z)=0.577 与噪声一起压掉。")
print("g=0 时 E|D| 约 0.16(纯抽样),所以标称 0.1/0.2 两档实际上几乎没有注入任何可分辨的东西。")
print("\n主臂最强对比(基准4560 − 样本2)的实测 D 是 -0.7193;梯级最高一档实际只注入到上表最后一行。")
