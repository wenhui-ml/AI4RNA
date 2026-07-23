#!/usr/bin/env python3
"""Statistical separation of the panel-optimizing operator's panel-unanimous ascent
from the single-oracle gaming baseline (chomsky_climb positive side).

Reads the per-target panel adjudications (pKiss AND ProbKnot AND ShapeKnots -> unanimous)
for panel_auto (3 seeds) and the autonomous gamer (seed 0), both autonomous and on the SAME
60 held-out Pseudobase crossing targets. Reports:
  (1) per-seed + pooled panel-unanimous rates with percentile-bootstrap 95% CIs;
  (2) paired McNemar exact test (seed 0): panel_auto vs gamer on identical targets;
  (3) exact one-sided binomial of each panel_auto seed / pool vs the gamer's fixed 1/60 rate.
No live episodes: pure offline re-use of saved adjudications.
"""
import json, math, os, random

random.seed(20260723)
HERE = os.path.dirname(os.path.abspath(__file__))
IDS = [f"pseudobase:{i}" for i in range(60)]          # the frozen 60-target held-out pool
GAMER_BASELINE = 1 / 60                                 # single-oracle gamer, panel-unanimous

def uni_set(path):
    d = json.load(open(os.path.join(HERE, path)))
    return {r["id"] for r in d["rows"] if r.get("unanimous")}

def vec(s):  # binary unanimous vector over the 60 held-out ids
    return [1 if i in s else 0 for i in IDS]

def boot_ci(v, B=20000, lo=2.5, hi=97.5):
    n = len(v); rates = []
    for _ in range(B):
        rates.append(sum(v[random.randrange(n)] for _ in range(n)) / n)
    rates.sort()
    return rates[int(lo/100*B)], rates[int(hi/100*B)]

def binom_tail_ge(k, n, p):   # P(X >= k) for X~Bin(n,p), exact one-sided
    return sum(math.comb(n, j) * p**j * (1-p)**(n-j) for j in range(k, n+1))

def mcnemar_exact(b, c):      # two-sided exact McNemar on discordant counts b,c
    n = b + c
    if n == 0: return 1.0
    m = min(b, c)
    p = 2 * sum(math.comb(n, k) * 0.5**n for k in range(0, m+1))
    return min(p, 1.0)

gamer = uni_set("generous/panel_autonomous.json")
seeds = {
    "panel_auto seed0": uni_set("panel_auto/panel_panel_auto.json"),
    "panel_auto seed1": uni_set("seed1/panel/panel_panel_auto.json"),
    "panel_auto seed2": uni_set("seed2/panel/panel_panel_auto.json"),
}

print(f"held-out pool n=60 ; gamer panel-unanimous = {len(gamer)}/60 = {len(gamer)/60:.4f}  ids={sorted(gamer)}\n")

print("(1) panel_auto panel-unanimous rate + percentile-bootstrap 95% CI (B=20000):")
pooled = []
for name, s in seeds.items():
    v = vec(s); pooled += v
    lo, hi = boot_ci(v)
    clears = "CLEARS" if lo > GAMER_BASELINE else "does NOT clear"
    print(f"  {name}: {len(s):2d}/60 = {len(s)/60:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  ({clears} gamer {GAMER_BASELINE:.4f})")
lo, hi = boot_ci(pooled)
clears = "CLEARS" if lo > GAMER_BASELINE else "does NOT clear"
print(f"  POOLED 3 seeds: {sum(pooled)}/{len(pooled)} = {sum(pooled)/len(pooled):.4f}  95% CI [{lo:.4f}, {hi:.4f}]  ({clears} gamer {GAMER_BASELINE:.4f})")

print("\n(2) paired McNemar exact (SAME 60 targets), panel_auto seedX vs gamer:")
for name, s in seeds.items():
    b = len(s - gamer)          # panel_auto unanimous, gamer not
    c = len(gamer - s)          # gamer unanimous, panel_auto not
    p = mcnemar_exact(b, c)
    print(f"  {name}: b(only panel_auto)={b}  c(only gamer)={c}  discordant={b+c}  McNemar exact p={p:.4f}")

print("\n(3) exact one-sided binomial vs gamer's fixed rate p0=1/60:")
for name, s in seeds.items():
    k = len(s); p = binom_tail_ge(k, 60, GAMER_BASELINE)
    print(f"  {name}: P(X>={k}/60 | p0=1/60) = {p:.2e}")
kk = sum(pooled); nn = len(pooled)
print(f"  POOLED: P(X>={kk}/{nn} | p0=1/60) = {binom_tail_ge(kk, nn, GAMER_BASELINE):.2e}")
