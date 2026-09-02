"""M66: the control the headline itself needs.  If the SAME scaffold resubmitted moves
the score as much as CHANGING the scaffold does, then 'the scaffold carries the number'
is not the right sentence."""
import os, re, sys, json, collections, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m66_load import load, UNIVERSE
SEED = 20260902; rng = np.random.default_rng(SEED)
BAD = {"Undisclosed", "NA"}
R = [r for r in load() if not (isinstance(r["info_resolved"], (int, float))
     and abs(r["info_resolved"] - 100.0*len(r["resolved"])/UNIVERSE) > 0.25)]
for r in R: r["rate"] = len(r["resolved"])/UNIVERSE
R = [r for r in R if r["model"] not in BAD and not r["model"].startswith("http")]

# A: within (model, agent) -- the same scaffold family resubmitted
A = collections.defaultdict(list)
for r in R: A[(r["model"], r["agent"])].append(r["rate"])
Asp = [ (max(v)-min(v))*100 for v in A.values() if len(v) >= 2 ]
# B: within model, ACROSS agents -- one best entry per agent so resubmission cannot leak in
Bm = collections.defaultdict(dict)
for r in R:
    d = Bm[r["model"]]
    if r["agent"] not in d or r["rate"] > d[r["agent"]]: d[r["agent"]] = r["rate"]
Bsp = [ (max(v.values())-min(v.values()))*100 for v in Bm.values() if len(v) >= 2 ]

print(f"A  within (model, SAME agent), resubmissions : n={len(Asp):3d}  median={np.median(Asp):5.2f}pp  max={max(Asp):.1f}pp")
print(f"B  within model, ACROSS agents (best/agent)  : n={len(Bsp):3d}  median={np.median(Bsp):5.2f}pp  max={max(Bsp):.1f}pp")
bs = np.array([np.median(rng.choice(Bsp, len(Bsp), True)) - np.median(rng.choice(Asp, len(Asp), True))
               for _ in range(20000)])
lo, hi = np.percentile(bs, [2.5, 97.5])
d = np.median(Bsp)-np.median(Asp)
print(f"\nB - A = {d:+5.2f}pp   95% CI [{lo:+.2f}, {hi:+.2f}]  -> "
      f"{'SEPARATED: changing the scaffold moves it more' if lo>0 else 'NOT separated; interval covers 0'}")

print("\n### matched: models measured BOTH ways, so the same model supplies both numbers")
pair = []
for m, agents in Bm.items():
    if len(agents) < 2: continue
    within = [ (max(v)-min(v))*100 for (mm, a), v in A.items() if mm == m and len(v) >= 2 ]
    if not within: continue
    across = (max(agents.values())-min(agents.values()))*100
    pair.append((m, max(within), across, len(agents)))
print(f"  models with both quantities: {len(pair)}")
for m, w, a, k in sorted(pair, key=lambda x: -x[2]):
    print(f"    {m[:30]:30s} agents={k:2d}  same-scaffold_max={w:5.1f}pp   across-scaffold={a:5.1f}pp   "
          f"{'across wins' if a>w else 'SAME-SCAFFOLD WINS'}")
if pair:
    dd = np.array([a-w for _, w, a, _ in pair])
    b2 = np.array([np.median(rng.choice(dd, len(dd), True)) for _ in range(20000)])
    l2, h2 = np.percentile(b2, [2.5, 97.5])
    print(f"\n  paired (across - same), median = {np.median(dd):+5.2f}pp  95% CI [{l2:+.2f}, {h2:+.2f}]  n={len(dd)}")
    print(f"  models where changing the scaffold moves it MORE than resubmitting the same one: "
          f"{sum(1 for _, w, a, _ in pair if a > w)}/{len(pair)}")
json.dump(dict(A_n=len(Asp), A_median=float(np.median(Asp)), A_max=float(max(Asp)),
               B_n=len(Bsp), B_median=float(np.median(Bsp)), B_max=float(max(Bsp)),
               diff=float(d), ci=[float(lo), float(hi)],
               paired_n=len(pair), paired_median=float(np.median(dd)) if pair else None,
               paired_ci=[float(l2), float(h2)] if pair else None, seed=SEED),
          open("/data/wenhui/openclaude_rna/rna_harness/runs/ceiling/m66_headline_control.json","w"), indent=1)
