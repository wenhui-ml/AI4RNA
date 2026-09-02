"""M66: the actual instrument port.  Everything before this was entry-level aggregate
spread.  R1's estimator works inside strata, and the stratum here is the instance."""
import os, re, sys, json, collections, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m66_load import load, UNIVERSE

SEED = 20260902
rng = np.random.default_rng(SEED)
BAD = {"Undisclosed", "NA"}
R = [r for r in load() if not (isinstance(r["info_resolved"], (int, float))
     and abs(r["info_resolved"] - 100.0*len(r["resolved"])/UNIVERSE) > 0.25)]
def version(e):
    m = re.search(r'[-_]v?([0-9]+\.[0-9]+\.[0-9]+)', e); return m.group(1) if m else "NA"
for r in R:
    r["cell"] = (r["agent"], version(r["entry"])); r["rate"] = len(r["resolved"])/UNIVERSE
R = [r for r in R if r["model"] not in BAD and not r["model"].startswith("http")]
best = {}
for r in R:
    k = (r["model"], r["cell"])
    if k not in best or r["rate"] > best[k]["rate"]: best[k] = r

by_model = collections.defaultdict(list)
for (m, c), r in best.items(): by_model[m].append(r)

def mcnemar_exact(b, c):
    """two-sided exact binomial on the discordant pairs"""
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    from math import comb
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2.0 ** n) * 2.0
    return min(1.0, p)

print("### the scaffold does not only move the score: it moves WHICH instances are solved")
print("    (paired at the instance; b/c are the discordant instances)\n")
recs = []
for m, v in sorted(by_model.items()):
    if len(v) < 2: continue
    for x, y in itertools.combinations(sorted(v, key=lambda z: -z["rate"]), 2):
        A, B = x["resolved"], y["resolved"]
        b = len(A - B); c = len(B - A); both = len(A & B)
        disc = b + c
        union = len(A | B)
        p = mcnemar_exact(b, c)
        recs.append(dict(model=m, hi=x["agent"], lo=y["agent"],
                         rateA=len(A)/UNIVERSE, rateB=len(B)/UNIVERSE,
                         b=b, c=c, both=both, disc=disc, union=union, p=p,
                         jac=both/union if union else float("nan"),
                         churn=disc/union if union else float("nan")))
print(f"paired scaffold comparisons on a fixed model: {len(recs)}")
top = sorted(recs, key=lambda r: -r["disc"])[:12]
for r in top:
    print(f"  {r['model'][:26]:26s} {r['hi'][:16]:16s}({r['rateA']*100:4.1f}%) vs "
          f"{r['lo'][:16]:16s}({r['rateB']*100:4.1f}%)  b={r['b']:3d} c={r['c']:3d} "
          f"disc={r['disc']:3d}  churn={r['churn']:.2f}  p={r['p']:.2g}")

ch = np.array([r["churn"] for r in recs]); ja = np.array([r["jac"] for r in recs])
dr = np.array([abs(r["rateA"]-r["rateB"])*100 for r in recs])
print(f"\n  churn = discordant / union-of-solved")
print(f"    median churn   = {np.median(ch):.3f}   (IQR {np.percentile(ch,25):.3f}-{np.percentile(ch,75):.3f})")
print(f"    median Jaccard = {np.median(ja):.3f}")
print(f"    median |rate difference| = {np.median(dr):.1f}pp")

print("\n### the sharp form: pairs whose SCORES nearly agree but whose SOLVED SETS do not")
near = [r for r in recs if abs(r["rateA"]-r["rateB"]) <= 0.02]
print(f"  comparisons where the two scaffolds differ by <=2.0pp in score: {len(near)}")
for r in sorted(near, key=lambda r: -r["disc"])[:8]:
    print(f"    {r['model'][:24]:24s} {r['hi'][:15]:15s} {r['rateA']*100:4.1f}%  vs  "
          f"{r['lo'][:15]:15s} {r['rateB']*100:4.1f}%   -> {r['disc']:3d} instances disagree "
          f"({r['churn']*100:.0f}% of the union)")
if near:
    cn = np.array([r["churn"] for r in near])
    print(f"  median churn among near-tied pairs: {np.median(cn):.3f}  "
          f"(n={len(near)}, median score gap {np.median([abs(r['rateA']-r['rateB'])*100 for r in near]):.1f}pp)")

out = dict(n_comparisons=len(recs),
           median_churn=float(np.median(ch)), churn_iqr=[float(np.percentile(ch,25)), float(np.percentile(ch,75))],
           median_jaccard=float(np.median(ja)), median_rate_gap_pp=float(np.median(dr)),
           near_tied=dict(n=len(near),
                          median_churn=float(np.median([r["churn"] for r in near])) if near else None,
                          median_gap_pp=float(np.median([abs(r["rateA"]-r["rateB"])*100 for r in near])) if near else None),
           seed=SEED)
p = "/data/wenhui/openclaude_rna/rna_harness/runs/ceiling/m66_instance_level.json"
json.dump(out, open(p, "w"), indent=1)
print(f"\nwrote {p}")

print("\n" + "="*72)
print("### THE CONTROL: is the churn just run-to-run noise?")
print("    same model, SAME scaffold, different submissions -> the noise floor\n")
allrows = collections.defaultdict(list)
for r in R: allrows[(r["model"], r["agent"])].append(r)
same = []
for (m, a), v in sorted(allrows.items()):
    if len(v) < 2: continue
    for x, y in itertools.combinations(sorted(v, key=lambda z: -z["rate"]), 2):
        A, B = x["resolved"], y["resolved"]
        u = len(A | B)
        if u == 0: continue
        same.append(dict(model=m, agent=a, rateA=len(A)/UNIVERSE, rateB=len(B)/UNIVERSE,
                         disc=len(A-B)+len(B-A), churn=(len(A-B)+len(B-A))/u,
                         dA=x["date"], dB=y["date"]))
print(f"same-model same-scaffold pairs available: {len(same)}")
for s in sorted(same, key=lambda s: -s["churn"])[:10]:
    print(f"  {s['model'][:24]:24s} {s['agent'][:20]:20s} {s['rateA']*100:4.1f}% vs {s['rateB']*100:4.1f}%  "
          f"disc={s['disc']:3d} churn={s['churn']:.3f}  ({s['dA']} vs {s['dB']})")
if same:
    sc = np.array([s["churn"] for s in same])
    print(f"\n  NOISE FLOOR  median same-scaffold churn = {np.median(sc):.3f}  "
          f"(n={len(sc)}, IQR {np.percentile(sc,25):.3f}-{np.percentile(sc,75):.3f})")
    print(f"  CROSS        median cross-scaffold churn = {np.median(ch):.3f}  (n={len(ch)})")
    print(f"  near-tied    median churn                = {np.median(cn):.3f}  (n={len(cn)})")
    # bootstrap the gap
    rr = np.random.default_rng(SEED)
    bs = np.array([np.median(rr.choice(ch, len(ch), True)) - np.median(rr.choice(sc, len(sc), True))
                   for _ in range(20000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"  cross - same = {np.median(ch)-np.median(sc):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
          f"-> {'SEPARATED' if lo>0 else 'NOT separated; interval covers 0'}")
    bs2 = np.array([np.median(rr.choice(cn, len(cn), True)) - np.median(rr.choice(sc, len(sc), True))
                    for _ in range(20000)])
    lo2, hi2 = np.percentile(bs2, [2.5, 97.5])
    print(f"  near-tied - same = {np.median(cn)-np.median(sc):+.3f}  95% CI [{lo2:+.3f}, {hi2:+.3f}]  "
          f"-> {'SEPARATED' if lo2>0 else 'NOT separated; interval covers 0'}")

print("\n### how many distinct models carry the near-tied result?")
print("   models:", dict(collections.Counter(r['model'][:26] for r in near)))

# --- persist the control (the first dump above ran before the control existed)
out["control_same_scaffold"] = dict(n=len(same), median_churn=float(np.median(sc)),
                                    iqr=[float(np.percentile(sc,25)), float(np.percentile(sc,75))])
out["gap_cross_minus_same"] = dict(diff=float(np.median(ch)-np.median(sc)), ci=[float(lo), float(hi)],
                                   separated=bool(lo > 0))
out["gap_near_minus_same"] = dict(diff=float(np.median(cn)-np.median(sc)), ci=[float(lo2), float(hi2)],
                                  separated=bool(lo2 > 0))
out["near_tied_models"] = {k: v for k, v in collections.Counter(r['model'] for r in near).items()}
json.dump(out, open(p, "w"), indent=1)
print(f"re-wrote {p} with the control")
