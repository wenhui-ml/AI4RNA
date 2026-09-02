"""M66 inference: S5 (which factor moves the number more) with bootstrap intervals,
and the headline spreads with intervals.  Offline, CPU."""
import os, re, sys, json, collections, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m66_load import load, UNIVERSE

SEED, NBOOT = 20260902, 20000
rng = np.random.default_rng(SEED)
BAD = {"Undisclosed", "NA"}

R = [r for r in load() if not (isinstance(r["info_resolved"], (int, float))
     and abs(r["info_resolved"] - 100.0*len(r["resolved"])/UNIVERSE) > 0.25)]
def version(e):
    m = re.search(r'[-_]v?([0-9]+\.[0-9]+\.[0-9]+)', e); return m.group(1) if m else "NA"
for r in R:
    r["cell"] = (r["agent"], version(r["entry"])); r["rate"] = len(r["resolved"])/UNIVERSE
R = [r for r in R if r["model"] not in BAD and not r["model"].startswith("http")]

# best rate per (model, cell)
best = {}
for r in R:
    k = (r["model"], r["cell"])
    if k not in best or r["rate"] > best[k]["rate"]: best[k] = r
rows = list(best.values())

by_model = collections.defaultdict(list); by_cell = collections.defaultdict(list)
for r in rows: by_model[r["model"]].append(r); by_cell[r["cell"]].append(r)

def spread(v): return (max(v)-min(v))*100.0
WM = {m: spread([x["rate"] for x in v]) for m, v in by_model.items() if len(v) >= 2}
WC = {c: spread([x["rate"] for x in v]) for c, v in by_cell.items() if len(v) >= 2}
WM4 = {m: s for m, s in WM.items() if len(by_model[m]) >= 4}
WC4 = {c: s for c, s in WC.items() if len(by_cell[c]) >= 4}

def boot_med(vals):
    a = np.asarray(vals, float)
    b = np.array([np.median(rng.choice(a, len(a), replace=True)) for _ in range(NBOOT)])
    return float(np.median(a)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

out = {}
print("### headline spreads, with bootstrap intervals over the units")
for tag, D, unit in [("withinModel", WM, "models measured under >=2 scaffold cells"),
                     ("withinCell",  WC, "scaffold cells holding >=2 models"),
                     ("withinModel4", WM4, "models under >=4 cells"),
                     ("withinCell4",  WC4, "cells with >=4 models")]:
    if len(D) < 3: 
        print(f"  {tag}: n={len(D)} -- too few, reported as under-powered"); continue
    m, lo, hi = boot_med(list(D.values()))
    out[tag] = dict(n=len(D), median=m, ci=[lo, hi], max=max(D.values()))
    print(f"  {tag:13s} n={len(D):3d}  median spread={m:5.2f}pp  95% CI [{lo:.2f}, {hi:.2f}]  max={max(D.values()):.1f}pp   ({unit})")

print("\n### S5 -- does the scaffold move the number as much as the model does?")
if len(WM4) >= 3 and len(WC4) >= 3:
    a, b = list(WM4.values()), list(WC4.values())
    d = np.median(a) - np.median(b)
    bs = np.array([np.median(rng.choice(a, len(a), True)) - np.median(rng.choice(b, len(b), True))
                   for _ in range(NBOOT)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    out["S5"] = dict(within_model_median=float(np.median(a)), n_models=len(a),
                     within_cell_median=float(np.median(b)), n_cells=len(b),
                     diff=float(d), ci=[float(lo), float(hi)])
    print(f"  within-MODEL across scaffolds : median {np.median(a):5.2f}pp  (n={len(a)} models)")
    print(f"  within-SCAFFOLD across models : median {np.median(b):5.2f}pp  (n={len(b)} cells)")
    print(f"  difference                    : {d:+5.2f}pp   95% CI [{lo:+.2f}, {hi:+.2f}]")
    print(f"  S5 (within-model >= within-scaffold): {'HOLDS' if lo > 0 else ('UNRESOLVED, interval covers 0' if hi > 0 else 'FAILS')}")
else:
    print(f"  not identified: {len(WM4)} models / {len(WC4)} cells at the >=4 bar")

print("\n### the board's own scale")
allr = np.array([r["rate"] for r in rows])*100
out["board"] = dict(n_entries=len(rows), iqr=float(np.percentile(allr,75)-np.percentile(allr,25)),
                    spread=float(allr.max()-allr.min()), median=float(np.median(allr)))
print(f"  entries={len(rows)}  median={np.median(allr):.1f}%  IQR={out['board']['iqr']:.1f}pp  full spread={out['board']['spread']:.1f}pp")

print("\n### arm E -- non-attempts scored as failures")
ng = [(r["agent"], len(r["no_generation"])) for r in R if r["fmt"] == "results_json"]
per = collections.defaultdict(list)
for a, n in ng: per[a].append(n)
big = {a: float(np.mean(v))/5.0 for a, v in per.items() if len(v) >= 4}
out["noGen"] = dict(entries=len(ng), total=int(sum(n for _, n in ng)),
                    by_scaffold={a: round(v,2) for a, v in sorted(big.items(), key=lambda x:-x[1])},
                    max_entry_pp=float(max(n for _, n in ng))/5.0)
print(f"  entries={len(ng)}  total never-attempted instances={out['noGen']['total']}  "
      f"worst entry={out['noGen']['max_entry_pp']:.1f}pp")

out["deepest_case"] = {}
tgt = "claude-3-5-sonnet-20241022"
v = sorted([x for x in rows if x["model"] == tgt], key=lambda x: -x["rate"])
if v:
    out["deepest_case"] = dict(model=tgt, n_cells=len(v), hi=v[0]["rate"]*100, lo=v[-1]["rate"]*100,
                               spread=(v[0]["rate"]-v[-1]["rate"])*100,
                               hi_scaffold=v[0]["agent"], lo_scaffold=v[-1]["agent"])
    print(f"\n  deepest case {tgt}: {len(v)} cells, {v[-1]['rate']*100:.1f}% ({v[-1]['agent']}) "
          f"-> {v[0]['rate']*100:.1f}% ({v[0]['agent']})")

out["provenance"] = dict(corpus="SWE-bench/experiments", commit="8fa0575cea71b5258b6cd211ee0a3f42a2f5fa1e",
                         split="evaluation/verified", universe=UNIVERSE, seed=SEED, nboot=NBOOT,
                         entries_loaded=len(R), placeholder_labels_dropped=sorted(BAD))
p = "/data/wenhui/openclaude_rna/rna_harness/runs/ceiling/m66_swebench_placement.json"
json.dump(out, open(p, "w"), indent=1, default=str)
print(f"\nwrote {p}")
