"""M66 arm D, minimal identified form: does the ORDER of two models flip with the scaffold?"""
import os, re, sys, collections, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m66_load import load, UNIVERSE
R = [r for r in load() if not (isinstance(r["info_resolved"], (int, float))
     and abs(r["info_resolved"] - 100.0*len(r["resolved"])/UNIVERSE) > 0.25)]
def version(e):
    m = re.search(r'[-_]v?([0-9]+\.[0-9]+\.[0-9]+)', e); return m.group(1) if m else "NA"
BAD = {"Undisclosed", "NA"}
for r in R:
    r["cell"] = (r["agent"], version(r["entry"])); r["rate"] = len(r["resolved"])/UNIVERSE
R = [r for r in R if r["model"] not in BAD and not r["model"].startswith("http")]

cm = collections.defaultdict(dict)      # cell -> model -> best rate
for r in R:
    if r["model"] not in cm[r["cell"]] or r["rate"] > cm[r["cell"]][r["model"]]:
        cm[r["cell"]][r["model"]] = r["rate"]

pairs = collections.defaultdict(list)   # (mA,mB) -> [(cell, rA, rB)]
for c, d in cm.items():
    for a, b in itertools.combinations(sorted(d), 2):
        pairs[(a, b)].append((c, d[a], d[b]))
shared = {k: v for k, v in pairs.items() if len(v) >= 2}
print(f"model pairs co-measured in >=2 scaffold cells: {len(shared)}")
flips = []
for (a, b), v in shared.items():
    signs = {np.sign(x - y) for _, x, y in v}
    if len({s for s in signs if s != 0}) > 1:
        flips.append((a, b, v))
print(f"pairs whose ORDER flips between cells: {len(flips)}  ({100*len(flips)/max(len(shared),1):.0f}%)")
for a, b, v in sorted(flips, key=lambda t: -max(abs(x-y) for _, x, y in t[2]))[:10]:
    print(f"\n  {a[:34]}   vs   {b[:34]}")
    for c, x, y in sorted(v, key=lambda t: -(t[1]-t[2])):
        w = a if x > y else b
        print(f"     {str(c)[:36]:36s} {x*100:5.1f}% vs {y*100:5.1f}%  -> {w[:28]} wins by {abs(x-y)*100:.1f}pp")

print("\n### how large are the flips?")
if flips:
    mag = []
    for a, b, v in flips:
        d = [x-y for _, x, y in v]
        mag.append((max(d)-min(d))*100)
    mag = np.array(mag)
    print(f"  swing between the two cells' margins: median={np.median(mag):.1f}pp  max={mag.max():.1f}pp")
allm = np.array([abs(x-y)*100 for v in shared.values() for _, x, y in v])
print(f"  for scale: median |margin| between two models inside one cell = {np.median(allm):.1f}pp")
