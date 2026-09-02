"""M66 arm D, corrected.  The first design compared a model's board number with
itself whenever its board entry WAS the cell entry, which is an identity and not
a test.  The identified comparison is between two cells that share models."""
import os, re, sys, collections, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m66_load import load, UNIVERSE

SEED = 20260902
R = [r for r in load() if not (isinstance(r["info_resolved"], (int, float))
     and abs(r["info_resolved"] - 100.0*len(r["resolved"])/UNIVERSE) > 0.25)]
def version(e):
    m = re.search(r'[-_]v?([0-9]+\.[0-9]+\.[0-9]+)', e)
    return m.group(1) if m else "NA"
for r in R:
    r["cell"] = (r["agent"], version(r["entry"])); r["rate"] = len(r["resolved"])/UNIVERSE

cellmod = collections.defaultdict(dict)
for r in R:
    c = r["cell"]
    if r["model"] not in cellmod[c] or r["rate"] > cellmod[c][r["model"]]:
        cellmod[c][r["model"]] = r["rate"]

def spearman(a, b):
    ra = np.argsort(np.argsort(-np.asarray(a, float))).astype(float)
    rb = np.argsort(np.argsort(-np.asarray(b, float))).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra*ra).sum()*(rb*rb).sum())
    return float((ra*rb).sum()/d) if d > 0 else float("nan")

print("### cell pairs sharing at least 3 models -- the identified arm D")
pairs = []
cells = list(cellmod)
for c1, c2 in itertools.combinations(cells, 2):
    sh = sorted(set(cellmod[c1]) & set(cellmod[c2]))
    if len(sh) >= 3:
        a = [cellmod[c1][m] for m in sh]; b = [cellmod[c2][m] for m in sh]
        ra = np.argsort(np.argsort(-np.asarray(a))); rb = np.argsort(np.argsort(-np.asarray(b)))
        pairs.append((len(sh), spearman(a, b), int(np.abs(ra-rb).max()), c1, c2, sh, a, b))
pairs.sort(key=lambda x: (-x[0], x[1]))
print(f"  identified cell pairs: {len(pairs)}")
for n, rho, mx, c1, c2, sh, a, b in pairs:
    print(f"  n={n}  spearman={rho:+.3f}  max_rank_shift={mx}")
    print(f"     {str(c1)[:38]:38s} vs {str(c2)[:38]}")
    for m, x, y in zip(sh, a, b):
        print(f"        {m[:36]:36s} {x*100:5.1f}%  vs {y*100:5.1f}%   delta={100*(x-y):+5.1f}pp")

if pairs:
    ns = np.array([p[0] for p in pairs]); rs = np.array([p[1] for p in pairs])
    w = ns / ns.sum()
    print(f"\n  n-weighted mean Spearman across identified pairs: {float((w*rs).sum()):+.3f}")
    print(f"  pairs with Spearman < 0.9 (S3's bar): {int((rs < 0.9).sum())} / {len(rs)}")
    print(f"  pairs with a rank shift >= 3 (S4's bar): {sum(1 for p in pairs if p[2] >= 3)} / {len(pairs)}")

print("\n### the same question without ranks: how much does a model's number move between cells?")
m2 = collections.defaultdict(dict)
for c, d in cellmod.items():
    for m, v in d.items(): m2[m][c] = v
sp = [(m, max(d.values())-min(d.values()), len(d)) for m, d in m2.items() if len(d) >= 2]
sp.sort(key=lambda x: -x[1])
print(f"  models measured in >=2 cells: {len(sp)}")
for m, s, k in sp[:12]:
    print(f"    {m[:38]:38s} cells={k}  spread={s*100:5.1f}pp")
if sp:
    v = np.array([s for _, s, _ in sp])
    print(f"  median within-model spread across scaffolds: {np.median(v)*100:.1f}pp")
allr = np.array([r["rate"] for r in R])
print(f"  for scale: full-board spread across ALL entries = {(allr.max()-allr.min())*100:.1f}pp, "
      f"IQR = {(np.percentile(allr,75)-np.percentile(allr,25))*100:.1f}pp")

print("\n### the two alternative explanations, tested")
BAD = {"Undisclosed", "NA"}
rows = collections.defaultdict(list)
for r in R:
    if r["model"] in BAD or r["model"].startswith("http"): continue
    rows[r["model"]].append(r)
keep = {m: v for m, v in rows.items() if len({x["cell"] for x in v}) >= 2}
print(f"  models kept after dropping placeholder labels: {len(keep)}")
sp = sorted(((max(x['rate'] for x in v)-min(x['rate'] for x in v), m, v) for m, v in keep.items()), reverse=True)
print(f"  median within-model spread (cleaned): {np.median([s for s,_,_ in sp])*100:.1f}pp")

print("\n  [1] is the spread just time?  span of submission dates vs spread")
xs, ys = [], []
for s, m, v in sp:
    ds = sorted(int(x["date"][:4])*12+int(x["date"][4:6]) for x in v)
    span = ds[-1]-ds[0]
    xs.append(span); ys.append(s*100)
    print(f"    {m[:34]:34s} cells={len({x['cell'] for x in v}):2d} spread={s*100:5.1f}pp  date_span={span:3d}mo")
xs, ys = np.array(xs, float), np.array(ys, float)
if len(xs) > 3 and xs.std() > 0:
    r_ = float(np.corrcoef(xs, ys)[0,1]); print(f"    corr(date_span, spread) = {r_:+.3f}  (n={len(xs)})")

print("\n  [2] same model, same YEAR, different scaffold -- time held fixed")
n_ok = 0
for s, m, v in sp:
    byyr = collections.defaultdict(list)
    for x in v: byyr[x["date"][:4]].append(x)
    for y, xs_ in sorted(byyr.items()):
        cs = {x["cell"] for x in xs_}
        if len(cs) >= 2:
            rr = [x["rate"] for x in xs_]
            n_ok += 1
            print(f"    {m[:30]:30s} {y}  cells={len(cs):2d}  spread={100*(max(rr)-min(rr)):5.1f}pp  "
                  f"[{', '.join(sorted(x['agent'][:14] for x in xs_))}]"[:150])
print(f"    model-years with >=2 scaffolds: {n_ok}")

print("\n### the deepest single case: one model across every scaffold that ran it")
tgt = "claude-3-5-sonnet-20241022"
v = sorted(rows.get(tgt, []), key=lambda x: -x["rate"])
print(f"  {tgt}: {len(v)} entries")
for x in v:
    print(f"    {x['rate']*100:5.1f}%  {x['agent'][:30]:30s} {x['date']}  no_gen={len(x['no_generation'])}")
