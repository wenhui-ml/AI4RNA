"""M66 arms C/D/E on the cleaned corpus.  Offline, CPU, no API."""
import json, os, re, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m66_load import load, UNIVERSE

SEED = 20260902
R = load()

# --- integrity gate declared in A2: drop entries whose per-instance file contradicts its own entry
drop = []
for r in R:
    if isinstance(r["info_resolved"], (int, float)):
        rec = 100.0 * len(r["resolved"]) / UNIVERSE
        if abs(r["info_resolved"] - rec) > 0.25:
            drop.append((r["entry"], r["info_resolved"], rec))
D = {e for e, _, _ in drop}
R = [r for r in R if r["entry"] not in D]
print(f"entries dropped for self-inconsistency: {len(drop)}")
for e, a, b in drop: print(f"   {e[:56]:56s} stated={a} recomputed={b:.1f}")
print(f"entries retained: {len(R)}")

def version(e):
    m = re.search(r'-v?([0-9]+\.[0-9]+\.[0-9]+)', e) or re.search(r'_v?([0-9]+\.[0-9]+\.[0-9]+)', e)
    return m.group(1) if m else "NA"
for r in R:
    r["cell"] = (r["agent"], version(r["entry"]))
    r["rate"] = len(r["resolved"]) / UNIVERSE

print("\n### the leaderboard as it is read: models ranked by their best entry")
best = {}
for r in R:
    if r["model"] not in best or r["rate"] > best[r["model"]]["rate"]:
        best[r["model"]] = r
print(f"  distinct models on the board: {len(best)}")

print("\n### arm D -- models that appear in more than one (scaffold,version) cell")
m2c = collections.defaultdict(set)
for r in R: m2c[r["model"]].add(r["cell"])
multi = {m for m, c in m2c.items() if len(c) >= 2}
print(f"  models in >=2 cells: {len(multi)}")

# the controlled cells: a (scaffold,version) cell holding >=4 distinct models
cellmods = collections.defaultdict(dict)
for r in R:
    c = r["cell"]
    if r["model"] not in cellmods[c] or r["rate"] > cellmods[c][r["model"]]:
        cellmods[c][r["model"]] = r["rate"]
ctrl = {c: d for c, d in cellmods.items() if len(d) >= 4}
print(f"  controlled cells (>=4 models each): {len(ctrl)}")
for c, d in sorted(ctrl.items(), key=lambda x: -len(x[1])):
    print(f"    {str(c)[:44]:44s} models={len(d)}")

def spearman(a, b):
    ra = np.argsort(np.argsort(-np.asarray(a, float)))
    rb = np.argsort(np.argsort(-np.asarray(b, float)))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum()))

print("\n### arm D headline -- inside each controlled cell, does the cell order match the board order?")
for c, d in sorted(ctrl.items(), key=lambda x: -len(x[1])):
    ms = sorted(d)
    cell_rate = [d[m] for m in ms]
    board_rate = [best[m]["rate"] for m in ms]
    same_entry = sum(1 for m in ms if best[m]["cell"] == c)
    rho = spearman(cell_rate, board_rate)
    ranks_c = np.argsort(np.argsort(-np.asarray(cell_rate)))
    ranks_b = np.argsort(np.argsort(-np.asarray(board_rate)))
    mx = int(np.abs(ranks_c - ranks_b).max())
    print(f"  {str(c)[:40]:40s} n={len(ms):2d}  board-entry-is-this-cell for {same_entry}/{len(ms)}  "
          f"spearman={rho:+.3f}  max_rank_shift={mx}")

print("\n### arm E -- the non-attempt, which the board scores as a failure")
ng = [(r["agent"], len(r["no_generation"]), r["rate"]) for r in R if r["fmt"] == "results_json"]
per = collections.defaultdict(list)
for a, n, _ in ng: per[a].append(n)
tot = sum(n for _, n, _ in ng); ent = len(ng)
print(f"  entries in results_json format: {ent};  total no_generation instances: {tot}")
big = [(a, np.mean(v), max(v), len(v)) for a, v in per.items() if len(v) >= 4]
for a, m, mx, n in sorted(big, key=lambda x: -x[1]):
    print(f"    {a[:30]:30s} n={n:2d} mean={m:6.2f}/500 ({m/5:.1f}pp) max={mx}")
print("\n  effect of scoring a non-attempt as a failure, per entry (pp of the 500 denominator):")
eff = sorted(((len(r['no_generation'])/5.0, r['entry'], r['agent']) for r in R if r['fmt']=='results_json'), reverse=True)
for v, e, a in eff[:8]:
    print(f"    {v:5.1f}pp  {a[:22]:22s} {e[:44]}")
