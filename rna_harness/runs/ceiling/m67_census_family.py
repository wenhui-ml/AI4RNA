"""M67 arm B: the census over every SWE-bench split in the clone, not just two."""
import os, sys, json, itertools, collections
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from m66_load import load, UNIVERSE
BASE = "/mnt/beegfs/scratch/wenhui_swebench/experiments/evaluation"
SPLITS = ["verified", "lite", "multilingual", "multimodal", "test"]
BAD = {"Undisclosed", "NA"}

DENOM = {"verified": UNIVERSE}   # 500 is SWE-bench Verified's size and nothing else's

def keep(r, split):
    """The retained-set rule needs a denominator. It is known for Verified, and for any entry
    whose per-instance file names the set it was run on. Where neither holds the entry is kept
    and the check is recorded as not applicable, rather than applying Verified's 500 to a split
    of a different size, which silently deletes entries."""
    v = r.get("info_resolved")
    if not isinstance(v, (int, float)):
        return True, "no stated score"
    n = DENOM.get(split) or (len(r["attempted"]) if r.get("attempted") else None)
    if not n:
        return True, "denominator unknown"
    return abs(v - 100.0 * len(r["resolved"]) / n) <= 0.25, "checked"

def cen(rows):
    am, ma = collections.defaultdict(set), collections.defaultdict(set)
    for r in rows: am[r["agent"]].add(r["model"]); ma[r["model"]].add(r["agent"])
    ap = list(itertools.combinations(sorted(am), 2))
    sh = [len(am[a] & am[b]) for a, b in ap]
    solo = sum(1 for m, a in ma.items() if len(a) == 1)
    mp = collections.Counter()
    for a, ms in am.items():
        for x, y in itertools.combinations(sorted(ms), 2): mp[(x, y)] += 1
    npairs = len(list(itertools.combinations(sorted(ma), 2)))
    return dict(entries=len(rows), models=len(ma), agents=len(am),
                agent_pairs=len(ap), ge1=sum(1 for s in sh if s >= 1),
                ge2=sum(1 for s in sh if s >= 2), ge3=sum(1 for s in sh if s >= 3),
                solo=solo, solo_frac=round(solo / max(len(ma), 1), 4),
                model_pairs=npairs, co_measured=len(mp),
                co_ge2=sum(1 for v in mp.values() if v >= 2))

OUT, allrows = {"splits": {}}, []
print(f"{'split':14s} {'entr':>5s} {'mods':>5s} {'scaf':>5s} {'pairs':>6s} {'>=1':>5s} {'>=2':>4s} "
      f"{'>=3':>4s} {'solo':>6s} {'solo%':>6s}")
for sp in SPLITS:
    d = os.path.join(BASE, sp)
    if not os.path.isdir(d): print(f"{sp}: absent"); continue
    R0 = [r for r in load(d) if r["model"] not in BAD and not r["model"].startswith("http")]
    kept, why = [], collections.Counter()
    for r in R0:
        ok, tag = keep(r, sp)
        why[tag if ok else "dropped:" + tag] += 1
        if ok: kept.append(r)
    R = kept
    OUT.setdefault("retention", {})[sp] = dict(why)
    if not R: print(f"{sp}: no usable entries"); continue
    c = cen(R); c["retention"] = dict(why); OUT["splits"][sp] = c; allrows += R
    print(f"{sp:14s} {c['entries']:>5d} {c['models']:>5d} {c['agents']:>5d} {c['agent_pairs']:>6d} "
          f"{c['ge1']:>5d} {c['ge2']:>4d} {c['ge3']:>4d} {c['solo']:>6d} {100*c['solo_frac']:>5.0f}%")
p = cen(allrows); OUT["pooled"] = p
print(f"\n{'POOLED':14s} {p['entries']:>5d} {p['models']:>5d} {p['agents']:>5d} {p['agent_pairs']:>6d} "
      f"{p['ge1']:>5d} {p['ge2']:>4d} {p['ge3']:>4d} {p['solo']:>6d} {100*p['solo_frac']:>5.0f}%")
print(f"\nS6 no split adds a scaffold pair sharing three models: "
      f"{all(c['ge3'] == 0 for c in OUT['splits'].values())}")
print(f"S7 pooled single-scaffold fraction above 0.75: {p['solo_frac'] > 0.75}  ({p['solo_frac']:.3f})")
print(f"S8 pooled scaffold pairs sharing three models at most 5: {p['ge3'] <= 5}  ({p['ge3']})")
OUT["S6"] = all(c["ge3"] == 0 for c in OUT["splits"].values())
OUT["S7"] = bool(p["solo_frac"] > 0.75); OUT["S8"] = bool(p["ge3"] <= 5)
json.dump(OUT, open(os.path.join(HERE, "m67_census_family.json"), "w"), indent=1)
print("\nwrote m67_census_family.json")
