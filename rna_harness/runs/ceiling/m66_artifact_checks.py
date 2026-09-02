"""M66 step 0 -- the three construction checks the declaration requires before any arm is read."""
import json, os, sys, collections, yaml

ROOT = "/mnt/beegfs/scratch/wenhui_swebench/experiments/evaluation/verified"

def load():
    out = []
    for e in sorted(os.listdir(ROOT)):
        md = os.path.join(ROOT, e, "metadata.yaml")
        rj = os.path.join(ROOT, e, "results", "results.json")
        if not (os.path.exists(md) and os.path.exists(rj)):
            continue
        try:
            m = yaml.safe_load(open(md)) or {}
            r = json.load(open(rj))
        except Exception:
            continue
        t = m.get("tags") or {}
        i = m.get("info") or {}
        a = m.get("assets") or {}
        mods = t.get("model") or []
        mod = mods[0] if isinstance(mods, list) and mods else (t.get("model_display") or "NA")
        out.append(dict(entry=e, model=str(mod), agent=str(t.get("agent") or "NA"),
                        agent_org=str(t.get("agent_org") or "NA"),
                        repo=str((a.get("repo") or "")), date=e.split("_")[0],
                        results=r, info=i, tags=t))
    return out

R = load()
print(f"entries usable (metadata + results.json): {len(R)}")

print("\n### A1 -- is the largest scaffold arm one submitter's sweep?")
by = collections.defaultdict(list)
for r in R: by[r["agent"]].append(r)
for ag, rs in sorted(by.items(), key=lambda x: -len(x[1]))[:4]:
    owners = collections.Counter(x["repo"].split("/")[3] if x["repo"].count("/") >= 3 else "NONE" for x in rs)
    dates = collections.Counter(x["date"] for x in rs)
    print(f"  {ag[:30]:30s} entries={len(rs):3d} models={len(set(x['model'] for x in rs)):3d} "
          f"owners={dict(owners)} distinct_dates={len(dates)}")

print("\n### A2 -- do entries share an attempted set?")
keys = collections.Counter()
for r in R: keys.update(k for k, v in r["results"].items() if isinstance(v, list))
print("  list-valued keys in results.json:", dict(keys))
cov = {}
for r in R:
    ids = set()
    for k, v in r["results"].items():
        if isinstance(v, list): ids |= set(v)
    cov[r["entry"]] = ids
sz = sorted(len(v) for v in cov.values())
print(f"  ids named per entry: min={sz[0]} p25={sz[len(sz)//4]} median={sz[len(sz)//2]} max={sz[-1]}")
full = [e for e, v in cov.items() if len(v) >= 500]
print(f"  entries naming >=500 ids: {len(full)}")
allids = set()
for v in cov.values(): allids |= v
print(f"  union of all ids named anywhere: {len(allids)}")

print("\n### A3 -- no_generation: how often is a non-attempt scored as a failure?")
ng = [(r["entry"], len(r["results"].get("no_generation") or []),
       len(r["results"].get("no_logs") or []), len(r["results"].get("resolved") or []),
       r["agent"]) for r in R]
nz = [x for x in ng if x[1] > 0]
print(f"  entries with at least one no_generation: {len(nz)} / {len(R)}")
if nz:
    v = sorted(x[1] for x in nz)
    print(f"  no_generation count among those: min={v[0]} median={v[len(v)//2]} max={v[-1]}")
    per = collections.defaultdict(list)
    for e, n, nl, res, ag in ng: per[ag].append(n)
    print("  mean no_generation by scaffold (arms with >=4 entries):")
    for ag, xs in sorted(per.items(), key=lambda x: -sum(x[1]) / max(len(x[1]), 1)):
        if len(xs) >= 4:
            print(f"    {ag[:32]:32s} n={len(xs):3d} mean={sum(xs)/len(xs):7.2f} max={max(xs)}")
