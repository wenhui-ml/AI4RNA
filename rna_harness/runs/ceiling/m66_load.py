"""M66 loader -- the corpus stores per-instance outcomes in TWO formats and the
controlled sweep is entirely in the second one, so both are read here."""
import json, os, collections, yaml

ROOT = "/mnt/beegfs/scratch/wenhui_swebench/experiments/evaluation/verified"
UNIVERSE = 500  # SWE-bench Verified

def load(root=ROOT):
    out = []
    for e in sorted(os.listdir(root)):
        d = os.path.join(root, e)
        md = os.path.join(d, "metadata.yaml")
        if not os.path.exists(md):
            continue
        try:
            m = yaml.safe_load(open(md)) or {}
        except Exception:
            continue
        t = m.get("tags") or {}
        i = m.get("info") or {}
        mods = t.get("model") or []
        mod = mods[0] if isinstance(mods, list) and mods else (t.get("model_display") or "NA")

        resolved, nogen, fmt, attempted = None, None, None, None
        pid = os.path.join(d, "per_instance_details.json")
        rj = os.path.join(d, "results", "results.json")
        if os.path.exists(pid):
            try:
                p = json.load(open(pid))
                resolved = {k for k, v in p.items() if v.get("resolved")}
                attempted = set(p)          # this format names exactly what was run
                nogen = set()
                fmt = "per_instance"
            except Exception:
                pass
        if resolved is None and os.path.exists(rj):
            try:
                r = json.load(open(rj))
                resolved = set(r.get("resolved") or [])
                nogen = set(r.get("no_generation") or [])
                attempted = None            # not recorded; the leaderboard denominator is the universe
                fmt = "results_json"
            except Exception:
                pass
        if resolved is None:
            continue
        out.append(dict(entry=e, model=str(mod), agent=str(t.get("agent") or "NA"),
                        model_org=str(t.get("model_org") or "NA"),
                        agent_org=str(t.get("agent_org") or "NA"),
                        os_model=t.get("os_model"), os_system=t.get("os_system"),
                        date=e.split("_")[0], fmt=fmt,
                        resolved=resolved, no_generation=nogen, attempted=attempted,
                        info_resolved=i.get("resolved"), calls=i.get("instance_calls")))
    return out

if __name__ == "__main__":
    R = load()
    print(f"entries with a per-instance outcome: {len(R)}")
    print("  by format:", dict(collections.Counter(r['fmt'] for r in R)))

    print("\n### is the leaderboard denominator the 500-instance universe?")
    ck = [(r["entry"], r["info_resolved"], 100.0 * len(r["resolved"]) / UNIVERSE)
          for r in R if isinstance(r["info_resolved"], (int, float))]
    bad = [(e, a, b) for e, a, b in ck if abs(a - b) > 0.25]
    print(f"  entries checkable: {len(ck)};  |info.resolved - len(resolved)/500| > 0.25pp: {len(bad)}")
    for e, a, b in bad[:5]:
        print(f"    {e[:52]:52s} stated={a} recomputed={b:.1f}")

    print("\n### A1 (redone) -- the largest scaffold arms")
    by = collections.defaultdict(list)
    for r in R: by[r["agent"]].append(r)
    for ag, rs in sorted(by.items(), key=lambda x: -len(x[1]))[:6]:
        dt = collections.Counter(x["date"] for x in rs)
        print(f"  {ag[:26]:26s} entries={len(rs):3d} models={len(set(x['model'] for x in rs)):3d} "
              f"fmt={dict(collections.Counter(x['fmt'] for x in rs))} distinct_dates={len(dt)} "
              f"top_date={dt.most_common(1)[0]}")
