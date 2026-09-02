"""M66 independent re-derivation of the census.

Deliberately a different code path from m66_load.py + m66_identifiability.py: the
inclusion test is written from scratch, and the pair counts come from a bipartite
incidence matrix rather than from set intersections over itertools.combinations.
A disagreement anywhere means one of the two is wrong.  Offline, CPU, $0.
"""
import os, json, glob, sys
import numpy as np
import yaml

BASE = "/mnt/beegfs/scratch/wenhui_swebench/experiments/evaluation"
PLACEHOLDER = {"Undisclosed", "NA"}

DROPPED = {}

def entries(split):
    """inclusion, rewritten: a directory counts if it names a model, a scaffold, and at
    least one resolved instance in EITHER of the two formats."""
    out = []
    for md in sorted(glob.glob(os.path.join(BASE, split, "*", "metadata.yaml"))):
        d = os.path.dirname(md)
        try:
            m = yaml.safe_load(open(md)) or {}
        except Exception:
            continue
        t = (m.get("tags") or {})
        mods = t.get("model")
        name = (mods[0] if isinstance(mods, list) and mods else t.get("model_display"))
        agent = t.get("agent")
        if not name or not agent:
            continue
        name, agent = str(name), str(agent)
        if name in PLACEHOLDER or name.startswith("http"):
            continue
        n_res, seen = 0, False
        pid = os.path.join(d, "per_instance_details.json")
        rj = os.path.join(d, "results", "results.json")
        if os.path.isfile(pid):
            try:
                n_res = sum(1 for v in json.load(open(pid)).values() if v.get("resolved")); seen = True
            except Exception:
                pass
        if not seen and os.path.isfile(rj):
            try:
                n_res = len(json.load(open(rj)).get("resolved") or []); seen = True
            except Exception:
                pass
        if not seen:
            continue
        # the retained-set rule, written out rather than imported: an entry whose per-instance
        # record contradicts its own stated leaderboard number is not a measurement.
        stated = (m.get("info") or {}).get("resolved")
        if isinstance(stated, (int, float)) and abs(stated - 100.0 * n_res / 500.0) > 0.25:
            DROPPED[split] = DROPPED.get(split, 0) + 1
            continue
        out.append((name, agent))
    return out

def census(rows, split=None):
    models = sorted({m for m, _ in rows})
    agents = sorted({a for _, a in rows})
    mi = {m: i for i, m in enumerate(models)}
    ai = {a: i for i, a in enumerate(agents)}
    A = np.zeros((len(agents), len(models)), dtype=np.int64)   # incidence
    for m, a in rows:
        A[ai[a], mi[m]] = 1                                    # presence, not multiplicity
    SS = A @ A.T                                               # scaffold x scaffold shared models
    MM = A.T @ A                                               # model x model shared scaffolds
    iu_s = np.triu_indices(len(agents), 1)
    iu_m = np.triu_indices(len(models), 1)
    sh = SS[iu_s]
    co = MM[iu_m]
    per_model_scaffolds = A.sum(axis=0)                        # scaffolds each model appears under
    return dict(entries=len(rows), dropped_self_inconsistent=DROPPED.get(split, 0),
                models=len(models), agents=len(agents),
                agent_pairs=int(len(sh)),
                agent_pairs_sharing_ge1=int((sh >= 1).sum()),
                agent_pairs_sharing_ge2=int((sh >= 2).sum()),
                agent_pairs_sharing_ge3=int((sh >= 3).sum()),
                models_under_1_agent=int((per_model_scaffolds == 1).sum()),
                models_under_ge2_agents=int((per_model_scaffolds >= 2).sum()),
                model_pairs_possible=int(len(co)),
                model_pairs_ever_co_measured=int((co >= 1).sum()),
                model_pairs_co_measured_ge2_agents=int((co >= 2).sum()))

ref = json.load(open("/data/wenhui/openclaude_rna/rna_harness/runs/ceiling/m66_identifiability.json"))
bad = 0
for split in ("verified", "lite"):
    got = census(entries(split), split)
    want = ref[split]
    print(f"### {split}")
    for k in sorted(want):
        if k.startswith("_"): continue
        g, w = got.get(k), want[k]
        ok = (g == w)
        bad += (not ok)
        print(f"  {k:38s} recheck={str(g):>6s}  published={str(w):>6s}  {'ok' if ok else '<<< MISMATCH'}")
    print()
rows = entries("verified") + entries("lite")
comb = census(rows)
w = ref["combined"]
print("### combined")
for k, wk in (("entries","entries"), ("models","models"), ("agents","agents"),
              ("agent_pairs","agent_pairs"), ("agent_pairs_sharing_ge3","sharing_ge3"),
              ("models_under_1_agent","models_under_1_agent")):
    g, v = comb[k], w[wk]
    ok = (g == v); bad += (not ok)
    print(f"  {wk:38s} recheck={str(g):>6s}  published={str(v):>6s}  {'ok' if ok else '<<< MISMATCH'}")
print(f"\n{'ALL CENSUS NUMBERS REPRODUCE' if bad==0 else str(bad)+' MISMATCH(ES)'}")
sys.exit(1 if bad else 0)
