"""M66: the structural finding.  Not 'we could not decompose the leaderboard' but
'the leaderboard was never crossed, so nobody can'.  Needs no control: it is a census."""
import os, sys, json, collections, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m66_load import load
BASE = "/mnt/beegfs/scratch/wenhui_swebench/experiments/evaluation"
BAD = {"Undisclosed", "NA"}

out = {}
for split in ["verified", "lite"]:
    R = [r for r in load(os.path.join(BASE, split))
         if r["model"] not in BAD and not r["model"].startswith("http")]
    am = collections.defaultdict(set); ma = collections.defaultdict(set)
    for r in R: am[r["agent"]].add(r["model"]); ma[r["model"]].add(r["agent"])
    agents, models = sorted(am), sorted(ma)
    apairs = list(itertools.combinations(agents, 2))
    sh = [len(am[a] & am[b]) for a, b in apairs]
    mp = collections.Counter()
    for a, ms in am.items():
        for x, y in itertools.combinations(sorted(ms), 2): mp[(x, y)] += 1
    mpairs = list(itertools.combinations(models, 2))
    solo = [m for m, a in ma.items() if len(a) == 1]
    d = dict(entries=len(R), models=len(models), agents=len(agents),
             agent_pairs=len(apairs),
             agent_pairs_sharing_ge1=sum(1 for s in sh if s >= 1),
             agent_pairs_sharing_ge2=sum(1 for s in sh if s >= 2),
             agent_pairs_sharing_ge3=sum(1 for s in sh if s >= 3),
             models_under_1_agent=len(solo),
             models_under_ge2_agents=len(models) - len(solo),
             model_pairs_possible=len(mpairs),
             model_pairs_ever_co_measured=len(mp),
             model_pairs_co_measured_ge2_agents=sum(1 for v in mp.values() if v >= 2))
    out[split] = d
    print(f"### {split}")
    print(f"  entries {d['entries']},  models {d['models']},  scaffolds {d['agents']}")
    print(f"  a model's number comes from exactly ONE scaffold for "
          f"{d['models_under_1_agent']}/{d['models']} models "
          f"({100*d['models_under_1_agent']/d['models']:.0f}%)")
    print(f"  scaffold pairs sharing >=1 model: {d['agent_pairs_sharing_ge1']}/{d['agent_pairs']} "
          f"({100*d['agent_pairs_sharing_ge1']/d['agent_pairs']:.1f}%);  "
          f">=2: {d['agent_pairs_sharing_ge2']};  >=3: {d['agent_pairs_sharing_ge3']}")
    print(f"  model pairs ever compared inside one scaffold: "
          f"{d['model_pairs_ever_co_measured']}/{d['model_pairs_possible']} "
          f"({100*d['model_pairs_ever_co_measured']/d['model_pairs_possible']:.1f}%);  "
          f"compared inside >=2 scaffolds: {d['model_pairs_co_measured_ge2_agents']}")
    print()

# combined
R = []
for split in ["verified", "lite"]:
    R += [dict(r, split=split) for r in load(os.path.join(BASE, split))
          if r["model"] not in BAD and not r["model"].startswith("http")]
am = collections.defaultdict(set); ma = collections.defaultdict(set)
for r in R: am[r["agent"]].add(r["model"]); ma[r["model"]].add(r["agent"])
ap = list(itertools.combinations(sorted(am), 2))
sh = [len(am[a] & am[b]) for a, b in ap]
solo = sum(1 for m, a in ma.items() if len(a) == 1)
out["combined"] = dict(entries=len(R), models=len(ma), agents=len(am),
                       agent_pairs=len(ap), sharing_ge3=sum(1 for s in sh if s >= 3),
                       models_under_1_agent=solo)
print(f"### combined  entries={len(R)} models={len(ma)} scaffolds={len(am)}")
print(f"  scaffold pairs sharing >=3 models: {sum(1 for s in sh if s>=3)} / {len(ap)}")
print(f"  models measured under exactly one scaffold: {solo}/{len(ma)} ({100*solo/len(ma):.0f}%)")
json.dump(out, open("/data/wenhui/openclaude_rna/rna_harness/runs/ceiling/m66_identifiability.json","w"), indent=1)
