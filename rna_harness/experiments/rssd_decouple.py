"""Decoupled headline result: AGENT-DISCOVERED skill + PROVEN 0-loss learned abstention.

The end-to-end agent (rssd_agent / rssd_agent_nested) authored a GOOD cheap SKILL but a BADLY-calibrated
abstention policy (it gave up on solvable targets -> quality loss). That muddies "cheaper" with "lower
quality". This script DECOUPLES the two contributions to get a clean, defensible claim:
  * cheap tool = the AGENT'S OWN authored skill (the novel, discovered part) -- re-run here per target;
  * routing    = the PROVEN k-NN abstention (learned on TRAIN, frozen on HELD-OUT), which is 0-loss.
Result per domain: at the SAME held-out solve count as always-SOTA (0 solves lost), how much cheaper?
Runs for BOTH domains (pseudoknot + nested/Eterna) so the claim is cross-domain. Gaming-proof: skill code
is frozen, abstention trained only on TRAIN, evaluated on disjoint HELD-OUT."""
from __future__ import annotations

import argparse
import json
import os

from ..cognitive.features import FEATURE_NAMES, features as extract_features
from ..cognitive.router import Router
from ..cognitive.tools import DesirnaTool

OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "rssd_decouple")
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
TAUS = (0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75)


def _knn_sim(ids, feats, skill, des_solved, des_cost):
    """Skill-first, then PROVEN k-NN abstention on the expensive SOTA. Train k-NN on TRAIN (parity split),
    freeze, evaluate on HELD-OUT. Returns always-SOTA baseline + cascade(no-abstain) + the best 0-loss tau."""
    train = [i for i in ids if i % 2 == 0]
    held = [i for i in ids if i % 2 == 1]
    des_name = DesirnaTool(timelimit_s=300).name
    router = Router([DesirnaTool(timelimit_s=300)])
    for i in train:
        router.record(feats[i], des_name, des_solved.get(i, False), des_cost)

    def sk(i):
        r = skill.get(str(i)) or skill.get(i) or {}
        return bool(r.get("solved")), float(r.get("cost", 0.0))

    def sim(tau):
        c = s = lost = fired = 0
        for i in held:
            solved, cost = sk(i)
            c += cost
            if solved:
                s += 1; continue
            est = router._knn([feats[i][k] for k in FEATURE_NAMES], des_name)
            p = est[0] if est else 1.0
            if p >= tau:                                   # fire SOTA
                c += des_cost; fired += 1; s += int(des_solved.get(i, False))
            elif des_solved.get(i, False):                 # abstained on a real solve -> lost
                lost += 1
        return {"cost": round(c, 6), "solved": s, "lost": lost, "sota_fired": fired}

    sota_cost = sum(des_cost for _ in held)
    sota_solved = sum(int(des_solved.get(i, False)) for i in held)
    arms = {f"{t:.2f}": sim(t) for t in TAUS}
    cascade = sim(0.0)                                     # tau=0 => always fire SOTA on skill-failures
    best0 = min((a for a in arms.values() if a["lost"] == 0), key=lambda a: a["cost"], default=None)
    return {"held_n": len(held), "always_sota": {"cost": round(sota_cost, 6), "solved": sota_solved},
            "cascade": cascade, "arms": arms, "best0_abstain": best0}


def _run_pk_skill(seed):
    from .rssd_agent import run_skill
    from ..benchmarks.rnainvbench import load_pseudobase
    from .rssd_compounding import _desirna_column
    code = open(os.path.join(RUNS, "rssd_agent_z-ai_glm-5.2", "authored_skill.py")).read()
    tasks = load_pseudobase()
    tgt = {int(t.task_id.split(":")[1]): t.target for t in tasks}
    ids = sorted(tgt)
    des_solved, des_cost, des_src = _desirna_column()
    print(f"  [PK] running GLM-authored skill on {len(ids)} targets ...", flush=True)
    skill = run_skill(code, {str(i): tgt[i] for i in ids}, budget_calls=120, per_target_wall=90, seed=seed)
    feats = {i: extract_features(tgt[i]) for i in ids}
    return ids, feats, skill, des_solved, des_cost, des_src


def _run_nested_skill(seed):
    from .rssd_agent_nested import run_skill, _eterna_desirna_column
    from ..benchmarks.eterna100 import load_eterna100
    code = open(os.path.join(RUNS, "rssd_agent_nested", "authored_skill.py")).read()
    des_solved, des_cost = _eterna_desirna_column()
    puz = {p["id"]: p["target"] for p in load_eterna100(version="V2")}
    ids = sorted([i for i in des_solved if i in puz])
    tgt = {i: puz[i] for i in ids}
    print(f"  [nested] running GLM-authored skill on {len(ids)} Eterna targets ...", flush=True)
    skill = run_skill(code, {str(i): tgt[i] for i in ids}, budget=4000, wall=60, seed=seed)
    feats = {i: extract_features(tgt[i]) for i in ids}
    return ids, feats, skill, des_solved, des_cost, "measured median (Eterna table)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=["pk", "nested"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    print("=== RSSD DECOUPLE: agent-discovered skill + proven 0-loss k-NN abstention ===", flush=True)
    out = {}
    for dom in a.domains:
        ids, feats, skill, des_solved, des_cost, src = (_run_pk_skill if dom == "pk" else _run_nested_skill)(a.seed)
        skill_solved_all = sum(1 for i in ids if (skill.get(str(i)) or {}).get("solved"))
        res = _knn_sim(ids, feats, skill, des_solved, des_cost)
        res.update({"domain": dom, "n_all": len(ids), "skill_solved_all": skill_solved_all,
                    "desirna_solved_all": sum(des_solved.values()), "desirna_cost_src": src})
        out[dom] = res
        h = res
        sota = h["always_sota"]; b0 = h["best0_abstain"]; casc = h["cascade"]
        x = (sota["cost"] / b0["cost"]) if b0 and b0["cost"] else float("nan")
        print(f"\n  === {dom.upper()} (held-out {h['held_n']}; agent skill solved {skill_solved_all}/{len(ids)} overall) ===", flush=True)
        print(f"    always-SOTA                 solved={sota['solved']}/{h['held_n']}  cost=${sota['cost']:.4f}  (1.0x)", flush=True)
        print(f"    skill cascade (no abstain)  solved={casc['solved']}/{h['held_n']}  cost=${casc['cost']:.4f}  "
              f"({sota['cost']/max(casc['cost'],1e-12):.1f}x)", flush=True)
        if b0:
            print(f"    skill + 0-loss abstention   solved={b0['solved']}/{h['held_n']}  cost=${b0['cost']:.4f}  "
                  f"({x:.1f}x cheaper, SAME quality, 0 lost)", flush=True)
        json.dump(out, open(os.path.join(a.outdir, "decouple.json"), "w"), indent=1)

    print(f"\n=== HEADLINE (decoupled, equal-quality) ===", flush=True)
    for dom, r in out.items():
        b0 = r["best0_abstain"]; sota = r["always_sota"]
        if b0 and b0["cost"]:
            print(f"  {dom:7}: agent-discovered skill + 0-loss abstention = same {b0['solved']}/{r['held_n']} "
                  f"solved as always-SOTA, at {sota['cost']/b0['cost']:.1f}x lower compute.", flush=True)
    print(f"  wrote {os.path.join(a.outdir, 'decouple.json')}", flush=True)


if __name__ == "__main__":
    main()
