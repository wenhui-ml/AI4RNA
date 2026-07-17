"""F2 -- cross-domain cost-compounding on Eterna100 (nested MFE domain).

Tests whether the RSSD cost result ("cheap tool first + learned abstention from the expensive SOTA beats
always-SOTA on cost at EQUAL quality, held-out") is DOMAIN-GENERAL, not a pseudoknot artifact. Eterna differs
structurally: here the cheap REMC tools SOLVE many puzzles (unlike pseudoknots where REMC is doomed), so the
cascade itself saves a lot and the abstention lever targets the hard tail DesiRNA can't crack either.

Reuses the learning_vs_cold outcome table (each tool run once per puzzle: REMC@{5,30,120}, DesiRNA@{60,180},
with REAL measured solved+cost+wall). --build extends it to more puzzles (overnight, CPU+DesiRNA). --sim does
always_sota vs cheap-cascade vs learned-abstention on a TRAIN/HELD-OUT split, with the perfect-foresight
oracle upper bound. Mirrors rssd_compounding so the two domains are directly comparable."""
from __future__ import annotations

import argparse
import json
import os

from ..benchmarks.eterna100 import load_eterna100
from ..cognitive.features import FEATURE_NAMES, features as extract_features
from ..cognitive.router import Router
from ..cognitive.tools import default_toolbox
from .learning_vs_cold import build_table, _table_path

OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "rssd_eterna_cost")
SOTA = "desirna@180s"        # the expensive tool the abstention policy gates
CHEAP_FIRST = ["remc@5s", "remc@30s", "remc@120s", "desirna@60s"]   # always tried, cheapest-first


def _split(pids):
    return [p for p in pids if p % 2 == 0], [p for p in pids if p % 2 == 1]


def simulate(table_dir, outdir):
    table = json.load(open(_table_path(table_dir)))
    puz = {p["id"]: p for p in load_eterna100(version="V2")}
    pids = sorted({int(k.split(":")[0]) for k in table
                   if all(f"{int(k.split(':')[0])}:{t}" in table for t in ([SOTA] + CHEAP_FIRST))})
    feats = {p: extract_features(puz[p]["target"]) for p in pids}

    def solved(pid, tool): return bool(table[f"{pid}:{tool}"]["solved"])
    def cost(pid, tool):   return float(table[f"{pid}:{tool}"]["cost"])
    sota_cost_of = {p: cost(p, SOTA) for p in pids}

    train, held = _split(pids)
    # learn P(SOTA solves | features) on TRAIN
    router = Router(default_toolbox())
    for p in train:
        router.record(feats[p], SOTA, solved(p, SOTA), cost(p, SOTA))

    def cheap_cascade(pid):
        c, ok = 0.0, False
        for t in CHEAP_FIRST:
            c += cost(pid, t)
            if solved(pid, t):
                ok = True; break
        return c, ok

    def arm_always_sota(ids):
        return sum(cost(p, SOTA) for p in ids), sum(int(solved(p, SOTA)) for p in ids)

    def arm_cascade(ids):                       # cheap-first, fall to SOTA last (no abstention)
        C = S = 0
        for p in ids:
            c, ok = cheap_cascade(p)
            if not ok:
                c += cost(p, SOTA); ok = solved(p, SOTA)
            C += c; S += int(ok)
        return C, S

    def arm_abstain(ids, tau):                  # cheap-first, then fire SOTA only if predicted to solve
        C = S = lost = fired = 0
        for p in ids:
            c, ok = cheap_cascade(p)
            if not ok:
                x = [feats[p][k] for k in FEATURE_NAMES]
                est = router._knn(x, SOTA)
                pr = est[0] if est else 1.0
                if pr >= tau:
                    c += cost(p, SOTA); ok = solved(p, SOTA); fired += 1
                elif solved(p, SOTA):
                    lost += 1
            C += c; S += int(ok)
        return {"cost": C, "solved": S, "solves_lost": lost, "sota_fired": fired}

    def arm_oracle(ids):                        # perfect foresight: fire SOTA iff it will solve
        C = S = 0
        for p in ids:
            c, ok = cheap_cascade(p)
            if not ok and solved(p, SOTA):
                c += cost(p, SOTA); ok = True
            C += c; S += int(ok)
        return C, S

    TAUS = (0.15, 0.30, 0.50, 0.75)
    out = {"n_puzzles": len(pids), "train": train, "held": held, "held_result": {}}
    print(f"\n=== F2 Eterna cross-domain cost-compounding ({len(pids)} puzzles; "
          f"train={len(train)} held={len(held)}) ===", flush=True)
    for name, ids in (("HELD-OUT", held), ("ALL", pids)):
        cs, ss = arm_always_sota(ids)
        cc, sc = arm_cascade(ids)
        oc, so = arm_oracle(ids)
        abst = {str(t): arm_abstain(ids, t) for t in TAUS}
        best0 = min((a for a in abst.values() if a["solves_lost"] == 0), key=lambda a: a["cost"], default=None)
        print(f"\n  [{name}] ({len(ids)} puzzles):", flush=True)
        print(f"    always-SOTA         solved={ss}/{len(ids)}  cost=${cs:.4f}  (1.0x)", flush=True)
        print(f"    cheap cascade       solved={sc}/{len(ids)}  cost=${cc:.4f}  ({cs/max(cc,1e-12):.1f}x)", flush=True)
        for t in TAUS:
            a = abst[str(t)]
            print(f"    abstain@tau={t:<4}    solved={a['solved']}/{len(ids)}  cost=${a['cost']:.4f}  "
                  f"({cs/max(a['cost'],1e-12):.1f}x)  lost={a['solves_lost']}", flush=True)
        print(f"    ORACLE (upper bnd)  solved={so}/{len(ids)}  cost=${oc:.4f}  ({cs/max(oc,1e-12):.1f}x)  lost=0", flush=True)
        if best0:
            print(f"    -> best 0-loss abstention: ${best0['cost']:.4f} ({cs/max(best0['cost'],1e-12):.1f}x cheaper "
                  f"at same {best0['solved']} solves)", flush=True)
        if name == "HELD-OUT":
            out["held_result"] = {"always_sota": {"cost": cs, "solved": ss},
                                  "cascade": {"cost": cc, "solved": sc},
                                  "oracle": {"cost": oc, "solved": so}, "abstain": abst}
    os.makedirs(outdir, exist_ok=True)
    json.dump(out, open(os.path.join(outdir, "eterna_cost.json"), "w"), indent=1)
    print(f"\n  wrote {os.path.join(outdir, 'eterna_cost.json')}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=int, metavar="N", help="build/extend the Eterna table over the first N V2 puzzles")
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--table-dir", default=os.path.join(os.path.dirname(__file__), "..", "runs", "learning_vs_cold"))
    ap.add_argument("--outdir", default=OUT)
    a = ap.parse_args()
    if a.build:
        ids = [p["id"] for p in load_eterna100(version="V2")][:a.build]
        build_table(ids, a.table_dir)
    if a.sim:
        simulate(a.table_dir, a.outdir)
    if not (a.build or a.sim):
        ap.error("pass --build N and/or --sim")
