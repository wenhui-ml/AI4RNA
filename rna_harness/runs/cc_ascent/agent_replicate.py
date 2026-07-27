#!/usr/bin/env python3
"""R7: replicate the frozen agent-written operators across seeds and executions.

carryover.py surfaced a result the paper currently buries in a supplement: on carry-over to ShapeKnots --
the one predictor in no arm's loop -- the LLM-invented operators beat every hand-built arm, difficulty-matched
and paired, with the loser winning 0 or 1 discordant targets:

    gemini-3.1-pro  vs DesiRNA  +0.42 (p=3.1e-5) | vs gamer +0.37 (p=1.2e-4) | vs panel operator +0.38 (p=2.4e-4)
    kimi-k3         vs DesiRNA  +0.21 (p=0.0039) | vs gamer +0.17 (p=0.039)  | vs panel operator +0.19 (p=0.016)

Those are SINGLE RUNS. This study has already shown once (Table 9) that a single run of a wall-clock-bounded
operator is not a reliable estimate -- execution spread at a fixed nominal seed was the largest source of
variation we measured. So the finding is not reportable until it is replicated at the same depth as the
headline comparison, which is what this script does: the two zero-error operators, frozen code re-executed
at 5 nominal seeds x 3 independent executions, on both pools.

The other four operators are excluded on purpose: they time out on 19-32 of the 60 targets, so their
denominators are survivorship-filtered towards easy targets and a carry-over rate computed on them is not
comparable. That exclusion is itself reported.

No LLM calls -- the operator code is frozen in result.json and only re-executed. Cost is CPU plus oracle time.
"""
import json, os, sys, time
import multiprocessing as mp
import multiprocessing.pool as mpp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

from rna_harness.benchmarks.rnainvbench import load_pseudobase          # noqa: E402
from rna_harness.oracles_pk import (pkiss_pairs, probknot_pairs,        # noqa: E402
                                    shapeknots_pairs, rnastructure_available)

sys.path.insert(0, HERE)
from invent_operator import make_design, forms_target                   # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
EXECS = [0, 1, 2]
OPERATORS = {"gemini31pro": "invent_operator/google_gemini31propreview/result.json",
             "kimik3": "invent_operator/moonshotai_kimik3/result.json"}
NWORKERS = int(os.environ.get("NWORKERS", "30"))


class _NDProcess(mp.get_context("fork").Process):
    """Non-daemonic worker. make_design() enforces its 20 s per-call cap by forking a grandchild and
    SIGKILLing its process group on timeout, and a daemonic pool worker is not allowed to have children.
    We keep make_design's mechanism rather than substituting a signal-based timeout, because the whole point
    of this run is to reproduce the frozen single-run evaluation faithfully -- a different timeout mechanism
    is a different experiment. Nesting is bounded at NWORKERS x 1 grandchild."""

    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class _NDContext(type(mp.get_context("fork"))):
    Process = _NDProcess


class NDPool(mpp.Pool):
    def __init__(self, *a, **kw):
        kw["context"] = _NDContext()
        super().__init__(*a, **kw)


def _one(args):
    """Adjudicate one (operator, target) under the full panel. Runs in a pool worker."""
    code, target, seed, tid = args
    try:
        design = make_design(code, per_call_s=20)
        s = design(target, seed)
    except Exception as e:                                   # noqa: BLE001
        return {"id": tid, "error": repr(e)[:120]}
    if not s:
        return {"id": tid, "pkiss": False, "probknot": False, "shapeknots": False, "unanimous": False,
                "seq": None}
    try:
        pk = bool(forms_target(pkiss_pairs(s), target))
        pb = bool(forms_target(probknot_pairs(s), target))
        sk = bool(forms_target(shapeknots_pairs(s), target))
    except Exception as e:                                   # noqa: BLE001
        return {"id": tid, "error": "adjudication: " + repr(e)[:100]}
    return {"id": tid, "pkiss": pk, "probknot": pb, "shapeknots": sk,
            "unanimous": pk and pb and sk, "seq": s}


def run_pool(name, code, targets, tag):
    out = os.path.join(HERE, f"arep_{name}_{tag}")
    os.makedirs(out, exist_ok=True)
    dest = os.path.join(out, "panel.json")
    if os.path.exists(dest):
        print(f"  [skip] {dest} exists", flush=True)
        return
    for seed in SEEDS:
        for e in EXECS:
            key = f"s{seed}_e{e}"
            f = os.path.join(out, f"{key}.json")
            if os.path.exists(f):
                continue
            t0 = time.time()
            # seed the operator with the nominal seed; execution index varies only machine state,
            # exactly as in the gamer-vs-panel replication.
            args = [(code, t["target"], seed, t["id"]) for t in targets]
            with NDPool(NWORKERS) as pool:
                rows = pool.map(_one, args)
            npk = sum(bool(r.get("pkiss")) for r in rows)
            nin = sum(bool(r.get("pkiss")) and bool(r.get("probknot")) for r in rows)
            nsk = sum(bool(r.get("shapeknots")) for r in rows)
            nun = sum(bool(r.get("unanimous")) for r in rows)
            nerr = sum(1 for r in rows if r.get("error"))
            json.dump({"operator": name, "pool": tag, "seed": seed, "exec": e, "rows": rows,
                       "n_pkiss": npk, "n_in": nin, "n_shapeknots": nsk, "n_unanimous": nun,
                       "n_error": nerr}, open(f, "w"), indent=1)
            print(f"  {name} {tag} {key}: pKiss={npk} pk&pb={nin} SK={nsk} unan={nun} err={nerr} "
                  f"({time.time()-t0:.0f}s)", flush=True)


def main():
    if not rnastructure_available():
        raise SystemExit("RNAstructure not available; ShapeKnots is required for this analysis")
    dev = [{"id": f"pseudobase:{k}", "target": t.target} for k, t in enumerate(load_pseudobase(limit=60))]
    sd = json.load(open(os.path.join(HERE, "struct_split.json")))
    sdis = [{"id": r["id"], "target": r["target"]} for r in sd["rows"]]
    print(f"pools: development n={len(dev)}, structurally disjoint n={len(sdis)}; "
          f"{len(SEEDS)}x{len(EXECS)} cells per operator per pool; workers={NWORKERS}", flush=True)
    for name, path in OPERATORS.items():
        p = os.path.join(HERE, path)
        if not os.path.exists(p):
            print(f"MISSING {p}", flush=True)
            continue
        code = json.load(open(p))["best_code"]
        print(f"\n=== {name} ===", flush=True)
        run_pool(name, code, dev, "dev")
        run_pool(name, code, sdis, "sdis")


if __name__ == "__main__":
    main()
