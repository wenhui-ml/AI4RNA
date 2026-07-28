#!/usr/bin/env python3
"""Does structural retrieval move anything DesiRNA cannot reach on its own? The last lever with a mechanism.

WHY THIS ONE AND NOT ANOTHER. Today's tally: re-analysing signals the system already produces is 0/7; adding
an information source is 3/4 (the multi-seed probe failed, because re-running the same solver with a new seed
is a fresh SAMPLE, not fresh information about designability). Retrieval is different in exactly that respect:
the information injected is VERIFIED SUB-SEQUENCES FROM OTHER TARGETS, which no amount of re-running this
target can produce.

MECHANISM AND PLUMBING, both pre-existing, neither re-derived here:
  * experiments/retrieval.py builds an L2 unit library keyed by a leaf unit's dot-bracket shape, ingesting only
    ORACLE-VERIFIED solves, with a leakage guard. It is tested (tests/test_retrieval.py) and imported by four
    experiment scripts -- but by nothing under cognitive/, so it has never been in a solve path.
  * experiments/rssd_constrained.py::ConstrainedDesiRNA already threads an IUPAC restraint into DesiRNA via
    task.meta["seq_restr"], replacing _build_input's hardcoded 'N'*n. That is the injection point.
So the experiment is: retrieved units -> IUPAC restraint -> the existing constrained adapter.

LIBRARY IS BUILT FROM OUR OWN VERIFIED SOLVES ONLY -- the 84 sequences stored by sota_curve.py. No Eterna
sample solution is ever ingested, and the evaluation targets are passed to lib.forbid().

ARMS. `retrieval` pins ONLY the positions of units the library actually HIT. `prior` pins the SAME positions
with retrieval.py's canonical prior instead of retrieved sequence -- the control that separates "transferred
experience" from "any constraint helps", which is the control the last four nulls all needed.
`retrieval_clamp` additionally GC-clamps each transplanted unit's terminal pair (the module's own
compositional-fidelity move: adapt the template's boundary, do not transplant verbatim).

TWO QUESTIONS, separate results:
  A CAPABILITY  the 16 targets no tier solves, at t=180. Nothing to break here, so it is upside-only.
  B EFFICIENCY  the 12 targets that need escalation, at t=5 with their OWN solution forbidden. If retrieval
                makes them solve at the cheap tier, that is a real cost win even with the ceiling unmoved.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
OUT = os.path.join(HERE, "retrieval_arm")


def build_restraint(target, lib, mode):
    """(iupac, n_hit, n_units, n_pinned). `retrieval*` pins only library HITS; `prior` pins the same
    positions with the canonical prior; `none` pins nothing."""
    from rna_harness.experiments.retrieval import _canonical_unit_seq
    from rna_harness.mcp.motif_kb import decompose_structure
    from rna_harness.structure_utils import pair_table
    n = len(target)
    r = ["N"] * n
    dec = decompose_structure(target)
    pt = pair_table(target)
    units = dec.get("units", [])
    n_hit = 0
    for u in units:
        a, b = u["positions"]
        sub = u["sub_target"]
        got = lib._best_unit(sub)
        is_hit = got is not None and len(got) == (b - a + 1)
        if not is_hit:
            continue                                    # only HIT units are pinned, in every arm
        n_hit += 1
        fill = got if mode.startswith("retrieval") else _canonical_unit_seq(sub)
        if len(fill) != (b - a + 1):
            continue
        for k, ch in enumerate(fill):
            r[a + k] = ch
        if mode == "retrieval_clamp" and pt[a] == b:
            r[a], r[b] = "G", "C"
    return "".join(r), n_hit, len(units), sum(1 for c in r if c != "N")


def work(item):
    pid, target, mode, tl, forbid_self, libpath = item
    dest = os.path.join(OUT, f"p{pid}_{mode}_t{tl}.json")
    if os.path.exists(dest):
        return dest
    from rna_harness.cognitive.cost import solver_cost_usd
    from rna_harness.experiments.retrieval import RetrievalLibrary
    from rna_harness.experiments.rssd_constrained import ConstrainedDesiRNA
    from rna_harness.generators.desirna import DesiRNAGenerator
    from rna_harness.task import DesignTask, OracleConfig
    lib = RetrievalLibrary.load(libpath)
    if forbid_self:
        lib.forbid([target])                            # belt and braces; see also the builder
    oc = OracleConfig(param_set="turner2004", temperature=37.0, dangles=2)
    restr, n_hit, n_units, n_pin = build_restraint(target, lib, mode)
    task = DesignTask(task_id=str(pid), target=target, oracle=oc,
                      meta={"seq_restr": restr} if mode != "none" else {})
    gen = (DesiRNAGenerator if mode == "none" else ConstrainedDesiRNA)(
        replicas=10, timelimit_s=tl, results_number=20)
    t0 = time.time()
    try:
        res = gen.design(task, seed=0)
        w = time.time() - t0
        rec = {"id": pid, "n": len(target), "mode": mode, "t": tl, "solved": bool(res.solved),
               "bp": int(res.best_bp_distance), "wall_s": round(w, 1),
               "cost_usd": solver_cost_usd(w, 10), "n_hit": n_hit, "n_units": n_units,
               "n_pinned": n_pin, "frac_pinned": round(n_pin / max(len(target), 1), 3),
               "seq": res.best_sequence or None}
    except Exception as e:                                                    # noqa: BLE001
        rec = {"id": pid, "mode": mode, "t": tl, "error": repr(e)[:200]}
    with open(dest, "w") as f:
        json.dump(rec, f)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--hard-tier", type=int, default=180)
    ap.add_argument("--cheap-tier", type=int, default=5)
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--set-c", action="store_true",
                    help="set C: does the restraint BREAK any target the plain cheap tier already solved? "
                         "E2 measured exactly this risk for biological restraints (0/20 broken); it has "
                         "never been measured for RETRIEVED ones, and without it the arm cannot ship.")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    base = collections.defaultdict(dict)
    for f in glob.glob(os.path.join(HERE, "sota_curve", "*.json")):
        d = json.load(open(f))
        base[d["t"]][d["id"]] = d
    B = sorted(base)
    ids = sorted(base[B[0]])
    from rna_harness.benchmarks.eterna100 import load_eterna100
    tgt = {p["id"]: p["target"] for p in load_eterna100(version="V2")}

    solved_at = {}
    for i in ids:
        for t in B:
            if base[t][i].get("solved"):
                solved_at[i] = t
                break
    HOPELESS = [i for i in ids if i not in solved_at]
    ESCALATE = [i for i in ids if solved_at.get(i, 0) > B[0]]
    print(f"retrieval arm: {len(HOPELESS)} hopeless targets (A, t={a.hard_tier}), "
          f"{len(ESCALATE)} escalation targets (B, t={a.cheap_tier})")

    # ---- build the library from OUR OWN verified solves, with the eval sets forbidden ------------
    from rna_harness.experiments.retrieval import RetrievalLibrary
    libdir = os.path.join(OUT, "_lib")
    os.makedirs(libdir, exist_ok=True)
    libA = os.path.join(libdir, "libA.json")
    libB = {i: os.path.join(libdir, f"libB_{i}.json") for i in ESCALATE}
    donors = [(i, base[solved_at[i]][i].get("seq")) for i in solved_at]
    donors = [(i, s) for i, s in donors if s]
    print(f"  donors: {len(donors)} of our OWN oracle-verified solves (sequences stored by sota_curve)")
    if not os.path.exists(libA):
        L = RetrievalLibrary()
        L.forbid([tgt[i] for i in HOPELESS])
        n_add = sum(int(L.add_solution(tgt[i], s, source="sota_curve")["added"]) for i, s in donors)
        L.save(libA)
        print(f"  libA ingested {n_add}/{len(donors)} donors -> {L.stats()}")
    for i in ESCALATE:                                   # leave-one-out: never ingest the eval target
        if not os.path.exists(libB[i]):
            L = RetrievalLibrary()
            L.forbid([tgt[i]])
            for j, s in donors:
                if j != i:
                    L.add_solution(tgt[j], s, source="sota_curve")
            L.save(libB[i])
    print(f"  libB: {len(ESCALATE)} leave-one-out libraries\n", flush=True)

    MODES = ("retrieval", "retrieval_clamp", "prior")
    ALREADY = [i for i in ids if solved_at.get(i) == B[0]]          # solved by the PLAIN cheap tier
    if a.set_c:
        libC = {i: os.path.join(libdir, f"libC_{i}.json") for i in ALREADY}
        for i in ALREADY:
            if not os.path.exists(libC[i]):
                L = RetrievalLibrary()
                L.forbid([tgt[i]])
                for j, s_ in donors:
                    if j != i:
                        L.add_solution(tgt[j], s_, source="sota_curve")
                L.save(libC[i])
        print(f"  set C: {len(ALREADY)} targets the plain t={a.cheap_tier} tier already solves, "
              f"leave-one-out libraries built\n", flush=True)
        # Both modes, to separate "PINNING breaks it" from "the RETRIEVED CONTENT breaks it".
        # Same positions in both arms, different fill.
        jobs = [(i, tgt[i], m, a.cheap_tier, True, libC[i])
                for i in ALREADY for m in ("retrieval", "prior")]
    else:
        jobs = [(i, tgt[i], m, a.hard_tier, True, libA) for i in HOPELESS for m in MODES]
        jobs += [(i, tgt[i], m, a.cheap_tier, True, libB[i]) for i in ESCALATE for m in MODES]
    if not a.analyse_only:
        todo = [j for j in jobs
                if not os.path.exists(os.path.join(OUT, f"p{j[0]}_{j[2]}_t{j[3]}.json"))]
        print(f"  {len(todo)} cells to run, {a.workers} workers", flush=True)
        from concurrent.futures import ProcessPoolExecutor, as_completed
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(work, j) for j in todo]
            for n, _f in enumerate(as_completed(futs), 1):
                if n % 10 == 0 or n == len(todo):
                    el = time.time() - t0
                    print(f"    {n}/{len(todo)}  {el/60:.1f} min  "
                          f"eta {el/max(n,1)*(len(todo)-n)/60:.1f} min", flush=True)

    got = collections.defaultdict(dict)
    for f in glob.glob(os.path.join(OUT, "*.json")):
        d = json.load(open(f))
        if "error" not in d:
            got[(d["id"], d["t"])][d["mode"]] = d

    print(f"\n=== A. capability: the {len(HOPELESS)} targets no tier solves, at t={a.hard_tier} ===")
    print(f"  baseline (all-N, measured by sota_curve): 0/{len(HOPELESS)}")
    for m in MODES:
        cells = [got[(i, a.hard_tier)].get(m) for i in HOPELESS]
        cells = [c for c in cells if c]
        w = [c["id"] for c in cells if c["solved"]]
        bp = [c["bp"] for c in cells]
        hit = [c["n_hit"] for c in cells]
        print(f"  {m:<18} {len(w)}/{len(cells)} solved {('-> ' + str(w)) if w else '':<22} "
              f"median bp {sorted(bp)[len(bp)//2] if bp else '-':>4}  "
              f"units hit {sum(hit)}/{sum(c['n_units'] for c in cells)}  "
              f"pinned {100*sum(c['n_pinned'] for c in cells)/max(sum(c['n'] for c in cells),1):.0f}% of bases")

    print(f"\n=== B. efficiency: the {len(ESCALATE)} escalation targets at the CHEAP tier "
          f"t={a.cheap_tier} (leave-one-out library) ===")
    print(f"  baseline (all-N at t={a.cheap_tier}): 0/{len(ESCALATE)} by construction")
    for m in MODES:
        cells = [got[(i, a.cheap_tier)].get(m) for i in ESCALATE]
        cells = [c for c in cells if c]
        w = [c["id"] for c in cells if c["solved"]]
        print(f"  {m:<18} {len(w)}/{len(cells)} solved {('-> ' + str(w)) if w else ''}")
    print(f"\n  the control that decides it: `retrieval` must beat `prior` at the SAME pinned positions.")

    if a.set_c:
        print(f"\n=== C. does the restraint BREAK what the plain tier already solved? ===")
        print(f"  plain t={a.cheap_tier} (no restraint): {len(ALREADY)}/{len(ALREADY)} by construction")
        brk = {}
        for m in ("retrieval", "prior"):
            cells = [got[(i, a.cheap_tier)].get(m) for i in ALREADY]
            cells = [c for c in cells if c]
            brk[m] = [c["id"] for c in cells if not c["solved"]]
            print(f"  {m:<12} kept {len(cells)-len(brk[m])}/{len(cells)}  BROKE {len(brk[m])}")
        if brk.get("prior") is not None and brk.get("retrieval") is not None:
            r, pr = set(brk["retrieval"]), set(brk["prior"])
            print(f"\n  broken by BOTH: {len(r & pr)}   only by retrieval: {len(r - pr)}   "
                  f"only by prior: {len(pr - r)}")
            print(f"  -> {'PINNING itself is the problem (both arms break it)' if len(r & pr) >= 0.6*len(r|pr) else 'the RETRIEVED CONTENT is the problem'}")
        print(f"\n  net at the cheap tier = (B gained) - (C broken) = "
              f"4 - {len(brk.get('retrieval', []))} = {4 - len(brk.get('retrieval', []))}")

    json.dump({"hopeless": HOPELESS, "escalate": ESCALATE, "modes": list(MODES),
               "hard_tier": a.hard_tier, "cheap_tier": a.cheap_tier,
               "cells": {f"{k[0]}_{k[1]}_{m}": v for k, d in got.items() for m, v in d.items()}},
              open(os.path.join(HERE, "retrieval_arm.json"), "w"), indent=1)
    print(f"\n  wrote rna_harness/runs/ceiling/retrieval_arm.json")


if __name__ == "__main__":
    main()
