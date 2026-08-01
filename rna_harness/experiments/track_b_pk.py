"""Track B -- open-domain RNA design under a SWITCHED oracle (pseudoknots, verified by pKiss).

Track A (Eterna100) lives entirely inside ViennaRNA MFE. Track B moves to RNAInvBench / Pseudobase++
pseudoknot targets, where the CORRECT verifier is pKiss, not MFE. It exercises the cognitive core on the
verifier's axis 1 (WHICH oracle) -- the part that search-only difficulty never touches:

  * oracle-switch is AUTOMATIC: verify() routes has_pseudoknot(target) -> pKiss. Every candidate --
    including DesiRNA's own output -- is re-folded by pKiss; the MFE oracle is never trusted on a pseudoknot.
  * economy under an EXPENSIVE oracle: pKiss costs seconds/call (vs microseconds for MFE), and that cost
    now lives inside ToolResult.cost_usd (solver + oracle). "Stay cheap" now means "don't burn pKiss calls
    on tools that cannot design a pseudoknot" -- our REMC optimises the MFE skeleton and cannot form the
    crossing stem, so the core must route past it to DesiRNA (the only potentially pk-capable tool).
  * WHY the switch matters (the control): a pseudoknot-BLIND harness checks fold(seq) == round_only(T)
    under MFE and reports FALSE POSITIVES -- a sequence that forms only the nested skeleton "passes". The
    fp-demo prints naive-MFE vs honest-pKiss side by side to quantify that gap.

v1 scope: designs the STRUCTURE (task.target); the Pseudobase IUPAC sequence restraints are DROPPED (the
cognitive tools take a target string, not a DesignTask). Constraint-aware design is a later extension.
Red line unchanged: solved := pk-fold(seq) has EXACTLY the target's pairs, under pKiss -- nothing more.
"""
from __future__ import annotations

import argparse
import json
import os

import RNA

from ..benchmarks.rnainvbench import (load_pseudobase, verify_pk, pkiss_available,
                                      _round_only, pk_pairs, n_pk_pairs)
from ..cognitive.core import CognitiveCore, FixedPolicy
from ..cognitive.features import features
from ..cognitive.tools import DesirnaTool, RemcTool, default_toolbox

OUT_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "runs", "track_b_pk")


def _naive_mfe_solved(seq, target) -> bool:
    """What a pseudoknot-BLIND MFE harness would report: does seq fold (MFE) to the NESTED SKELETON of
    the target? This is the false-positive-prone check the correct pKiss verifier replaces."""
    if not seq or len(seq) != len(target):
        return False
    st, _ = RNA.fold(seq)
    return st == _round_only(target)


# ---- cheap sanity: list tasks + confirm they really are pseudoknots ------------------------------
def list_tasks(limit):
    tasks = load_pseudobase(limit=limit)
    print(f"loaded {len(tasks)} Pseudobase++ tasks; pKiss available: {pkiss_available()}", flush=True)
    for t in tasks:
        f = features(t.target)
        print(f"  {t.task_id:14} n={t.n:>3} pairs={len(pk_pairs(t.target)):>2} "
              f"pk_pairs_lost_by_MFE={t.meta['n_pk_pairs']:>2} has_pseudoknot(feat)={int(f['has_pseudoknot'])} "
              f"restr_fixed={len(t.constraints.motifs):>2}", flush=True)


# ---- the cheap, high-value control: MFE false-positives that pKiss catches ------------------------
def fp_demo(limit, outdir, seed=0):
    """Show that the MFE oracle is UNSAFE on pseudoknots. Our REMC optimises the nested skeleton
    round_only(T); its best sequence tends to satisfy naive-MFE (fold==round_only) yet FAIL pKiss on the
    full target (the crossing stem never forms). Each such case is a false positive the switch removes."""
    os.makedirs(outdir, exist_ok=True)
    from .anneal import replica_exchange
    tasks = load_pseudobase(limit=limit)
    print(f"fp-demo on {len(tasks)} tasks (REMC targets the MFE skeleton; pKiss judges the truth)", flush=True)
    rows, fp = [], 0
    for t in tasks:
        skel = _round_only(t.target)
        r = replica_exchange(skel, n_replicas=8, wall_s=20.0, seed=seed)   # solve the NESTED skeleton
        seq = r.get("best_seq")
        naive = _naive_mfe_solved(seq, t.target)          # MFE oracle's verdict on the full target
        honest = verify_pk(seq, t.target)                 # pKiss verdict on the full target
        is_fp = bool(naive and not honest.get("solved"))
        fp += int(is_fp)
        rows.append({"id": t.task_id, "n": t.n, "n_pk_pairs": t.meta["n_pk_pairs"],
                     "naive_mfe_solved": naive, "pkiss_solved": honest.get("solved"),
                     "pkiss_bp": honest.get("bp"), "false_positive": is_fp})
        print(f"  {t.task_id:14} naive_MFE={int(naive)} pKiss={int(bool(honest.get('solved')))} "
              f"pkiss_bp={honest.get('bp')}  {'<== FALSE POSITIVE' if is_fp else ''}", flush=True)
    print(f"\nMFE-oracle false positives caught by the pKiss switch: {fp}/{len(tasks)}", flush=True)
    json.dump({"false_positives": fp, "n": len(tasks), "rows": rows},
              open(os.path.join(outdir, "fp_demo.json"), "w"), indent=1)


# ---- one DesiRNA pseudoknot attempt: can the SOTA tool actually design a pk under our adapter? -----
def desirna_smoke(idx, timelimit_s, seed=0):
    tasks = load_pseudobase(limit=idx + 1)
    t = tasks[idx]
    print(f"DesiRNA pk-smoke on {t.task_id} n={t.n} (target={t.target})", flush=True)
    r = DesirnaTool(timelimit_s=timelimit_s).attempt(t.target, seed=seed)
    print(f"  solved(pKiss)={r.solved} bp={r.bp} oracle={r.oracle} cost=${r.cost_usd:.2e} "
          f"solver_wall={r.wall_s:.1f}s pkiss_wall={r.oracle_s:.1f}s available={r.available}", flush=True)
    if r.seq:
        print(f"  seq={r.seq}", flush=True)
    return r


# ---- full harness (cognitive vs fixed baselines under pKiss) -- EXPENSIVE, scaffolded for later ----
def build_policies(which):
    pol = {}
    if "cognitive" in which:
        pol["cognitive"] = CognitiveCore(default_toolbox(), learn=True)
    if "always_cheap" in which:
        pol["always_cheap"] = FixedPolicy([RemcTool(wall_s=30)], name="always_cheap")
    if "always_sota" in which:
        pol["always_sota"] = FixedPolicy([DesirnaTool(timelimit_s=120)], name="always_sota")
    # R7 M5 / R8: PKX_DETERMINISTIC=1 puts every operator in the counted-budget, RNG-seeded regime, so a
    # nominal seed fixes the run bit-for-bit AND every arm receives the SAME pKiss allowance. Under the
    # default (0) the operators keep the historical wall-clock stopping rules, which is what every
    # measurement up to R7 used. One switch for the whole sweep: mixing regimes across arms would silently
    # unmatch the budget the ablation is supposed to hold fixed.
    _det = os.environ.get("PKX_DETERMINISTIC", "0") == "1"
    _maxpk = int(os.environ.get("PKX_MAXPKISS", "150"))

    if "autonomous" in which:
        # The INVENTED MCF operator: decompose T -> CF skeleton -> compose crossing pairs -> pKiss-guided
        # repair. It NEVER calls DesiRNA (nor any external pseudoknot solver), so autonomy is STRUCTURAL:
        # a nonzero held-out solve rate here is a certified CF->MCF ascent by the agent's own procedure,
        # not a borrowed one. Budgets are env-tunable for calibration (PKD_SKEL_S/PKD_REPAIR_S/PKD_ITERS).
        from ..cognitive.skills.pk_decompose import DecomposePKTool
        _sk = float(os.environ.get("PKD_SKEL_S", "12"))
        _rp = float(os.environ.get("PKD_REPAIR_S", "60"))
        _it = int(os.environ.get("PKD_ITERS", "400"))
        _abl = os.environ.get("PKD_ABLATE", "none")   # P2: none|norepair|nocompose|nodecompose
        pol["autonomous"] = FixedPolicy(
            [DecomposePKTool(skel_wall_s=_sk, repair_wall_s=_rp, max_repair_iters=_it, ablate=_abl,
                             deterministic=_det, max_pkiss=_maxpk)],
            name="autonomous")
    if "panel_auto" in which:
        # PANEL-OPTIMIZING operator: optimizes pKiss AND ProbKnot agreement (lexicographic), not a single
        # oracle -> tests whether model-ROBUST autonomous ascent is attainable where pKiss-gaming (2/43) is not.
        from ..cognitive.skills.pk_decompose_panel import DecomposePKPanelTool
        _psk = float(os.environ.get("PKP_SKEL_S", "12"))
        _prp = float(os.environ.get("PKP_REPAIR_S", "90"))
        _pit = int(os.environ.get("PKP_ITERS", "800"))
        _pgate = int(os.environ.get("PKP_GATE", "3"))
        _pmpb = int(os.environ.get("PKP_MAXPB", "120"))
        pol["panel_auto"] = FixedPolicy(
            [DecomposePKPanelTool(skel_wall_s=_psk, repair_wall_s=_prp, max_repair_iters=_pit,
                                  panel_gate=_pgate, max_probknot=_pmpb,
                                  deterministic=_det, max_pkiss=_maxpk)], name="panel_auto")
    # R4: per-component ablation of the PANEL operator + a matched-budget undirected-search control, each
    # scored under the same panel. Arm name 'panel_<mode>' => DecomposePKPanelTool(ablate=<mode>) at
    # IDENTICAL budgets/oracle access/initialization as panel_auto; only the named component is removed.
    for _mode in ("randwalk", "blindrand", "norepair", "nodecompose", "nocompose"):
        if f"panel_{_mode}" in which:
            from ..cognitive.skills.pk_decompose_panel import DecomposePKPanelTool
            pol[f"panel_{_mode}"] = FixedPolicy(
                [DecomposePKPanelTool(skel_wall_s=float(os.environ.get("PKP_SKEL_S", "12")),
                                      repair_wall_s=float(os.environ.get("PKP_REPAIR_S", "90")),
                                      max_repair_iters=int(os.environ.get("PKP_ITERS", "800")),
                                      panel_gate=int(os.environ.get("PKP_GATE", "3")),
                                      max_probknot=int(os.environ.get("PKP_MAXPB", "120")),
                                      ablate=_mode, deterministic=_det,
                                      max_pkiss=_maxpk)], name=f"panel_{_mode}")
    # R8 attribution: arm name 'pv_<variant>' => the panel operator with ONE agent-written mechanism
    # transplanted in, at otherwise identical settings. See DecomposePKPanelTool.VARIANTS for the chain.
    for _v in ("ungated", "scalar1", "scalar2", "anneal", "bigpk", "tinypk", "nodecoy"):
        if f"pv_{_v}" in which:
            from ..cognitive.skills.pk_decompose_panel import DecomposePKPanelTool
            pol[f"pv_{_v}"] = FixedPolicy(
                [DecomposePKPanelTool(skel_wall_s=float(os.environ.get("PKP_SKEL_S", "12")),
                                      repair_wall_s=float(os.environ.get("PKP_REPAIR_S", "90")),
                                      max_repair_iters=int(os.environ.get("PKP_ITERS", "800")),
                                      panel_gate=int(os.environ.get("PKP_GATE", "3")),
                                      max_probknot=int(os.environ.get("PKP_MAXPB", "120")),
                                      deterministic=_det, max_pkiss=_maxpk,
                                      variant=_v)], name=f"pv_{_v}")
    if "probknot_auto" in which:
        # P4 symmetry control: optimizes ProbKnot in-loop (analog of autonomous, which optimizes pKiss).
        from ..cognitive.skills.pk_decompose_probknot import DecomposeProbKnotTool
        _bsk = float(os.environ.get("PKB_SKEL_S", "12"))
        _brp = float(os.environ.get("PKB_REPAIR_S", "90"))
        _bit = int(os.environ.get("PKB_ITERS", "400"))
        pol["probknot_auto"] = FixedPolicy(
            [DecomposeProbKnotTool(skel_wall_s=_bsk, repair_wall_s=_brp, max_repair_iters=_bit)],
            name="probknot_auto")
    return pol


def _oracle_s(res):
    return round(sum(s.get("oracle_s", 0.0) for s in res.get("trace", [])), 3)


def run(limit, which, outdir, seed=0):
    os.makedirs(outdir, exist_ok=True)
    if not pkiss_available():
        raise SystemExit("pKiss unavailable -- set PKISS_CMD or install conda env 'pk'")
    tasks = load_pseudobase(limit=limit)
    pols = build_policies(which)
    print(f"Track B run: {len(tasks)} pseudoknot tasks; oracle=pKiss; policies={list(pols)}", flush=True)
    rows = []
    agg = {n: {"solved": 0, "cost_usd": 0.0, "oracle_s": 0.0} for n in pols}
    for t in tasks:
        assert features(t.target)["has_pseudoknot"] == 1.0     # these ARE pseudoknots
        rec = {"id": t.task_id, "n": t.n, "n_pk_pairs": t.meta["n_pk_pairs"], "by_policy": {}}
        for name, pol in pols.items():
            r = pol.solve(t.target, seed=seed)
            nm = _naive_mfe_solved(r.get("seq"), t.target) if r.get("seq") else None
            os_ = _oracle_s(r)
            rec["by_policy"][name] = {"solved": r["solved"], "cost_usd": r["cost_usd"],
                                      "n_actions": r["n_actions"], "oracle_s": os_,
                                      "naive_mfe_solved": nm,
                                      "seq": r.get("seq")}   # saved for offline cross-oracle panel adjudication
            agg[name]["solved"] += int(r["solved"])
            agg[name]["cost_usd"] += r["cost_usd"]
            agg[name]["oracle_s"] += os_
        rows.append(rec)
        print(f"  {t.task_id:14} | " + " ".join(
            f"{nm}:{v['solved']}/${v['cost_usd']:.2e}(pk{v['oracle_s']:.0f}s)"
            for nm, v in rec["by_policy"].items()), flush=True)
        json.dump({"agg": agg, "rows": rows}, open(os.path.join(outdir, "run.json"), "w"), indent=1)
    print("\n=== AGGREGATE (solved / total $ / pKiss seconds) ===", flush=True)
    for name, a in agg.items():
        print(f"  {name:14} solved={a['solved']}/{len(tasks)}  total=${a['cost_usd']:.3e}  "
              f"pKiss={a['oracle_s']:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", type=int, metavar="N", help="load+describe first N tasks (cheap)")
    ap.add_argument("--fp-demo", type=int, metavar="N", help="MFE false-positive control on N tasks")
    ap.add_argument("--desirna-smoke", type=int, metavar="IDX", help="one DesiRNA pk attempt on task IDX")
    ap.add_argument("--run", type=int, metavar="N", help="full cognitive-vs-baselines on N tasks (expensive)")
    ap.add_argument("--timelimit-s", type=int, default=60)
    ap.add_argument("--policies", nargs="+", default=["cognitive", "always_cheap", "always_sota"])
    ap.add_argument("--outdir", default=OUT_DEFAULT)
    ap.add_argument("--seed", type=int, default=0)      # P6: multi-seed for bootstrap CIs
    a = ap.parse_args()
    if a.list is not None:
        list_tasks(a.list)
    elif a.fp_demo is not None:
        fp_demo(a.fp_demo, a.outdir)
    elif a.desirna_smoke is not None:
        desirna_smoke(a.desirna_smoke, a.timelimit_s)
    elif a.run is not None:
        run(a.run, a.policies, a.outdir, seed=a.seed)
    else:
        ap.error("pass one of --list N | --fp-demo N | --desirna-smoke IDX | --run N")
