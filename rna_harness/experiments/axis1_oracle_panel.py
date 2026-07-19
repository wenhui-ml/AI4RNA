"""R3.1 -- Axis-1 oracle validity: is the pseudoknot "false-accept" verdict a property of the DESIGNS, or
of the REFERENCE ORACLE (pKiss)?

A reviewer's central objection to Axis-1 is that pKiss is a HEURISTIC pseudoknot predictor, not ground
truth, so "the design does/does not form the target" -- and hence the 28% false-accept figure -- is defined
only *relative to pKiss configuration C0*. This experiment re-verifies the DesiRNA designs (saved by
run_pk_desirna) under a panel of THREE pseudoknot folders, two of which are algorithmically INDEPENDENT of
pKiss (RNAstructure ProbKnot: maximum-expected-accuracy over a partition-function ensemble; RNAstructure
ShapeKnots: heuristic MFE search). We then report:

  (1) REPRESENTABILITY GATE: tasks for which some oracle cannot return a verdict are abstained, not scored.
  (2) PER-ORACLE false-accept rate: the 28% recomputed under each oracle -> a RANGE, not a point.
  (3) CROSS-ORACLE AGREEMENT: pairwise raw agreement + Cohen's kappa on the binary "forms T", 3-way unanimity.
  (4) ABSTENTION-ON-DISAGREEMENT: on the unanimous subset, the (now oracle-robust) false-accept rate; plus
      the abstention fraction (how much we must decline to call because the oracles disagree).
  (5) POSITIVE CONTROL (sensitivity): on designs pKiss certifies as forming T, the fraction the two
      independent oracles CONFIRM -- a cross-predictor sensitivity check on designed pseudoknots (we do not
      claim experimental ground truth; see the paper's oracle-validity caveat).

Each oracle gets the same benefit of the doubt pKiss got: it "forms T" if ANY candidate in the saved pool
folds (pseudoknot-aware) to exactly the target's pair set. Run after run_pk_desirna saves sequences."""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations

from ..benchmarks.rnainvbench import pk_pairs
from ..oracles_pk import ORACLES, rnastructure_available

PK_JSON = os.path.join(os.path.dirname(__file__), "..", "runs", "pk_desirna", "pk_desirna.json")
OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "axis1_oracle_panel")
ORACLE_NAMES = ["pkiss", "probknot", "shapeknots"]


def _verdict_one_oracle(name, cands, target, timeout_s):
    """forms_T under one oracle over the candidate pool (benefit of the doubt = best over pool).
    Returns (forms_T:bool|None, min_bp:int|None). None => oracle failed on EVERY candidate (abstain)."""
    fn = ORACLES[name]
    tgt = pk_pairs(target)
    min_bp = None
    any_ok = False
    for c in cands:
        if not c:
            continue
        pairs = fn(c, timeout_s=timeout_s)
        if pairs is None:
            continue
        any_ok = True
        bp = len(pairs ^ tgt)
        if min_bp is None or bp < min_bp:
            min_bp = bp
        if bp == 0:
            return True, 0
    if not any_ok:
        return None, None
    return (min_bp == 0), min_bp


def _eval_task(row, timeout_s):
    """All three oracles on one task's candidate pool."""
    target = row["target"]
    cands = row.get("candidates") or [row.get("mfe_best_seq")]
    out = {"task_id": row["task_id"], "n": row.get("n"), "n_pk_pairs": row.get("n_pk_pairs"),
           "mfe_reported_solved": bool(row.get("mfe_reported_solved"))}
    for nm in ORACLE_NAMES:
        ft, bp = _verdict_one_oracle(nm, cands, target, timeout_s)
        out[f"forms_T_{nm}"] = ft
        out[f"minbp_{nm}"] = bp
    return out


def _kappa(a, b):
    """Cohen's kappa for two binary label lists (drop positions where either is None)."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return None, 0
    n = len(pairs)
    po = sum(1 for x, y in pairs if x == y) / n
    pa1 = sum(1 for x, _ in pairs if x) / n
    pb1 = sum(1 for _, y in pairs if y) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    k = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return k, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=PK_JSON)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--outdir", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    if not rnastructure_available():
        raise SystemExit("RNAstructure (ProbKnot/ShapeKnots) not found -- set RNASTRUCTURE_BIN/DATAPATH.")
    d = json.load(open(a.inp))
    rows = [r for r in d["rows"] if r.get("candidates") or r.get("mfe_best_seq")]
    print(f"Axis-1 oracle panel on {len(rows)} tasks x 3 oracles (pkiss, probknot, shapeknots); "
          f"pool benefit-of-doubt. Independent oracles = ProbKnot, ShapeKnots.", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_eval_task, r, a.timeout) for r in rows]
        done = 0
        for fut in as_completed(futs):
            results.append(fut.result()); done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(rows)} tasks", flush=True)

    # ---- (1) representability gate: any oracle returning None on every candidate => abstain that task
    gated, scored = [], []
    for r in results:
        if any(r[f"forms_T_{nm}"] is None for nm in ORACLE_NAMES):
            gated.append(r)
        else:
            scored.append(r)

    # ---- (2) per-oracle false-accept rate among MFE-reported-solved
    mfe_solved = [r for r in scored if r["mfe_reported_solved"]]
    per_oracle = {}
    for nm in ORACLE_NAMES:
        solved = sum(1 for r in scored if r[f"forms_T_{nm}"])
        fa = sum(1 for r in mfe_solved if not r[f"forms_T_{nm}"])
        per_oracle[nm] = {"forms_T_count": solved, "false_accepts": fa,
                          "false_accept_pct": 100 * fa / max(1, len(mfe_solved))}

    # ---- (3) agreement: pairwise raw + kappa, 3-way unanimity
    labels = {nm: [r[f"forms_T_{nm}"] for r in scored] for nm in ORACLE_NAMES}
    agree = {}
    for x, y in combinations(ORACLE_NAMES, 2):
        k, n = _kappa(labels[x], labels[y])
        raw = sum(1 for i in range(len(scored)) if labels[x][i] == labels[y][i]) / max(1, len(scored))
        agree[f"{x}|{y}"] = {"raw_pct": 100 * raw, "kappa": k, "n": n}
    unanimous = [r for r in scored if len({r[f"forms_T_{nm}"] for nm in ORACLE_NAMES}) == 1]
    disagree = [r for r in scored if len({r[f"forms_T_{nm}"] for nm in ORACLE_NAMES}) > 1]

    # ---- (4) abstention rule: on unanimous subset, false-accept rate is oracle-robust
    unan_mfe = [r for r in unanimous if r["mfe_reported_solved"]]
    unan_fa = sum(1 for r in unan_mfe if not r["forms_T_pkiss"])  # unanimous => any oracle same verdict
    abstain_pct = 100 * len(disagree) / max(1, len(scored))

    # ---- (5) positive control / cross-predictor sensitivity: on pKiss-certified solves, do the two
    #          INDEPENDENT oracles confirm?
    pkiss_solved = [r for r in scored if r["forms_T_pkiss"]]
    confirm = {nm: sum(1 for r in pkiss_solved if r[f"forms_T_{nm}"]) for nm in ("probknot", "shapeknots")}
    pos = {nm: {"confirmed": confirm[nm], "of": len(pkiss_solved),
                "pct": 100 * confirm[nm] / max(1, len(pkiss_solved))} for nm in confirm}

    summary = {"n_total": len(results), "n_gated_abstained": len(gated), "n_scored": len(scored),
               "n_mfe_solved": len(mfe_solved), "per_oracle_false_accept": per_oracle,
               "agreement": agree, "n_unanimous": len(unanimous), "n_disagree": len(disagree),
               "abstain_pct": abstain_pct,
               "unanimous_false_accept": {"num": unan_fa, "den": len(unan_mfe),
                                          "pct": 100 * unan_fa / max(1, len(unan_mfe))},
               "positive_control_sensitivity": pos}
    json.dump({"summary": summary, "rows": results}, open(os.path.join(a.outdir, "panel.json"), "w"), indent=1)

    # ---- report
    print(f"\n=== R3.1 AXIS-1 ORACLE VALIDITY ({len(scored)} scored, {len(gated)} gated/abstained) ===", flush=True)
    print(f"  MFE-reported solved: {len(mfe_solved)}", flush=True)
    print("  per-oracle false-accept rate among MFE-solved (the '28%' recomputed per oracle):", flush=True)
    for nm in ORACLE_NAMES:
        po = per_oracle[nm]
        print(f"    {nm:10}: {po['false_accepts']:>3}/{len(mfe_solved)} = {po['false_accept_pct']:5.1f}%  "
              f"(forms T on {po['forms_T_count']})", flush=True)
    fa_pcts = [per_oracle[nm]["false_accept_pct"] for nm in ORACLE_NAMES]
    print(f"    -> RANGE across oracles: [{min(fa_pcts):.1f}, {max(fa_pcts):.1f}]%", flush=True)
    print("  cross-oracle agreement on 'forms T' (raw %, Cohen kappa):", flush=True)
    for k, v in agree.items():
        kk = f"{v['kappa']:.2f}" if v["kappa"] is not None else "n/a"
        print(f"    {k:22}: raw {v['raw_pct']:5.1f}%  kappa {kk}", flush=True)
    print(f"  3-way unanimous: {len(unanimous)}/{len(scored)}  |  disagreement (abstain): "
          f"{len(disagree)}/{len(scored)} = {abstain_pct:.1f}%", flush=True)
    ufa = summary["unanimous_false_accept"]
    print(f"  ORACLE-ROBUST false-accept (unanimous subset): {ufa['num']}/{ufa['den']} = {ufa['pct']:.1f}%", flush=True)
    print("  positive control (independent oracles confirming pKiss-certified solves):", flush=True)
    for nm, v in pos.items():
        print(f"    {nm:10}: {v['confirmed']}/{v['of']} = {v['pct']:.1f}%", flush=True)
    print(f"\n  wrote {os.path.join(a.outdir, 'panel.json')}", flush=True)


if __name__ == "__main__":
    main()
