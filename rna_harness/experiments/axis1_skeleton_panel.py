"""R3.1 (reviewer 3.1/3.2): run the FULL 3-oracle panel on the SKELETON-SEARCH designs.

The paper's headline 99.6% Axis-1 false-accept rate comes from a pseudoknot-BLIND REMC search that designs
against the nested skeleton round_only(T); it was verified only by pKiss. The multi-oracle panel
(axis1_oracle_panel.py) was run on the *DesiRNA* designs, not these. This script closes that gap: it
regenerates the skeleton-search designs (the sequences were never saved) and re-verifies EACH with all three
independent pseudoknot folders (pKiss, ProbKnot, ShapeKnots), reporting per oracle:
  * exact pair-set recovery of the FULL target (forms T),
  * crossing-pair RECALL (fraction of the target's crossing/pseudoknot pairs present in the predicted fold),
  * base-pair distance to the full target,
and the 3-way-unanimous subset. One design per target (the REMC best that the MFE oracle accepted) -- no
candidate pool, so the analysis unit is unambiguous. Phase 1 (REMC) is sequential because replica_exchange
uses its own process pool; phase 2 (panel folding) is parallelized."""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import RNA

from ..benchmarks.rnainvbench import load_pseudobase, _round_only, pk_pairs
from ..oracles_pk import ORACLES, rnastructure_available

OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "axis1_skeleton_panel")
DESIGNS = os.path.join(OUT, "skeleton_designs.json")


def _crossing_pairs(target):
    """The pseudoknot/crossing pairs = full pair set minus the nested-skeleton pair set."""
    return pk_pairs(target) - pk_pairs(_round_only(target))


def _naive_mfe_solved(seq, target):
    if not seq or len(seq) != len(target):
        return False
    st, _ = RNA.fold(seq)
    return st == _round_only(target)


def regen_designs(seed=0, wall_s=20.0):
    """Phase 1: regenerate the skeleton-search designs (REMC on round_only(T)); save sequences."""
    from .anneal import replica_exchange
    tasks = load_pseudobase()
    print(f"[phase1] regenerating skeleton designs for {len(tasks)} targets (REMC, wall={wall_s}s, seed={seed})", flush=True)
    designs = []
    for k, t in enumerate(tasks):
        skel = _round_only(t.target)
        r = replica_exchange(skel, n_replicas=8, wall_s=wall_s, seed=seed)
        seq = r.get("best_seq")
        naive = _naive_mfe_solved(seq, t.target)
        designs.append({"id": t.task_id, "n": t.n, "target": t.target, "seq": seq,
                        "naive_mfe_solved": bool(naive), "n_cross": len(_crossing_pairs(t.target))})
        if (k + 1) % 10 == 0 or k + 1 == len(tasks):
            nfp = sum(d["naive_mfe_solved"] for d in designs)
            print(f"  {k+1:>3}/{len(tasks)}  skeleton-solved so far={nfp}", flush=True)
            os.makedirs(OUT, exist_ok=True)
            json.dump({"designs": designs}, open(DESIGNS, "w"), indent=1)
    return designs


def _panel_one(rec):
    """Phase 2 worker: fold one design with all 3 oracles; compute exact / crossing-recall / bp-dist."""
    seq, target = rec["seq"], rec["target"]
    allp = pk_pairs(target)
    cross = _crossing_pairs(target)
    out = {"id": rec["id"], "n": rec["n"], "n_cross": len(cross)}
    for name, fn in ORACLES.items():
        pairs = None
        try:
            pairs = fn(seq)
        except Exception:
            pairs = None
        if pairs is None:
            out[name] = {"ok": False}
            continue
        exact = (pairs == allp)
        recall = (len(pairs & cross) / len(cross)) if cross else None
        bpdist = len(pairs ^ allp)                                  # symmetric difference = bp distance
        out[name] = {"ok": True, "exact": bool(exact),
                     "cross_recall": recall, "bp_dist": bpdist, "n_pred": len(pairs)}
    return out


def run_panel(workers=32):
    """Phase 2: panel-fold every MFE-accepted skeleton design; summarize per oracle."""
    if not rnastructure_available():
        raise SystemExit("RNAstructure unavailable -- set RNASTRUCTURE_BIN / DATAPATH (conda env 'rnastr')")
    designs = json.load(open(DESIGNS))["designs"]
    solved = [d for d in designs if d["naive_mfe_solved"]]
    print(f"[phase2] panel on {len(solved)} MFE-accepted skeleton designs (of {len(designs)}) with {workers} workers", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_panel_one, d): d["id"] for d in solved}
        for i, fut in enumerate(as_completed(futs)):
            rows.append(fut.result())
            if (i + 1) % 20 == 0 or i + 1 == len(solved):
                print(f"  folded {i+1}/{len(solved)}", flush=True)
    rows.sort(key=lambda r: r["id"])
    summary = _summarize(rows, len(solved))
    os.makedirs(OUT, exist_ok=True)
    json.dump({"summary": summary, "rows": rows}, open(os.path.join(OUT, "panel.json"), "w"), indent=1)
    _print_summary(summary)
    return summary


def _summarize(rows, n_solved):
    names = list(ORACLES)
    summ = {"n_designs": n_solved, "per_oracle": {}}
    for nm in names:
        ok = [r[nm] for r in rows if r[nm].get("ok")]
        n_ok = len(ok)
        exact = sum(int(x["exact"]) for x in ok)
        recalls = [x["cross_recall"] for x in ok if x["cross_recall"] is not None]
        bpd = [x["bp_dist"] for x in ok]
        summ["per_oracle"][nm] = {
            "n_folded": n_ok,
            "exact": exact,
            "exact_rate": round(exact / n_ok, 4) if n_ok else None,
            "false_accept_rate": round(1 - exact / n_ok, 4) if n_ok else None,
            "mean_cross_recall": round(sum(recalls) / len(recalls), 4) if recalls else None,
            "zero_cross_recall_frac": round(sum(1 for x in recalls if x == 0.0) / len(recalls), 4) if recalls else None,
            "median_bp_dist": sorted(bpd)[len(bpd) // 2] if bpd else None,
        }
    # 3-way unanimous exact (all three folded and agree)
    unan = [r for r in rows if all(r[nm].get("ok") for nm in names)]
    unan_exact_all = sum(1 for r in unan if all(r[nm]["exact"] for nm in names))
    unan_fa_all = sum(1 for r in unan if all(not r[nm]["exact"] for nm in names))
    summ["unanimous"] = {"n": len(unan), "all_exact": unan_exact_all, "all_false_accept": unan_fa_all,
                         "unanimous_fa_rate": round(unan_fa_all / len(unan), 4) if unan else None}
    return summ


def _print_summary(s):
    print("\n=== SKELETON-DESIGN 3-ORACLE PANEL ===", flush=True)
    print(f"designs (MFE-accepted): {s['n_designs']}", flush=True)
    for nm, d in s["per_oracle"].items():
        print(f"  {nm:10} folded={d['n_folded']:>3}  exact={d['exact']:>3} "
              f"({100*(d['exact_rate'] or 0):.1f}%)  false-accept={100*(d['false_accept_rate'] or 0):.1f}%  "
              f"mean_cross_recall={d['mean_cross_recall']}  zero_recall_frac={d['zero_cross_recall_frac']}  "
              f"med_bp_dist={d['median_bp_dist']}", flush=True)
    u = s["unanimous"]
    print(f"  3-way unanimous: n={u['n']}  all-false-accept={u['all_false_accept']} "
          f"({100*(u['unanimous_fa_rate'] or 0):.1f}%)  all-exact={u['all_exact']}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["regen", "panel", "all"], default="all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wall-s", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    if a.phase in ("regen", "all") and not os.path.exists(DESIGNS):
        regen_designs(seed=a.seed, wall_s=a.wall_s)
    elif a.phase in ("regen", "all"):
        print(f"[phase1] designs already exist at {DESIGNS}; skipping regen", flush=True)
    if a.phase in ("panel", "all"):
        run_panel(workers=a.workers)
