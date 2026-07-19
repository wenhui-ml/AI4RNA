"""R3-3.2: FIXED-candidate cross-oracle comparison for the DesiRNA designs.

The panel in axis1_oracle_panel.py judges each target via best-of-pool (some candidate folds to T), which
confounds oracle differences with candidate selection. Here we fix ONE canonical design per target (DesiRNA's
mfe_best_seq) and judge that identical sequence under all three predictors, reporting exact-match,
crossing-pair recall, and bp-distance per oracle -- the pure fixed-sequence re-judgement the reviewer asks for
(best-of-pool remains the supplementary view)."""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..benchmarks.rnainvbench import load_pseudobase, _round_only, pk_pairs
from ..oracles_pk import ORACLES, rnastructure_available

PK_JSON = os.path.join(os.path.dirname(__file__), "..", "runs", "pk_desirna", "pk_desirna.json")
OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "axis1_desirna_fixedcand")


def _crossing(target):
    return pk_pairs(target) - pk_pairs(_round_only(target))


def _one(rec):
    seq, target = rec["seq"], rec["target"]
    allp = pk_pairs(target); cross = _crossing(target)
    out = {"id": rec["id"], "n": rec["n"], "n_cross": len(cross)}
    for nm, fn in ORACLES.items():
        try:
            pairs = fn(seq)
        except Exception:
            pairs = None
        if pairs is None:
            out[nm] = {"ok": False}; continue
        out[nm] = {"ok": True, "exact": bool(pairs == allp),
                   "cross_recall": (len(pairs & cross) / len(cross)) if cross else None,
                   "bp_dist": len(pairs ^ allp)}
    return out


def run(workers=32):
    if not rnastructure_available():
        raise SystemExit("RNAstructure unavailable (env rnastr)")
    os.makedirs(OUT, exist_ok=True)
    pk = json.load(open(PK_JSON))
    tgt = {t.task_id: t.target for t in load_pseudobase()}
    # id in pk_desirna rows is like "pseudobase:k" or an int index; map to target via mfe_best_seq present
    recs = []
    for r in pk["rows"]:
        seq = r.get("mfe_best_seq")
        tid = r.get("id") or r.get("task_id")
        target = tgt.get(tid) or r.get("target")
        if seq and target and len(seq) == len(target) and r.get("mfe_reported_solved", True):
            recs.append({"id": tid, "n": len(target), "seq": seq, "target": target})
    print(f"fixed-candidate cross-oracle on {len(recs)} DesiRNA mfe_best designs", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_one, r) for r in recs]
        for i, f in enumerate(as_completed(futs)):
            rows.append(f.result())
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(recs)}", flush=True)
    summ = {"n": len(rows), "per_oracle": {}}
    for nm in ORACLES:
        ok = [r[nm] for r in rows if r[nm].get("ok")]
        n_ok = len(ok); exact = sum(int(x["exact"]) for x in ok)
        rec = [x["cross_recall"] for x in ok if x["cross_recall"] is not None]
        summ["per_oracle"][nm] = {"n_folded": n_ok, "exact": exact,
                                  "exact_rate": round(exact / n_ok, 4) if n_ok else None,
                                  "false_accept_rate": round(1 - exact / n_ok, 4) if n_ok else None,
                                  "mean_cross_recall": round(sum(rec) / len(rec), 4) if rec else None}
    json.dump({"summary": summ, "rows": rows}, open(os.path.join(OUT, "fixedcand.json"), "w"), indent=1)
    print("\n=== FIXED-CANDIDATE DesiRNA cross-oracle (one design/target) ===", flush=True)
    for nm, d in summ["per_oracle"].items():
        print(f"  {nm:10} exact={d['exact']}/{d['n_folded']} ({100*(d['exact_rate'] or 0):.1f}%)  "
              f"false-accept={100*(d['false_accept_rate'] or 0):.1f}%  cross_recall={d['mean_cross_recall']}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=32)
    run(workers=ap.parse_args().workers)
