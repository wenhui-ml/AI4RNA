"""R3.3 (reviewer 3.3): the native-sequence POSITIVE CONTROL and pseudoknot-class breakdown.

The reviewer's decisive point: a predictor's exact-match FAILURE on a *designed* sequence is only cleanly a
design failure if the predictor can recover the target from its *native* (wild-type) sequence. If a predictor
cannot fold the native sequence to T, then T is effectively outside that predictor's reach and the "false
accept" on designs is confounded with representability. We operationalize representability empirically as
NATIVE RECOVERY.

This script:
  1. parses the wild-type Pseudobase++ sequences from RNAInvBench's antaRNA constraint files,
  2. matches each of the 251 Pseudobase targets to its native sequence by (length, pair-set) -- robust to
     bracket-symbol conventions,
  3. folds each native with pKiss / ProbKnot / ShapeKnots and reports, per predictor,
       native exact recovery of T  and  native crossing-pair recall,
  4. reports a pseudoknot-class histogram (number of crossing stems) of the 251 targets,
  5. joins the DesiRNA-design exact-recovery (from the existing panel) for the reviewer's comparison table.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..benchmarks.rnainvbench import load_pseudobase, _round_only, pk_pairs
from ..oracles_pk import ORACLES, rnastructure_available

NATIVE_FILES = [
    "/data/wenhui/RNAInvBench/rna_design_algorithms/antaRNA/Constraints/Pseudobase++",
    "/data/wenhui/RNAInvBench/rna_design_algorithms/antaRNA/Constraints/Pseudobase++_TEST",
    "/data/wenhui/RNAInvBench/rna_design_algorithms/antaRNA/Constraints/Pseudobase++_TRAIN",
]
DESIRNA_PANEL = os.path.join(os.path.dirname(__file__), "..", "runs", "axis1_oracle_panel", "panel.json")
OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "axis1_native_control")


def parse_natives():
    """Parse antaRNA constraint files (>name / native-seq / dot-bracket triples)."""
    entries = {}
    for path in NATIVE_FILES:
        if not os.path.exists(path):
            continue
        lines = [ln.rstrip("\n") for ln in open(path) if ln.strip()]
        i = 0
        while i + 2 < len(lines):
            if not lines[i].startswith(">"):
                i += 1
                continue
            name, seq, db = lines[i], lines[i + 1], lines[i + 2]
            i += 3
            seq = seq.strip().upper().replace("T", "U")
            db = db.strip()
            if len(seq) != len(db) or any(c not in "ACGU" for c in seq):
                continue
            key = (len(db), frozenset(pk_pairs(db)))
            entries.setdefault(key, (name, seq, db))    # first native wins on collision
    return entries


def _crossing(target):
    return pk_pairs(target) - pk_pairs(_round_only(target))


def _n_cross_stems(pairset):
    """Number of crossing 'stems' = maximal diagonally-contiguous runs of crossing pairs."""
    ps = sorted(pairset)
    stems, used = 0, set()
    for (i, j) in ps:
        if (i, j) in used:
            continue
        stems += 1
        a, b = i, j
        while (a + 1, b - 1) in pairset:                 # extend the stem
            used.add((a + 1, b - 1)); a, b = a + 1, b - 1
    return stems


def _pk_class(target):
    ns = _n_cross_stems(_crossing(target))
    if ns <= 0:
        return "nested"
    if ns == 1:
        return "H-type (1 crossing stem)"
    if ns == 2:
        return "kissing/2-stem"
    return "complex (>=3 crossing stems)"


def _fold_one(rec):
    seq, target = rec["seq"], rec["target"]
    allp = pk_pairs(target)
    cross = _crossing(target)
    out = {"id": rec["id"], "n": rec["n"], "n_cross": len(cross)}
    for name, fn in ORACLES.items():
        try:
            pairs = fn(seq)
        except Exception:
            pairs = None
        if pairs is None:
            out[name] = {"ok": False}
            continue
        out[name] = {"ok": True, "exact": bool(pairs == allp),
                     "cross_recall": (len(pairs & cross) / len(cross)) if cross else None,
                     "bp_dist": len(pairs ^ allp)}
    return out


def _designed_recovery():
    """Pull DesiRNA-design exact-recovery ('forms T') per oracle from the existing panel."""
    if not os.path.exists(DESIRNA_PANEL):
        return None
    try:
        p = json.load(open(DESIRNA_PANEL))
    except Exception:
        return None
    # be tolerant of schema; look for per-oracle forms_T / exact counts
    return p.get("summary", p)


def run(workers=32):
    if not rnastructure_available():
        raise SystemExit("RNAstructure unavailable -- set RNASTRUCTURE_BIN / DATAPATH (conda env 'rnastr')")
    os.makedirs(OUT, exist_ok=True)
    natives = parse_natives()
    tasks = load_pseudobase()
    print(f"parsed {len(natives)} distinct native structures; {len(tasks)} Pseudobase targets", flush=True)

    matched, unmatched = [], []
    classes = {}
    for t in tasks:
        cls = _pk_class(t.target)
        classes[cls] = classes.get(cls, 0) + 1
        key = (len(t.target), frozenset(pk_pairs(t.target)))
        hit = natives.get(key)
        if hit:
            matched.append({"id": t.task_id, "n": t.n, "target": t.target, "seq": hit[1], "native_name": hit[0]})
        else:
            unmatched.append(t.task_id)
    print(f"native match: {len(matched)}/{len(tasks)} targets have a wild-type sequence "
          f"({len(unmatched)} unmatched)", flush=True)
    print(f"pseudoknot-class histogram: {classes}", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fold_one, m): m["id"] for m in matched}
        for i, fut in enumerate(as_completed(futs)):
            rows.append(fut.result())
            if (i + 1) % 20 == 0 or i + 1 == len(matched):
                print(f"  folded native {i+1}/{len(matched)}", flush=True)
    rows.sort(key=lambda r: r["id"])

    summ = {"n_targets": len(tasks), "n_matched": len(matched), "pk_classes": classes, "per_oracle": {}}
    for nm in ORACLES:
        ok = [r[nm] for r in rows if r[nm].get("ok")]
        n_ok = len(ok)
        exact = sum(int(x["exact"]) for x in ok)
        recalls = [x["cross_recall"] for x in ok if x["cross_recall"] is not None]
        summ["per_oracle"][nm] = {
            "n_folded": n_ok,
            "native_exact": exact,
            "native_exact_rate": round(exact / n_ok, 4) if n_ok else None,
            "native_mean_cross_recall": round(sum(recalls) / len(recalls), 4) if recalls else None,
        }
    summ["designed_recovery_panel"] = _designed_recovery()
    json.dump({"summary": summ, "rows": rows, "unmatched": unmatched},
              open(os.path.join(OUT, "native_control.json"), "w"), indent=1)

    print("\n=== NATIVE POSITIVE CONTROL (wild-type recovery) ===", flush=True)
    for nm, d in summ["per_oracle"].items():
        print(f"  {nm:10} folded={d['n_folded']:>3}  native-exact={d['native_exact']:>3} "
              f"({100*(d['native_exact_rate'] or 0):.1f}%)  native mean crossing-recall={d['native_mean_cross_recall']}",
              flush=True)
    print(f"  pk-class histogram: {classes}", flush=True)
    return summ


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    run(workers=a.workers)
