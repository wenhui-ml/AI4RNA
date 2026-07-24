#!/usr/bin/env python3
"""Panel calibration on NATIVE pseudoknots (reviewer concern 1a). The design benchmarks mask sequences, so we
pull native (sequence, pseudoknot-structure) pairs from bpRNA (multimolecule/bprna, HF) -- see calib_natives.json.
For each native we fold its OWN sequence under each panel member and ask whether the member recovers the
native's crossing pair set (crossing RECALL / sensitivity), and whether the panel is unanimous. This quantifies
the precision--recall trade-off of the unanimity rule: high specificity (E0: 0/60 on skeletons) but recall < 1,
and it shows how much unanimity costs over single predictors. No design here -- pure verifier calibration."""
import json, os, sys, time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from rna_harness.oracles_pk import (pkiss_pairs, probknot_pairs, shapeknots_pairs, rnastructure_available)

HERE = os.path.dirname(os.path.abspath(__file__))
_OPEN = "([{<A"; _CLOSE = ")]}>a"; _P = dict(zip(_CLOSE, _OPEN))


def to_pairs(s):
    st = {c: [] for c in _OPEN}; pr = []
    for i, c in enumerate(s):
        if c in _OPEN: st[c].append(i)
        elif c in _CLOSE:
            o = _P[c]
            if not st[o]: return None
            pr.append((st[o].pop(), i))
    return None if any(st[c] for c in _OPEN) else pr


def crosses(a, b):
    (i, j), (k, l) = a, b; return (i < k < j < l) or (k < i < l < j)


def crossing_pairs(pr):
    return set(pr[a] for a in range(len(pr)) for b in range(len(pr)) if a != b and crosses(pr[a], pr[b]))


def recovers_crossing(pred_pairs, native_cross):
    pset = set(tuple(sorted(p)) for p in pred_pairs)
    return native_cross.issubset(pset)


def main():
    rnastr = rnastructure_available()
    natives = json.load(open(os.path.join(HERE, "calib_natives.json")))["rows"]
    print(f"panel calibration on {len(natives)} native pseudoknots; rnastr={rnastr}", flush=True)
    rows = []
    agg = {"pkiss": 0, "probknot": 0, "shapeknots": 0, "panel_all3": 0, "n": 0}
    for k, nt in enumerate(natives):
        seq, ss = nt["seq"], nt["str"]
        native_cross = set(tuple(sorted(p)) for p in crossing_pairs(to_pairs(ss)))
        if not native_cross:
            continue
        t0 = time.time()
        try:
            pk = recovers_crossing(pkiss_pairs(seq), native_cross)
            pb = recovers_crossing(probknot_pairs(seq), native_cross) if rnastr else None
            sk = recovers_crossing(shapeknots_pairs(seq), native_cross) if rnastr else None
            allp = bool(pk) and bool(pb) and bool(sk) if rnastr else None
            agg["n"] += 1; agg["pkiss"] += int(bool(pk))
            if rnastr:
                agg["probknot"] += int(bool(pb)); agg["shapeknots"] += int(bool(sk)); agg["panel_all3"] += int(bool(allp))
            rows.append({"id": nt["id"], "len": nt["len"], "n_cross": len(native_cross),
                         "pkiss": bool(pk), "probknot": None if pb is None else bool(pb),
                         "shapeknots": None if sk is None else bool(sk), "panel_all3": allp,
                         "s": round(time.time() - t0, 1)})
            if k % 10 == 0:
                print(f"  [{agg['n']}] {nt['id']:14} len={nt['len']} pk={pk} pb={pb} sk={sk} all3={allp}", flush=True)
        except Exception as e:
            rows.append({"id": nt["id"], "error": repr(e)[:120]})
        json.dump({"n": agg["n"], "recall": agg, "rows": rows},
                  open(os.path.join(HERE, "native_calib.json"), "w"), indent=1)
    n = agg["n"] or 1
    print(f"\n=== NATIVE crossing RECALL (sensitivity), n={agg['n']} ===", flush=True)
    print(f"  pKiss       {agg['pkiss']}/{agg['n']} = {agg['pkiss']/n:.2f}", flush=True)
    print(f"  ProbKnot    {agg['probknot']}/{agg['n']} = {agg['probknot']/n:.2f}", flush=True)
    print(f"  ShapeKnots  {agg['shapeknots']}/{agg['n']} = {agg['shapeknots']/n:.2f}", flush=True)
    print(f"  panel (all 3 unanimous) {agg['panel_all3']}/{agg['n']} = {agg['panel_all3']/n:.2f}", flush=True)
    print("  -> unanimity recall < min single-predictor recall = the precision/recall cost of the AND-rule.", flush=True)


if __name__ == "__main__":
    main()
