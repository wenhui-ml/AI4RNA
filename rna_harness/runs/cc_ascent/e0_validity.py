#!/usr/bin/env python3
"""E0: verifier-validity control (chomsky_climb). Does the panel actually discriminate a crossing design
from a pseudoknot-FREE (CF-skeleton) design, or would it over-accept skeletons as pseudoknots?

For each held-out target T we design its CF skeleton (round brackets only) with REMC, then fold that
skeleton sequence under the full panel and ask whether the panel reports the FULL crossing target T. A valid
(non-over-accepting) panel should almost never say yes: the skeleton has no crossing stem. Contrast the
naive-MFE oracle, which credits 80/80 such skeletons (Sec. 5, the false-positive control). Expected: panel
crossing-recall on skeletons ~ 0 -> the panel is a sound instrument for the positive side.
"""
import json, os, sys, time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from rna_harness.experiments.track_b_pk import build_policies, _round_only
from rna_harness.benchmarks.rnainvbench import load_pseudobase
from rna_harness.oracles_pk import (forms_target, pkiss_pairs, probknot_pairs,
                                     shapeknots_pairs, rnastructure_available)

HERE = os.path.dirname(os.path.abspath(__file__))


def main(limit=60, seed=0):
    rnastr = rnastructure_available()
    tasks = load_pseudobase(limit=limit)
    remc = build_policies(["always_cheap"])["always_cheap"]
    print(f"E0 verifier-validity: {len(tasks)} skeletons; rnastr={rnastr}", flush=True)
    rows, n_skel, n_panel_fp, n_nested_ok = [], 0, 0, 0
    for t in tasks:
        skel = _round_only(t.target)
        t0 = time.time()
        try:
            r = remc.solve(skel, seed=seed)           # design a sequence realizing the CF skeleton
            seq = r.get("seq")
            if not seq:
                rows.append({"id": t.task_id, "skeleton_designed": False}); continue
            n_skel += 1
            # panel verdict on the skeleton sequence vs the FULL crossing target (should be NO)
            pk = bool(forms_target(pkiss_pairs(seq), t.target))
            pb = bool(forms_target(probknot_pairs(seq), t.target)) if rnastr else None
            sk = bool(forms_target(shapeknots_pairs(seq), t.target)) if rnastr else None
            any_pk = pk or bool(pb) or bool(sk)       # any panel member crediting the crossing = a panel FP
            unan = (pk and bool(pb) and bool(sk)) if rnastr else None
            n_panel_fp += int(any_pk)
            # sanity: panel recovers the NESTED skeleton itself (soundness on nested)
            nested_ok = bool(forms_target(pkiss_pairs(seq), skel))
            n_nested_ok += int(nested_ok)
            rows.append({"id": t.task_id, "skeleton_designed": True, "vs_full_target": {"pkiss": pk, "probknot": pb,
                         "shapeknots": sk, "any_member": any_pk, "unanimous": unan}, "nested_recovered": nested_ok,
                         "s": round(time.time() - t0, 1)})
            print(f"  {t.task_id:14} skel->full: any_member={any_pk} unan={unan}  nested_ok={nested_ok}", flush=True)
        except Exception as e:
            rows.append({"id": t.task_id, "error": repr(e)[:200]})
        json.dump({"n_skeletons": n_skel, "panel_any_member_FP": n_panel_fp, "panel_unanimous_FP":
                   sum(1 for r in rows if (r.get("vs_full_target") or {}).get("unanimous")),
                   "nested_recovered": n_nested_ok, "rows": rows},
                  open(os.path.join(HERE, "e0_validity.json"), "w"), indent=1)
    print(f"\nE0 summary: skeletons designed={n_skel}; panel credits the CROSSING on {n_panel_fp}/{n_skel} "
          f"(any member), {sum(1 for r in rows if (r.get('vs_full_target') or {}).get('unanimous'))}/{n_skel} "
          f"panel-unanimous; nested skeleton recovered {n_nested_ok}/{n_skel}. "
          f"(Naive-MFE credits 80/80 -- Sec. 5.)", flush=True)


if __name__ == "__main__":
    main(limit=int(sys.argv[1]) if len(sys.argv) > 1 else 60)
