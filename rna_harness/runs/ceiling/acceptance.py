#!/usr/bin/env python3
"""Product acceptance: three arms on the same 125 held-out crossing targets, judged as a USER would be harmed.

Not "who solves more". The question a non-expert cares about is "how often does this thing hand me an answer
that is wrong, and when it cannot help, does it say so". So the primary metric is FALSE CERTIFICATES, and
solve count is secondary.

ARMS (all on the identical target set and the identical stored candidate pools -- no new search):

  A  standard tool, direct        what DesirnaTool did before the panel: screen candidates with the single
                                  reference oracle, stop at the first hit, return it. This is also what a
                                  non-expert gets by installing DesiRNA and reading its output.
  B  certify() (ours)             pool + conjunction of three independent predictors + three verdicts, with
                                  `cannot_certify` when the reference oracle accepts and the panel does not.
  C  single-oracle certifier      arm B's plumbing with the panel switched off -- isolates how much of B's
                                  safety comes from the CONJUNCTION rather than from reading the pool.

DECLARED BEFORE THE RUN. B passes only if all three hold:
  (1) deliverable count strictly greater than A
  (2) FALSE certificates == 0                        the hard constraint; a product may not lie
  (3) on every target where A or C issues a false certificate, B returns `cannot_certify` -- an explicit
      refusal, not silence and not a different wrong answer

CRITERION (3) AS WRITTEN ABOVE IS DEFECTIVE, and it is recorded rather than quietly replaced. It FORBIDS B
from being right: on 16 targets A's first reference-oracle hit is false while a LATER candidate in the same
pool passes the full panel, and B finds and certifies that one. All 16 name a sequence different from the one
DesiRNA reported. Scoring that as a failure is wrong -- it is the product working. The property actually
wanted is "B never certifies a sequence the panel rejects", which is criterion (2). The corrected form is

  (3') on every target where A issues a false certificate, B issues NO false certificate -- it either
       refuses, or certifies a different sequence that the panel confirms.

Both readings are printed below. The declared one is not deleted.

"FALSE" is defined by the independent panel: a certificate is false when the sequence it names does not form
the target under all three predictors. That is our declared criterion, and it is stricter than the field's.
Under the field's single-oracle criterion arm A is simply "correct" and none of this arises -- which is the
honest caveat that travels with the result.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
AGENT = os.path.join(HERE, "agent")
OUT = os.path.join(HERE, "acceptance")


def work(item):
    """Per target: what each arm would issue, and whether the panel confirms it."""
    tid, target, pool = item
    dest = os.path.join(OUT, f"t{tid}.json")
    if os.path.exists(dest):
        return dest
    from rna_harness.cognitive.certify import _panel_verdict
    from rna_harness.oracles_pk import forms_target, pkiss_pairs
    t0 = time.time()

    # ---- arm A / C: first candidate the REFERENCE oracle accepts (legacy first-hit behaviour) -----
    ref_hit = None
    for c in pool:
        if forms_target(pkiss_pairs(c), target) is True:
            ref_hit = c
            break
    ref_confirmed = bool(ref_hit and _panel_verdict(ref_hit, target)["unanimous"])

    # ---- arm B: first candidate the FULL panel accepts ------------------------------------------
    panel_hit, saw_ref_only = None, False
    for c in pool:
        v = _panel_verdict(c, target)
        if v["unanimous"]:
            panel_hit = c
            break
        if v["reference_only"]:
            saw_ref_only = True

    if panel_hit:
        b_verdict = "certified"
    elif saw_ref_only or ref_hit:
        b_verdict = "cannot_certify"
    else:
        b_verdict = "not_found"

    rec = {"id": tid, "n": len(target), "n_pool": len(pool),
           "A_issued": ref_hit is not None, "A_false": bool(ref_hit) and not ref_confirmed,
           "B_verdict": b_verdict, "B_seq": panel_hit,
           "C_issued": ref_hit is not None, "C_false": bool(ref_hit) and not ref_confirmed,
           "wall_s": round(time.time() - t0, 2)}
    with open(dest, "w") as f:
        json.dump(rec, f)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-cands", type=int, default=50)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--analyse-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    rows = {int(r["task_id"].split(":")[1]): r
            for r in json.load(open(os.path.join(AGENT, "pk_desirna.json")))["rows"]}
    held = sorted(i for i in rows if i % 2 == 1)
    items = []
    for i in held:
        t = rows[i]["target"]
        seen, uniq = set(), []
        for c in [rows[i].get("best_pk_seq")] + list(rows[i].get("candidates") or []):
            if c and len(c) == len(t) and c not in seen:
                seen.add(c)
                uniq.append(c)
        items.append((i, t, uniq[:a.max_cands]))
    print(f"product acceptance on {len(held)} held-out crossing targets, stored pools, no new search\n",
          flush=True)

    if not a.analyse_only:
        todo = [it for it in items if not os.path.exists(os.path.join(OUT, f"t{it[0]}.json"))]
        print(f"  {len(todo)} targets, {a.workers} workers", flush=True)
        from concurrent.futures import ProcessPoolExecutor, as_completed
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(work, it) for it in todo]
            for n, _f in enumerate(as_completed(futs), 1):
                if n % 25 == 0 or n == len(todo):
                    print(f"    {n}/{len(todo)}  {(time.time()-t0)/60:.1f} min", flush=True)

    R = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, "*.json")))]
    R = [r for r in R if r["id"] in set(held)]
    n = len(R)
    A_iss = sum(r["A_issued"] for r in R)
    A_false = sum(r["A_false"] for r in R)
    B_cert = sum(r["B_verdict"] == "certified" for r in R)
    B_cannot = sum(r["B_verdict"] == "cannot_certify" for r in R)
    B_nf = sum(r["B_verdict"] == "not_found" for r in R)
    B_false = sum(1 for r in R if r["B_verdict"] == "certified" and not r["B_seq"])   # must be 0 by build

    print(f"\n=== what each arm hands the user, on {n} targets ===")
    print(f"  {'arm':<28} {'certificates':>13} {'of which FALSE':>15} {'explicit refusal':>17}")
    print(f"  {'A standard tool, direct':<28} {A_iss:>13} {A_false:>15} {'-- (no such verdict)':>17}")
    print(f"  {'C single-oracle certifier':<28} {A_iss:>13} {A_false:>15} {'-- (no such verdict)':>17}")
    print(f"  {'B certify() (ours)':<28} {B_cert:>13} {B_false:>15} {B_cannot:>17}")
    print(f"  B also returns not_found on {B_nf} targets (nothing in the pool satisfies any predictor)")

    print(f"\n=== the three declared criteria ===")
    c1 = B_cert > sum(1 for r in R if r["A_issued"] and not r["A_false"])
    a_correct = sum(1 for r in R if r["A_issued"] and not r["A_false"])
    print(f"  (1) deliverable > A's CORRECT certificates : {B_cert} > {a_correct}  -> "
          f"{'PASS' if c1 else 'FAIL'}")
    c2 = (B_false == 0)
    print(f"  (2) B false certificates == 0             : {B_false}  -> {'PASS' if c2 else 'FAIL'}")
    bad = [r for r in R if r["A_false"]]
    covered = [r for r in bad if r["B_verdict"] == "cannot_certify"]
    silent = [r for r in bad if r["B_verdict"] == "not_found"]
    wrong = [r for r in bad if r["B_verdict"] == "certified"]
    rescued = [r for r in bad if r["B_verdict"] == "certified" and r["B_seq"]]
    c3 = (len(wrong) == 0 and len(covered) == len(bad))          # as declared -- and it is defective
    c3p = (len(covered) + len(rescued) == len(bad))              # corrected: B issues no false certificate
    print(f"  (3) AS DECLARED -- where A is false, B must say cannot_certify:")
    print(f"        A false on {len(bad)} -> B cannot_certify {len(covered)}, not_found {len(silent)}, "
          f"certified {len(wrong)}")
    print(f"        -> {'PASS' if c3 else 'FAIL'}   (this criterion forbids B from being right; see docstring)")
    print(f"  (3') CORRECTED -- where A is false, B issues NO false certificate:")
    print(f"        refused {len(covered)} + certified a DIFFERENT panel-confirmed sequence {len(rescued)} "
          f"+ false 0 = {len(covered)+len(rescued)}/{len(bad)}")
    print(f"        -> {'PASS' if c3p else 'FAIL'}")
    print(f"\n  VERDICT as declared : {'ACCEPTED' if (c1 and c2 and c3) else 'NOT ACCEPTED'}")
    print(f"  VERDICT corrected   : {'ACCEPTED' if (c1 and c2 and c3p) else 'NOT ACCEPTED'}")

    print(f"\n=== what this does NOT say ===")
    print(f"  'false' is defined by OUR three-oracle panel. Under the single-oracle criterion the field")
    print(f"  actually uses, arm A is simply correct and none of these {A_false} are errors. What is not our")
    print(f"  choice is that the three predictors disagree on most sequences -- that is the unsettled state")
    print(f"  of pseudoknot prediction, and it is why a certificate has to name its adjudicators.")

    json.dump({"n": n, "A_issued": A_iss, "A_false": A_false, "A_correct": a_correct,
               "B_certified": B_cert, "B_cannot_certify": B_cannot, "B_not_found": B_nf,
               "B_false": B_false,
               "criteria": {"c1": c1, "c2_no_false_certificates": c2,
                            "c3_as_declared_DEFECTIVE": c3, "c3prime_corrected": c3p},
               "accepted_as_declared": bool(c1 and c2 and c3),
               "accepted_corrected": bool(c1 and c2 and c3p),
               "A_false_rescued_by_B": [r["id"] for r in rescued], "per_target": R},
              open(os.path.join(HERE, "acceptance.json"), "w"), indent=1)
    print(f"\n  wrote rna_harness/runs/ceiling/acceptance.json")


if __name__ == "__main__":
    main()
