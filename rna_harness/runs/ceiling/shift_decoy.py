#!/usr/bin/env python3
"""The sharpest negative control available: a REAL pseudoknot, in the WRONG REGISTER.

WHAT THE EXISTING DECOY DOES NOT TEST. `nested` decoys are designed for the target's pseudoknot-free
skeleton, so they have no crossing stem at all, and the shipped rule rejects them 366 times out of 367. That
establishes "we do not confirm a sequence with NO knot". It leaves the harder question untouched, and it is
the question the verdict's own name makes:

    does `knot_confirmed` confirm THE knot, or merely A knot in roughly the right place?

A referee asks it in one line and nothing measured so far answers it. The audit of extra crossing pairs
(47 of 56 delivered designs carry none) is an after-the-fact statistic on sequences that passed; it is not an
adversarial sample built to break the rule.

THE CONSTRUCT. Take the target's crossing stem and slide it by 1 or 2 nucleotides -- both halves, so the stem
keeps its length -- giving a DIFFERENT but perfectly valid pseudoknot. Then design for THAT with the same
solver, at the same budget, screened by the same machinery. The only difference between a positive and one of
these is which structure the solver was pointed at.

A REGISTER SHIFT IS THE HARD CASE, and the reason to prefer it over a large displacement. Slide a 5 bp helix
by one base and the sequence is still nearly complementary in the original register, so a predictor could
plausibly draw either. A big shift would produce a sequence with no complementarity at the original positions
at all, the rule would reject it trivially, and the test would prove nothing. Biologically this is also the
mistake that matters: a helix slipped by one pair is a different structure made of almost the same sequence.

ONLY DECOYS THAT ACTUALLY FORM THEIR OWN SHIFTED KNOT ARE COUNTED. A design that simply failed is not a
decoy, it is a failed design, and including it would let a vacuous test look like a pass. So each candidate is
first judged BY OUR OWN RULE against the SHIFTED target; only those that pass become part of the null, and
only then are they judged against the ORIGINAL target.

DECLARED BEFORE THE RUN:
  * acceptance below 5% -> the verdict is position-specific and the name `knot_confirmed` is earned
  * acceptance above 20% -> the rule confirms "a knot near here", the name is wrong, and the operating point
    published for it does not describe the population it actually faces
  * in between -> report the number and stop calling the rule position-specific without qualification

The count of targets where no shifted variant is constructible, and where the solver could not build one that
forms the shifted knot, are both reported. A null whose members were quietly dropped is not a null.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)

SHIFTS = (1, 2)


def _try_shift(target: str, k: int):
    """One signed attempt: slide the crossing stem by exactly k nt."""
    from rna_harness.benchmarks.rnainvbench import _round_only, pk_pairs
    full, nested = pk_pairs(target), pk_pairs(_round_only(target))
    cross = full - nested
    if not cross:
        return None
    n = len(target)
    moved = {(i + k, j + k) for (i, j) in cross}
    if any(not (0 <= a < b < n) for a, b in moved):
        return None
    if moved & full:                                   # must be a genuinely different set of pairs
        return None
    keep = nested | moved
    # every position may take part in at most one pair, and the moved stem must still CROSS the nested one
    seen = set()
    for a, b in keep:
        if a in seen or b in seen:
            return None
        seen.add(a)
        seen.add(b)
    if not any(a < c < b < d or c < a < d < b for (a, b) in nested for (c, d) in moved):
        return None
    out = ["."] * n
    for a, b in nested:
        out[a], out[b] = "(", ")"
    for a, b in moved:
        out[a], out[b] = "[", "]"
    return "".join(out)


def shifted_target(target: str, k: int):
    """Slide the crossing stem by k nt in EITHER direction, keeping its length.

    Both halves move together so the stem stays the same length and stays a stem; only WHERE it sits changes,
    which is the distinction under test. BOTH SIGNS ARE TRIED because one of them usually has nowhere to go:
    in these targets the crossing stem's 5' side sits flush against the nested stem, so a +1 slide walks
    straight into it and the first version of this function returned None on 12 of 12 sampled targets. That
    was a construction failure, not a property of the structures, and had it gone unnoticed the benchmark
    would have reported an empty null as though it were a clean pass.
    """
    for signed in (k, -k):
        got = _try_shift(target, signed)
        if got is not None:
            return got
    return None


def work(job):
    tid, target, k, timelimit, seed = job
    from rna_harness.cognitive.certify import _knot_recall
    from rna_harness.generators.desirna import DesiRNAGenerator
    from rna_harness.task import DesignTask, OracleConfig
    shifted = shifted_target(target, k)
    if shifted is None:
        return {"task_id": tid, "shift": k, "constructible": False}
    t0 = time.time()
    oc = OracleConfig(param_set="turner2004", temperature=37.0, dangles=2)
    try:
        res = DesiRNAGenerator(replicas=10, timelimit_s=timelimit, results_number=50).design(
            DesignTask(task_id="shift", target=shifted, oracle=oc), seed=seed)
    except Exception as e:                                                    # noqa: BLE001
        return {"task_id": tid, "shift": k, "constructible": True, "error": repr(e)[:160]}
    pool, seen = [], set()
    for s in ([res.best_sequence] if res.best_sequence else []) + list(res.candidates or []):
        if s and len(s) == len(target) and s not in seen:
            seen.add(s)
            pool.append(s)
    # A DECOY MUST FIRST BE A REAL PSEUDOKNOT IN THE WRONG PLACE. Judge against the SHIFTED target by our own
    # rule; a candidate that fails that is a failed design, not a decoy, and must not enter the null.
    decoy = None
    for s in pool[:25]:
        if _knot_recall(s, shifted).get("knot_confirmed"):
            decoy = s
            break
    if decoy is None:
        return {"task_id": tid, "shift": k, "constructible": True, "decoy_built": False,
                "n_pool": len(pool), "wall_s": round(time.time() - t0, 1)}
    # now the actual question: judged against the ORIGINAL target, does our rule accept it?
    kr = _knot_recall(decoy, target)
    return {"task_id": tid, "shift": k, "constructible": True, "decoy_built": True,
            "sequence": decoy, "shifted_target": shifted,
            "accepted_against_original": bool(kr.get("knot_confirmed")),
            "recall_against_original": max([v for v in kr["per_oracle"].values() if v is not None] or [0.0]),
            "confirmed_by": kr.get("confirmed_by"), "wall_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timelimit", type=int, default=60)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--group", default="kissing / chain")
    ap.add_argument("--out", default=os.path.join(HERE, "shift_decoy.json"))
    a = ap.parse_args()

    top = {r["task_id"]: r["topology"] for r in
           json.load(open(os.path.join(HERE, "pk_topology.json")))["rows"]}
    pk = {r["task_id"]: r for r in
          json.load(open(os.path.join(HERE, "agent", "pk_desirna.json")))["rows"]}
    ids = [t for t in top if top[t] == a.group and t in pk]
    jobs = [(t, pk[t]["target"], k, a.timelimit, 0) for k in SHIFTS for t in ids]
    print(f"移位诱饵:{len(ids)} 条 {a.group} 靶标 × 位移 {SHIFTS} nt,"
          f"每条用同一求解器 {a.timelimit}s 设计\n", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    R, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, f in enumerate(as_completed([ex.submit(work, j) for j in jobs]), 1):
            R.append(f.result())
            if n % 20 == 0 or n == len(jobs):
                print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)

    from scipy.stats import beta
    print(f"\n=== 结果 ===")
    out = {"shifts": list(SHIFTS), "group": a.group, "by_shift": {}}
    for k in SHIFTS:
        S = [r for r in R if r["shift"] == k]
        nc = sum(1 for r in S if not r.get("constructible"))
        nb = sum(1 for r in S if r.get("constructible") and not r.get("decoy_built")
                 and not r.get("error"))
        valid = [r for r in S if r.get("decoy_built")]
        acc = sum(1 for r in valid if r["accepted_against_original"])
        lo = 0.0 if acc == 0 else float(beta.ppf(0.025, acc, len(valid) - acc + 1))
        hi = 1.0 if acc == len(valid) else float(beta.ppf(0.975, acc + 1, len(valid) - acc))
        out["by_shift"][k] = {"n_targets": len(S), "not_constructible": nc, "decoy_not_built": nb,
                              "n_valid_decoys": len(valid), "accepted": acc,
                              "lo95": lo, "hi95": hi}
        print(f"  位移 {k} nt:  可构造 {len(S)-nc}/{len(S)},其中求解器真的做出移位假结 {len(valid)}")
        print(f"            这些【真假结、错位置】的序列,拿【原靶标】判 -> 接受 "
              f"{acc}/{len(valid)} = {acc/max(1,len(valid)):.1%} [{lo:.1%}, {hi:.1%}]")
        if valid:
            rs = sorted(r["recall_against_original"] for r in valid)
            print(f"            对原靶标交叉对的最好复现率:中位 {rs[len(rs)//2]:.2f}  最大 {max(rs):.2f}")
    tot_v = sum(o["n_valid_decoys"] for o in out["by_shift"].values())
    tot_a = sum(o["accepted"] for o in out["by_shift"].values())
    lo = 0.0 if tot_a == 0 else float(beta.ppf(0.025, tot_a, tot_v - tot_a + 1))
    hi = 1.0 if tot_a == tot_v else float(beta.ppf(0.975, tot_a + 1, tot_v - tot_a))
    print(f"\n  合计:{tot_a}/{tot_v} = {tot_a/max(1,tot_v):.1%} [{lo:.1%}, {hi:.1%}]")
    print(f"  预注册判据: <5% -> 判据是位置特异的,`knot_confirmed` 这个名字站得住;"
          f" >20% -> 名字是错的")
    out["total"] = {"accepted": tot_a, "n": tot_v, "lo95": lo, "hi95": hi}
    out["rows"] = R
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
