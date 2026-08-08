#!/usr/bin/env python3
"""Stage A: run OUR product on the 57 OpenKnot targets, so that for the first time we can be scored beside
the methods that were actually synthesised.

THE QUESTION THIS FINALLY MAKES ANSWERABLE. Yesterday's report recorded that OpenKnot lists thirteen design
methods with wet-lab scores and that DesiRNA -- our entire design capability -- does not appear on that
table, so we could not place ourselves on it. That was true as long as the only way onto the table was a
laboratory. It stopped being true today: RibonanzaNet-SS now runs locally and reproduces the published
`RNet_structure` exactly (median base-pair F1 1.000 over 300 designs, crossed-pair Spearman +0.988), and the
same model's prediction is the strongest wet-lab proxy we have measured (+0.364 within-target against the
tube, against our own panel's +0.155). So we can score our own designs on the same instrument that scores
theirs.

THE ISOLATION RULE, stated before anything runs and non-negotiable. RibonanzaNet is the JUDGE here and must
never enter the design loop. We have already paid for breaking this once: an operator that used pKiss
in-loop reported 43 of 60 solved and the independent panel confirmed 1 of 60 -- a 43x inflation produced
entirely by optimising against the thing that would grade it. Our pipeline in this run uses only its own
thermodynamic panel, exactly as it ships. Stage B scores the output. Nothing flows backwards.

    Corollary that must be reported and not buried: Struct2SeQ-SHAPE OPTIMISES against RibonanzaNet. Scoring
    it with RibonanzaNet grades it on its own objective. Every comparison against that arm is therefore
    reported separately from the rest, and a loss to it means less than a loss to Eterna or Rosetta.

WHAT STAGE A DOES. One target per puzzle (57 of them, 48-240 nt), our shipped `certify_design` at the
`standard` budget -- the ladder a real user actually gets, (5, 15, 60) seconds -- and the sequence it
returns is what we will be judged on. `best_effort` would flatter us and is not what the product hands back,
so it is run as a declared secondary on the targets `standard` fails, and never mixed into the primary.

    A target we return `not_found` for counts as a LOSS in stage B, not as missing data. Dropping our own
    failures would be the exact move that makes a leaderboard meaningless.

COST: $0 API. DesiRNA and the panel are local binaries.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)

DATA = "/data/wenhui/openknot_data/Data/OpenKnotBench_data.v4.5.1.csv"


def targets():
    """One target structure per puzzle -- the MODAL one, not the first row's.

    Six of the 57 puzzles carry more than one target string, and four of those differ in LENGTH: W01
    (85/88), W06 (98/100), W10 (98/100), W14 (95/97), plus W09 and P06 which the v4.5.0 notes describe as
    two slightly different structures released at different times. Taking the first row gave a minority
    variant for P06 (100 of 449 designs) and for W06 (10 of 1046) -- we would have designed against a
    target almost none of the real designs were scored on. The modal string is what the field actually
    competed against, and stage B additionally restricts each comparison pool to designs carrying the same
    string, so the head-to-head is exact rather than approximate.
    """
    csv.field_size_limit(10 ** 9)
    import collections
    votes, meta = collections.defaultdict(collections.Counter), {}
    with open(DATA, newline="") as fh:
        r = csv.reader(fh)
        idx = {c: i for i, c in enumerate(next(r))
               if c in ("puzzle", "round", "target_structure", "design_sequence", "SN_filter")}
        for x in r:
            if x[idx["SN_filter"]] != "1":
                continue
            p, t = x[idx["puzzle"]], x[idx["target_structure"]]
            if not t:
                continue
            votes[p][t] += 1
            meta.setdefault(p, x[idx["round"]])
    out = {}
    for p, c in votes.items():
        t, n = c.most_common(1)[0]
        out[p] = {"puzzle": p, "round": meta[p], "target": t, "n": len(t),
                  "modal_share": n / sum(c.values()), "n_variants": len(c)}
    return out


def work(item):
    pid, target, budget, seed = item
    from rna_harness.cognitive.certify import certify_design
    t0 = time.time()
    try:
        c = certify_design(target, budget=budget, seed=seed)
    except Exception as e:                                                     # noqa: BLE001
        return {"puzzle": pid, "budget": budget, "error": repr(e)[:200]}
    return {"puzzle": pid, "budget": budget, "seed": seed, "verdict": c.verdict,
            "sequence": c.sequence, "n": c.n, "strictness": c.strictness,
            "checked_by": c.checked_by, "cost_usd": c.cost_usd, "wall_s": round(time.time() - t0, 1),
            "n_screened": c.detail.get("n_candidates_screened"),
            "reason": c.reason[:160]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", default="standard")
    ap.add_argument("--rescue", action="store_true",
                    help="declared secondary: re-run the standard failures at best_effort")
    ap.add_argument("--only", default="", help="comma-separated puzzles; used to re-run the ones whose "
                    "target string was fixed after the audit, without repeating the other 55")
    ap.add_argument("--merge-into", default="", help="existing leaderboard_ours.json to update in place")
    ap.add_argument("--out", default=os.path.join(HERE, "leaderboard_ours.json"))
    a = ap.parse_args()

    # The first run of this script returned not_found on all 57 targets in 8.8 s each -- a ladder of
    # (5, 15, 60) seconds cannot finish in 8.8 s, and the reason was that DESIRNA_CMD was unset so the
    # solver never ran. The product reported "no sequence survived the criterion" rather than "the solver
    # was not available", which is a real defect in its own right: a caller would read an unavailable tool
    # as a hard target. Asserted here so a missing solver can never again be published as a capability
    # measurement.
    cmd = os.environ.get("DESIRNA_CMD")
    assert cmd and os.path.exists(cmd.split()[-1]), (
        "DESIRNA_CMD is unset or points nowhere; without it every target silently returns not_found and "
        "this run would measure our environment rather than our capability")
    print(f"求解器:{cmd}")

    T = targets()
    print(f"OpenKnot 靶标 {len(T)} 个,长度 {min(v['n'] for v in T.values())}-"
          f"{max(v['n'] for v in T.values())} nt")
    print(f"用产品出厂预算 `{a.budget}`(用户真正拿到的那条阶梯),seed={a.seed}\n", flush=True)

    if a.only:
        keep = {x.strip() for x in a.only.split(",") if x.strip()}
        missing = keep - set(T)
        assert not missing, f"unknown puzzles {missing}"
        T = {p: v for p, v in T.items() if p in keep}
        print(f"只跑 {sorted(T)}(审计后靶标串已改为众数,其余 55 个不受影响,不重跑)")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    jobs = [(p, v["target"], a.budget, a.seed) for p, v in sorted(T.items())]
    R, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, f in enumerate(as_completed([ex.submit(work, j) for j in jobs]), 1):
            R.append(f.result())
            if n % 5 == 0 or n == len(jobs):
                print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)

    ok = [r for r in R if r.get("sequence")]
    verd = {}
    for r in R:
        verd[r.get("verdict", "error")] = verd.get(r.get("verdict", "error"), 0) + 1
    print(f"\n=== 我们的交付 ===")
    for k, v in sorted(verd.items(), key=lambda kv: -kv[1]):
        print(f"  {k:18s} {v:3d}")
    print(f"  拿到序列的靶标 {len(ok)}/{len(R)}")
    if ok:
        print(f"  中位耗时 {sorted(r['wall_s'] for r in ok)[len(ok)//2]:.1f}s,"
              f"总成本 ${sum(r['cost_usd'] for r in R):.4f}")

    failed = [r["puzzle"] for r in R if not r.get("sequence")]
    if failed:
        print(f"  未拿到序列:{sorted(failed)}")
        print(f"  -> 这些在打分阶段【计为失败】,不是缺失数据")

    if a.rescue and failed:
        print(f"\n=== 声明为次要:对 {len(failed)} 个失败靶标再跑 best_effort ===", flush=True)
        jobs2 = [(p, T[p]["target"], "best_effort", a.seed) for p in failed]
        R2 = []
        with ProcessPoolExecutor(max_workers=min(a.workers, len(jobs2))) as ex:
            for n, f in enumerate(as_completed([ex.submit(work, j) for j in jobs2]), 1):
                R2.append(f.result())
                print(f"    {n}/{len(jobs2)}", flush=True)
        got = [r for r in R2 if r.get("sequence")]
        print(f"  best_effort 救回 {len(got)}/{len(jobs2)}")
        R += R2

    if a.merge_into:
        # Replace only the re-run puzzles, and replace their TARGET too -- the whole point of the re-run is
        # that the old entry designed against a minority variant, so keeping the old target would leave the
        # scorer comparing the new sequence against the old structure.
        base = json.load(open(a.merge_into))
        base["rows"] = [r for r in base["rows"] if r["puzzle"] not in T] + R
        base["targets"].update(T)
        base.setdefault("amendments", []).append(
            {"what": "re-ran with the modal target string", "puzzles": sorted(T)})
        json.dump(base, open(a.merge_into, "w"), ensure_ascii=False, indent=1)
        print(f"\n  merged {sorted(T)} into {a.merge_into}")
        return

    json.dump({"budget": a.budget, "seed": a.seed, "targets": T, "rows": R},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}   (下一步:用 rnet venv 跑 leaderboard_score.py)")


if __name__ == "__main__":
    main()
