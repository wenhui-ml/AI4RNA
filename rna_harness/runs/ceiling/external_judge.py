#!/usr/bin/env python3
"""A FOURTH judge that was never in the loop. The one remaining charge is self-validation.

THE CHARGE, stated as a referee would. The confirmation rule is built from pKiss, ProbKnot and ShapeKnots,
and every number defending it was computed with those same three. Two of them agree at Cohen's kappa 0.74, so
the panel is nearer two voices than three. Nothing so far rules out the reading that `knot_confirmed` selects
sequences THIS PANEL happens to like.

The only thing that answers it is a predictor that took no part in the rule, was not consulted during
screening, and does not appear anywhere in the pipeline. RNAPKplex qualifies: a different algorithm (an
RNAplex-style interior-loop approximation over pseudoknot candidates) in a different codebase, run here from
a separate environment so it cannot be reached by accident from the shipped path.

WHY THE OBVIOUS COMPARISON IS WRONG, and this is the whole design. "What fraction of our accepted designs
does the fourth judge confirm?" has no interpretation on its own, because the fourth judge has its OWN
sensitivity -- if it recognises only 40% of pseudoknots at all, then 40% on our set is a perfect score and
80% would be impossible. Every predictor in this field misses most of nature; that was the finding that
started this whole line of work.

So the reference is the judge's own rate on GROUND TRUTH, measured in the same run:

    S_ext   the fourth judge's rate on NATIVE sequences        its ceiling, whatever that turns out to be
    X_ours  the same judge's rate on OUR ACCEPTED designs      the quantity under test
    X_dec   the same judge on NESTED DECOYS                    designs our rule rejected, folding the
                                                               pseudoknot-free skeleton by construction

    X_ours ~= S_ext  and  X_dec << S_ext   ->  an outsider sees our accepted set as true positives and our
                                               rejected set as negatives. That is external validation.
    X_ours << S_ext                        ->  our set is enriched in what OUR panel likes and an
                                               independent judge does not. The rule is panel-specific and
                                               the certificate has to say so.
    X_dec ~= X_ours                        ->  the fourth judge cannot separate them either, which says the
                                               task is beyond all current instruments rather than that we
                                               are right.

DECLARED BEFORE THE RUN: the criterion is X_ours within the 95% interval of S_ext, and X_dec strictly below
it. Anything else is reported as it comes, including the outcome where the fourth judge simply disagrees
with us -- which is a real possibility and the reason this is worth running rather than assuming.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)

PKPLEX = os.environ.get("RNAPKPLEX", "/home/ubuntu/miniconda3/envs/vpk/bin/RNAPKplex")


def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    from scipy.stats import beta
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return (lo, hi)


def pkplex_pairs(seq: str, timeout_s: float = 60.0):
    """Fold with the OUTSIDE judge and return its pair set. None means it failed to run, never a rejection."""
    from rna_harness.benchmarks.rnainvbench import pk_pairs
    if not os.path.exists(PKPLEX):
        return None
    try:
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run([PKPLEX], input=seq + "\n", capture_output=True, text=True,
                               timeout=timeout_s, cwd=d)
        for ln in (r.stdout or "").splitlines():
            ln = ln.strip()
            body = ln.split()[0] if ln else ""
            if body and set(body) <= set(".()[]{}<>") and len(body) == len(seq):
                return pk_pairs(body)
    except Exception:                                                          # noqa: BLE001
        return None
    return None


def judge(item):
    """Both verdicts on ONE sequence: ours and the outsider's. Same string, two juries."""
    tid, pop, target, seq = item
    from rna_harness.benchmarks.rnainvbench import _round_only, pk_pairs
    from rna_harness.cognitive.certify import _knot_recall
    cross = pk_pairs(target) - pk_pairs(_round_only(target))
    if not cross or not seq:
        return None
    ours = _knot_recall(seq, target)
    ps = pkplex_pairs(seq)
    rec = {"id": tid, "pop": pop, "ours_confirmed": bool(ours.get("knot_confirmed")),
           "ours_confirmed_by": ours.get("confirmed_by")}
    if ps is None:
        rec["ran"] = False
        return rec
    rec.update({"ran": True, "recall": len(ps & cross) / len(cross),
                "confirmed": (ps & cross) == cross})
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(HERE, "external_judge.json"))
    a = ap.parse_args()
    if not os.path.exists(PKPLEX):
        print(f"!! 找不到外部裁判 {PKPLEX} —— 先装好再跑,不要用三家里的任何一家顶替")
        return

    import importlib.util
    spec = importlib.util.spec_from_file_location("_cal", os.path.join(HERE, "calibrate.py"))
    cal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cal)
    from rna_harness.benchmarks.rnainvbench import load_pseudobase
    natives = cal._native_sequences()
    tgt = {t.task_id: t.target for t in load_pseudobase()}

    items = []
    # POSITIVE CONTROL: the judge's own ceiling on ground truth, measured in this same run
    for tid, s in natives.items():
        if tid in tgt:
            items.append((tid, "native", tgt[tid], s))
    # UNDER TEST: the sequences our rule accepted
    topo = {r["task_id"]: r["topology"]
            for r in json.load(open(os.path.join(HERE, "pk_topology.json")))["rows"]}
    deliv = json.load(open(os.path.join(HERE, "kissing_deliver.json")))
    for r in deliv["rows"]:
        if r.get("delivered") and r.get("sequence") and r["task_id"] in tgt:
            items.append((r["task_id"] + "#ours", "ours", tgt[r["task_id"]], r["sequence"]))
    # NEGATIVE: decoys designed for the pseudoknot-free skeleton. Generated FRESH here rather than reused
    # from calibrate_strict.json, which stored verdicts but not sequences -- and inverse_fold is unseeded, so
    # those exact strings cannot be recovered. Generating them here is better anyway: the same sequences get
    # judged by BOTH our rule and the outsider, so the comparison is on one population rather than two.
    spec2 = importlib.util.spec_from_file_location("_cs", os.path.join(HERE, "calibrate_strict.py"))
    cs = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(cs)
    n_dec = 0
    for tid, t in tgt.items():
        if tid not in natives:
            continue
        d = cs.nested_decoy(t)
        if d:
            items.append((tid + "#nested", "nested", t, d))
            n_dec += 1
    print(f"  新生成嵌套诱饵 {n_dec} 条(同一批序列将同时交给我们的判据和外部裁判)", flush=True)
    items = [it for it in items if it[3]]
    print(f"外部裁判 RNAPKplex(从未进过回路):{len(items)} 条待判\n", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        recs = [r for r in ex.map(judge, items) if r]

    out = {"judge": "RNAPKplex", "path": PKPLEX, "pops": {}}
    print(f"  {'总体':<10}{'跑通':>8}{'完全复现交叉茎':>28}")
    for pop in ("native", "ours", "nested"):
        sub = [r for r in recs if r["pop"] == pop and r.get("ran")]
        k = sum(1 for r in sub if r["confirmed"])
        lo, hi = clopper_pearson(k, len(sub))
        out["pops"][pop] = {"k": k, "n": len(sub), "lo95": lo, "hi95": hi}
        print(f"  {pop:<10}{len(sub):>8}{k:>10}/{len(sub):<4} = {k/max(1,len(sub)):5.1%} "
              f"[{lo:5.1%}, {hi:5.1%}]")
    n_, o_, d_ = out["pops"]["native"], out["pops"]["ours"], out["pops"]["nested"]
    if n_["n"] and o_["n"]:
        inside = n_["lo95"] <= (o_["k"] / o_["n"]) <= n_["hi95"]
        below = (d_["k"] / d_["n"]) < n_["lo95"] if d_["n"] else None
        print(f"\n  预注册判据(合计):X_ours 落在 S_ext 的 95% 区间内 = {inside};"
              f"  X_decoy 严格低于该区间 = {below}")
        out["preregistered"] = {"ours_within_native_ci": inside, "decoy_below_native_ci": below}

    # STRATIFY BY TOPOLOGY, because the aggregate comparison above is confounded and should not be read on
    # its own. S_ext is the judge's rate on NATIVES, and natives are 135 H-type against 42 kissing while the
    # design set is 187 against 51 -- so the aggregate compares one topology mixture with a different one,
    # and the judge's accuracy plainly depends on topology (71% on H-type natives, 55% on kissing natives).
    # The stratum is not chosen after the fact: this project has reported by topology since kissing_deliver.
    print(f"\n  分拓扑重算(合计口径混了拓扑构成,不能单独读):")
    print(f"    {'拓扑':<18}{'S_ext(天然)':>22}{'X_ours(设计)':>20}{'判据':>10}")
    strat = {}
    for g in ("kissing / chain", "higher-order", "H-type"):
        sn = [r for r in recs if r["pop"] == "native" and topo.get(r["id"].split("#")[0]) == g
              and r.get("ran")]
        so = [r for r in recs if r["pop"] == "ours" and topo.get(r["id"].split("#")[0]) == g
              and r.get("ran")]
        if not sn or not so:
            continue
        ka = sum(1 for r in sn if r["confirmed"])
        kb = sum(1 for r in so if r["confirmed"])
        la, ha = clopper_pearson(ka, len(sn))
        ok = la <= kb / len(so) <= ha
        strat[g] = {"native_k": ka, "native_n": len(sn), "native_lo": la, "native_hi": ha,
                    "ours_k": kb, "ours_n": len(so), "within": ok}
        print(f"    {g:<18}{ka:>5}/{len(sn):<4}={ka/len(sn):6.1%} [{la:.0%},{ha:.0%}]"
              f"{kb:>7}/{len(so):<4}={kb/len(so):6.1%}{'  通过' if ok else '  不通过':>10}")
    out["stratified"] = strat
    # THE 2x2 THAT MATTERS: the same sequences, our verdict against the outsider's.
    print(f"\n  同一批序列上,两套判据的 2x2(仅取外部裁判跑通的):")
    print(f"    {'总体':<10}{'我们过/它过':>14}{'我们过/它不过':>16}{'我们不过/它过':>16}{'都不过':>10}")
    tab = {}
    for pop in ("native", "ours", "nested"):
        sub = [r for r in recs if r["pop"] == pop and r.get("ran")]
        a_ = sum(1 for r in sub if r["ours_confirmed"] and r["confirmed"])
        b_ = sum(1 for r in sub if r["ours_confirmed"] and not r["confirmed"])
        c_ = sum(1 for r in sub if not r["ours_confirmed"] and r["confirmed"])
        d_ = sum(1 for r in sub if not r["ours_confirmed"] and not r["confirmed"])
        tab[pop] = {"both": a_, "ours_only": b_, "ext_only": c_, "neither": d_, "n": len(sub)}
        print(f"    {pop:<10}{a_:>14}{b_:>16}{c_:>16}{d_:>10}")
    out["agreement_2x2"] = tab
    out["records"] = recs
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
