"""R2-3.1: CORRECT cascade cost accounting -- gold INVOCATION rate, not gold ACCEPTANCE rate.

A reviewer caught a logic error: the paper described the cheapest-first cascade as running the gold (pfe)
rung 'only for the 8.2% survivors', but 8.2% is P(A_pfe | A_match) -- the gold ACCEPTANCE rate. A sound
cascade must run gold on every candidate that PASSES the cheaper rungs, i.e. on P(pass uMFE) of the stream
(the gold INVOCATION rate), which on the exact-match solution set is 91.4%, not 8.2%.

This script recomputes the cascade cost with the correct invocation-gated formula
    E[C] = C_match + P(pass match)*C_uMFE + P(pass uMFE)*C_pfe
on three workloads and reports, per workload: P(A_match), P(A_uMFE), P(A_pfe), the gold-invocation rate
(fraction reaching pfe = P(pass uMFE)), gold calls, the cascade cost, always-gold cost, and the ratio.
Uses per-rung standalone verify() costs (the same ones sound_cascade uses)."""
from __future__ import annotations

import json
import os
import random

from .verifier_routing import build_workload, RUNGS
from ..benchmarks.eterna100 import load_eterna100

OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "cascade_cost")


def _analyze(W, label):
    n = len(W)
    gold = RUNGS[-1]                                        # pfe
    p = {r: sum(w["verdict"][r] for w in W) / n for r in RUNGS}
    # invocation-gated cascade: run match on all; uMFE on match-passers; pfe on uMFE-passers
    total = gold_calls = 0.0
    for w in W:
        c = w["cost"]["match"]
        if w["verdict"]["match"]:
            c += w["cost"]["umfe"]
            if w["verdict"]["umfe"]:                        # reaches gold iff it passed uMFE
                c += w["cost"][gold]; gold_calls += 1
        total += c
    cascade = total / n
    always_gold = sum(w["cost"][gold] for w in W) / n
    return {"workload": label, "n": n,
            "P_match": round(p["match"], 4), "P_uMFE": round(p["umfe"], 4), "P_pfe": round(p["pfe"], 4),
            "gold_invocation_rate": round(gold_calls / n, 4), "gold_calls": int(gold_calls),
            "cascade_ms": round(1e3 * cascade, 4), "always_gold_ms": round(1e3 * always_gold, 4),
            "cascade_over_always_gold": round(cascade / always_gold, 4) if always_gold else None}


def run(seed=0, n_targets=15, max_n=65):
    os.makedirs(OUT, exist_ok=True)
    puz = load_eterna100(version="V2")
    ids = [q["id"] for q in sorted(puz, key=lambda q: len(q["target"])) if len(q["target"]) <= max_n][:n_targets]
    rng = random.Random(seed)

    rows = []
    # reject-heavy: 12 random (reject-side) + 1 solver solution per target
    W_rej = build_workload(ids, rng, rand_per=12, sols_per=1, seed=seed)
    rows.append(_analyze(W_rej, "reject-heavy (12 rand + 1 sol/target)"))
    # accept-heavy: 6 solver solutions, no random
    W_acc = build_workload(ids, rng, rand_per=0, sols_per=6, seed=seed)
    rows.append(_analyze(W_acc, "accept-heavy (6 sol/target, no random)"))
    # exact-match solution stream: the accept-heavy candidates that actually exact-match (P(A_match)=1 by
    # construction) -- this is the Table-7 workload; here gold-invocation = P(A_uMFE|A_match), NOT 8.2%
    W_exact = [w for w in W_acc if w["verdict"]["match"]]
    if W_exact:
        rows.append(_analyze(W_exact, "exact-match solution stream (all pass match)"))

    json.dump({"rows": rows, "seed": seed, "ids": ids}, open(os.path.join(OUT, "cascade_cost.json"), "w"), indent=1)
    print("=== CORRECT cascade cost (gold INVOCATION-gated) ===", flush=True)
    hdr = f"{'workload':42} {'n':>4} {'P(match)':>9} {'P(uMFE)':>8} {'P(pfe)':>7} {'goldInvoc':>10} {'casc/gold':>10}"
    print(hdr, flush=True)
    for r in rows:
        print(f"{r['workload']:42} {r['n']:>4} {r['P_match']:>9} {r['P_uMFE']:>8} {r['P_pfe']:>7} "
              f"{r['gold_invocation_rate']:>10} {r['cascade_over_always_gold']:>10}", flush=True)
    print(f"\n  KEY: gold-invocation rate = P(pass uMFE), NOT the gold-acceptance rate P(pfe).", flush=True)
    print(f"  wrote {os.path.join(OUT, 'cascade_cost.json')}", flush=True)
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-targets", type=int, default=15)
    ap.add_argument("--max-n", type=int, default=65)
    a = ap.parse_args()
    run(seed=a.seed, n_targets=a.n_targets, max_n=a.max_n)
