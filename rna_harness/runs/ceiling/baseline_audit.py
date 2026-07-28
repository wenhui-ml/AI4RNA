#!/usr/bin/env python3
"""Priority-1 audit: do this branch's headline efficiency multiples divide by FLAT or by CASCADE?

WHY. frontier.py showed that on the freshly measured DesiRNA curve the FLAT -> CASCADE step is worth 2.66x and
is FREE -- any scheduler that retires solved targets gets it with zero learning. So a headline "we cut SOTA
cost by Nx" is only a claim about learning if N is measured against the cascade. Measured against FLAT it
bundles the free gain with the learned one, and a policy could even be WORSE than cascading while still
reporting a multiple above 1.

WHAT IS AUDITED. The two archived results this branch quotes:
  audit/rssd_compounding.json  crossing / pseudobase, held-out n = 125
  audit/rssd_eterna_cost.json  Eterna100, held-out n = 50
Both were copied verbatim from the rna-specialization branch; nothing is re-run.

GOOD NEWS FOUND FIRST. Both experiments ALREADY contain the correct baseline -- `cold` and `cascade`
respectively, defined in their own sources as cheap-first-then-SOTA with no learning, plus a perfect-foresight
`oracle`/`oracle_abstain` upper bound. The defect is not the experiment; it is which ratio got quoted.

The decomposition below is the honest one:
    FLAT -> CASCADE            free, no learning
    CASCADE -> learned arm     the actual contribution of learning
    CASCADE -> ORACLE          the prize that was available
    captured = (cascade - learned) / (cascade - oracle)
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
AUD = os.path.join(HERE, "audit")


def ratio(a, b):
    return a / b if b else float("nan")


def audit(name, flat, casc, oracle, learned_arms, n, solved):
    print(f"\n=== {name}  (held-out n = {n}, all arms solve {solved}) ===")
    print(f"  FLAT (always_sota)            ${flat:.4f}")
    print(f"  CASCADE (zero knowledge)      ${casc:.4f}    FLAT/CASCADE = {ratio(flat, casc):.3f}x  <- FREE")
    print(f"  ORACLE (perfect foresight)    ${oracle:.4f}    CASCADE/ORACLE = {ratio(casc, oracle):.3f}x"
          f"  <- the whole prize")
    print(f"\n  {'learned arm':<22} {'$':>9} {'vs FLAT':>9} {'vs CASCADE':>11} {'prize captured':>15} "
          f"{'solves lost':>12}")
    best = None
    for lbl, c, lost in learned_arms:
        cap = (casc - c) / (casc - oracle) if (casc - oracle) else float("nan")
        print(f"  {lbl:<22} {c:>9.4f} {ratio(flat, c):>8.3f}x {ratio(casc, c):>10.3f}x "
              f"{100*cap:>14.1f}% {str(lost):>12}")
        # The claim being audited is "at IDENTICAL held-out quality", so the best arm must be the cheapest
        # one that loses NO solves. Choosing by cost alone lets a high-tau arm that abstains on solvable
        # targets look like the winner (and even report >100% of the prize, which is the tell).
        if lost == 0 and (best is None or c < best[1]):
            best = (lbl, c, cap, lost)
    print(f"\n  best arm AT ZERO SOLVES LOST        {best[0]}  ${best[1]:.4f}")
    print(f"  headline as quoted (FLAT/best)      {ratio(flat, best[1]):.3f}x")
    print(f"  honest learned gain (CASCADE/best)  {ratio(casc, best[1]):.3f}x")
    print(f"  free part that was bundled in       {ratio(flat, casc):.3f}x")
    verdict = ("LEARNING ADDS NOTHING -- the headline is the cascade"
               if ratio(casc, best[1]) < 1.05 else
               f"learning is real but SMALLER than quoted: {ratio(casc, best[1]):.2f}x, not "
               f"{ratio(flat, best[1]):.2f}x")
    print(f"  VERDICT: {verdict}")
    return {"n": n, "solved": solved, "flat": flat, "cascade": casc, "oracle": oracle,
            "best_arm": best[0], "best_cost": best[1], "prize_captured": best[2],
            "solves_lost": best[3], "headline_vs_flat": ratio(flat, best[1]),
            "honest_vs_cascade": ratio(casc, best[1]), "free_flat_to_cascade": ratio(flat, casc)}


def main():
    out = {}

    d = json.load(open(os.path.join(AUD, "rssd_compounding.json")))
    h = d["arms"]["held"]
    arms = [("learned ordering", h["learned"]["cost"], 0)]
    arms += [(f"abstain@tau={t}", v["cost"], v.get("solves_lost"))
             for t, v in sorted(h["abstain"].items(), key=lambda kv: float(kv[0]))]
    out["crossing_held"] = audit("crossing / pseudobase", h["always_sota"]["cost"], h["cold"]["cost"],
                                 h["oracle_abstain"]["cost"], arms, h["n"], h["always_sota"]["solved"])

    e = json.load(open(os.path.join(AUD, "rssd_eterna_cost.json")))["held_result"]
    arms = [(f"abstain@tau={t}", v["cost"], v.get("solves_lost"))
            for t, v in sorted(e["abstain"].items(), key=lambda kv: float(kv[0]))]
    out["eterna_held"] = audit("Eterna100", e["always_sota"]["cost"], e["cascade"]["cost"],
                               e["oracle"]["cost"], arms, len(json.load(
                                   open(os.path.join(AUD, "rssd_eterna_cost.json")))["held"]),
                               e["always_sota"]["solved"])

    print(f"\n=== which archived ratio equals the numbers this branch has been quoting? ===")
    targets = {"1.54x (Eterna)": 1.54, "3.51x (crossing)": 3.51,
               "13x (memory upper)": 13.0, "8x (memory lower)": 8.0}
    cand = []
    for dom, r in out.items():
        for lbl, v in (("FLAT/best", r["headline_vs_flat"]), ("FLAT/CASCADE", r["free_flat_to_cascade"]),
                       ("CASCADE/best", r["honest_vs_cascade"]),
                       ("FLAT/ORACLE", r["flat"] / r["oracle"]),
                       ("CASCADE/ORACLE", r["cascade"] / r["oracle"])):
            cand.append((dom, lbl, v))
    for tl, tv in targets.items():
        m = min(cand, key=lambda c: abs(c[2] - tv))
        flag = "MATCH" if abs(m[2] - tv) < 0.03 * max(tv, 1) else "no close match"
        print(f"  {tl:<18} closest archived ratio = {m[0]}/{m[1]} = {m[2]:.3f}x   [{flag}]")

    json.dump(out, open(os.path.join(HERE, "baseline_audit.json"), "w"), indent=1)
    print(f"\n  wrote rna_harness/runs/ceiling/baseline_audit.json")


if __name__ == "__main__":
    main()
