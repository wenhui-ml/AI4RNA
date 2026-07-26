#!/usr/bin/env python3
"""Panel-operator component ablation + matched-budget undirected-search control (reviewer R4 §5.1/§5.2).

Reads pablate_s{0,1,2}/sweep.json (all six arms run in the SAME sweep, so every comparison is paired on
seed AND target at matched budgets/oracle access/initialization). Reports the three named metrics of the
paper -- r_in = pKiss AND ProbKnot (the panel operator's own objective), r_ho = ShapeKnots only (the one
genuinely held-out oracle), r_mix = all three -- per arm, per seed, plus the paired McNemar of each ablation
against the full operator and a two-way (target+seed) bootstrap of the paired difference.

The question this answers: is the panel-agreement OBJECTIVE doing the work, or would a matched-budget
undirected search with the same oracle access get to the same place?
"""
import json, math, os, random

random.seed(20260726)
HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1, 2]
ARMS = ["panel_auto", "panel_randwalk", "panel_blindrand", "panel_norepair",
        "panel_nodecompose", "panel_nocompose"]
LABEL = {"panel_auto": "full operator (reference)",
         "panel_randwalk": "- objective (undirected walk, oracle-informed proposals)",
         "panel_blindrand": "- objective, - proposal bias (oracle-blind random search)",
         "panel_norepair": "- repair loop entirely",
         "panel_nodecompose": "- CF-skeleton decomposition (random init)",
         "panel_nocompose": "- complementarity composition"}
METRICS = [("r_in", "pKiss AND ProbKnot (in-objective)"),
           ("r_ho", "ShapeKnots only (held out)"),
           ("r_mix", "all three (mixed)")]


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(2 * sum(math.comb(n, k) * 0.5 ** n for k in range(0, min(b, c) + 1)), 1.0)


def load(s):
    """(seed, target, arm) -> {r_in, r_ho, r_mix}. Errored arm-target cells are dropped for ALL arms on that
    target, so every arm is scored on an identical target set (paired)."""
    d = json.load(open(os.path.join(HERE, f"pablate_s{s}", "sweep.json")))
    out = {}
    for r in d["rows"]:
        cells, ok = {}, True
        for a in ARMS:
            v = r["arms"].get(a) or {}
            if v.get("error"):
                ok = False
                break
            p = v.get("panel") or {}
            cells[a] = {"r_in": bool(v.get("solved")),          # solved == pKiss AND ProbKnot exact
                        "r_ho": bool(p.get("shapeknots")),
                        "r_mix": bool(p.get("unanimous"))}
        if ok:
            out[r["id"]] = cells
    return out


def main():
    data = {s: load(s) for s in SEEDS}
    ids = sorted(set.intersection(*[set(v) for v in data.values()]))
    n = len(ids)
    print(f"panel-operator ablation: n={n} targets common to seeds {SEEDS}, all six arms paired per target\n")

    for key, desc in METRICS:
        print(f"=== {key}: {desc} ===")
        print(f"  {'arm':56} " + "  ".join(f"s{s}" for s in SEEDS) + "   mean   McNemar vs full (b,c,p)")
        for a in ARMS:
            counts = [sum(data[s][t][a][key] for t in ids) for s in SEEDS]
            mean = sum(counts) / len(counts)
            if a == "panel_auto":
                tail = "--"
            else:
                b = sum(1 for s in SEEDS for t in ids
                        if data[s][t]["panel_auto"][key] and not data[s][t][a][key])
                c = sum(1 for s in SEEDS for t in ids
                        if data[s][t][a][key] and not data[s][t]["panel_auto"][key])
                tail = f"b={b},c={c},p={mcnemar_exact(b, c):.3f}"
            print(f"  {LABEL[a]:56} " + "  ".join(f"{c:2d}" for c in counts) + f"   {mean:5.2f}  {tail}")
        # two-way bootstrap of the paired difference full - ablation, pooled over seeds
        print("    two-way (target+seed) bootstrap 95% CI on Delta = full - ablation:")
        for a in ARMS[1:]:
            D = {(s, t): int(data[s][t]["panel_auto"][key]) - int(data[s][t][a][key])
                 for s in SEEDS for t in ids}
            obs = sum(D.values()) / len(D)
            B, rates = 20000, []
            for _ in range(B):
                ss = [SEEDS[random.randrange(len(SEEDS))] for _ in SEEDS]
                ts = [ids[random.randrange(n)] for _ in range(n)]
                rates.append(sum(D[(s, t)] for s in ss for t in ts) / (len(ss) * len(ts)))
            rates.sort()
            lo, hi = rates[int(0.025 * B)], rates[int(0.975 * B)]
            verdict = "SEPARATED" if lo > 0 else "not separated (CI spans 0)"
            print(f"      vs {a:20} Delta={obs:+.4f}  CI [{lo:+.4f},{hi:+.4f}]  {verdict}")
        print()


if __name__ == "__main__":
    main()
