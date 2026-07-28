#!/usr/bin/env python3
"""Statistics and mechanism check for Experiment A (agent_panel.py).

agent_panel.py produced the counts. This asks three questions the counts alone cannot answer:

  1. Which contrasts are STATISTICALLY real? Exact McNemar on the paired (target) design, because all three
     systems are scored on the SAME 125 held-out crossing targets. Bonferroni over the tests reported here.
  2. Is the operator specifically worse than a SINGLE DesiRNA candidate, or does every single-sequence system
     collapse under the panel? These are different conclusions and only the paired test separates them.
  3. MECHANISM. The operator's `fix_crossing` slams every crossing pair to G-C. If that is how it wins pKiss,
     its pKiss-passing designs should be GC-saturated on crossing positions while its panel-passing designs
     (all 1 of them) and DesiRNA's should not. Stated before looking.
"""
from __future__ import annotations

import json
import os
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
AGENT = os.path.join(HERE, "agent")


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar: b = A-only wins, c = B-only wins. p = 2 * P(X <= min(b,c)) under Bin(n, 1/2)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def load():
    d = json.load(open(os.path.join(HERE, "agent_panel.json")))
    rec = {int(k): v for k, v in d["per_target"].items()}
    held = d["held"]
    pool = {int(r["task_id"].split(":")[1]): r
            for r in json.load(open(os.path.join(AGENT, "desirna_panel_pool.json")))["rows"]}
    fixed = {int(r["id"].split(":")[1]): r
             for r in json.load(open(os.path.join(AGENT, "desirna_panel_fixed.json")))["rows"]}
    cols = {}
    cols["operator"] = {i: {"pkiss": rec[i].get("solved") is True,
                            "probknot": rec[i].get("probknot") is True,
                            "shapeknots": rec[i].get("shapeknots") is True} for i in rec}
    cols["desirna_pool"] = {i: {"pkiss": pool[i].get("forms_T_pkiss") is True,
                               "probknot": pool[i].get("forms_T_probknot") is True,
                               "shapeknots": pool[i].get("forms_T_shapeknots") is True}
                            for i in held if i in pool}
    cols["desirna_fixed"] = {i: {k: (isinstance(fixed[i].get(k), dict) and fixed[i][k].get("exact") is True)
                                 for k in ("pkiss", "probknot", "shapeknots")}
                             for i in held if i in fixed}
    for c in cols.values():
        for v in c.values():
            v["unanimous"] = v["pkiss"] and v["probknot"] and v["shapeknots"]
    return d, rec, held, cols


def main():
    d, rec, held, cols = load()
    print(f"Experiment A statistics -- {len(held)} held-out crossing targets, paired design\n")

    TESTS = [("operator", "desirna_fixed", "pkiss"),
             ("operator", "desirna_fixed", "probknot"),
             ("operator", "desirna_fixed", "shapeknots"),
             ("operator", "desirna_fixed", "unanimous"),
             ("operator", "desirna_pool", "unanimous"),
             ("desirna_fixed", "desirna_pool", "unanimous")]
    print(f"  {'contrast':<34} {'criterion':<11} {'A':>4} {'B':>4} {'A-only':>7} {'B-only':>7} "
          f"{'exact p':>10}")
    ps = []
    for A, B, crit in TESTS:
        ids = [i for i in held if i in cols[A] and i in cols[B]]
        a = sum(cols[A][i][crit] for i in ids)
        b = sum(cols[B][i][crit] for i in ids)
        ao = sum(cols[A][i][crit] and not cols[B][i][crit] for i in ids)
        bo = sum(cols[B][i][crit] and not cols[A][i][crit] for i in ids)
        p = mcnemar_exact(ao, bo)
        ps.append(p)
        print(f"  {A+' vs '+B:<34} {crit:<11} {a:>4} {b:>4} {ao:>7} {bo:>7} {p:>10.3g}")
    bonf = 0.05 / len(TESTS)
    print(f"\n  Bonferroni threshold over {len(TESTS)} tests: {bonf:.4f}")
    print(f"  surviving: " + ", ".join(f"{A} vs {B} / {c}" for (A, B, c), p in zip(TESTS, ps) if p < bonf))

    # ---- 2. does the pKiss claim survive dropping EACH oracle in turn (which oracle kills it) --------
    print(f"\n=== which oracle kills the operator's 91 (pKiss-passing subset only) ===")
    pk_ids = [i for i in cols["operator"] if cols["operator"][i]["pkiss"]]
    for name in ("probknot", "shapeknots"):
        agr = sum(cols["operator"][i][name] for i in pk_ids)
        print(f"  of the operator's {len(pk_ids)} pKiss solves, {name:<11} also confirms {agr:>3} "
              f"({100.0*agr/len(pk_ids):.1f}%)")
    for name in ("probknot", "shapeknots"):
        dids = [i for i in cols["desirna_fixed"] if cols["desirna_fixed"][i]["pkiss"]]
        agr = sum(cols["desirna_fixed"][i][name] for i in dids)
        print(f"  of DesiRNA-fixed's {len(dids)} pKiss solves, {name:<11} also confirms {agr:>3} "
              f"({100.0*agr/len(dids):.1f}%)")

    # ---- 3. mechanism: is fix_crossing's G-C saturation the tell? ------------------------------------
    from rna_harness.benchmarks.rnainvbench import load_pseudobase, pk_pairs, _round_only
    tgt = {int(t.task_id.split(":")[1]): t.target for t in load_pseudobase()}
    print(f"\n=== mechanism: fix_crossing sets every crossing pair to G-C. Is that the tell? ===")

    def gc_frac(seq, target):
        cross = pk_pairs(target) - pk_pairs(_round_only(target))
        if not cross or not seq:
            return None
        gc = sum(1 for i, j in cross
                 if {seq[i], seq[j]} == {"G", "C"} and i < len(seq) and j < len(seq))
        return gc / len(cross)

    groups = {"operator, pKiss-pass": [], "operator, pKiss-fail": [], "DesiRNA best_pk_seq": []}
    for i, r in rec.items():
        f = gc_frac(r.get("seq"), tgt[i])
        if f is None:
            continue
        groups["operator, pKiss-pass" if r.get("solved") else "operator, pKiss-fail"].append(f)
    des = {int(r["task_id"].split(":")[1]): r for r in
           json.load(open(os.path.join(AGENT, "pk_desirna.json")))["rows"]}
    for i in held:
        if i in des:
            f = gc_frac(des[i].get("best_pk_seq"), tgt[i])
            if f is not None:
                groups["DesiRNA best_pk_seq"].append(f)
    print(f"  {'group':<24} {'n':>4} {'mean GC-saturation of crossing pairs':>38} {'==1.0':>7}")
    for k, v in groups.items():
        if v:
            print(f"  {k:<24} {len(v):>4} {sum(v)/len(v):>38.3f} "
                  f"{sum(1 for x in v if x >= 0.999):>7}")

    out = {"tests": [{"A": A, "B": B, "criterion": c, "p_exact": p} for (A, B, c), p in zip(TESTS, ps)],
           "bonferroni": bonf,
           "gc_saturation": {k: {"n": len(v), "mean": (sum(v)/len(v) if v else None),
                                 "n_fully_saturated": sum(1 for x in v if x >= 0.999)}
                             for k, v in groups.items()}}
    json.dump(out, open(os.path.join(HERE, "agent_panel_stats.json"), "w"), indent=1)
    print(f"\n  wrote {os.path.relpath(os.path.join(HERE, 'agent_panel_stats.json'), REPO)}")


if __name__ == "__main__":
    main()
