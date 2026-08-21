#!/usr/bin/env python3
"""gamma* = F_nested - E_skeleton - dG_stack(crossed): the critical pseudoknot penalty, per molecule.

Recovered from the 8-21 analysis document (docs/weekly-report/2026-08-21-analysis.md, section five). The
arithmetic there was done in a throwaway heredoc, so this module re-derives it and `--reproduce` asserts the
published 12-row table comes back. Nothing downstream may be built on the table until that assertion passes.

    E(target) = E_skeleton  +  dG_stack(crossed)  +  gamma
                  exact         APPROXIMATE          unknown, one scalar per pseudoknot type

    gamma* = F_nested - E_skeleton - dG_stack(crossed)      target beats the whole nested world <=> gamma < gamma*

`F_nested` is the ensemble free energy of every nested structure (ViennaRNA `pf()`, O(L^3), measured
parameters). `E_skeleton` is the target with its crossing brackets dropped, evaluated exactly. `dG_stack` is
the ONLY approximation: the sum of nearest-neighbour stack terms over adjacent crossing pairs, taken from
ViennaRNA's own `eval_int_loop` rather than hand-indexed out of stack37 so that the number is the measured
term and not my transcription of it. It OMITS loop-entropy coupling, dangling ends / terminal mismatches, and
coaxial stacking -- that omission is what g3_stack_error.py exists to bound.

WHAT THIS IS NOT. The comparison is target vs the entire NESTED world, which is a NECESSARY condition for the
design to fold as intended, not a sufficient one: a design can also lose to another CROSSING structure, and
the nested partition function cannot bound that. So a positive verdict here means nothing and a negative
verdict is a proof of failure. One-sided, deliberately (section 1.2 of the analysis retracts the two-sided
reading).

COST: $0, local ViennaRNA only. Needs the `rna` env (RNA 2.7.2).
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import RNA                                                                     # noqa: E402

CANON = {frozenset("GC"), frozenset("AU"), frozenset("GU")}


class Undefined(ValueError):
    """gamma* is not defined for this (sequence, target) pair.

    Rule 38, second half: name an input for which the quantity is undefined and check the code RAISES
    rather than returning a number. Both ViennaRNA entry points this module uses will happily evaluate a
    non-canonical pair -- `energy_of_struct` charges a penalty, and `eval_int_loop` on an A-A stack returns
    +1.30 -- so nothing downstream would have noticed. 22.5% of the OpenKnot designs cannot form the target
    skeleton at all and were half of the gamma*<0 column (g2_audit.md); a further 10.6% cannot form the
    crossing stem, which nothing gated at all until g_review_0821.md found it.
    """


def check_domain(seq: str, target: str) -> None:
    """Raise unless `seq` can actually form every pair the target asks for, crossing pairs included."""
    from rna_harness.benchmarks.rnainvbench import _round_only, pk_pairs
    if len(seq) != len(target):
        raise Undefined(f"length {len(seq)} != target {len(target)}")
    bad_skel = [(i, j) for (i, j) in pk_pairs(_round_only(target))
                if frozenset({seq[i], seq[j]}) not in CANON]
    bad_cross = [(i, j) for (i, j) in crossing_set(target)
                 if frozenset({seq[i], seq[j]}) not in CANON]
    if bad_skel or bad_cross:
        raise Undefined(f"non-canonical pairs: {len(bad_skel)} in the skeleton, "
                        f"{len(bad_cross)} in the crossing stem -- E_skeleton and/or dG_stack meaningless")

# The published table, section five of docs/weekly-report/2026-08-21-analysis.md. Kept here so the
# reproduction is an assertion against a frozen record rather than against whatever the code prints today.
PUBLISHED = {
    "P01": (52, -19.98, -19.70, -7.80, 3, 7.52), "P02": (75, -22.30, -20.40, -11.10, 4, 9.20),
    "P03": (73, -21.75, -16.40, -12.40, 4, 7.05), "P04": (93, -36.19, -30.20, -9.10, 3, 3.11),
    "P05": (59, -23.75, -22.60, -12.30, 5, 11.15), "P07": (80, -28.52, -26.80, -11.60, 4, 9.88),
    "P08": (96, -34.86, -34.80, -11.60, 4, 11.54), "P09": (68, -26.77, -9.50, -10.30, 6, -6.97),
    "P10": (76, -27.90, -17.70, -19.90, 7, 9.70), "P11": (100, -46.34, -38.80, -14.90, 5, 7.36),
    "P12": (100, -29.95, -25.10, -16.80, 8, 11.95), "P13": (100, -47.12, -46.20, -14.80, 5, 13.88),
}


def crossing_set(target: str) -> set:
    """The target's crossing pairs: the ones its pseudoknot brackets carry."""
    from rna_harness.benchmarks.rnainvbench import _round_only, pk_pairs
    return pk_pairs(target) - pk_pairs(_round_only(target))


def stack_terms(seq: str, pairs) -> list:
    """Every nearest-neighbour stack term inside `pairs`, in kcal/mol, from ViennaRNA's own evaluator.

    A term exists for each (i,j) in `pairs` whose (i+1,j-1) is also in `pairs` -- i.e. once per adjacent
    pair of stacked base pairs. `eval_int_loop` with a zero-size interior loop IS the stack term, so this
    reads the measured parameter instead of re-deriving the pair-type indexing into stack37.
    """
    fc = RNA.fold_compound(seq)
    S = set(pairs)
    out = []
    for (i, j) in sorted(S):
        if (i + 1, j - 1) in S:
            out.append(((i, j), fc.eval_int_loop(i + 1, j + 1, i + 2, j) / 100.0))
    return out


def gamma_star(seq: str, target: str, strict: bool = True) -> dict:
    """One molecule's gamma*, with every component reported separately so the approximate one is visible.

    `strict` (the default) enforces the domain: a sequence that cannot form the target's pairs raises
    rather than returning a number. Pass strict=False only to reproduce a pre-rule-38 population-wide
    figure, and say so when you quote it.
    """
    if strict:
        check_domain(seq, target)
    cross = crossing_set(target)
    from rna_harness.benchmarks.rnainvbench import _round_only
    skel = _round_only(target)
    f_nested = RNA.fold_compound(seq).pf()[1]
    e_skel = RNA.energy_of_struct(seq, skel)
    terms = stack_terms(seq, cross)
    dg = sum(v for _, v in terms)
    return {"L": len(seq), "F_nested": round(f_nested, 2), "E_skeleton": round(e_skel, 2),
            "dG_stack": round(dg, 2), "n_stack": len(terms), "n_cross": len(cross),
            "gamma_star": round(f_nested - e_skel - dg, 2)}


def best_wet_per_target(src: str = "panel_f1_fourth.json") -> dict:
    """One design per target: the highest tube score in the fourth batch. What the published table used."""
    d = json.load(open(os.path.join(HERE, src)))
    best = {}
    for r in d["designs"]:
        if "error" in r or r.get("wet") is None:
            continue
        p = r["puzzle"]
        if p not in best or r["wet"] > best[p]["wet"]:
            best[p] = r
    return best


def main() -> int:
    from leaderboard_design import targets
    T = targets()
    best = best_wet_per_target()
    rows, mism, undef = {}, [], {}
    print(f"{'target':6} {'L':>4} {'F_nested':>9} {'E_skel':>8} {'dG_stack':>9} {'nst':>4} {'gamma*':>8}   published")
    for p in sorted(best):
        tgt = T[p]["target"]
        if not crossing_set(tgt):
            continue
        try:
            g = gamma_star(best[p]["seq"], tgt)
        except Undefined as exc:                                               # noqa: PERF203
            undef[p] = str(exc)
            continue
        rows[p] = g
        pub = PUBLISHED.get(p)
        tag = ""
        if pub:
            want = {"L": pub[0], "F_nested": pub[1], "E_skeleton": pub[2], "dG_stack": pub[3],
                    "n_stack": pub[4], "gamma_star": pub[5]}
            bad = {k: (g[k], v) for k, v in want.items() if abs(g[k] - v) > 0.011}
            tag = "  OK" if not bad else f"  MISMATCH {bad}"
            if bad:
                mism.append((p, bad))
        print(f"{p:6} {g['L']:4d} {g['F_nested']:9.2f} {g['E_skeleton']:8.2f} {g['dG_stack']:9.2f} "
              f"{g['n_stack']:4d} {g['gamma_star']:8.2f}{tag}")
    extra = sorted(set(PUBLISHED) - set(rows))
    print(f"\n  定义域闸门(第 38 条):{len(undef)} 个靶标的 best-wet 设计形不成靶标配对,已抛出而非返回数")
    for p, why in sorted(undef.items()):
        print(f"    {p}: {why}")
    print(f"\n  reproduced {len(rows)} rows, published {len(PUBLISHED)}, "
          f"mismatches {len(mism)}, published-but-absent {extra}")
    json.dump({"rows": rows, "mismatches": mism, "published_absent": extra,
               "undefined_domain": undef,
               "reproduces": not mism and not extra},
              open(os.path.join(HERE, "gamma_star_reproduce.json"), "w"), indent=1)
    return 0 if (not mism and not extra) else 1


if __name__ == "__main__":
    sys.exit(main())
