#!/usr/bin/env python3
"""R6 Major Issue 5: is the DesiRNA-vs-our-operator ProbKnot contrast a PAIRED comparison, or two
unpaired 43-target subsets?

The paper's Table 9 reported "26/43 vs 2/43" -- the ProbKnot carry-over of each arm's own 43 pKiss-solves.
The reviewer's objection is exact: those two 43s need not be the SAME 43 targets, so the gap can absorb
target composition and target difficulty on top of method. This script replaces the unpaired ratio with
three paired statistics on the identical 60-target pool:

  (A) the intersection / symmetric difference of the two pKiss-solved sets;
  (B) a PAIRED McNemar of ProbKnot carry-over restricted to the common pKiss-solved subset;
  (C) the same comparison at the full n=60 denominator (a target neither arm pKiss-solves counts as a
      ProbKnot failure for both), which is the composition-free version;
  (D) crossing-pair-count difficulty profiles of the two solved sets, so a composition explanation can be
      inspected rather than assumed away.

Pure offline re-use of the saved per-target adjudications; no oracle calls, no episodes.
"""
import json, math, os, random, sys

random.seed(20260726)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
from rna_harness.benchmarks.rnainvbench import load_pseudobase   # noqa: E402

IDS = [f"pseudobase:{i}" for i in range(60)]


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    m = min(b, c)
    return min(2 * sum(math.comb(n, k) * 0.5 ** n for k in range(0, m + 1)), 1.0)


def load_adj(path):
    """id -> {'pkiss','probknot','shapeknots','unanimous'} over an arm's pKiss-solved set."""
    d = json.load(open(os.path.join(HERE, path)))
    return {r["id"]: r for r in d["rows"]}


def boot_paired_diff(pairs, B=20000):
    """Percentile bootstrap over TARGETS of the paired risk difference mean(x - y)."""
    n = len(pairs)
    if n == 0:
        return (float("nan"), float("nan"))
    out = []
    for _ in range(B):
        s = 0
        for _ in range(n):
            a, b = pairs[random.randrange(n)]
            s += a - b
        out.append(s / n)
    out.sort()
    return out[int(0.025 * B)], out[int(0.975 * B)]


def main():
    des = load_adj("desirna_seqs/panel_always_sota.json")   # DesiRNA (MFE objective)
    our = load_adj("generous/panel_autonomous.json")        # our pKiss-trained gamer
    npk = {t.task_id: t.meta.get("n_pk_pairs") for t in load_pseudobase(limit=60)}

    D, O = set(des), set(our)
    both, d_only, o_only = D & O, D - O, O - D
    print("=== (A) are the two '43 pKiss-solves' the same targets? ===")
    print(f"  DesiRNA pKiss-solved            : {len(D)}/60")
    print(f"  our pKiss-trained operator      : {len(O)}/60")
    print(f"  INTERSECTION (common subset)    : {len(both)}")
    print(f"  DesiRNA-only / ours-only        : {len(d_only)} / {len(o_only)}")
    print(f"  Jaccard overlap                 : {len(both)/len(D | O):.3f}")
    print(f"  -> the two denominators are {'NOT ' if both != D or both != O else ''}identical; an unpaired "
          f"26/43-vs-2/43 ratio therefore mixes method with target composition.\n")

    def pb(rec):
        return 1 if (rec and rec.get("probknot")) else 0

    print("=== (B) PAIRED ProbKnot carry-over on the COMMON pKiss-solved subset ===")
    common = sorted(both, key=lambda x: int(x.split(":")[1]))
    b = sum(1 for t in common if pb(des[t]) and not pb(our[t]))     # DesiRNA only
    c = sum(1 for t in common if pb(our[t]) and not pb(des[t]))     # ours only
    kd = sum(pb(des[t]) for t in common)
    ko = sum(pb(our[t]) for t in common)
    p = mcnemar_exact(b, c)
    lo, hi = boot_paired_diff([(pb(des[t]), pb(our[t])) for t in common])
    print(f"  n(common) = {len(common)}")
    print(f"  DesiRNA ProbKnot-confirmed      : {kd}/{len(common)} = {kd/len(common):.3f}")
    print(f"  ours    ProbKnot-confirmed      : {ko}/{len(common)} = {ko/len(common):.3f}")
    print(f"  paired McNemar: b(DesiRNA only)={b}, c(ours only)={c}, exact two-sided p={p:.3g}")
    print(f"  paired risk difference          : {(kd-ko)/len(common):+.3f}  "
          f"bootstrap 95% CI [{lo:+.3f},{hi:+.3f}]\n")

    print("=== (C) same comparison at the FULL n=60 denominator (composition-free) ===")
    b60 = sum(1 for t in IDS if pb(des.get(t)) and not pb(our.get(t)))
    c60 = sum(1 for t in IDS if pb(our.get(t)) and not pb(des.get(t)))
    kd60 = sum(pb(des.get(t)) for t in IDS)
    ko60 = sum(pb(our.get(t)) for t in IDS)
    p60 = mcnemar_exact(b60, c60)
    lo60, hi60 = boot_paired_diff([(pb(des.get(t)), pb(our.get(t))) for t in IDS])
    print(f"  DesiRNA pKiss AND ProbKnot      : {kd60}/60 = {kd60/60:.3f}")
    print(f"  ours    pKiss AND ProbKnot      : {ko60}/60 = {ko60/60:.3f}")
    print(f"  paired McNemar: b={b60}, c={c60}, exact two-sided p={p60:.3g}")
    print(f"  paired risk difference          : {(kd60-ko60)/60:+.3f}  "
          f"bootstrap 95% CI [{lo60:+.3f},{hi60:+.3f}]\n")

    print("=== (D) difficulty profile of each solved set (crossing-pair count) ===")
    def prof(S, label):
        v = [npk[t] for t in S if npk.get(t) is not None]
        mean = sum(v) / len(v)
        return f"  {label:34s} n={len(v):2d} mean crossing pairs={mean:.2f} " \
               f"min/max={min(v)}/{max(v)}"
    allv = [npk[t] for t in IDS if npk.get(t) is not None]
    print(prof(D, "DesiRNA pKiss-solved"))
    print(prof(O, "ours pKiss-solved"))
    print(prof(both, "common subset"))
    print(f"  {'whole pool':34s} n={len(allv)} mean crossing pairs={sum(allv)/len(allv):.2f} "
          f"min/max={min(allv)}/{max(allv)}")
    print("\n  -> Read (B) and (C), not the unpaired ratio, as the evidence. The design is observational in\n"
          "     the arms (two different methods, not a randomized intervention), so even the paired form is\n"
          "     an ASSOCIATION between 'was trained on pKiss' and 'loses ProbKnot agreement', not a causal\n"
          "     localization of gaming.")


if __name__ == "__main__":
    main()
