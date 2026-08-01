#!/usr/bin/env python3
"""R11 (reviewer Q2), D3 seventh test: whole-distribution comparison, not hand-picked features.

The state of D3. Six candidate mechanisms are excluded by measurement: oracle-call volume in both directions,
target selectivity, GC composition, denominator collapse, four mechanisms legible in the agent source, and one
structural feature (`decoy`) that survived a composition-adjusted observational gate and then reversed under
intervention. In the last revision we wrote that we could not name a seventh candidate that was both legible
and testable here. A reviewer named one, and it is a better question than the six before it.

The question. Every test so far compared a HAND-PICKED scalar. A difference could instead live in the overall
distribution of the accepted designs -- their energetics or loop geometry as a whole -- and a battery of
single features would miss it while each individual test came back null. So we compare distributions.

What makes this a fair test rather than another fishing expedition, and it is the same guard as before:
  * Quantities are chosen to be NOT fixed by the target. Both arms are scored on the same targets, so length,
    pair count and crossing count are matched by construction and would show a spurious null. We use only
    per-design quantities the operator is free to vary: folding energetics under a model NEITHER arm
    optimizes (ViennaRNA's nested MFE and ensemble free energy) and loop-length geometry.
  * Two-sample Kolmogorov-Smirnov on the full empirical CDF, so a shift, a spread change, or a shape change
    all register -- which is the point of asking distributionally.
  * A PERMUTATION null on arm labels, because 1104 vs 530 designs are not independent draws (60 targets,
    repeated seeds) and an asymptotic KS p-value would be anticonservative in exactly the way the positive
    control's McNemar was.
  * COMPOSITION-MATCHED, and it took two attempts to get an honest test. Unstratified, all six quantities came
    back significant, which is what a non-discriminating test looks like: folding energy per nucleotide is
    largely GC content in energy units, and GC is already known to differ (0.529 vs 0.460). Decile
    stratification did not fix it either -- the negative control (GC tested against itself within GC bins)
    came back significant, because with distributions this far apart a decile still contains a systematic
    within-bin imbalance. So we CALIPER-MATCH: each hand-built design is paired with an agent design whose GC
    is within +/-0.01, without replacement, and the comparison runs on the matched sets only. The negative
    control is run on the matched sets too and MUST be null; if it is not, no row below can be read, and we
    say so rather than reporting the confounded numbers.
  * The interpretation is stated first, both ways. A difference here is a lead, not a mechanism, and would
    face the same transplant test everything else did. NO difference is the informative outcome: it would say
    the gap is not a global property of the accepted designs at all, which points at topology-specific
    handling and away from anything a distributional summary can capture.

Offline over stored sequences; ViennaRNA folds only (cheap, nested model, in no arm's loop).
"""
import glob, json, math, os, random, sys

random.seed(20260801)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
sys.path.insert(0, HERE)

import RNA                                                                     # noqa: E402
from rna_harness.benchmarks.rnainvbench import load_pseudobase                 # noqa: E402

OPS = ["gemini31pro", "kimik3"]
SEEDS, EXECS = [0, 1, 2, 3, 4], [0, 1, 2]
NPERM = 2000


def features(seq, target):
    """Per-design quantities the target does not fix. Folded under ViennaRNA's nested model, which is in
    neither arm's optimization loop, so this is not scoring either arm on its own objective."""
    n = len(seq)
    st, mfe = RNA.fold(seq)
    fc = RNA.fold_compound(seq)
    _, efe = fc.pf()
    loops, run = [], 0
    for ch in st:
        if ch == ".":
            run += 1
        elif run:
            loops.append(run); run = 0
    if run:
        loops.append(run)
    return {"mfe_per_nt": mfe / n,
            "efe_per_nt": efe / n,
            "ens_gap": (mfe - efe) / n,          # how far the MFE sits below the ensemble: foldedness
            "loop_mean": (sum(loops) / len(loops)) if loops else 0.0,
            "loop_max": float(max(loops)) if loops else 0.0,
            "n_loops": float(len(loops)),
            "gc_all": sum(c in "GC" for c in seq) / n}


KEYS = ["mfe_per_nt", "efe_per_nt", "ens_gap", "loop_mean", "loop_max", "n_loops"]


def caliper_match(agent, hand, key="gc_all", cal=0.01):
    """Pair each hand-built design with an unused agent design of near-identical composition.

    Greedy nearest-neighbour on the caliper variable, without replacement, hand-built as the smaller set. The
    result is two sets with (by construction) almost the same composition distribution, so any surviving
    difference in another quantity is not composition. Returns (matched_agent, matched_hand, diagnostics).
    """
    pool = sorted(range(len(agent)), key=lambda i: agent[i][key])
    vals = [agent[i][key] for i in pool]
    used = set()
    ma, mh = [], []
    import bisect
    for h in hand:
        g = h[key]
        j = bisect.bisect_left(vals, g)
        best, bestd = None, None
        for cand in range(max(0, j - 40), min(len(pool), j + 40)):
            if pool[cand] in used:
                continue
            d = abs(vals[cand] - g)
            if d <= cal and (bestd is None or d < bestd):
                best, bestd = cand, d
        if best is not None:
            used.add(pool[best]); ma.append(agent[pool[best]]); mh.append(h)
    diag = (len(mh), len(hand),
            sum(x[key] for x in ma) / max(1, len(ma)),
            sum(x[key] for x in mh) / max(1, len(mh)))
    return ma, mh, diag


def strat_ks(a_pairs, b_pairs, nbins=10):
    """KS computed WITHIN matched GC bins and pooled, weighted by bin size.

    Each element is (value, gc). Bins are deciles of the pooled GC distribution, so the two arms are compared
    only against designs of comparable composition. A quantity that is really just GC re-expressed collapses
    here; one that differs at matched composition survives.
    """
    allgc = sorted(g for _, g in a_pairs + b_pairs)
    if not allgc:
        return float("nan")
    edges = [allgc[min(len(allgc) - 1, int(len(allgc) * i / nbins))] for i in range(1, nbins)]

    def binof(g):
        for i, e in enumerate(edges):
            if g <= e:
                return i
        return nbins - 1

    num = den = 0.0
    for b_ in range(nbins):
        A = [v for v, g in a_pairs if binof(g) == b_]
        B = [v for v, g in b_pairs if binof(g) == b_]
        if len(A) < 10 or len(B) < 10:
            continue
        w = len(A) + len(B)
        num += w * ks(A, B); den += w
    return num / den if den else float("nan")


def ks(a, b):
    """Two-sample KS statistic on the pooled support."""
    if not a or not b:
        return float("nan")
    sa, sb = sorted(a), sorted(b)
    allv = sorted(set(sa) | set(sb))
    d = 0.0
    ia = ib = 0
    for v in allv:
        while ia < len(sa) and sa[ia] <= v:
            ia += 1
        while ib < len(sb) and sb[ib] <= v:
            ib += 1
        d = max(d, abs(ia / len(sa) - ib / len(sb)))
    return d


def perm_p(a, b, stat, nperm=NPERM):
    """Label-permutation null: exchange arm labels and recompute. No independence assumption on the draws."""
    obs = stat(a, b)
    pool = list(a) + list(b)
    na = len(a)
    hit = 0
    for _ in range(nperm):
        random.shuffle(pool)
        hit += stat(pool[:na], pool[na:]) >= obs
    return obs, (hit + 1) / (nperm + 1)


def collect(targets):
    agent, hand = [], []
    for op in OPS:
        for f in sorted(glob.glob(os.path.join(HERE, f"arep_{op}_dev", "s*_e*.json"))):
            for r in json.load(open(f))["rows"]:
                if r.get("seq") and r.get("pkiss") and r.get("probknot"):
                    agent.append(features(r["seq"], targets[r["id"]]))
    for s in SEEDS:
        for e in EXECS:
            p = os.path.join(HERE, f"gvp_e{e}_s{s}", "sweep.json")
            if not os.path.exists(p):
                continue
            d = json.load(open(p))
            if len(d.get("rows") or []) < 60:
                continue
            for r in d["rows"]:
                v = r["arms"].get("panel_auto") or {}
                if v.get("solved") and v.get("seq") and not v.get("error"):
                    hand.append(features(v["seq"], targets[r["id"]]))
    return agent, hand


def mean_sd(xs):
    m = sum(xs) / len(xs)
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1))


def main():
    tasks = load_pseudobase(limit=60)
    targets = {f"pseudobase:{i}": t.target for i, t in enumerate(tasks)}
    agent, hand = collect(targets)
    print("=" * 100)
    print("D3, seventh test: whole-distribution comparison of accepted designs (reviewer Q2)")
    print("=" * 100)
    print(f"  agent designs {len(agent)}   hand-built {len(hand)}   folded under ViennaRNA (in no arm's loop)")
    print("  Quantities the target does not fix; KS on the full CDF; permutation null on arm labels.\n")
    mA, mH, diag = caliper_match(agent, hand)
    print(f"  caliper-matched on GC (+/-0.01, no replacement): {diag[0]}/{diag[1]} hand-built designs paired; "
          f"matched mean GC {diag[2]:.4f} (agent) vs {diag[3]:.4f} (hand-built)\n")
    _, p_neg = perm_p([f["gc_all"] for f in mA], [f["gc_all"] for f in mH], ks)
    print(f"  NEGATIVE CONTROL on the matched sets: GC itself, perm p={p_neg:.4g} "
          f"({'PASS -- test is calibrated' if p_neg >= 0.05 else 'FAIL -- nothing below is readable'})\n")
    print(f"  {'quantity':12} {'agent (matched)':>19} {'hand-built (matched)':>21} {'KS D':>8} {'perm p':>10}")
    sig = []
    for k in KEYS:
        A = [f[k] for f in mA]; H = [f[k] for f in mH]
        maa, sa = mean_sd(A); mhh, sh = mean_sd(H)
        d, p = perm_p(A, H, ks)
        if p < 0.05:
            sig.append((k, d, p))
        print(f"  {k:12} {maa:11.3f}+-{sa:<6.3f} {mhh:13.3f}+-{sh:<6.3f} {d:8.3f} {p:10.4g}")
    if p_neg < 0.05:
        sig = None

    print("\n  " + "-" * 96)
    if sig is None:
        print("  The negative control did not pass even after caliper matching, so this question is NOT")
        print("  answerable on this dataset: the two arms' composition distributions overlap too poorly for")
        print("  any conditioning available at this sample size to equalise them, and every quantity above")
        print("  remains confounded with GC. We report the attempt and the failed control rather than the")
        print("  numbers, and record the distributional question as open for want of a design, not answered.")
    elif not sig:
        print("  NO distributional difference on any quantity. Combined with the six scalar mechanisms already")
        print("  excluded, the gap is not a global property of the accepted designs -- not their energetics,")
        print("  not their loop geometry, not their composition, not their search cost. That points away from")
        print("  anything a distributional summary can capture and toward how the two operators handle")
        print("  particular targets, which is the reviewer's own reading and the one we now adopt.")
    else:
        print("  Distributional differences found: " + ", ".join(f"{k} (D={d:.3f}, p={p:.3g})"
                                                                 for k, d, p in sig))
        print("  A lead, not a mechanism. Each faces the same transplant test every earlier candidate faced:")
        print("  bias the operator toward it and require carry-over to rise while r_in holds. We report it as")
        print("  Stage 1 evidence only -- `decoy` cleared exactly this bar and then reversed under")
        print("  intervention, which is why an observational difference is never the end of the argument.")
    print("\n  Six quantities on one dataset is multiple comparison; permutation p-values are uncorrected and")
    print("  nothing here is read from a bare p<0.05.")


if __name__ == "__main__":
    main()
