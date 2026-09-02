#!/usr/bin/env python3
"""M68 arm E: buy 10^6-draw precision for the excision contrast, and repair M64's sidedness.

DEFECT FOUND WHILE RUNNING THIS (recorded as E2 in the erratum): M64's purchase engine computes a
ONE-SIDED p, (null >= d), while every published p in the frozen family comes from
designaudit.core.contrast, which is TWO-SIDED, (|null| >= |d|). M64's equivalence gate compared the
statistic and the null width, both of which are sidedness-blind, and every arm it bought sat at the
permutation floor where the two conventions agree, so nothing caught it. The five purchased values
in M64 are therefore understated by about a factor of two and were substituted into a family whose
other members are two-sided. This script reports both conventions, uses the two-sided one to match
the family, and re-derives M64's five purchased arms on identical seeds so the correction can be
checked rather than asserted.

Declaration: m68_review_gaps_declaration.json, arm E.

The manuscript says cxP3 does not survive the family correction of M64 and that buying precision
cannot change that, because 0.0305 is not at the permutation floor. That is an argument. The
declaration says to buy the resolution, so it is bought and the sentence becomes a measurement.

The engine is M64's, unchanged: the same within-stratum label permutation written as random
subset-sums. It is gated exactly as M64 gated it -- the vectorised statistic and the null width must
reproduce the published ones at the published draw count before any purchased p is read.

COST: $0, offline, CPU only.
"""
from __future__ import annotations
import json, math, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, "/data/wenhui/openclaude_rna/rna_harness/tools")
import m61_readout_and_consequence as M61
import m61b_readout_and_consequence as M61B
import m64_multiplicity_and_shape as M64

RADIUS = 3
NPERM_CHECK = 2000
NPERM_BUY = 1_000_000
SEED = 20260902


def two_sided(null, d, nperm):
    """The convention every published p in this family uses (designaudit.core.contrast)."""
    return (float((np.abs(null) >= abs(d)).sum()) + 1.0) / (nperm + 1.0)


def strat(r):
    """M61.ctr's own strata: the design cell crossed with depth and crossing."""
    return (r["design"], r["depth3"], r["crossed"])


def main():
    pubj = json.load(open(os.path.join(HERE, "m61b_readout_and_consequence.json")))
    pub = pubj["A6_off_mutation_cut"][str(RADIUS)]
    print(f"published radius-{RADIUS}: effect {pub['effect']:+.5f}  p {pub['p']:.5f}  "
          f"weight {pub['weight']:.0f}", flush=True)

    recs, _et = M61.build()
    rows = M61.flat(recs)
    newrows = []
    for r, f in zip(recs, rows):
        P = r["prof"]; L = len(P[0])
        keep = np.ones(L, bool)
        lo1, hi1 = max(0, r["a_idx"] - RADIUS), min(L, r["a_idx"] + RADIUS + 1)
        lo2, hi2 = max(0, r["b_idx"] - RADIUS), min(L, r["b_idx"] + RADIUS + 1)
        keep[lo1:hi1] = False; keep[lo2:hi2] = False
        ri = M61B.rmsd_mask(P[0], P[1], keep); rj = M61B.rmsd_mask(P[0], P[2], keep)
        rij = M61B.rmsd_mask(P[0], P[3], keep)
        if not (np.isfinite(ri) and np.isfinite(rj) and np.isfinite(rij)):
            continue
        D = math.sqrt(ri ** 2 + rj ** 2)
        if D <= 0:
            continue
        g = dict(f); g["rex"] = 1 - rij / D
        newrows.append(g)
    print(f"records rebuilt: {len(newrows)}  (published n {pub['n']})", flush=True)

    S = M64.prep(newrows, "CG", "AU", strat, field="rex")
    d, w, ns = M64.obs(S)
    nl = M64.null_vec(S, NPERM_CHECK, SEED)
    p_check = two_sided(nl, d, NPERM_CHECK)
    dok = abs(d - pub["effect"]) <= 1e-6 + 5e-5
    wok = abs(w - pub["weight"]) < 0.5
    print(f"\nequivalence gate at {NPERM_CHECK} draws:", flush=True)
    print(f"  statistic {d:+.6f} vs published {pub['effect']:+.5f}   -> {dok}", flush=True)
    print(f"  weight {w:.0f} vs published {pub['weight']:.0f}   strata {ns}   -> {wok}", flush=True)
    p_check_one = M64.pval(nl, d, NPERM_CHECK)
    print(f"  p at {NPERM_CHECK}: two-sided {p_check:.5f} vs published {pub['p']:.5f}"
          f"   (one-sided would be {p_check_one:.5f})", flush=True)
    OUT = {"seed": SEED, "radius": RADIUS, "published": pub,
           "gate": {"d": round(d, 6), "weight": w, "n_strata": ns,
                    "p_at_check": round(p_check, 5), "d_matches": bool(dok),
                    "weight_matches": bool(wok), "pass": bool(dok and wok)}}
    if not (dok and wok):
        OUT["verdict"] = "gate_failed_no_purchased_p_is_read"
        print("\nGATE FAILED: no purchased p is read; the published value stands.", flush=True)
        json.dump(OUT, open(os.path.join(HERE, "m68g_buy_excision_precision.json"), "w"), indent=1)
        return

    print(f"\nbuying {NPERM_BUY} draws ...", flush=True)
    nlb = M64.null_vec(S, NPERM_BUY, SEED + 1000)
    p = two_sided(nlb, d, NPERM_BUY)
    p_one = M64.pval(nlb, d, NPERM_BUY)
    ge = int((np.abs(nlb) >= abs(d)).sum())
    sd = float(nlb.std())
    print(f"  null SD {sd:.6f}   z {d/sd:+.3f}   |null| >= |d| in {ge} / {NPERM_BUY}"
          f"   two-sided p = {p:.3e}   (published {pub['p']:.5f}; one-sided would be "
          f"{p_one:.3e})", flush=True)
    OUT["bought"] = {"d": round(d, 6), "p": p, "p_one_sided": p_one, "n_ge": ge,
                     "nperm": NPERM_BUY,
                     "null_sd": round(sd, 6), "z": round(d / sd, 3) if sd > 0 else None,
                     "published_p": pub["p"]}

    # re-run the frozen family's Holm with cxP3 resolved, changing nothing else
    m64j = json.load(open(os.path.join(HERE, "m64_multiplicity_and_shape.json")))
    resolved = dict(m64j["B_corrected"]["resolved_p"])
    before = dict(m64j["B_corrected"]["holm_reject"])
    resolved["cxP3"] = p
    hol, thr = M64.holm(resolved)
    bhr = M64.bh(resolved)
    OUT["family_after_purchase"] = {
        "m": len(resolved), "cxP3_p": p, "cxP3_holm_threshold": thr["cxP3"],
        "cxP3_holm_reject": bool(hol["cxP3"]), "cxP3_bh_reject": bool(bhr["cxP3"]),
        "n_holm_reject_before": sum(1 for v in before.values() if v),
        "n_holm_reject_after": sum(1 for v in hol.values() if v),
        "changed": sorted(k for k in hol if bool(hol[k]) != bool(before.get(k)))}
    print(f"\nfamily of {len(resolved)} with cxP3 resolved:", flush=True)
    print(f"  cxP3 p={p:.3e}   Holm threshold {thr['cxP3']:.3e}   "
          f"reject: {bool(hol['cxP3'])}   BH reject: {bool(bhr['cxP3'])}", flush=True)
    print(f"  Holm rejections {OUT['family_after_purchase']['n_holm_reject_before']} -> "
          f"{OUT['family_after_purchase']['n_holm_reject_after']}"
          f"   changed members: {OUT['family_after_purchase']['changed'] or 'none'}", flush=True)
    # ---- E2: re-derive M64's five purchased arms two-sided, on identical seeds
    print("\n=== E2: M64's five purchased arms, re-derived two-sided on identical seeds ===",
          flush=True)
    import m16_rescue as M16, m21_thermo_mediator as M21
    recs16 = M16.build()
    recs21 = M21.build_with_dg()
    edges = M21.bin_edges(recs21, M21.NBINS)
    for r in recs21:
        r["bin"] = M21.assign_bin(r["dG"], edges)
    ARMS = [
        ("rescueAUP", recs16, "CG", "AU", M16.stratum),
        ("rescueGUP", recs16, "CG", "GU", M16.stratum),
        ("eP", recs21, "CG", "AU", M21.strat_dg),
        ("eDZeroP", [r for r in recs21 if r["depth"] == 0], "CG", "AU",
         (lambda r: (r["design"], r["crossed"]))),
        ("eDTwoP", [r for r in recs21 if r["depth"] >= 2], "CG", "AU",
         (lambda r: (r["design"], r["crossed"]))),
    ]
    rep = {}
    for i, (nm, rr, A, B, sfn) in enumerate(ARMS):
        Si = M64.prep(rr, A, B, sfn)
        di, wi, nsi = M64.obs(Si)
        nli = M64.null_vec(Si, NPERM_BUY, M64.SEED + 1000 + 100 * i)
        p2 = two_sided(nli, di, NPERM_BUY)
        p1 = M64.pval(nli, di, NPERM_BUY)
        was = m64j["B"][nm]["p"]
        rep[nm] = {"d": round(di, 6), "p_two_sided": p2, "p_one_sided": p1,
                   "m64_reported": was, "one_sided_reproduces": bool(abs(p1 - was) < 1e-12)}
        print(f"  {nm:11s} one-sided {p1:.3e} (M64 printed {was:.3e}, reproduces "
              f"{rep[nm]['one_sided_reproduces']})   two-sided {p2:.3e}", flush=True)
        resolved[nm] = p2
    OUT["E2_m64_sidedness_repair"] = rep
    hol3, thr3 = M64.holm(resolved)
    bh3 = M64.bh(resolved)
    OUT["family_fully_two_sided"] = {
        "m": len(resolved),
        "n_holm_reject": sum(1 for v in hol3.values() if v),
        "holm_members": sorted(k for k in hol3 if hol3[k]),
        "n_bh_reject": sum(1 for v in bh3.values() if v),
        "changed_vs_m64": sorted(k for k in hol3 if bool(hol3[k]) != bool(before.get(k))),
        "cxP3_holm_reject": bool(hol3["cxP3"]), "cxP3_bh_reject": bool(bh3["cxP3"])}
    F = OUT["family_fully_two_sided"]
    print(f"\nfamily of {F['m']} with every purchased p two-sided:", flush=True)
    print(f"  Holm rejects {F['n_holm_reject']}   BH rejects {F['n_bh_reject']}", flush=True)
    print(f"  members: {F['holm_members']}", flush=True)
    print(f"  decisions changed against M64 as published: {F['changed_vs_m64'] or 'NONE'}",
          flush=True)
    OUT["verdict"] = "complete"
    json.dump(OUT, open(os.path.join(HERE, "m68g_buy_excision_precision.json"), "w"), indent=1)
    print("\nwrote m68g_buy_excision_precision.json", flush=True)


if __name__ == "__main__":
    main()
