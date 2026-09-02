#!/usr/bin/env python3
"""M68 arm B, second half: the numerator/denominator decomposition AFTER excision, at each radius.

Declaration: m68_review_gaps_declaration.json, arm B, "also"; prediction S6.

Section 5.3 reports that the whole of any base-composition effect on R sits in the denominator, and
that the denominator does carry most of the movement -- but it reports that BEFORE excision only.
The excision sweep then reports the composite R at each radius and never says what happened to the
two pieces. S6 predicts the denominator's two-arm difference shrinks with radius. This runs the
same estimator M61b used, on the same rows M61b built, on D and rho_ij separately.

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

RADII = (None, 0, 1, 2, 3)


def main():
    recs, _et = M61.build()
    rows = M61.flat(recs)
    print(f"records {len(recs)}  (M61b's own set)", flush=True)

    OUT = {"radii": [], "note": "same rows, same estimator, same strata as M61b's A6"}
    print(f"\n  {'radius':>7}{'n':>7}{'D contrast':>13}{'rho_ij contrast':>17}"
          f"{'R contrast':>13}{'from den':>11}{'den share':>11}", flush=True)
    for rad in RADII:
        newrows = []
        for r, f in zip(recs, rows):
            P = r["prof"]; L = len(P[0])
            keep = np.ones(L, bool)
            if rad is not None:
                lo1, hi1 = max(0, r["a_idx"] - rad), min(L, r["a_idx"] + rad + 1)
                lo2, hi2 = max(0, r["b_idx"] - rad), min(L, r["b_idx"] + rad + 1)
                keep[lo1:hi1] = False; keep[lo2:hi2] = False
            ri = M61B.rmsd_mask(P[0], P[1], keep); rj = M61B.rmsd_mask(P[0], P[2], keep)
            rij = M61B.rmsd_mask(P[0], P[3], keep)
            if not (np.isfinite(ri) and np.isfinite(rj) and np.isfinite(rij)):
                continue
            D = math.sqrt(ri ** 2 + rj ** 2)
            if D <= 0:
                continue
            g = dict(f)
            g["Dex"] = D; g["Nex"] = rij; g["Rex"] = 1 - rij / D
            newrows.append(g)
        eD = M61.ctr(newrows, "Dex")
        eN = M61.ctr(newrows, "Nex")
        eR = M61.ctr(newrows, "Rex")
        Dm = float(np.mean([g["Dex"] for g in newrows]))
        Nm = float(np.mean([g["Nex"] for g in newrows]))
        # the same first-order decomposition M61 used on the unexcised readout
        from_num = -eN.effect / Dm
        from_den = Nm * eD.effect / Dm ** 2
        tot = abs(from_num) + abs(from_den)
        rec = {"radius": rad, "n": len(newrows),
               "D": {"effect": round(eD.effect, 5), "ci": [round(x, 4) for x in eD.ci],
                     "p": round(eD.p, 5)},
               "rho_ij": {"effect": round(eN.effect, 5), "ci": [round(x, 4) for x in eN.ci],
                          "p": round(eN.p, 5)},
               "R": {"effect": round(eR.effect, 5), "ci": [round(x, 4) for x in eR.ci]},
               "D_mean": round(Dm, 5), "rho_ij_mean": round(Nm, 5),
               "from_numerator": round(from_num, 5), "from_denominator": round(from_den, 5),
               "denominator_share": round(abs(from_den) / tot, 4) if tot else None}
        OUT["radii"].append(rec)
        print(f"  {str(rad):>7}{len(newrows):>7}{eD.effect:>+13.5f}{eN.effect:>+17.5f}"
              f"{eR.effect:>+13.5f}{from_den:>+11.5f}{100*rec['denominator_share']:>10.1f}%",
              flush=True)

    d0 = abs(OUT["radii"][0]["D"]["effect"])
    d3 = abs(OUT["radii"][-1]["D"]["effect"])
    OUT["S6"] = bool(d3 < d0)
    OUT["D_shrink_frac"] = round(1 - d3 / d0, 4) if d0 else None
    print(f"\nS6 (denominator's two-arm difference shrinks with radius): {OUT['S6']}"
          f"   |D| falls {100*(1-d3/d0):.1f}% from no-excision to +-3", flush=True)
    # where the composite ends up relative to the two pieces
    OUT["den_share_no_excision"] = OUT["radii"][0]["denominator_share"]
    OUT["den_share_radius_three"] = OUT["radii"][-1]["denominator_share"]
    json.dump(OUT, open(os.path.join(HERE, "m68d_denominator_by_radius.json"), "w"), indent=1)
    print("\nwrote m68d_denominator_by_radius.json", flush=True)


if __name__ == "__main__":
    main()
