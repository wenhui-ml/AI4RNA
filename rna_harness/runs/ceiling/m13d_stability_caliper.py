#!/usr/bin/env python3
"""M13d: M13b's mechanism arm M1 was run with a caliper wider than the effect it had to hold fixed.

M13b matched ViennaRNA's target free energy at 0.2 x the POPULATION SD = 5.98 kcal/mol. The treatment
itself -- a gc_paired gap of 0.088 -- moves that same quantity by -5.47 kcal/mol. The caliper was 1.1x
the thing it was supposed to control, so "the effect survives dG matching" was not a finding. It is
withdrawn and re-measured here in WITHIN-CELL SD units (7.05 kcal/mol), swept, every rung reported.

If the effect decays to zero as the caliper tightens, the mechanism is thermodynamic stabilisation of
the intended base pairs -- which is what M13c's arms independently point at. If it survives a caliper
far below the treatment's own dG footprint, it is something else.

COST: $0, offline.
"""
from __future__ import annotations
import argparse, collections, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from m13_cache import POPS                                                            # noqa: E402
from m13b_gc_placement import assemble, match, point, pval, DELTA                     # noqa: E402

KS = (1.0, 0.5, 0.25, 0.125, 0.0625)


def within_sd(D, key):
    return float(np.std(np.concatenate([c[key] - c[key].mean() for c in D["cells"]
                                        if len(c[key]) >= 2]))) or 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--nperm", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join(HERE, "m13d_stability_caliper.json"))
    a = ap.parse_args()
    print("=== M13d:把目标结构自由能的 caliper 换成【格内 SD】单位并扫描 ===")
    print("  M13b 的 M1 臂用的是 0.2 x 全体SD = 5.98 kcal/mol,而处理本身移动 dG 达 -5.47 kcal/mol。")
    print("  **那一臂据此撤回。** 本轮重测。离线 $0。\n")
    C = json.load(open(os.path.join(HERE, "m13_features.json")))
    E = json.load(open(os.path.join(HERE, "m13b_extra.json")))
    D = {fn: assemble(C[fn]["rows"], E[fn], a.seed) for _nm, fn in POPS}
    OUT = {}
    for nm, fn in POPS:
        d = D[fn]; w = within_sd(d, "dg")
        treat = [c["gcp"] for c in d["cells"]]
        M0 = match(d, treat, DELTA)
        base = point(M0); n0 = M0["npair"]
        # dG footprint of the treatment, within cell
        num = den = 0.0
        for c in d["cells"]:
            t, h = c["gcp"] - c["gcp"].mean(), c["dg"] - c["dg"].mean()
            num += float((t * h).sum()); den += float((t * t).sum())
        foot = (num / den) * float(M0["dose"].mean()) if den else float("nan")
        print(f"=== {nm} ===  格内 dG SD {w:.2f} kcal/mol;处理的 dG 足迹 {foot:+.2f} kcal/mol;"
              f"无 dG 配平时 {base:+.3f}(n={n0})")
        rows = []
        for k in KS:
            cal = k * w
            d2 = dict(d); d2["sdg"] = cal / 0.20          # match() uses 0.20 * D["sdg"]
            M = match(d2, treat, DELTA, dg=True)
            if M is None or M["npair"] < 30:
                print(f"   caliper {cal:6.2f} kcal ({k:6.4f}x格内SD)  配对不足 "
                      f"({0 if M is None else M['npair']})")
                rows.append({"k": k, "cal": round(cal, 3), "npair": 0 if M is None else M["npair"]})
                continue
            obs = point(M)
            rng = np.random.default_rng(a.seed); null = []
            for _ in range(a.nperm):
                cols = [c["gcp"][rng.permutation(len(c["gcp"]))] for c in d["cells"]]
                Mp = match(d2, cols, DELTA, dg=True)
                null.append(point(Mp) if Mp else 0.0)
            null = np.sort(np.array(null)); p = pval(obs, null)
            adg = float(np.abs(M["dose"]).mean())
            rows.append({"k": k, "cal": round(cal, 3), "effect": round(obs, 3),
                         "p": round(p, 5), "npair": M["npair"], "ntgt": M["ntgt"],
                         "retain": round(obs / (base or 1e-9), 3)})
            print(f"   caliper {cal:6.2f} kcal ({k:6.4f}x格内SD)  效应 {obs:+7.3f}  "
                  f"保留 {obs/(base or 1e-9):6.1%}  p {p:.4f}  配对 {M['npair']:5d}/{M['ntgt']:3d}靶")
        OUT[fn] = {"pop": nm, "within_sd_dg": round(w, 3), "footprint": round(foot, 3),
                   "base": round(base, 3), "base_npair": n0, "rungs": rows}
        print()
    json.dump(OUT, open(a.out, "w"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
