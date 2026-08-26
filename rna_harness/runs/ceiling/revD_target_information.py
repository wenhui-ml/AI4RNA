#!/usr/bin/env python3
"""REVIEW D: does optimal experimental design have anything to work with?

WHAT I SAID. Borrow OED / IRT to choose which targets and designs to measure. That is infrastructure,
not a discovery, and I said so. What I did NOT check is whether its premise holds at all.

D's premise is that targets differ in how much information they carry about a criterion. Localise
r1b's variance decomposition to one target:

    rho_t = 1 - sigma_noise^2 / sigma_within-cell(t)^2 ,     sigma_noise = 3.7716 (r1b, 557 groups)

If rho_t is near zero for most targets, then the wet spread on those targets is mostly tube noise, no
criterion can rank designs there, and no experimental design fixes it -- D would be pointless. If
rho_t varies a lot, D has something to select on, and the selection is worth its cost.

Also reported: rho_t against designs-per-target and against target length, because M9-pre-d found the
per-target quantity is DESIGN-COUNT limited, and this is the same question on a different quantity.

Offline, $0.
"""
from __future__ import annotations
import argparse, collections, json, math, os, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from scipy.stats import spearmanr                                              # noqa: E402
from e1_ensemble_marginals import load_pop                                     # noqa: E402

POPS = (("样本2(主)", "panel_f1_fresh.json"), ("基准4560", "panel_f1_designs.json"),
        ("样本3", "panel_f1_third.json"), ("第四批", "panel_f1_fourth.json"))
SIGMA_NOISE = 3.7716          # r1b, variance-moment estimator on 557 repeat groups


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-designs", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(HERE, "revD_target_information.json"))
    a = ap.parse_args()
    print("=== REVIEW D:最优实验设计有没有东西可以选 ===")
    print(f"  事后复核。把 r1b 的方差分解落到单个靶标上,sigma_noise = {SIGMA_NOISE}(557 组)\n")
    from leaderboard_design import targets as load_targets
    T = load_targets()
    OUT = {}
    for nm, fn in POPS:
        rec, _s, _m = load_pop(fn)
        by = collections.defaultdict(list)
        for r in rec:
            if r.get("wet") is not None:
                by[r["puzzle"]].append(float(r["wet"]))
        rows = []
        for puz, ys in sorted(by.items()):
            if len(ys) < a.min_designs:
                continue
            sd = st.pstdev(ys)
            rho = 1.0 - (SIGMA_NOISE ** 2) / (sd ** 2) if sd > 0 else float("-inf")
            L = len(T[puz]["target"]) if puz in T else None
            rows.append({"puzzle": puz, "n": len(ys), "sd": round(sd, 3),
                         "rho_t": round(rho, 4), "len": L})
        if not rows:
            print(f"  {nm}: 无数据\n"); continue
        rows.sort(key=lambda d: -d["rho_t"])
        pos = [r for r in rows if r["rho_t"] > 0]
        print(f"=== {nm} ===  靶标 {len(rows)}")
        print(f"  rho_t > 0 的靶标:**{len(pos)}/{len(rows)}** "
              f"({len(pos)/len(rows)*100:.1f}%)  <- 只有这些上面排序才有意义")
        print(f"  rho_t 中位 {st.median([r['rho_t'] for r in rows]):+.4f},"
              f"上四分位 {sorted(r['rho_t'] for r in rows)[int(.75*len(rows))]:+.4f}")
        print(f"  最好的五个:", ", ".join(f"{r['puzzle']}({r['rho_t']:+.2f},n={r['n']})" for r in rows[:5]))
        L = [(r["len"], r["rho_t"]) for r in rows if r["len"]]
        N = [(r["n"], r["rho_t"]) for r in rows]
        rl = spearmanr([x for x, _ in L], [y for _, y in L])[0] if len(L) > 5 else None
        rn = spearmanr([x for x, _ in N], [y for _, y in N])[0] if len(N) > 5 else None
        print(f"  rho_t vs 靶标长度 rho = {'--' if rl is None else f'{rl:+.4f}'}"
              f" | rho_t vs 该靶标设计数 rho = {'--' if rn is None else f'{rn:+.4f}'}")
        OUT[fn] = {"pop": nm, "n_targets": len(rows), "n_positive": len(pos),
                   "frac_positive": round(len(pos) / len(rows), 4),
                   "median_rho_t": round(st.median([r["rho_t"] for r in rows]), 4),
                   "rho_vs_len": None if rl is None else round(float(rl), 4),
                   "rho_vs_ndesign": None if rn is None else round(float(rn), 4),
                   "rows": rows}
        print()
    if OUT:
        f = st.median([v["frac_positive"] for v in OUT.values()])
        v = ("no_target_carries_signal__OED_pointless" if f < 0.15 else
             "few_targets_carry_signal__OED_has_a_lot_to_select_on" if f < 0.6 else
             "most_targets_carry_signal__OED_buys_little")
        print(f"=== 裁定 ===\n  **{v}**  (rho_t>0 的靶标比例中位 {f:.3f})")
        OUT["_summary"] = {"verdict": v, "median_frac_positive": f}
    json.dump(OUT, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
