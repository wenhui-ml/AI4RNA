#!/usr/bin/env python3
"""R2b: put intervals on the operating-point grid, and say which rungs are actually distinguishable.

`calibrate_precision` landed 2026-08-24 and nobody acted on it. The ceiling-branch memory still says
"do not change the verdict until calibrate_precision lands" -- it landed, and the three tables were never
written up. Worse, the grid is being read as if its numbers were exact:

    SHIPPED k>=1 recall==1     sensitivity 0.840   false accept 0.0040
    max f1 >= 0.4              sensitivity 0.860   false accept 0.0040   <- "假收不升、灵敏度更高"

with n_native = 250 and n_random = 250. A false accept of 0.0040 is ONE random out of 250, and 0.0000 is
ZERO out of 250, whose 95% upper bound is about 1.2%. Choosing an operating point by comparing 0.0040
against 0.0000 on 250 draws is the same class of error as quoting the 0.811 ceiling without an interval
(`r1b_audit.md`): a point estimate deciding something it cannot resolve.

This run adds Clopper-Pearson intervals to every rung, and -- the part that matters for a decision --
McNemar's exact paired test between the shipped rung and each candidate, since every rung is scored on
the SAME 250 natives and SAME 250 randoms. Paired, not two independent proportions.

COST: $0, offline, pure re-analysis of calibrate_precision.json.
"""
from __future__ import annotations

import json, math, os, sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "calibrate_precision.json")


def cp_interval(k, n, alpha=0.05):
    """Clopper-Pearson exact interval, computed by bisection on the binomial tail."""
    def tail_le(p, k_, n_):
        return sum(comb(n_, i) * p ** i * (1 - p) ** (n_ - i) for i in range(0, k_ + 1))
    lo, hi = 0.0, 1.0
    if k > 0:
        a, b = 0.0, 1.0
        for _ in range(200):
            m = (a + b) / 2
            if tail_le(m, k - 1, n) > 1 - alpha / 2:
                a = m
            else:
                b = m
        lo = (a + b) / 2
    if k < n:
        a, b = 0.0, 1.0
        for _ in range(200):
            m = (a + b) / 2
            if tail_le(m, k, n) < alpha / 2:
                b = m
            else:
                a = m
        hi = (a + b) / 2
    return lo, hi


def mcnemar_exact(b, c):
    """exact two-sided McNemar on the discordant pairs (b, c). Returns p."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n) * 2
    return min(1.0, p)


def main() -> int:
    d = json.load(open(SRC))
    nN, nR = d["n_native"], d["n_random"]
    rows = d["rows"]
    raw = d["raw"]
    print("=== R2b:给操作点网格装区间,并判哪些梯级真的分得开 ===")
    print(f"  来源 calibrate_precision.json,天然 {nN} 条、随机 {nR} 条,离线 $0\n")
    print(f"  {'规则':<32}{'灵敏度':>8}{'95% (Clopper-Pearson)':>26}{'假收':>8}{'95%':>24}")
    OUT = []
    for r in rows:
        kS = round(r["sensitivity"] * nN); kF = round(r["false_accept"] * nR)
        sl, sh = cp_interval(kS, nN); fl, fh = cp_interval(kF, nR)
        OUT.append({**r, "k_sens": kS, "ci_sens": [round(sl, 4), round(sh, 4)],
                    "k_fa": kF, "ci_fa": [round(fl, 4), round(fh, 4)]})
        print(f"  {r['rule']:<32}{r['sensitivity']:8.3f}   [{sl:.3f},{sh:.3f}]"
              f"{r['false_accept']:12.4f}   [{fl:.4f},{fh:.4f}]")

    print(f"\n  **假收 0.0040 = {round(0.004*nR)}/250,其 95% 是 [{cp_interval(1,nR)[0]:.4f},{cp_interval(1,nR)[1]:.4f}];"
          f"\n    假收 0.0000 = 0/250,其 95% 上界是 {cp_interval(0,nR)[1]:.4f} —— 两者区间几乎完全重叠。**")
    print("  ⇒ **这张网格分不开任何一对『假收』。用它挑操作点,挑的是灵敏度,不是假收。**")

    print("\n=== 必响探针:先用逐条记录重算出原表,一位不差,再谈比较 ===")

    def decide(rec, rule):
        """calibrate_precision.py 第 64 行的原语义:『None from an oracle is never an accept』。
        某个 oracle 返回 None 时,那一路不参与、也永不算接受。天然条无 None,随机条有 16 条含 None。"""
        f1 = rec["f1"]; rc = rec["recall"]; pr = rec["precision"]
        ors = sorted(f1)
        if rule == "SHIPPED k>=1 recall==1":
            return any(rc[o] is not None and rc[o] >= 1.0 for o in ors)
        if rule.startswith("max f1 >= "):
            t = float(rule.split(">=")[1])
            return any(f1[o] is not None and f1[o] >= t for o in ors)
        if rule.startswith("k>=1 recall==1 AND prec >= "):
            t = float(rule.split(">=")[-1])
            return any(rc[o] is not None and rc[o] >= 1.0
                       and pr[o] is not None and pr[o] >= t for o in ors)
        if rule.startswith("SCREEN mean recall >= "):
            t = float(rule.split(">=")[1])
            v = [rc[o] for o in ors if rc[o] is not None]
            return bool(v) and (sum(v) / len(v)) >= t
        raise ValueError(rule)

    nat = [r for r in raw if r["kind"] == "native"]
    ran = [r for r in raw if r["kind"] != "native"]
    DEC = {}
    bad = []
    for r in rows:
        dn = [decide(x, r["rule"]) for x in nat]
        dr = [decide(x, r["rule"]) for x in ran]
        DEC[r["rule"]] = (dn, dr)
        s_ = sum(dn) / len(dn); f_ = sum(dr) / len(dr)
        if abs(s_ - r["sensitivity"]) > 1e-9 or abs(f_ - r["false_accept"]) > 1e-9:
            bad.append((r["rule"], s_, r["sensitivity"], f_, r["false_accept"]))
    if bad:
        print(f"  ✗ 没响:{len(bad)} 条规则重算不上,例:{bad[0]}")
        print("  **本次运行作废(第 54 条):重算不出原表就不许拿它做任何比较。**")
        json.dump({"verdict": "void_cannot_reproduce_grid", "mismatch": bad[:5]},
                  open(os.path.join(HERE, "r2b_calibration_intervals.json"), "w"),
                  indent=1, ensure_ascii=False)
        return 0
    print(f"  ✓ 响:{len(rows)} 条规则的灵敏度与假收全部逐位重算一致"
          f"(天然 {len(nat)}、随机 {len(ran)})")

    print("\n=== 配对 McNemar:候选梯级 vs 出厂梯级(同一批样本,精确二项)===")
    SH = "SHIPPED k>=1 recall==1"
    sn, sr = DEC[SH]
    print(f"  {'规则':<32}{'灵敏度差':>10}{'b/c':>9}{'p(配对)':>10}{'假收差':>9}{'b/c':>9}{'p(配对)':>10}")
    PAIR = {}
    for r in rows:
        if r["rule"] == SH:
            continue
        dn, dr = DEC[r["rule"]]
        b = sum(1 for a, c in zip(dn, sn) if a and not c)
        c_ = sum(1 for a, c in zip(dn, sn) if c and not a)
        pS = mcnemar_exact(b, c_)
        b2 = sum(1 for a, c in zip(dr, sr) if a and not c)
        c2 = sum(1 for a, c in zip(dr, sr) if c and not a)
        pF = mcnemar_exact(b2, c2)
        PAIR[r["rule"]] = {"sens_delta": round(r["sensitivity"] - rows[0]["sensitivity"], 4),
                           "sens_bc": [b, c_], "p_sens": round(pS, 4),
                           "fa_delta": round(r["false_accept"] - rows[0]["false_accept"], 4),
                           "fa_bc": [b2, c2], "p_fa": round(pF, 4)}
        print(f"  {r['rule']:<32}{r['sensitivity']-rows[0]['sensitivity']:+10.3f}{str(b)+'/'+str(c_):>9}"
              f"{pS:10.4f}{r['false_accept']-rows[0]['false_accept']:+9.4f}{str(b2)+'/'+str(c2):>9}{pF:10.4f}")

    print("\n=== 这张网格到底能决定什么 ===")
    f4 = PAIR["max f1 >= 0.4"]
    print(f"  `max f1 >= 0.4` 相对出厂:灵敏度 {f4['sens_delta']:+.3f}(不一致对 {f4['sens_bc'][0]}/{f4['sens_bc'][1]},"
          f"配对 p = {f4['p_sens']:.4f});假收 {f4['fa_delta']:+.4f}(不一致对 {f4['fa_bc'][0]}/{f4['fa_bc'][1]},"
          f"p = {f4['p_fa']:.4f})")

    json.dump({"n_native": nN, "n_random": nR, "rows": OUT,
               "fa_1_of_250": [round(x, 4) for x in cp_interval(1, nR)],
               "fa_0_of_250": [round(x, 4) for x in cp_interval(0, nR)],
               "paired": PAIR},
              open(os.path.join(HERE, "r2b_calibration_intervals.json"), "w"),
              indent=1, ensure_ascii=False)
    print(f"\n  -> r2b_calibration_intervals.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
