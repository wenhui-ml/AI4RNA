#!/usr/bin/env python3
"""G2: molecules the panel certifies while the exact nested comparison says they cannot fold that way.

Pre-registered in g2_provable_false_accepts_declaration.json. Kill conditions there are frozen.

WHY THIS RUNS AFTER G1 CAME BACK NULL. G1's audit showed gamma*'s ranking content is reproduced by counting
G-C pairs in the crossing stem (M2 - GC = -0.0020 [-0.0791, +0.0728]). That kills the PREDICTION in section
five of the analysis document. It does not kill this run, because this one is a LOGICAL claim:

    F_nested > E_skeleton + dG_stack  =>  at ANY positive gamma the target loses to the exactly
                                          computed nested ensemble  =>  it cannot fold as intended

A one-sided rejector does not need to correlate with a continuous score to be sound. But the weakening must
be carried into the conclusion: with G1 null, this run's value rests entirely on the energy arithmetic being
right, and the independent empirical support from G1 is zero.

TWO thresholds, both reported: gamma* < 0 (raw) and gamma* < +2.4 (corrected; G3 measured dG_stack as
over-stabilising by 2.400 kcal/mol, so true gamma* = computed - 2.4). The correction can only ADD rejections,
so reporting only the favourable one would be cherry-picking. The correction is itself indicative, not exact
-- it was measured on a nested analogue.

COST: $0, local ViennaRNA only. Needs the `rna` env.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

TUBE_PASS = 80.0             # OpenKnot's own pass mark, used throughout this project
GAMMA_BIAS = 2.4             # g3_audit.md section five


def work(item):
    puz, seq, target, wet, f1c, method = item
    import RNA
    from gamma_star import crossing_set, stack_terms
    from rna_harness.benchmarks.rnainvbench import _round_only
    try:
        cross = crossing_set(target)
        if not cross:
            return {"puzzle": puz, "error": "target has no crossing pairs"}
        f = RNA.fold_compound(seq).pf()[1]
        e = RNA.energy_of_struct(seq, _round_only(target))
        terms = stack_terms(seq, cross)
        dg = sum(v for _, v in terms)
        gc = 0
        S = set(cross)
        for (i, j) in sorted(S):
            if (i + 1, j - 1) in S:
                for (x, y) in ((i, j), (i + 1, j - 1)):
                    if {seq[x], seq[y]} == {"G", "C"}:
                        gc += 1
    except Exception as exc:                                                   # noqa: BLE001
        return {"puzzle": puz, "error": repr(exc)[:120]}
    return {"puzzle": puz, "method": method, "wet": wet, "f1c": f1c,
            "gamma_star": round(f - e - dg, 3), "gc_cross": gc, "n_cross_stacks": len(terms)}


def rate(rows, key):
    n = len(rows)
    return (sum(1 for r in rows if key(r)) / n) if n else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(HERE, "g2_provable_false_accepts.json"))
    a = ap.parse_args()

    from leaderboard_design import targets
    from novelty_stratified_ruler import replay_sample
    T = targets()
    jobs_raw, _ = replay_sample(0, 80)

    # ---- void gate 1: the join must be asserted, not assumed ------------------------------------------
    src = json.load(open(os.path.join(HERE, "criterion_vs_tube.json")))["designs"]
    if len(src) != len(jobs_raw):
        print(f"  VOID: 条数不符 {len(src)} vs {len(jobs_raw)}")
        return 1
    join_bad = []
    for k, ((p, _s, f1c, wet, m), d) in enumerate(zip(jobs_raw, src)):
        if (p != d["puzzle"] or m != d["method"]
                or abs(wet - d["wet"]) > 1e-6 or abs(f1c - d["f1c"]) > 1e-6):
            join_bad.append(k)
    print(f"  void gate 1 -- 顺序连接断言:{len(src)} 条,不一致 {len(join_bad)}")
    if join_bad:
        print("  VOID: criterion_vs_tube.json 与 replay_sample 顺序不一致,连接不可信。")
        json.dump({"verdict": "void", "join_mismatches": join_bad[:50]}, open(a.out, "w"), indent=1)
        return 1

    # ---- void gate 3: confirmed semantics ------------------------------------------------------------
    xt = collections.Counter((d["panel_k"], d["confirmed"]) for d in src)
    ok_sem = all((k >= 1) == c for (k, c) in xt)
    print(f"  void gate 3 -- confirmed == (panel_k >= 1):{'成立' if ok_sem else '不成立'}  {dict(xt)}")
    if not ok_sem:
        json.dump({"verdict": "void", "why": "confirmed semantics changed"}, open(a.out, "w"), indent=1)
        return 1

    jobs = [(p, s, T[p]["target"], wet, f1c, m) for p, s, f1c, wet, m in jobs_raw]
    from concurrent.futures import ProcessPoolExecutor
    R = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, r in enumerate(ex.map(work, jobs, chunksize=16), 1):
            R.append(r)
            if n % 1200 == 0 or n == len(jobs):
                print(f"    {n}/{len(jobs)}", flush=True)
    for r, d in zip(R, src):
        r["panel_k"] = d["panel_k"]
        r["confirmed"] = d["confirmed"]

    bad = [r for r in R if "error" in r]
    rows = [r for r in R if "error" not in r]
    print(f"  void gate 4 -- 成功 {len(rows)},失败/无交叉 {len(bad)}")

    # ---- void gate 2: identity against the stored gamma* record --------------------------------------
    rec = json.load(open(os.path.join(HERE, "gamma_star_reproduce.json")))["rows"]
    from gamma_star import best_wet_per_target
    best = best_wet_per_target()
    mism = [p for p, want in rec.items()
            if abs(work((p, best[p]["seq"], T[p]["target"], 0.0, 0.0, "x"))["gamma_star"]
                   - want["gamma_star"]) > 0.011]
    print(f"  void gate 2 -- 对已存 γ* 记录的恒等校验:{len(rec)} 行,不一致 {len(mism)}")
    if mism:
        json.dump({"verdict": "void", "identity_mismatches": mism}, open(a.out, "w"), indent=1)
        return 1

    conf = [r for r in rows if r["confirmed"]]
    print(f"\n  面板 confirmed(panel_k >= 1)的设计:{len(conf)}/{len(rows)}")

    out = {"n": len(rows), "n_confirmed": len(conf), "n_failed": len(bad),
           "gates": {"join_mismatches": 0, "identity_mismatches": 0,
                     "confirmed_semantics_ok": True}}

    # ---- primary: the disagreement count, by threshold and by panel_k --------------------------------
    print(f"\n{'阈值':22}{'γ*<阈值总数':>12}{'其中面板 confirmed':>20}   分层 k=1 / k=2 / k=3")
    prim = {}
    for name, thr in (("原始 γ* < 0", 0.0), ("修正 γ* < +2.4", GAMMA_BIAS)):
        lo = [r for r in rows if r["gamma_star"] < thr]
        loc = [r for r in lo if r["confirmed"]]
        byk = collections.Counter(r["panel_k"] for r in loc)
        prim[name] = {"threshold": thr, "n_below": len(lo), "n_below_and_confirmed": len(loc),
                      "by_panel_k": {str(k): byk.get(k, 0) for k in (1, 2, 3)},
                      "frac_of_confirmed": round(len(loc) / len(conf), 5) if conf else None,
                      "n_targets_touched": len({r["puzzle"] for r in loc})}
        print(f"{name:22}{len(lo):12d}{len(loc):20d}   "
              f"{byk.get(1,0)} / {byk.get(2,0)} / {byk.get(3,0)}"
              f"   ({len(loc)/len(conf)*100:.2f}% of confirmed, {prim[name]['n_targets_touched']} 个靶标)")
    out["primary"] = prim

    K1 = prim["原始 γ* < 0"]["n_below_and_confirmed"] == 0
    print(f"\n  K1 单侧检验在本人群无咬合力(原始阈值下计数为 0):{'FIRES' if K1 else '不触发'}")

    # ---- secondary: does the tube back the rejector, vs the GC control G1 forces -----------------------
    print(f"\n  次观测:面板 confirmed 的设计里,试管通过率(通过线 {TUBE_PASS})")
    sec = {}
    byt = collections.defaultdict(list)
    for r in conf:
        byt[r["puzzle"]].append(r)
    tgts = sorted(byt)

    for name, thr in (("原始 γ* < 0", 0.0), ("修正 γ* < +2.4", GAMMA_BIAS)):
        lo = [r for r in conf if r["gamma_star"] < thr]
        hi = [r for r in conf if r["gamma_star"] >= thr]
        if len(lo) < 30:
            print(f"    {name}: γ*<阈值组 n={len(lo)} < 30,按闸门 5 只报计数不报率")
            sec[name] = {"n_low": len(lo), "too_small": True}
            continue
        # GC control at the SAME quantile, so the two splits cut the same fraction of the population
        q = len(lo) / len(conf)
        gcs = sorted(r["gc_cross"] for r in conf)
        gcut = gcs[max(0, min(len(gcs) - 1, int(q * len(gcs))))]
        glo = [r for r in conf if r["gc_cross"] <= gcut]
        ghi = [r for r in conf if r["gc_cross"] > gcut]
        pl, ph = rate(lo, lambda r: r["wet"] >= TUBE_PASS), rate(hi, lambda r: r["wet"] >= TUBE_PASS)
        gl, gh = rate(glo, lambda r: r["wet"] >= TUBE_PASS), rate(ghi, lambda r: r["wet"] >= TUBE_PASS)

        # cluster bootstrap over targets, paired: the gamma* gap minus the GC gap
        rng = random.Random(4242)
        Bg, Bc, Bd = [], [], []
        for _ in range(a.boot):
            s = [tgts[rng.randrange(len(tgts))] for _ in tgts]
            pool = [r for p in s for r in byt[p]]
            L = [r for r in pool if r["gamma_star"] < thr]
            H = [r for r in pool if r["gamma_star"] >= thr]
            GL = [r for r in pool if r["gc_cross"] <= gcut]
            GH = [r for r in pool if r["gc_cross"] > gcut]
            if not (L and H and GL and GH):
                continue
            g = rate(L, lambda r: r["wet"] >= TUBE_PASS) - rate(H, lambda r: r["wet"] >= TUBE_PASS)
            c = rate(GL, lambda r: r["wet"] >= TUBE_PASS) - rate(GH, lambda r: r["wet"] >= TUBE_PASS)
            Bg.append(g); Bc.append(c); Bd.append(g - c)

        def ci(v):
            s = sorted(v)
            return [round(s[int(0.025 * len(s))], 4), round(s[int(0.975 * len(s))], 4)] if s else None
        cg, cc, cd = ci(Bg), ci(Bc), ci(Bd)
        sec[name] = {"n_low": len(lo), "n_high": len(hi), "pass_low": round(pl, 4), "pass_high": round(ph, 4),
                     "gap_gamma": round(pl - ph, 4), "ci_gap_gamma": cg,
                     "gc_cut": gcut, "quantile": round(q, 4), "n_gc_low": len(glo), "n_gc_high": len(ghi),
                     "pass_gc_low": round(gl, 4), "pass_gc_high": round(gh, 4),
                     "gap_gc": round(gl - gh, 4), "ci_gap_gc": cc,
                     "gamma_minus_gc": round((pl - ph) - (gl - gh), 4), "ci_diff": cd,
                     "n_boot_used": len(Bd)}
        print(f"    {name}:  γ*<阈值 n={len(lo)} 通过率 {pl:.4f}   γ*>=阈值 n={len(hi)} 通过率 {ph:.4f}"
              f"   差 {pl-ph:+.4f} {cg}")
        print(f"      GC 对照(同分位切点 gc<={gcut}, q={q:.3f}): n={len(glo)} 通过率 {gl:.4f}"
              f" vs n={len(ghi)} {gh:.4f}   差 {gl-gh:+.4f} {cc}")
        print(f"      γ* 的差 − GC 的差 = {(pl-ph)-(gl-gh):+.4f}  {cd}"
              f"   {'排除 0' if cd and (cd[0] > 0 or cd[1] < 0) else '含 0'}")
    out["secondary"] = sec

    s0 = sec.get("原始 γ* < 0", {})
    K2 = not (s0.get("ci_diff") and (s0["ci_diff"][0] > 0 or s0["ci_diff"][1] < 0))
    K3 = s0.get("gap_gamma") is not None and s0["gap_gamma"] >= 0
    print(f"\n  K2 否决器无独立经验支持(γ* 切分不优于 GC 切分):{'FIRES' if K2 else '不触发'}")
    print(f"  K3 方向反了(γ*<0 组通过率不低于对照组)        :{'FIRES' if K3 else '不触发'}")

    verdict = ("no_bite" if K1 else "direction_reversed" if K3 else
               "count_only_no_empirical_support" if K2 else "survives")
    out["kills"] = {"K1": K1, "K2": K2, "K3": K3}
    out["verdict"] = verdict
    out["carried_weakening"] = ("G1 已测出构造一不携带超出平凡 GC 计数的排序信息,"
                               "所以本实验的价值完全押在能量算术本身正确上,来自 G1 的独立经验支持为零。")
    out["correction_is_indicative"] = ("+2.4 由 G3 在【嵌套】类比上量出;嵌套环闭合与交叉环闭合是"
                                       "不同的物理项,故修正阈值是指示性的,不是精确的。")
    print(f"\n  裁定:{verdict}")
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
