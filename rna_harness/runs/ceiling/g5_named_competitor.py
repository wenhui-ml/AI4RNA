#!/usr/bin/env python3
"""G5: list the 123, and replace the ensemble with a competitor that can be printed.

Pre-registered in g5_named_competitor_declaration.json.

TWO JOBS. First, produce the per-molecule listing of G2's 123 designs (skeleton formable, panel-confirmed,
gamma* < 0) with everything needed to interrogate the panel: sequence, which oracles confirmed, what they
predicted, the tube reading. Second, repair the defect g2_audit.md named as rule 36b -- F_nested <= E_skeleton
is an IDENTITY, so gamma* subtracts a single-structure quantity from an ensemble quantity and its sign is
partly structural rather than a comparison.

    gamma* = F_nested        - E_skeleton - dG_stack     ensemble vs one structure  (biased inequality)
    gamma+ = E_MFE_nested    - E_skeleton - dG_stack     one structure vs one structure

E_MFE_nested >= F_nested identically, so gamma+ >= gamma* -- a strictly stronger condition, and void gate 2
asserts it per molecule. A design with gamma+ < 0 loses to a structure that can be PRINTED, at a stated
energy gap. That is checkable by a reader; "loses to the aggregate of all nested alternatives" is not.

NOT a rehabilitation of gamma*. G1 showed the ranking content of this whole line is reproduced by counting
G-C pairs in the crossing stem, and this run reports no correlation and claims no advantage over that count.
NOT a proof of panel error either -- g2_audit.md withdrew that. A free-energy gap is relative population, not
impossibility.

COST: $0, local ViennaRNA only. Needs the `rna` env.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

TUBE_PASS = 80.0
GAMMA_BIAS = 2.4
CANON = {frozenset("GC"), frozenset("AU"), frozenset("GU")}
ORACLES = ("pkiss", "probknot", "shapeknots")


def work(item):
    puz, seq, target, wet, f1c, method = item
    import RNA
    from gamma_star import crossing_set, stack_terms
    from rna_harness.benchmarks.rnainvbench import _round_only, pk_pairs
    try:
        cross = crossing_set(target)
        if not cross:
            return {"puzzle": puz, "error": "no crossing pairs"}
        skel = _round_only(target)
        nc = sum(1 for (i, j) in pk_pairs(skel) if frozenset({seq[i], seq[j]}) not in CANON)
        f = RNA.fold_compound(seq).pf()[1]
        mfe_s, mfe_e = RNA.fold(seq)
        e = RNA.energy_of_struct(seq, skel)
        terms = stack_terms(seq, cross)
        dg = sum(v for _, v in terms)
        gc = 0
        S = set(cross)
        for (i, j) in sorted(S):
            if (i + 1, j - 1) in S:
                for (x, y) in ((i, j), (i + 1, j - 1)):
                    if {seq[x], seq[y]} == {"G", "C"}:
                        gc += 1
        # how different is the named competitor from the intended skeleton
        sp, mp = pk_pairs(skel), pk_pairs(mfe_s)
        shared = len(sp & mp)
        # Does the named competitor actually COMPETE? A nested MFE that merely closes extra pairs
        # ELSEWHERE is not a competitor -- gamma+ < 0 then just compares a bigger structure against a
        # smaller one. A competitor that takes the very nucleotides the crossing stem needs is the nested
        # alternative that EATS the pseudoknot. This column separates the two, and the k=3 read showed the
        # split is not incidental: over the 82 it runs from 0 to all.
        cnt = {x for p in cross for x in p}
        snt = {x for p in sp for x in p}
        mnt = {x for p in mp for x in p}
        stolen = [n for n in cnt if n in mnt and n not in snt]
    except Exception as exc:                                                   # noqa: BLE001
        return {"puzzle": puz, "error": repr(exc)[:120]}
    return {"puzzle": puz, "method": method, "seq": seq, "wet": wet, "f1c": f1c, "L": len(seq),
            "nc_skel": nc, "F_nested": round(f, 2), "E_mfe": round(mfe_e, 2), "mfe_struct": mfe_s,
            "E_skeleton": round(e, 2), "dG_stack": round(dg, 2), "n_cross_stacks": len(terms),
            "gc_cross": gc, "gamma_star": round(f - e - dg, 2), "gamma_dag": round(mfe_e - e - dg, 2),
            "skel_pairs": len(sp), "mfe_shares_skel_pairs": shared,
            "cross_nt": len(cnt), "stolen_nt": len(stolen),
            "stolen_frac": round(len(stolen) / len(cnt), 3) if cnt else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(HERE, "g5_named_competitor.json"))
    ap.add_argument("--md", default=os.path.join(HERE, "g5_the_123.md"))
    a = ap.parse_args()

    from leaderboard_design import targets
    from novelty_stratified_ruler import replay_sample
    T = targets()
    jobs_raw, _ = replay_sample(0, 80)
    cvt = json.load(open(os.path.join(HERE, "criterion_vs_tube.json")))["designs"]
    pf1 = json.load(open(os.path.join(HERE, "panel_f1_designs.json")))["designs"]

    # ---- void gate 1: three-way order join, asserted ---------------------------------------------------
    bad = []
    for k, ((p, _s, f1c, wet, m), c, q) in enumerate(zip(jobs_raw, cvt, pf1)):
        for d in (c, q):
            if (p != d["puzzle"] or m != d["method"] or abs(wet - d["wet"]) > 1e-6
                    or abs(f1c - d["f1c"]) > 1e-6):
                bad.append(k)
                break
    print(f"  闸门1 三方顺序连接:{len(jobs_raw)} 条,不一致 {len(bad)}")
    if bad:
        print("  VOID"); json.dump({"verdict": "void"}, open(a.out, "w")); return 1

    # ---- void gate 4: recomputed k must equal the stored panel_k ---------------------------------------
    kmis = 0
    conf_by = []
    for c, q in zip(cvt, pf1):
        po = q.get("per_oracle") or {}
        cb = [o for o in ORACLES if (po.get(o) or {}).get("recall_product") is not None
              and po[o]["recall_product"] >= 1.0]
        conf_by.append(cb)
        kmis += (len(cb) != c["panel_k"])
    print(f"  闸门4 重算 k 与已存 panel_k:不一致 {kmis}")
    if kmis:
        print("  VOID"); json.dump({"verdict": "void", "k_mismatches": kmis}, open(a.out, "w")); return 1

    jobs = [(p, s, T[p]["target"], wet, f1c, m) for p, s, f1c, wet, m in jobs_raw]
    from concurrent.futures import ProcessPoolExecutor
    R = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, r in enumerate(ex.map(work, jobs, chunksize=16), 1):
            R.append(r)
            if n % 1500 == 0 or n == len(jobs):
                print(f"    {n}/{len(jobs)}", flush=True)
    for r, c, cb in zip(R, cvt, conf_by):
        r["panel_k"], r["confirmed"], r["confirmed_by"] = c["panel_k"], c["confirmed"], cb

    err = [r for r in R if "error" in r]
    rows = [r for r in R if "error" not in r]
    print(f"  闸门5 失败条数 {len(err)}")

    # ---- void gate 2: gamma_dag >= gamma_star, per molecule (E_MFE >= F_nested is an identity) ---------
    viol = [r for r in rows if r["gamma_dag"] < r["gamma_star"] - 1e-6]
    print(f"  闸门2 γ† >= γ* 逐条自检:违反 {len(viol)}")
    if viol:
        print("  VOID: 恒等式被违反,估计量坏了")
        json.dump({"verdict": "void", "violations": len(viol)}, open(a.out, "w")); return 1

    # ---- void gate 3: skeleton formable ----------------------------------------------------------------
    clean = [r for r in rows if r["nc_skel"] == 0]
    print(f"  闸门3 骨架可形成:{len(clean)}/{len(rows)}(剔除 {len(rows)-len(clean)} 条)")

    conf = [r for r in clean if r["confirmed"]]
    L = sorted([r for r in conf if r["gamma_star"] < 0],
               key=lambda r: (-r["panel_k"], r["gamma_star"]))
    print(f"\n  G2 的那一批:骨架干净 + 面板 confirmed + γ*<0 = {len(L)} 条")

    def pr(g):
        return sum(1 for r in g if r["wet"] >= TUBE_PASS) / len(g) if g else None

    dag0 = [r for r in L if r["gamma_dag"] < 0]
    dag24 = [r for r in L if r["gamma_dag"] < GAMMA_BIAS]
    print(f"\n  ── 单竞争者加强(γ† = E_MFE_nested − E_skeleton − dG_stack)──")
    print(f"  γ† < 0     :{len(dag0):3d} 条 / {len(L)}   试管通过率 "
          f"{('%.4f' % pr(dag0)) if dag0 else 'n/a'}   k=1/2/3 = "
          f"{'/'.join(str(sum(1 for r in dag0 if r['panel_k']==k)) for k in (1,2,3))}")
    print(f"  γ† < +2.4  :{len(dag24):3d} 条 / {len(L)}   试管通过率 "
          f"{('%.4f' % pr(dag24)) if dag24 else 'n/a'}   k=1/2/3 = "
          f"{'/'.join(str(sum(1 for r in dag24 if r['panel_k']==k)) for k in (1,2,3))}")
    print(f"  对照:γ*<0 全体 {len(L)} 条   试管通过率 {pr(L):.4f}")

    K1 = len(dag0) == 0
    K2 = (not K1) and pr(dag0) >= pr(L)
    print(f"\n  K1 单竞争者版本无咬合力(γ†<0 计数为 0):{'FIRES' if K1 else '不触发'}")
    print(f"  K2 加强是空的(γ†<0 通过率不低于 γ*<0)  :{'FIRES' if K2 else '不触发'}")
    verdict = "no_bite" if K1 else "sharpening_empty" if K2 else "sharpening_bites"
    print(f"  裁定:{verdict}")

    # ---- the listing -----------------------------------------------------------------------------------
    with open(a.md, "w") as fh:
        w = fh.write
        w("# G5:那 123 条,逐条列出(2026-08-21)\n\n")
        w("声明 `g5_named_competitor_declaration.json`。数据 `g5_named_competitor.json`。$0,离线。\n\n")
        w("**这是什么:** G2 那批骨架可形成、被出厂面板确认(`panel_k >= 1`)、而 `γ* < 0` 的设计。\n")
        w("**这不是什么:** 不是面板误收的证明(`g2_audit.md` 已撤回那个说法),也不是对 G1 那个 NULL 的翻案。\n")
        w("自由能差说的是**相对布居**,不是不可能;整条线的排序信息已被测出可由交叉茎 GC 计数复现。\n\n")
        w("**γ† = E_MFE_nested − E_skeleton − dG_stack** —— 把 γ\\* 里的【嵌套集合】换成【嵌套 MFE 这一个\n")
        w("能打印出来的结构】。`E_MFE >= F_nested` 是恒等式,所以 `γ† >= γ*`,**γ†<0 是更强的条件**:\n")
        w("满足它的分子输给的是一个具体结构,不是一个总和。逐条自检违反 0 处。\n\n")
        w(f"| | 条数 | 试管通过率 | k=1/2/3 |\n|---|---|---|---|\n")
        w(f"| γ\\* < 0(G2 那批) | {len(L)} | {pr(L):.4f} | "
          f"{'/'.join(str(sum(1 for r in L if r['panel_k']==k)) for k in (1,2,3))} |\n")
        w(f"| **γ† < 0**(输给一个可点名的结构) | **{len(dag0)}** | "
          f"{('%.4f' % pr(dag0)) if dag0 else 'n/a'} | "
          f"{'/'.join(str(sum(1 for r in dag0 if r['panel_k']==k)) for k in (1,2,3))} |\n")
        w(f"| γ† < +2.4(按 dG_stack 偏差修正) | {len(dag24)} | "
          f"{('%.4f' % pr(dag24)) if dag24 else 'n/a'} | "
          f"{'/'.join(str(sum(1 for r in dag24 if r['panel_k']==k)) for k in (1,2,3))} |\n\n")
        w("列的含义:`k` 面板确认数;`确认者` 哪几家判官;`γ*` 对集合;`γ†` 对单个 MFE 结构;\n")
        w("`GC` 交叉茎里的 G-C 配对数(G1 测出它单独就能复现整条线的排序信息,放在这里供对照);\n")
        w("`共享` = 嵌套 MFE 与靶标骨架共享的配对数 / 骨架配对总数 —— 竞争者离预期结构有多远;\n")
        w("**`抢位` = 嵌套 MFE 用掉了【交叉茎需要、而骨架里本来空着】的核苷酸数 / 交叉茎需要的总数。**\n")
        w("**抢位 = 0 ⇒ 这个『竞争者』根本不跟假结争碱基,γ†<0 只是拿大结构比小结构,应当剔除。**\n")
        w("抢位越接近 1 ⇒ 它就是『把假结吃掉的那个嵌套替代品』,即真正的竞争者。见 `g5_k3_read.md`。\n\n")
        cur = None
        for r in L:
            if r["panel_k"] != cur:
                cur = r["panel_k"]
                w(f"\n## k = {cur}(共 {sum(1 for x in L if x['panel_k']==cur)} 条)\n\n")
                w("| # | 靶标 | 方法 | L | γ\\* | **γ†** | E_MFE | E_skel | dG_stack | GC | 共享 | **抢位** | 试管 | |\n")
                w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
                n = 0
            n += 1
            w(f"| {n} | {r['puzzle']} | {r['method'][:22]} | {r['L']} | {r['gamma_star']:+.2f} | "
              f"**{r['gamma_dag']:+.2f}** | {r['E_mfe']:.2f} | {r['E_skeleton']:.2f} | "
              f"{r['dG_stack']:.2f} | {r['gc_cross']} | "
              f"{r['mfe_shares_skel_pairs']}/{r['skel_pairs']} | "
              f"**{r['stolen_nt']}/{r['cross_nt']}** | {r['wet']:.1f} | "
              f"{'通过' if r['wet']>=TUBE_PASS else '**失败**'} |\n")
        w("\n---\n\n## 序列与竞争者结构(供独立复核)\n\n")
        w("每条给出序列、靶标骨架、以及**嵌套 MFE 这个具体竞争者**。任何人可以用 ViennaRNA 重跑三行验证。\n\n")
        from rna_harness.benchmarks.rnainvbench import _round_only
        for i, r in enumerate(L, 1):
            w(f"### {i}. {r['puzzle']} / {r['method']}  ")
            w(f"k={r['panel_k']} 确认者={','.join(r['confirmed_by']) or '-'}  ")
            w(f"γ\\*={r['gamma_star']:+.2f} γ†={r['gamma_dag']:+.2f} 试管={r['wet']:.1f}\n\n")
            w(f"```\nseq  {r['seq']}\nskel {_round_only(T[r['puzzle']]['target'])}\nmfe  {r['mfe_struct']}\n```\n\n")
    print(f"\n  -> {a.md}")

    json.dump({"verdict": verdict, "kills": {"K1": K1, "K2": K2},
               "n_total": len(rows), "n_clean": len(clean), "n_confirmed_clean": len(conf),
               "n_list": len(L), "n_gamma_dag_lt0": len(dag0), "n_gamma_dag_lt24": len(dag24),
               "pass_rate": {"gamma_star_lt0": pr(L),
                             "gamma_dag_lt0": pr(dag0), "gamma_dag_lt24": pr(dag24)},
               "gates": {"join_bad": 0, "k_mismatch": 0, "identity_violations": 0,
                         "dropped_unformable": len(rows) - len(clean), "failed": len(err)},
               "rows": L,
               "not_a_claim": ("不是面板误收的证明,也不是对 G1 那个 NULL 的翻案;"
                               "自由能差是相对布居,不是不可能。")},
              open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
