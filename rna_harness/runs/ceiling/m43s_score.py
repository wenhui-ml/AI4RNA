#!/usr/bin/env python3
"""M43s: score M43's frozen declaration, which M43 itself did not.

`m43_reorder_declaration.json` stakes four predictions. `m43_reorder.json` carries
`staked: null`. The run computed almost everything the stakes need and then never graded them,
which breaks this paper's own rule -- a declared check that is only written down is not a check --
and would have left Appendix "prereg" claiming a scorecard that does not exist.

Stakes 2, 3 and 4 are gradable from the published artifact and are graded from it here, untouched.

Stake 1 is not: it asks whether the two nearest-neighbour implementations agree more with EACH
OTHER than either does with the transformer, and M43 computed only ONE cell of that matrix
(ViennaRNA vs RNAstructure, its V2 wiring check). The rest of the matrix is computed here, on the
same 158 wild types, through M43's own structure code so that the diagonal cell it already
published must come back identical.

Checks, before any stake is graded:
  identity   every implementation against itself is exactly 1.0
  symmetry   the matrix equals its transpose to 0.0
  reverse    the ViennaRNA/RNAstructure cell reproduces M43's published V2 median

The declaration is NOT edited. Offline, $0.
"""
from __future__ import annotations
import collections, csv, json, os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
OUTJ = os.path.join(HERE, "m43s_score.json")
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)


def main():
    t0 = time.time()
    import m43_reorder as M43
    import m16_rescue as M16
    import RNA

    DECL = json.load(open(os.path.join(HERE, "m43_reorder_declaration.json")))
    PUB = json.load(open(os.path.join(HERE, "m43_reorder.json")))
    out = {"scores": "m43_reorder_declaration.json", "reads": "m43_reorder.json",
           "cost": "offline, $0", "declaration_not_edited": True,
           "why": "m43_reorder.json carries staked:null; the declaration stakes four predictions"}

    # ---------------------------------------------------------------- the wild types M43 used
    csv.field_size_limit(10 ** 9)
    cellinfo = {}
    for d in csv.DictReader(open(M43.M26_SRC)):
        if d["mutA"] in ("", "NA", "nan") and d["mutB"] in ("", "NA", "nan"):
            cellinfo[(d["puzzle"], d["method"])] = d
    recs = [r for r in csv.DictReader(open(os.path.join(HERE, "m43_predictor_scores.csv")))]
    used = sorted({(r["puzzle"], r["method"]) for r in recs})
    cells = [(k, cellinfo[k]) for k in used if k in cellinfo]
    wts = [d["sequence"] for _k, d in cells]
    log(f"设计格 {len(cells)}(M43 发表 {PUB['n_cells']})  野生型 {len(set(wts))}")

    # ---------------------------------------------------------------- structures, M43's own code
    log("\n=== 四个实现在同一批野生型上的原生结构")
    rs_bpp, rs_mfe = M43.rnastructure_all(sorted(set(wts)))
    ef = M43.eternafold_structures(sorted(set(wts)))
    rnet_by_design = {}
    for d in csv.DictReader(open(M43.BENCHFILE)):
        ds, st = d.get("design_sequence"), d.get("RNet_structure")
        if ds and st and st not in ("", "NA", "nan") and len(st) == len(ds) and ds not in rnet_by_design:
            rnet_by_design[ds] = st
    rnet = {}
    for _k, d in cells:
        ss_, se_ = int(float(d["sub_start"])), int(float(d["sub_end"]))
        st = rnet_by_design.get(d["sequence"][ss_ - 1:se_])
        if st is None:
            continue
        full = ["."] * len(d["sequence"])
        for j, ch in enumerate(st):
            full[ss_ - 1 + j] = ch
        rnet[d["sequence"]] = "".join(full)

    IMPL = ("ViennaRNA", "EternaFold", "RNAstructure", "RibonanzaNet")
    struct = {}
    for _k, d in cells:
        w = d["sequence"]
        struct[w] = {"ViennaRNA": RNA.fold(w)[0], "RNAstructure": rs_mfe.get(w),
                     "EternaFold": ef.get(w), "RibonanzaNet": rnet.get(w)}
    cov = {n: sum(1 for w in struct if struct[w][n]) for n in IMPL}
    log(f"  覆盖 {cov}")

    # ---------------------------------------------------------------- pairwise agreement matrix
    # scored INSIDE the design window, exactly as R_agree is: outside it the constructs carry
    # adapters and barcodes that no implementation was asked to fold.
    log("\n=== 实现之间两两一致(设计窗口内,配对集合的 F1,按格取中位)")
    pair = collections.defaultdict(list)
    for _k, d in cells:
        w = d["sequence"]; ss, se = int(float(d["sub_start"])), int(float(d["sub_end"]))
        got = {}
        for n in IMPL:
            db = struct[w][n]
            if db and len(db) == len(w):
                got[n] = {(i, j) for i, j in M43.parse_db(db) if ss - 1 <= i < se and ss - 1 <= j < se}
        for a in IMPL:
            for b in IMPL:
                if a in got and b in got:
                    pair[(a, b)].append(M43.f1(got[a], got[b]))
    MAT = {f"{a}|{b}": round(float(np.median(pair[(a, b)])), 5) for a in IMPL for b in IMPL
           if pair[(a, b)]}
    out["pairwise_median_f1"] = MAT
    out["n_cells_pairwise"] = {f"{a}|{b}": len(pair[(a, b)]) for a in IMPL for b in IMPL}
    hdr = "  " + " " * 14 + "".join(f"{n[:12]:>14s}" for n in IMPL)
    log(hdr)
    for a in IMPL:
        log("  " + f"{a:14s}" + "".join(f"{MAT.get(f'{a}|{b}', float('nan')):14.4f}" for b in IMPL))

    # ---------------------------------------------------------------- checks
    log("\n=== 检验(先于任何评分)")
    C = {}
    C["identity"] = {n: MAT.get(f"{n}|{n}") for n in IMPL}
    ok_id = all(abs(v - 1.0) < 1e-12 for v in C["identity"].values())
    log(f"  自比恒等 {C['identity']}  -> {'通过' if ok_id else '**未通过**'}")
    asym = max(abs(MAT[f"{a}|{b}"] - MAT[f"{b}|{a}"]) for a in IMPL for b in IMPL
               if f"{a}|{b}" in MAT and f"{b}|{a}" in MAT)
    C["symmetry_max_abs"] = asym
    log(f"  对称性最大偏差 {asym:.2e}  -> {'通过' if asym < 1e-12 else '**未通过**'}")
    # The reverse control must recompute the PUBLISHED quantity, not a neighbouring one. M43's V2
    # is the median F1 between the two MFE structures over the WHOLE construct; the matrix above is
    # restricted to the design window, as R_agree is. Comparing them caught this: they differ by
    # 0.0001, which is agreement between two different quantities and not a reproduction of either.
    v2_pub = PUB["V"]["V2_mfe_agreement_vienna_vs_rnastructure"]
    full = [M43.f1(M43.parse_db(struct[w]["ViennaRNA"]), M43.parse_db(struct[w]["RNAstructure"]))
            for w in wts if struct[w]["RNAstructure"]]
    v2_now = round(float(np.median(full)), 4)
    C["reverse_v2_full_length"] = {"published": v2_pub, "ours": v2_now,
                                   "abs_diff": round(abs(v2_pub - v2_now), 6)}
    C["window_restricted_same_pair"] = MAT["ViennaRNA|RNAstructure"]
    ok_rev = abs(v2_pub - v2_now) < 1e-9
    log(f"  反向对照 ViennaRNA|RNAstructure 全长 本次 {v2_now:.4f}  M43 发表 {v2_pub:.4f}  "
        f"-> {'通过' if ok_rev else '**未通过**'}"
        f"   (窗口内的同一对是 {MAT['ViennaRNA|RNAstructure']:.4f},是另一个量)")
    C["pass"] = bool(ok_id and asym < 1e-12 and ok_rev)
    out["checks"] = C
    if not C["pass"]:
        log("  ** 检验未通过 ⇒ 不评分,按未通过写出 **")
        out["staked"] = None
        json.dump(out, open(OUTJ, "w"), ensure_ascii=False, indent=1); return 1

    # ---------------------------------------------------------------- the four stakes
    log("\n=== 记分(声明冻结于 425f3271,本次不改动它)")
    st, det = {}, {}

    nn = MAT["ViennaRNA|RNAstructure"]
    vr = MAT["ViennaRNA|RibonanzaNet"]; sr = MAT["RNAstructure|RibonanzaNet"]
    st["1_two_nn_implementations_cluster"] = bool(nn > vr and nn > sr)
    det["1"] = {"ViennaRNA_vs_RNAstructure": nn, "ViennaRNA_vs_RibonanzaNet": vr,
                "RNAstructure_vs_RibonanzaNet": sr,
                "note": "M43 computed only the first of these three and did not grade the stake"}

    ra, rr = PUB["E2"]["ranking_agree"], PUB["E2"]["ranking_resp"]
    st["2_rankings_are_not_identical"] = bool(any(ra[n] != rr[n] for n in ra))
    det["2"] = {"ranking_agree": ra, "ranking_resp": rr,
                "kendall_tau": PUB["E2"]["kendall_tau"]}

    st["3_at_least_one_swap_resolves"] = bool(PUB["E2"]["n_resolved_swaps"] >= 1)
    det["3"] = {"n_swaps": len(PUB["E2"]["swaps"]),
                "n_resolved": PUB["E2"]["n_resolved_swaps"],
                "swaps": PUB["E2"]["swaps"]}

    si, sv = PUB["span"]["implementations"], PUB["span"]["parameter_variants"]
    st["4_variants_span_less_than_implementations"] = bool(sv < si)
    det["4"] = {"implementations": si, "parameter_variants": sv,
                "ratio": round(si / sv, 2) if sv else None}

    for k, v in st.items():
        log(f"  {k:48s} {'成立' if v else '**失败**'}")
    out["staked"] = st; out["staked_detail"] = det
    out["staked_text"] = DECL["staked"]
    out["n_failed"] = sum(1 for v in st.values() if not v)

    out["elapsed_s"] = round(time.time() - t0, 1); out["log"] = LOG
    json.dump(out, open(OUTJ, "w"), ensure_ascii=False, indent=1)
    log(f"\n写出 {OUTJ}  用时 {out['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
