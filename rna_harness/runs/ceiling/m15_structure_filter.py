#!/usr/bin/env python3
"""M15: does the core result survive the dataset authors' own structure filter?

Declaration: m15_structure_filter_declaration.json, frozen and pushed at 0ae2ad87 BEFORE this file
existed (the sixteenth).

OpenKnotBench's README says RNet_F1 exists to filter designs whose SHAPE profile matches the target
while the secondary structure is wrong -- mutate-map-rescue showed this happened in some Round 3
designs. So a high OpenKnot score is not by itself proof the molecule folded as intended. That is
STRUCTURAL DEGENERACY, a different thing from the base-composition artifact M13b/M13c/M14 excluded.

The filter is a predictor, and the declaration says so. It touches eligibility only -- never the
treatment, the calipers, or the outcome -- and the complement subset is reported beside it.

COST: $0, offline.
"""
from __future__ import annotations

import argparse, collections, csv, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from m13_cache import POPS                                                            # noqa: E402
from m13_gc_placement import (assemble, match, boot, quantile_matched_delta, synth,
                              perm_within_cell, DELTA, NBOOT)                          # noqa: E402

SRC = "/data/wenhui/openknot_data/Data/OpenKnotBench_data.v4.5.1.csv"
CUT, NPERM, R_NOISE, DRAWS, P_GATE = 0.80, 2000, 12, 20, 0.005   # the column is 0-1 (min 0, max 1,
# median 0.9388), so the declaration's ">= 80" reads as 0.80. Scale correction, not a criterion change.
RHOS = (0.00, 0.02, 0.05, 0.10, 0.15, 0.25, 0.35, 0.45, 0.60, 0.80)
MIN_PAIRS, PASS, SELF_MAX, TOP_MIN = 100, 0.8, 0.15, 4.0


def rnet_table():
    """(puzzle, design_sequence) -> (RNet_F1, target_openknot_score).

    The join key is `design_sequence`, not `sequence`: the panel stores the DESIGN region while
    `sequence` is the 177-nt padded construct. Joining on `sequence` matched 0 of 8787.
    """
    csv.field_size_limit(10 ** 9)
    out = {}
    with open(SRC) as f:
        for d in csv.DictReader(f):
            v = d.get("RNet_F1", "")
            w = d.get("target_openknot_score", "")
            if v in ("", "NA", "nan") or not d.get("design_sequence"):
                continue
            try:
                out[(d["puzzle"], d["design_sequence"])] = (float(v),
                                                     None if w in ("", "NA", "nan") else float(w))
            except ValueError:
                continue
    return out


def null_p(D, treat, delta, seed, obs, nperm=NPERM):
    rng = np.random.default_rng(seed); n = []
    for _ in range(nperm):
        M = match(D, perm_within_cell(D, rng), delta, use_panel=True)
        n.append(float(sum(M["sums"].values()) / sum(M["cnts"].values())) if M else 0.0)
    n = np.sort(np.array(n))
    return (float((n >= obs).sum()) + 1) / (nperm + 1), float(n.std()), n


def run(rows, seed, tag):
    D = assemble(rows, seed)
    if D is None:
        return None
    M = match(D, D["gcp"], DELTA, use_panel=True)
    if M is None or M["npair"] < 10:
        return {"npair": 0 if M is None else M["npair"], "tag": tag}
    b = boot(M, seed)
    p, sd, _n = null_p(D, D["gcp"], DELTA, seed, b["mean"])
    return {"tag": tag, "effect": b["mean"], "ci": b["ci"], "p_perm": round(p, 5),
            "null_sd": round(sd, 4), "npair": M["npair"], "ntgt": M["ntgt"],
            "n_mol": D["n"], "_D": D}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "m15_structure_filter.json"))
    a = ap.parse_args()
    print("=== M15:加上数据集作者自己的结构筛子(RNet_F1 >= 80)之后,核心结果还在吗 ===")
    print("  声明 0ae2ad87(代码存在之前已推送)。离线 $0。")
    print("  **筛子是一个预测器。它只作用于入选资格,不进处理、不进配平、不进结果;互补子集同样报出。**\n")
    RT = rnet_table()
    print(f"源表可用 (puzzle, sequence) -> RNet_F1 条目 {len(RT)}\n")
    C = json.load(open(os.path.join(HERE, "m13_features.json")))
    # build() does not keep `seq`; m13b_extra.json holds it, aligned row-by-row and already
    # asserted on (puzzle, method, wet) when it was written.
    E = json.load(open(os.path.join(HERE, "m13b_extra.json")))
    OUT, REF = {}, json.load(open(os.path.join(HERE, "m13b_gc_placement.json")))["pops"]

    for nm, fn in POPS:
        rows = C[fn]["rows"]
        if len(E[fn]) != len(rows):
            raise RuntimeError(f"{fn}: extra {len(E[fn])} != rows {len(rows)}")
        for r, e in zip(rows, E[fn]):
            r["seq"] = e["seq"]
        # -------- join with row-by-row assertion on wet
        hit = mism = miss = 0
        keep = []
        for r in rows:
            k = (r["puzzle"], r.get("seq") or "")
            v = RT.get(k)
            if v is None:
                miss += 1; continue
            f1, w = v
            if w is not None and abs(w - r["wet"]) > 1e-6:
                mism += 1; continue
            hit += 1
            r2 = dict(r); r2["rnet"] = f1; keep.append(r2)
        frac_miss = miss / max(1, len(rows))
        print(f"=== {nm} ===  分子 {len(rows)}  连上 {hit}  wet 不一致剔除 {mism}  未连上 {miss}"
              f" ({frac_miss:.1%})")
        if frac_miss > 0.20:
            print("  形状四:RNet_F1 缺失 > 20% ⇒ 该人群不可用\n")
            OUT[fn] = {"pop": nm, "usable": False, "frac_missing": round(frac_miss, 4)}
            continue
        if mism / max(1, hit + mism) > 0.05:
            print("  形状二:连接失败率 > 5% ⇒ 停,不做近似匹配\n")
            OUT[fn] = {"pop": nm, "usable": False, "join_fail": True}
            continue
        A = [r for r in keep if r["rnet"] >= CUT]
        B = [r for r in keep if r["rnet"] < CUT]
        print(f"  RNet_F1 >= {CUT:.0f}: {len(A)} ({len(A)/max(1,len(keep)):.1%})   < {CUT:.0f}: {len(B)}")
        ra = run(A, a.seed, "A_filtered"); rb = run(B, a.seed, "B_complement")
        ref = REF.get(fn, {}).get("A")
        ent = {"pop": nm, "usable": True, "n_join": hit, "frac_missing": round(frac_miss, 4),
               "n_pass": len(A), "n_fail": len(B),
               "reference_unfiltered": {"effect": ref["effect"], "p": ref["p_perm"],
                                        "npair": ref["npair"]} if ref else None}
        for r, lab in ((ra, "A 筛后"), (rb, "B 互补")):
            if r and r.get("effect") is not None:
                print(f"  {lab:8s} 效应 {r['effect']:+7.4f} {r['ci']}  置换 p {r['p_perm']:.5f}  "
                      f"零SD {r['null_sd']:.4f}  配对 {r['npair']:5d}/{r['ntgt']:3d}靶")
                ent[r["tag"]] = {k: v for k, v in r.items() if not k.startswith("_")}
            elif r:
                print(f"  {lab:8s} 配对不足 ({r['npair']})")
                ent[r["tag"]] = {"npair": r["npair"]}
        if ref:
            print(f"  参照(未筛) 效应 {ref['effect']:+7.4f}  配对 {ref['npair']}")
        # ---- probes on the filtered arm
        if ra and ra.get("_D") and ra.get("npair", 0) >= 50:
            D = ra["_D"]
            p0 = match(D, D["gcf"], DELTA, use_panel=True)
            p0n = 0 if p0 is None else p0["npair"]
            rng = np.random.default_rng(1000 + a.seed); pos = 0
            for k in range(R_NOISE):
                col = rng.standard_normal(D["n"]) * (float(np.std(D["gcp"])) or 1.0)
                for idx in D["cells"]:
                    col[idx] -= col[idx].mean()
                Mn = match(D, col, quantile_matched_delta(D, col), use_panel=True)
                if Mn and Mn["npair"] >= 10:
                    o = float(sum(Mn["sums"].values()) / sum(Mn["cnts"].values()))
                    p, _sd, _n = null_p(D, col, quantile_matched_delta(D, col), a.seed + k, o, nperm=400)
                    pos += (p < P_GATE)
            lad = {}
            import random as _r
            from m12_sequence_intrinsic import synth_col
            nullq = float(np.quantile(_nullcache(D, a.seed), .995))
            for rho in RHOS:
                hit2 = tot2 = 0
                for t in range(DRAWS):
                    col = synth_col(D["y"], D["ci"], rho, _r.Random(int(rho * 1e4) + t + a.seed))
                    Mn = match(D, col, quantile_matched_delta(D, col), use_panel=True)
                    if Mn is None or Mn["npair"] < 10:
                        continue
                    o = float(sum(Mn["sums"].values()) / sum(Mn["cnts"].values()))
                    hit2 += (o > nullq); tot2 += 1
                lad[rho] = round(hit2 / tot2, 3) if tot2 else None
            ent["P0"] = p0n; ent["P1_pos"] = pos; ent["ladder"] = {str(k): v for k, v in lad.items()}
            ent["ladder_rho0"] = lad.get(0.00)
            top = max(k for k in RHOS)
            print(f"  P0 {p0n}(须 0)  P1 {pos}/{R_NOISE}(须<=1)  "
                  f"梯级 rho=0 自检 {lad.get(0.00)}  " +
                  "  ".join(f"{r:.2f}:{lad[r]:.2f}" for r in RHOS if r > 0))
        OUT[fn] = ent
        print()

    # verdict
    us = [v for v in OUT.values() if v.get("usable")]
    sig = [v for v in us if v.get("A_filtered", {}).get("p_perm", 1) < P_GATE
           and v.get("A_filtered", {}).get("effect", 0) > 0]
    small = [v for v in us if v.get("A_filtered", {}).get("npair", 0) < MIN_PAIRS]
    if len(small) >= 4:
        vd = "underpowered_after_filter"
    elif len(sig) >= 4:
        vd = "survives_structure_filter"
    else:
        bs = [v for v in us if v.get("B_complement", {}).get("p_perm", 1) < P_GATE]
        vd = "driven_by_structurally_wrong_designs" if len(bs) >= 4 else "under_replicated"
    print(f"可用人群 {len(us)}/6;A 臂显著 {len(sig)};A 臂配对<{MIN_PAIRS} 的 {len(small)}")
    print("裁定:", vd)
    for v in OUT.values():
        v.pop("_D", None)
        for k in ("A_filtered", "B_complement"):
            if isinstance(v.get(k), dict):
                v[k].pop("_D", None)
    json.dump({"pops": OUT, "cut": CUT, "verdict": vd}, open(a.out, "w"),
              ensure_ascii=False, indent=1)
    return 0


_NC = {}


def _nullcache(D, seed):
    k = id(D)
    if k not in _NC:
        rng = np.random.default_rng(seed); n = []
        for _ in range(400):
            M = match(D, perm_within_cell(D, rng), DELTA, use_panel=True)
            n.append(float(sum(M["sums"].values()) / sum(M["cnts"].values())) if M else 0.0)
        _NC[k] = np.sort(np.array(n))
    return _NC[k]


if __name__ == "__main__":
    raise SystemExit(main())
