#!/usr/bin/env python3
"""M48: is a window-level outcome ground truth for a pair-level prediction?

Declaration `m48_locality_declaration.json`, frozen and pushed at d86621e6 BEFORE this file
existed. Offline, $0.

W1: no window is "the right one". This reports stability. The full-window result stays the
reported one whatever the others say.
"""
from __future__ import annotations
import collections, csv, json, os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m44v_verify as V44

DECL = json.load(open(os.path.join(HERE, "m48_locality_declaration.json")))
M26W = json.load(open(os.path.join(HERE, "m26w_rescue_provenance.json")))
SRC = "/data/wenhui/openknot_data/Data/OK7a_M2R_data.v4.5.1.csv"
SCORES = os.path.join(HERE, "m43_predictor_scores.csv")
WINDOWS = (5, 10, 15, 25, None)          # None = the full design window, as the corpus computes it
IMPL = V44.IMPL
COL = V44.COL
OUTJ = os.path.join(HERE, "m48_locality.json")
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)


def load():
    csv.field_size_limit(10 ** 9)
    rows = list(csv.DictReader(open(SRC)))
    rk = [c for c in rows[0] if c.startswith("reactivity_")]

    def prof(r):
        return np.array([float(r[c]) if r[c] not in ("", "NA", "nan") else np.nan for c in rk])

    by = collections.defaultdict(dict)
    for r in rows:
        a = r["mutA"] if r["mutA"] not in ("", "NA", "nan") else None
        b = r["mutB"] if r["mutB"] not in ("", "NA", "nan") else None
        by[(r["puzzle"], r["method"])][(a, b)] = r
    return by, prof


def rescue(di, dj, dij, mask):
    m = mask & ~(np.isnan(di) | np.isnan(dj) | np.isnan(dij))
    if m.sum() < 10:
        return None
    den = np.hypot(np.linalg.norm(di[m]), np.linalg.norm(dj[m]))
    if den <= 0:
        return None
    return 1 - np.linalg.norm(dij[m]) / den


def main():
    t0 = time.time()
    by, prof = load()
    out = {"declaration": "m48_locality_declaration.json", "cost": "offline, $0",
           "wording_rules": DECL["wording_rules"], "not_a_test": DECL["not_a_test"]}

    log("=== 逐条重算 rescue,窗口 = 以两个被干预位点为中心的 +-w,以及全设计窗")
    recs, v2_bad, v2_n, noop = [], 0, 0, []
    for cell, v in by.items():
        wt = v.get((None, None))
        if wt is None:
            continue
        pw = prof(wt)
        ss, se = int(float(wt["sub_start"])), int(float(wt["sub_end"]))
        L = len(pw)
        wseq = wt["sequence"]
        for (a, b), r in v.items():
            if not (a and b) or r["rescue_factor"] in ("", "NA", "nan"):
                continue
            ra, rb = v.get((a, None)), v.get((None, b))
            if ra is None or rb is None:
                continue
            # positions are 1-based inside the design region
            try:
                pa, pb = ss - 1 + int(a) - 1, ss - 1 + int(b) - 1
            except ValueError:
                continue
            # NO-OP CONTROLS, found by V2 and excluded before it rather than by relaxing it.
            #
            # V2 fired at 1.5785% against a 1% kill. The cause is not a mis-centred window: 74
            # constructs carry mutation tags whose mutation is null -- the id reads
            # `P01_MPNN-fixbb_A40A_A48A_WT`, an A at position 40 replaced by an A -- so their
            # design-region sequence is identical to the wild type and there is no intervention for
            # a window to be centred on. Three independent confirmations: the id string encodes the
            # null substitution, the windowed sequences are equal, and Definition 1 had already
            # refused all 74 under its exactly-two-positions clause, so none is in the admitted set.
            # They are therefore out of scope for a check about intervention position, and are
            # excluded before it. The declaration is NOT edited; this is recorded as an erratum.
            diff = {k for k in range(min(len(wseq), len(r["sequence"])))
                    if ss - 1 <= k < se and wseq[k] != r["sequence"][k]}
            if not diff:
                noop.append({"cell": list(cell), "id": r["id"],
                             "rescue_factor": float(r["rescue_factor"])})
                continue
            # V2: the parsed positions must be exactly where the double mutant differs from the WT
            v2_n += 1
            if diff != {pa, pb}:
                v2_bad += 1
                continue
            di, dj, dij = prof(ra) - pw, prof(rb) - pw, prof(r) - pw
            base = np.zeros(L, bool); base[ss - 1:se] = True
            vals = {}
            for w in WINDOWS:
                if w is None:
                    mask = base
                else:
                    mask = np.zeros(L, bool)
                    for p in (pa, pb):
                        mask[max(0, p - w):min(L, p + w + 1)] = True
                    mask &= base
                vals[str(w)] = rescue(di, dj, dij, mask)
            recs.append({"cell": cell, "pa": pa, "pb": pb,
                         "corpus": float(r["rescue_factor"]), "w": vals})
    # ---------------------------------------------------------------- E0 the empirical null
    # A gift from the failed check. These constructs have the wild type's design-region sequence,
    # were synthesised and measured independently, and still carry a rescue factor. That is the
    # outcome's noise floor, measured from the corpus's own controls rather than assumed.
    rf = np.array([x["rescue_factor"] for x in noop], float)
    E0 = {"n": len(noop), "mean": round(float(rf.mean()), 4), "sd": round(float(rf.std()), 4),
          "median": round(float(np.median(rf)), 4), "min": round(float(rf.min()), 4),
          "max": round(float(rf.max()), 4),
          "q": [round(float(x), 4) for x in np.quantile(rf, [0.1, 0.25, 0.5, 0.75, 0.9])],
          "frac_above_rescue_threshold": round(float((rf >= V44.THRESH).mean()), 4),
          "rescue_threshold": V44.THRESH,
          "what_it_is": ("constructs tagged with a null substitution (A40A), whose design-region "
                         "sequence equals the wild type's; independently synthesised and measured"),
          "none_admitted": True}
    log(f"\n=== E0 空操作对照(设计区序列与野生型完全相同){len(noop)} 条")
    log(f"  rescue 均值 {E0['mean']} 中位 {E0['median']} 标准差 {E0['sd']} 范围 [{E0['min']}, {E0['max']}]")
    log(f"  其中越过 rescue 判定阈 {V44.THRESH} 的比例 {E0['frac_above_rescue_threshold']:.2%}"
        f"  <- 这就是『把噪声叫成 rescue』的经验假阳性率")
    out["E0_noop_null"] = E0
    out["declaration_erratum"] = (
        "The declaration's V2 assumed every mutation-tagged double mutant carries a mutation. 74 "
        "carry a null substitution (A40A) and have the wild type's design-region sequence. V2 fired "
        "at 1.5785% against its 1% kill, the cause was diagnosed rather than the threshold relaxed, "
        "and those records are excluded before the check as out of its scope -- they contain no "
        "intervention to centre a window on, and Definition 1 had already refused all 74. The "
        "declaration is not edited.")

    frac_bad = v2_bad / max(1, v2_n)
    log(f"  可用双突变体 {len(recs)}   V2 位点核对:{v2_n} 条中不匹配 {v2_bad}"
        f" ({frac_bad:.4%})  -> {'通过' if frac_bad <= 0.01 else '**未通过**'}")
    out["V2_position_check"] = {"n_checked": v2_n, "n_mismatched": v2_bad,
                                "frac": round(frac_bad, 6), "pass": bool(frac_bad <= 0.01)}
    if frac_bad > 0.01:
        log("  ** 窗口没有以干预位点为中心 ⇒ 按 kill 作废 **")
        out["void"] = "V2"; json.dump(out, open(OUTJ, "w"), ensure_ascii=False, indent=1); return 1

    full = np.array([r["w"]["None"] for r in recs if r["w"]["None"] is not None])
    cor = np.array([r["corpus"] for r in recs if r["w"]["None"] is not None])
    pear = float(np.corrcoef(cor, full)[0, 1])
    mad = float(np.abs(cor - full).max())
    ok1 = pear >= 0.999 and mad <= 0.01
    log(f"  V1 全窗复现语料列:Pearson {pear:.6f}  最大绝对差 {mad:.6f}"
        f"  (M26w 发表 {M26W['pearson']}, {M26W['max_abs_diff']})"
        f"  -> {'通过' if ok1 else '**未通过**'}")
    out["V1_reproduces_column"] = {"pearson": round(pear, 6), "max_abs_diff": round(mad, 6),
                                   "m26w_pearson": M26W["pearson"], "pass": bool(ok1)}
    if not ok1:
        out["void"] = "V1"; json.dump(out, open(OUTJ, "w"), ensure_ascii=False, indent=1); return 1

    # ---------------------------------------------------------------- E1 agreement between windows
    log("\n=== E1 各窗口之间的一致性(与全窗的 Spearman)")
    agree = {}
    for w in WINDOWS:
        k = str(w)
        pair = [(r["w"][k], r["w"]["None"]) for r in recs
                if r["w"][k] is not None and r["w"]["None"] is not None]
        a_ = np.array([p[0] for p in pair]); b_ = np.array([p[1] for p in pair])
        rs = float(np.corrcoef(a_.argsort().argsort(), b_.argsort().argsort())[0, 1])
        agree[k] = {"n": len(pair), "spearman_vs_full": round(rs, 4),
                    "mean": round(float(a_.mean()), 4), "sd": round(float(a_.std()), 4)}
        log(f"  w={k:5s}  n={len(pair):5d}  与全窗 Spearman {rs:+.4f}  均值 {a_.mean():.4f}")
    out["E1_window_agreement"] = agree

    # ---------------------------------------------------------------- E2 headline quantities
    log("\n=== E2 用每个窗口的 rescue 重算本文的头条量")
    scores = list(csv.DictReader(open(SCORES)))
    keyed = {}
    for r in recs:
        keyed[(r["cell"][0], r["cell"][1], r["pa"], r["pb"])] = r
    # the score table keys on (puzzle, method); pair it back by matching rescue to the corpus value
    bycell = collections.defaultdict(list)
    for d in scores:
        bycell[(d["puzzle"], d["method"])].append(d)
    lut = collections.defaultdict(list)
    for r in recs:
        lut[r["cell"]].append(r)

    res = {}
    for w in WINDOWS:
        k = str(w)
        cells = {}
        for cell, ds in bycell.items():
            pool = {round(x["corpus"], 6): x for x in lut.get(cell, [])}
            ys, keep = [], []
            for d in ds:
                x = pool.get(round(float(d["rescue"]), 6))
                if x is None or x["w"][k] is None:
                    continue
                ys.append(x["w"][k]); keep.append(d)
            if len(keep) < V44.NMIN:
                continue
            c = {"y": np.array(ys, float),
                 "depth": np.array([float(d["depth"]) for d in keep]),
                 "puzzle": cell[0]}
            for n in IMPL:
                c[n] = np.array([float(d[COL[n]]) for d in keep])
            cells[cell] = c
        if len(cells) < 50:
            res[k] = {"underpowered": True, "n_cells": len(cells)}
            log(f"  w={k:5s}  可计分格仅 {len(cells)} ⇒ 判为功效不足,不比较")
            continue
        sgn = V44.depth_sign(cells)
        row = {"n_cells": len(cells), "n_interventions": sum(len(c["y"]) for c in cells.values()),
               "depth_sign": sgn}
        for f in V44.BUDGETS:
            d0 = V44.recover(cells, "depth", sgn, f)
            row[f"depth@{f}"] = round(d0, 5)
            for n in IMPL:
                row[f"{n}@{f}"] = round(V44.recover(cells, n, sgn, f) - d0, 5)
            row[f"RNet-Vie@{f}"] = round(V44.recover(cells, "RibonanzaNet", sgn, f)
                                         - V44.recover(cells, "ViennaRNA", sgn, f), 5)
        res[k] = row
        log(f"  w={k:5s} 格 {row['n_cells']}  干预 {row['n_interventions']}  "
            f"深度@25% {row['depth@0.25']:.4f}  "
            + "  ".join(f"{n[:4]}{row[f'{n}@0.25']:+.4f}" for n in IMPL)
            + f"   RNet-Vie@10% {row['RNet-Vie@0.1']:+.5f}")
    out["E2_headline_by_window"] = res

    # ---------------------------------------------------------------- staked
    live = {k: v for k, v in res.items() if not v.get("underpowered")}
    st = {
        "1_w15_ranks_with_full_above_0.8": bool(agree["15"]["spearman_vs_full"] > 0.8),
        "2_depth_matched_or_better_every_window": bool(all(
            v[f"{n}@{f}"] <= 0 or v[f"{n}@{f}"] < 0.02
            for v in live.values() for n in IMPL for f in V44.BUDGETS)),
        "3_small_budget_gap_negative_every_window": bool(all(
            v["RNet-Vie@0.1"] < 0 for v in live.values())),
        "4_no_window_makes_a_model_win": None}
    out["staked"] = st; out["staked_text"] = DECL["staked"]
    log("\n=== 预注册记分(第 4 条需区间,交由主文的 M44/M47 口径判定,此处只报点估计)")
    for k, v in st.items():
        log(f"  {k:44s} {'成立' if v else ('**失败**' if v is False else '见正文')}")

    out["elapsed_s"] = round(time.time() - t0, 1); out["log"] = LOG
    json.dump(out, open(OUTJ, "w"), ensure_ascii=False, indent=1)
    log(f"\n写出 {OUTJ}  用时 {out['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
