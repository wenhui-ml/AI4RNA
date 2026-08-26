#!/usr/bin/env python3
"""M13 probes: the four that must fire, exactly as frozen in c080083e.

P0 structural identity   -- treatment := gc_frac (already matched to 0.02) => zero pairs must survive
P1 gaussian noise        -- 12 synthetic columns, delta_min QUANTILE-MATCHED (rule 71 width matching)
P2 constructive ladder   -- rule 71: dial measured in the gate's units (wet), top rung >= main arm max
P3 permutation           -- rule 72: shuffle gc_paired within cell, leave wet / gc_frac / pB untouched

COST: $0, offline.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from m13_gc_placement import (assemble, match, boot, quantile_matched_delta, synth,
                              perm_within_cell, POPS, DELTA, CAL_GC, RHOS, R_NOISE, DRAWS)   # noqa: E402

PASS = 0.8


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default=os.path.join(HERE, "m13_features.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "m13_probes.json"))
    a = ap.parse_args()
    C = json.load(open(a.cache))
    MAIN = json.load(open(os.path.join(HERE, "m13_gc_placement.json")))["pops"]
    print("=== M13 四根必响探针(声明 c080083e) ===\n")
    D = {}
    for nm, fn in POPS:
        D[fn] = (nm, assemble(C[fn]["rows"], a.seed))
    OUT = {}

    # ---------------- P0 structural identity
    print("--- P0 结构性恒等式:处理列换成 gc_frac(已被 caliper 0.02 配平),delta 仍 0.05 ---")
    p0 = {}
    for nm, fn in POPS:
        d = D[fn][1]
        M = match(d, d["gcf"], DELTA, use_panel=True)
        p0[nm] = 0 if M is None else M["npair"]
        print(f"  {nm:12s} 存活配对 {p0[nm]}")
    OUT["P0"] = p0
    p0_ok = all(v == 0 for v in p0.values())
    print(f"  P0 {'通过(全部恰好 0)' if p0_ok else '不通过'}\n")

    # ---------------- P1 gaussian noise, width matched
    print(f"--- P1 等宽纯噪声 {R_NOISE} 根(delta 按分位数配平,不照抄 0.05) ---")
    p1 = {}
    for nm, fn in POPS:
        d = D[fn][1]; rng = np.random.default_rng(1000 + a.seed)
        sd = float(np.std(d["gcp"])) or 1.0
        pos = neg = 0; npairs = []
        for k in range(R_NOISE):
            col = rng.standard_normal(len(d["y"])) * sd
            for idx in d["cells"]:
                col[idx] -= col[idx].mean()
            dl = quantile_matched_delta(d, col)
            M = match(d, col, dl, use_panel=True)
            if M is None:
                continue
            b = boot(M, a.seed + k); npairs.append(M["npair"])
            pos += b["excl_pos"]; neg += (b["excl"] and not b["excl_pos"])
        p1[nm] = {"pos": pos, "neg": neg, "median_pairs": int(np.median(npairs)) if npairs else 0}
        real = MAIN[fn]["A"]["npair"] if MAIN.get(fn, {}).get("A") else 0
        print(f"  {nm:12s} 正向排除0 {pos}/{R_NOISE}  负向 {neg}  "
              f"噪声臂中位配对 {p1[nm]['median_pairs']} vs 真列 {real}")
    OUT["P1"] = p1
    p1_ok = all(v["pos"] <= 1 for v in p1.values())
    print(f"  P1 {'通过(各 <=1)' if p1_ok else '不通过'}\n")

    # ---------------- P3 permutation (rule 72)
    print(f"--- P3 第72条置换:格内打乱 gc_paired,wet / gc_frac / pB 一律不动 ---")
    p3 = {}
    for nm, fn in POPS:
        d = D[fn][1]; rng = np.random.default_rng(2000 + a.seed)
        pos = neg = 0; npairs = []
        for k in range(R_NOISE):
            col = perm_within_cell(d, rng)
            assert np.allclose(np.sort(col), np.sort(d["gcp"]))          # marginal unchanged
            M = match(d, col, DELTA, use_panel=True)
            if M is None:
                continue
            b = boot(M, a.seed + k); npairs.append(M["npair"])
            pos += b["excl_pos"]; neg += (b["excl"] and not b["excl_pos"])
        p3[nm] = {"pos": pos, "neg": neg, "median_pairs": int(np.median(npairs)) if npairs else 0}
        real = MAIN[fn]["A"]["npair"] if MAIN.get(fn, {}).get("A") else 0
        print(f"  {nm:12s} 正向排除0 {pos}/{R_NOISE}  负向 {neg}  "
              f"置换臂中位配对 {p3[nm]['median_pairs']} vs 真列 {real}")
    OUT["P3"] = p3
    p3_ok = all(v["pos"] <= 1 for v in p3.values())
    print(f"  P3 {'通过(各 <=1)' if p3_ok else '不通过'}\n")

    # ---------------- P2 ladder (rule 71)
    print("--- P2 构造性梯级(第71条:刻度实测,顶档必须 >= 主臂最大,六人群全做) ---")
    lad = {}
    for rho in RHOS:
        eff, hit, tot = [], 0, 0
        for nm, fn in POPS:
            d = D[fn][1]; rng = np.random.default_rng(int(rho * 1e4) + a.seed)
            for k in range(DRAWS):
                col = synth(d, rho, rng)
                dl = quantile_matched_delta(d, col)
                M = match(d, col, dl, use_panel=True)
                if M is None:
                    continue
                b = boot(M, a.seed + k, nb=500)
                eff.append(b["mean"]); hit += b["excl_pos"]; tot += 1
        lad[rho] = {"E_effect": round(float(np.mean(eff)), 4) if eff else None,
                    "detect": round(hit / tot, 3) if tot else None, "n": tot}
        print(f"  rho={rho:.2f}  实测 E[配对差]={lad[rho]['E_effect']:+.4f} wet  "
              f"达标比例 {lad[rho]['detect']:.2f}  (n={tot})")
    OUT["P2_ladder"] = {str(k): v for k, v in lad.items()}
    thr = next((lad[r]["E_effect"] for r in RHOS if lad[r]["detect"] and lad[r]["detect"] >= PASS), None)
    top = max(v["E_effect"] for v in lad.values() if v["E_effect"] is not None)
    mains = [MAIN[fn]["A"]["mean"] for _n, fn in POPS if MAIN.get(fn, {}).get("A")]
    mx = max(mains)
    OUT["threshold"], OUT["top_rung"], OUT["main_max"] = thr, top, mx
    print(f"\n  检出阈值(首个达标比例>={PASS} 的档) = {thr}   顶档 = {top:+.4f}   主臂最大 = {mx:+.4f}")
    p2_ok = top >= mx
    print(f"  P2(b) 顶档 >= 主臂最大: {'通过' if p2_ok else '不通过 => K0 作废'}")
    k4 = (thr is None) or (thr > mx)
    print(f"  K4 {'触发 => no_adjudicating_power' if k4 else f'不触发(余量 {mx/thr:.1f} 倍)'}\n")

    OUT["gates"] = {"P0": p0_ok, "P1": p1_ok, "P2": p2_ok, "P3": p3_ok, "K4": k4}
    json.dump(OUT, open(a.out, "w"), ensure_ascii=False, indent=1)
    print("K0:", "不触发" if (p0_ok and p1_ok and p2_ok and p3_ok) else "触发 => 整轮作废")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
