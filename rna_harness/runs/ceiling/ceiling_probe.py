#!/usr/bin/env python3
"""Is R1's ceiling a statement about verifiers, or about the tube?

WHY THIS EXISTS. R1 produced the number this project has been quoting all day: the tube's replicate noise
allows R^2 ~ 0.81 within cell, the shipped verdict reaches 0.047 and the whole panel reaches 0.103, from which
I concluded "the ceiling is upstream -- in folding prediction and generation, not in the verdict". I then used
that conclusion to bound every other result of the day. I never tested it.

It is testable in minutes, because RibonanzaNet is a sequence -> reactivity model trained on exactly this kind
of data and its crossed-pair F1 (`f1c`) is already a column in these files:

    if f1c alone reaches far above 0.103, then structure-side signal exists that our panel is missing, and
    "the ceiling is upstream" is WRONG -- the verdict side has several times the headroom I claimed;

    if f1c alone lands near 0.103 too, then no structural model does better on this yardstick, most of the
    0.81 is something structure cannot see, and a large amount of speculation closes.

Same machinery as R1 -- within-cell demeaning, in-sample R^2 on nested feature sets, within-cell permutation
null, cluster bootstrap over cells -- so the numbers are directly comparable to R1's.

COST: $0, offline.
"""
from __future__ import annotations
import collections, json, os, random, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from arbitration_confirm import load_fresh                                        # noqa: E402
from r1_info_ladder import feats, r2_within, demean_by_cell                       # noqa: E402


def gap_one(item):
    seq, nested = item
    import RNA
    try:
        return RNA.energy_of_struct(seq, nested) - RNA.fold(seq)[1]
    except Exception:                                                             # noqa: BLE001
        return None


def run(nm, fn, workers, nperm, draws, seed):
    from concurrent.futures import ProcessPoolExecutor
    from rna_harness.benchmarks.rnainvbench import _round_only
    from leaderboard_design import targets
    T = targets()
    rec, _sp, _m = load_fresh(os.path.join(HERE, fn), "panel_recall_product")
    nest = {p: _round_only(T[p]["target"]) for p in {r["puzzle"] for r in rec} if p in T}
    keys = [r for r in rec if feats(r) and r["puzzle"] in nest and r.get("f1c") is not None]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        gaps = list(ex.map(gap_one, [(r["seq"], nest[r["puzzle"]]) for r in keys], chunksize=64))
    cells = collections.defaultdict(list); base = {}
    for r, g in zip(keys, gaps):
        if g is None:
            continue
        f = feats(r)
        base[id(r)] = (f[0] + f[1] + f[2], float(r["f1c"]), g)
        cells[(r["puzzle"], r["method"])].append(r)
    rows = [(c, gg) for c, gg in cells.items() if len(gg) >= 3]
    SETS = (("F0 出厂档位 k", lambda b: b[0][:3]),
            ("F2 面板全部列", lambda b: b[0]),
            ("f1c 单独(RNet)", lambda b: [b[1]]),
            ("gap 单独", lambda b: [b[2]]),
            ("F2 ∪ f1c", lambda b: b[0] + [b[1]]),
            ("F2 ∪ f1c ∪ gap", lambda b: b[0] + [b[1], b[2]]))
    y, _X, ci = demean_by_cell(rows, {k: v[0] for k, v in base.items()})
    rng = np.random.default_rng(seed)
    print(f"\n=== {nm}:格 {len(rows)},分子 {len(y)} ===", flush=True)
    print(f"  {'特征集':20s} {'样本内R²':>9s} {'置换零':>8s} {'净R²':>8s}  {'对格聚类 CI(净)':>20s}")
    out = {}
    for lab, sel in SETS:
        cols = {k: sel(v) for k, v in base.items()}
        yy, X, _ = demean_by_cell(rows, cols)
        obs = r2_within(yy, X)
        nul = []
        for _ in range(nperm):
            yp = yy.copy()
            for c in range(ci.max() + 1):
                m = ci == c
                yp[m] = rng.permutation(yp[m])
            nul.append(r2_within(yp, X))
        nm_ = float(np.mean(nul))
        prng = random.Random(seed); bs = []
        for _ in range(draws):
            s2 = [rows[prng.randrange(len(rows))] for _ in range(len(rows))]
            yz, Xz, _ = demean_by_cell(s2, cols)
            v = r2_within(yz, Xz)
            if v is not None:
                bs.append(v - nm_)
        bs.sort()
        lo, hi = bs[int(0.025 * len(bs))], bs[min(len(bs) - 1, int(0.975 * len(bs)))]
        print(f"  {lab:20s} {obs:9.4f} {nm_:8.4f} {obs-nm_:8.4f}  [{lo:+.4f}, {hi:+.4f}]", flush=True)
        out[lab] = {"r2": obs, "null": nm_, "net": obs - nm_, "ci": [lo, hi]}
    wv = float(np.var(y)); sd = 4.097 / (2 ** 0.5)
    out["reference_ceiling"] = 1 - sd ** 2 / wv
    out["n_cells"], out["n_mol"] = len(rows), int(len(y))
    print(f"  参考上限(复现噪声)≈ {out['reference_ceiling']:.3f}"
          f"   |  f1c / F2 = {out['f1c 单独(RNet)']['net']/out['F2 面板全部列']['net']:.2f}×")
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--nperm", type=int, default=150)
    ap.add_argument("--draws", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "ceiling_probe.json"))
    a = ap.parse_args()
    res = {}
    for nm, fn in (("第四批", "panel_f1_fourth.json"), ("样本2", "panel_f1_fresh.json"),
                   ("样本3", "panel_f1_third.json")):
        res[fn] = run(nm, fn, a.workers, a.nperm, a.draws, a.seed)
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n写出 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
