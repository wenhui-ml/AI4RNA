#!/usr/bin/env python
"""D4S -- does the fourth domain have an out-of-loop calibrator at all?

Pre-registration: d4s_linker_smoke_declaration.json.

Objective 4 scores the RANGE of criterion-manufacturing domains, and section A of its itemised table holds
exactly three rows: the pseudoknot panel, co-transcriptional folding, verifier arbitration.  Every one of
those earned its row on a judge that was never in the loop -- the tube, Kinfold, held-out pairwise judgement.
So a fourth domain's first step is not an experiment about the domain, it is this: does an out-of-loop
calibrator exist?  Without one the row would be the fifth unsourced claim, and the four already in the
denominator are what is holding the score down.

The candidate domain is two-state switches, where the criterion genuinely is missing: nothing accepted
decides whether a designed switch actually switches.  IntaRNA looked like the calibrator and is not one --
different algorithm family, but it reports "using Vienna RNA package 2.7.0", so it shares Turner parameters
with RNAcofold.  Algorithm independence without energy-model independence is the same shared-error-path trap
that forced domain one to go to the tube.  The one candidate left is RibonanzaNet on a linked construct:
learned, no energy model at all.  But the linker trick has never been validated, and that is what this run
tests -- not the switches, the calibrator.

Rule 29 governs the shape: the smoke case must sit on the side where the property is TRUE and must assert
it.  The true side is the rows RNAcofold says are held, and the run asserts their inter-chain pair set is
non-empty.  Last time I checked a crossing-pair constraint with a nested example and the smoke "passed"
something it had never tested.
"""
from __future__ import annotations

import argparse, json, os, random, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)

LINKER = "AAAAAAAA"
FLOOR_ROWS = 10


def pair_table(struct):
    st_, out = [], set()
    for i, c in enumerate(struct):
        if c == "(":
            st_.append(i)
        elif c == ")" and st_:
            out.add((st_.pop(), i))
    return out


def boot(v, draws, seed):
    rng = random.Random(seed); n = len(v)
    m = sorted(st.mean([v[rng.randrange(n)] for _ in range(n)]) for _ in range(draws))
    return st.mean(v), m[int(0.025 * draws)], m[int(0.975 * draws)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "switch_bench.json"))
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--out", default=os.path.join(HERE, "d4s_linker_smoke.json"))
    a = ap.parse_args()

    rows = json.load(open(a.src))["rows"]
    cases, drop = [], {}
    for r in rows:
        ch = ((r.get("free") or {}).get("chosen")) or {}
        seq, trig, cs = ch.get("sequence"), ch.get("trigger"), ch.get("complex_structure")
        if not (seq and trig and cs) or not ch.get("held"):
            drop["缺列或 held != true"] = drop.get("缺列或 held != true", 0) + 1
            continue
        if len(cs) != len(seq) + len(trig):
            drop["complex_structure 长度对不上"] = drop.get("complex_structure 长度对不上", 0) + 1
            continue
        inter = {(i, j) for i, j in pair_table(cs) if i < len(seq) <= j}
        if not inter:
            drop["真链间对为空"] = drop.get("真链间对为空", 0) + 1
            continue
        cases.append({"id": r["task_id"], "seq": seq, "trig": trig, "inter": sorted(inter)})
    print(f"held == true 且真链间对非空的行 {len(cases)} / {len(rows)};剔除 {drop}")
    # rule 29: the property under test must be TRUE on the smoke side, and asserted
    assert all(c["inter"] for c in cases), "真侧有行的链间配对为空 —— 这一侧的性质不为真,smoke 不成立"
    if len(cases) < FLOOR_ROWS:
        print(f"  低于声明下限 {FLOOR_ROWS} -> 人群不足,不给数")
        json.dump({"verdict": "void", "n": len(cases), "drop": drop}, open(a.out, "w"),
                  indent=1, ensure_ascii=False)
        return

    sys.path.insert(0, "/data/wenhui/RibonanzaNet")
    from rnet_reproduce import build_model, pairwise_logits, pairs_from_logits
    model = build_model()

    def inter_recall(seq, trig, truth):
        """Recall of the TRUE inter-chain pairs, read off RibonanzaNet on the linked construct.
        The linker shifts chain B by len(LINKER), so truth indices are remapped, not the prediction."""
        con = seq + LINKER + trig
        got = pairs_from_logits(pairwise_logits(model, con))
        off = len(LINKER)
        want = {(i, j + off) for i, j in truth}
        return len(got & want) / len(want)

    rng = random.Random(0)
    d, tr, fa = [], [], []
    for n, c in enumerate(cases, 1):
        t = inter_recall(c["seq"], c["trig"], c["inter"])
        sh = list(c["trig"]); rng.shuffle(sh)
        f = inter_recall(c["seq"], "".join(sh), c["inter"])
        tr.append(t); fa.append(f); d.append(t - f)
        if n % 10 == 0:
            print(f"    {n}/{len(cases)}", flush=True)
    pt, lo, hi = boot(d, a.draws, 0)
    ex = lo > 0 or hi < 0
    print(f"\n  真侧链间召回 均值 {st.mean(tr):.4f} 中位 {st.median(tr):.4f}")
    print(f"  假侧(trigger 打乱)   均值 {st.mean(fa):.4f} 中位 {st.median(fa):.4f}")
    print(f"  配对差 真−假  均值 {pt:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
          f"{'排除 0' if ex else '跨过 0'};正/负 "
          f"{sum(1 for x in d if x>0)}/{sum(1 for x in d if x<0)}")

    v = "calibrator_exists" if (ex and pt > 0) else "calibrator_absent"
    why = ("linker 构造能让一个【能量模型无关】的判官看见链间结合 -> 第四个领域有环外校准物,"
           "可以立下一个声明去真正造那个判据"
           if v == "calibrator_exists" else
           "linker 构造给不出能量模型无关的链间读数 -> 【第四个领域按这条路走不通】。"
           "这本身是交付物:它把『开领域四』从一个想法变成一个有据的否定,而且花的是一晚而不是一周")
    print(f"\n=== 判读 ===\n  结论 {v}\n  {why}")
    print("  IntaRNA 不能充当这个校准物:它是不同算法族,但自报 using Vienna RNA package 2.7.0,")
    print("  与 RNAcofold 共用同一套 Turner 参数 —— 算法独立,能量模型不独立。")
    json.dump({"verdict": v, "why": why, "n": len(cases), "drop": drop, "linker": LINKER,
               "true_mean": st.mean(tr), "false_mean": st.mean(fa),
               "diff_mean": pt, "ci": [lo, hi], "excludes_zero": bool(ex),
               "per_case": [{"id": c["id"], "true": t, "false": f}
                            for c, t, f in zip(cases, tr, fa)]},
              open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n写出 {a.out}")


if __name__ == "__main__":
    main()
