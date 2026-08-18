#!/usr/bin/env python3
"""What does each of our certificate tiers actually MEAN in a test tube? The harness's core number.

WHY THIS IS THE ONE THAT WAS MISSING. The product's value is not a sequence -- a non-expert cannot tell a
good sequence from a bad one -- it is a sequence WITH a grade they can act on. Every measurement this
project has made about its criterion asked "does the grade correlate with quality". None asked the question
a user actually has: "if it says knot_confirmed, how often is it wrong in the tube?" A correlation does not
answer that. A failure rate does.

    THE FIELD'S OWN THRESHOLD, not one we pick. Eterna defines OpenKnot Score > 80 as "highly likely to
    form the target structure" and > 90 as a high-confidence successful design. Both are reported, so the
    conclusion cannot be tuned by moving a line we drew ourselves.

    THE POPULATION. 4,560 designs that were actually synthesised and chemically mapped, from 13 methods.
    Our panel adjudicates each one exactly as certify.py would, reconstructing the shipped tiers:
        knot_confirmed   at least one oracle reproduces ALL crossing pairs (panel_k >= 1)
        knot_screened    no oracle does, but mean crossing recall >= 0.1
        not_found        mean crossing recall < 0.1  -- the product refuses to answer
    The grade ladder k = 1, 2, 3 is broken out separately: if the ladder means anything, tube failure must
    fall as k rises, and that is a prediction that can fail.

    WHAT A HARNESS IS ALLOWED TO CLAIM, fixed before the numbers. Not "our designs are better" -- the
    designs here are other people's. The claim under test is that OUR LABEL IS INFORMATIVE ABOUT THE TUBE:

    C1  SEPARATION. Failure rate must fall monotonically across not_found -> knot_screened ->
        knot_confirmed. A non-monotone ladder means the tiers are not ordered and the certificate's
        wording is wrong.
    C2  SELECTIVE RISK. The failure rate inside knot_confirmed, with a Clopper-Pearson interval. This is
        the number the certificate should be quoting to users and currently does not.
    C3  THE PRICE OF COVERAGE. Refusing the bottom tier costs coverage and buys accuracy; the trade is
        reported as a curve, not as one flattering point.
    C4  IS IT BETTER THAN NO HARNESS AT ALL? The base rate over all designs. If knot_confirmed's failure
        rate is not clearly below it, the label carries no information and says so.

    CLUSTERING. A user experiences per-design rates, so those are reported as the headline. But designs
    from one molecule are not independent, so every comparison is ALSO computed per target and reported --
    pooling designs is the pseudo-replication behind this project's retracted p = 2.05e-28.

COST: $0. Everything is on disk.
"""
from __future__ import annotations
import argparse, collections, json, os, statistics as st, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SCREEN = 0.1
THRESHOLDS = (80.0, 90.0)


def tier(d):
    if d["panel_k"] >= 1:
        return "knot_confirmed"
    if d["panel_mean"] >= SCREEN:
        return "knot_screened"
    return "not_found"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "criterion_vs_tube.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "label_calibration.json"))
    a = ap.parse_args()
    from external_judge import clopper_pearson
    from m2r_pairwise import wilcoxon

    D = json.load(open(a.src))["designs"]
    for d in D:
        d["tier"] = tier(d)
    by_tier = collections.defaultdict(list)
    for d in D:
        by_tier[d["tier"]].append(d)
    ORDER = ["not_found", "knot_screened", "knot_confirmed"]
    out = {"n": len(D), "thresholds": list(THRESHOLDS), "tiers": {}}

    print(f"{len(D)} 条真正合成过的设计,用出厂判据逐条定档\n")
    print(f"  {'档位':16s} {'条数':>6s} {'占比':>7s} " +
          "".join(f"{'试管<'+str(int(t))+' 失败率':>18s}" for t in THRESHOLDS) + f"{'试管分中位':>11s}")
    base = {}
    for t in THRESHOLDS:
        k = sum(1 for d in D if d["wet"] < t)
        base[t] = k / len(D)
    for name in ORDER:
        ds = by_tier[name]
        if not ds:
            continue
        row = {"n": len(ds), "share": len(ds) / len(D), "wet_median": st.median([d["wet"] for d in ds])}
        cells = ""
        for t in THRESHOLDS:
            k = sum(1 for d in ds if d["wet"] < t)
            lo, hi = clopper_pearson(k, len(ds))
            row[f"fail_{int(t)}"] = k / len(ds)
            row[f"fail_{int(t)}_ci"] = [lo, hi]
            cells += f"{k/len(ds):>9.1%} [{lo:.2f},{hi:.2f}]"
        print(f"  {name:16s} {len(ds):6d} {len(ds)/len(D):7.1%} {cells} {row['wet_median']:11.1f}")
        out["tiers"][name] = row
    print(f"  {'(不用 harness)':16s} {len(D):6d} {'100.0%':>7s} " +
          "".join(f"{base[t]:>9.1%} {'':>9s}" for t in THRESHOLDS) +
          f"{st.median([d['wet'] for d in D]):11.1f}")
    out["base_rate"] = {str(int(t)): base[t] for t in THRESHOLDS}

    print(f"\n=== C1 分离度:失败率必须随档位单调下降 ===")
    seq = [out["tiers"][n]["fail_80"] for n in ORDER if n in out["tiers"]]
    mono = all(seq[i] >= seq[i + 1] for i in range(len(seq) - 1))
    print(f"  试管<80 失败率  " + "  ->  ".join(f"{x:.1%}" for x in seq) +
          f"   {'单调,档位是有序的' if mono else 'KILL —— 档位不是有序的,证书的措辞是错的'}")
    out["C1_monotone"] = bool(mono)

    print(f"\n=== C2 等级阶梯 k=1/2/3:如果阶梯有意义,失败率必须继续下降 ===")
    lad = {}
    for k in (1, 2, 3):
        ds = [d for d in D if d["panel_k"] == k]
        if len(ds) < 30:
            print(f"  k={k}  {len(ds)} 条,样本不足"); continue
        f = sum(1 for d in ds if d["wet"] < 80) / len(ds)
        lo, hi = clopper_pearson(sum(1 for d in ds if d["wet"] < 80), len(ds))
        print(f"  k={k}  {len(ds):5d} 条  试管<80 失败率 {f:.1%} [{lo:.2f},{hi:.2f}]  "
              f"试管分中位 {st.median([d['wet'] for d in ds]):.1f}")
        lad[k] = {"n": len(ds), "fail_80": f, "ci": [lo, hi]}
    out["ladder"] = lad

    print(f"\n=== C3 覆盖的代价:交付谁、失败率多少 ===")
    print(f"  {'策略':34s} {'覆盖':>8s} {'试管<80 失败率':>16s} {'相对不用 harness':>18s}")
    strat = [("全部交付(不用 harness)", ORDER),
             ("拒掉 not_found", ["knot_screened", "knot_confirmed"]),
             ("只交付 knot_confirmed", ["knot_confirmed"])]
    for lab, keep in strat:
        ds = [d for d in D if d["tier"] in keep]
        f = sum(1 for d in ds if d["wet"] < 80) / len(ds)
        print(f"  {lab:34s} {len(ds)/len(D):8.1%} {f:16.1%} {f-base[80.0]:+18.1%}")
        out.setdefault("coverage_curve", []).append(
            {"strategy": lab, "coverage": len(ds) / len(D), "fail_80": f})

    print(f"\n=== C4 逐靶标复核(设计不独立,所以同一比较在靶标层面再做一次)===")
    per = collections.defaultdict(dict)
    for p in {d["puzzle"] for d in D}:
        ds = [d for d in D if d["puzzle"] == p]
        for name in ORDER:
            g = [d for d in ds if d["tier"] == name]
            if len(g) >= 5:
                per[p][name] = sum(1 for d in g if d["wet"] < 80) / len(g)
    pairs = [(per[p]["not_found"], per[p]["knot_confirmed"]) for p in per
             if "not_found" in per[p] and "knot_confirmed" in per[p]]
    if len(pairs) >= 6:
        d2 = [x - y for x, y in pairs]
        _, pv, _ = wilcoxon(d2, "greater")
        print(f"  {len(pairs)} 个靶标同时有两档:not_found 的失败率 − knot_confirmed 的失败率")
        print(f"  中位差 {st.median(d2):+.1%}  更高 {sum(1 for x in d2 if x>0)}/{len(d2)}  "
              f"p = {'n/a' if pv is None else f'{pv:.3g}'}")
        out["C4"] = {"n_targets": len(pairs), "median_diff": st.median(d2), "p": pv}
    else:
        print(f"  只有 {len(pairs)} 个靶标同时有两档,不足以做靶标层面的检验")

    print(f"\n=== 这些数字【不】说明什么 ===")
    print(f"  这 4,560 条是别人的设计。标签用在【我们自己的】设计上是外推,没有直接证据,")
    print(f"  因为我们的设计没有、也不可能有湿实验对照 —— 这正是要去 OpenKnot 轮次前瞻验证的原因。")
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()


def curve(D, key, name, thresh=80.0):
    """Risk-coverage: sweep an acceptance threshold on `key` and record (coverage, tube failure rate)."""
    vals = sorted({d[key] for d in D})
    out = []
    for v in vals:
        ds = [d for d in D if d[key] >= v]
        if len(ds) < 100:
            continue
        out.append((len(ds) / len(D), sum(1 for d in ds if d["wet"] < thresh) / len(ds), v))
    return out


def compare():
    """Does our label beat the best available alternative AT MATCHED COVERAGE?

    The tier table shows knot_confirmed failing 20.8% at 57.8% coverage. But RibonanzaNet's crossed-pair F1
    is sitting right there as a filter anyone could use, and comparing two filters at DIFFERENT coverages
    says nothing -- a stricter filter is supposed to have a lower failure rate. The only honest comparison
    fixes coverage and asks which filter has fewer tube failures inside it. If RNet wins at matched
    coverage, our panel's label adds nothing a user could not get from a published model, and that is the
    finding.
    """
    import json as _j, os as _o, bisect
    D = _j.load(open(_o.path.join(HERE, "criterion_vs_tube.json")))["designs"]
    for d in D:
        d["tier"] = tier(d)
    ours = curve(D, "panel_mean", "panel")
    rnet = curve(D, "f1c", "rnet")
    print(f"\n=== 匹配覆盖率的对照:我们的面板 vs RibonanzaNet 当过滤器 ===")
    print(f"  {'目标覆盖':>8s} {'面板 覆盖/失败率':>22s} {'RNet 覆盖/失败率':>22s} {'谁更好':>8s}")
    rows = []
    for tgt in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3):
        a = min(ours, key=lambda x: abs(x[0] - tgt))
        b = min(rnet, key=lambda x: abs(x[0] - tgt))
        who = "面板" if a[1] < b[1] - 0.005 else ("RNet" if b[1] < a[1] - 0.005 else "打平")
        print(f"  {tgt:8.0%} {a[0]:12.1%}/{a[1]:<8.1%} {b[0]:12.1%}/{b[1]:<8.1%} {who:>8s}")
        rows.append({"target_coverage": tgt, "panel": a, "rnet": b, "winner": who})
    base = sum(1 for d in D if d["wet"] < 80) / len(D)
    print(f"  不过滤:覆盖 100.0% / 失败率 {base:.1%}")
    _j.dump(rows, open(_o.path.join(HERE, "label_calibration_vs_rnet.json"), "w"),
            ensure_ascii=False, indent=1)
    print(f"  wrote label_calibration_vs_rnet.json")


if __name__ == "__main__":
    compare()
