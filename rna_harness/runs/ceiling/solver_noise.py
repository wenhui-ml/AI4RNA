#!/usr/bin/env python3
"""How often does the SAME request give a DIFFERENT verdict? Measured as a by-product, and it is large.

WHERE THIS CAME FROM. grade_max_bench.py ran two arms that differ only in which knot_confirmed candidate is
kept. The `certified` path returns before any of that code is reached, so for targets that land in
certified/cannot_certify the two arms are SAME-CONFIGURATION REPLICATES of one request -- an unplanned but
exact reproducibility test, on the shipped path, with a fixed seed.

22.9% of them flipped (16 of 70; 9 one way, 7 the other -- symmetric, so it is noise and not the change).

THE SEED DOES NOT MAKE THIS DETERMINISTIC, and that is the point. DesiRNA is bounded by WALL CLOCK, so the
same seed explores a different number of candidates depending on machine load. At the `cheap` budget that is
a single 5s tier, and the verdict a caller receives is not reproducible.

WHY IT MATTERS BEYOND THIS RUN. Every paired experiment in this project that re-runs the solver is exposed:
a difference of a few points between arms can be load, not treatment. grade_max survives because its
direction is 17-0 (sign test p=7.6e-06), which noise cannot produce -- but a smaller effect would not have
survived, and nothing in the earlier runs was checked against this.
"""
from __future__ import annotations
import json, os, sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "grade_max_bench.json")
    R = [r for r in json.load(open(src))["rows"]
         if "error" not in r.get("old", {}) and "error" not in r.get("new", {})]
    rep = [r for r in R if {r["old"]["verdict"], r["new"]["verdict"]} <= {"certified", "cannot_certify"}]
    flip = [r for r in rep if r["old"]["verdict"] != r["new"]["verdict"]]
    a = sum(1 for r in flip if r["old"]["verdict"] == "certified")
    print(f"同配置重复(certified/cannot_certify 路径,不经过被改动的代码):{len(rep)} 个靶标")
    print(f"  判决翻转 {len(flip)} 个 = {len(flip)/max(1,len(rep)):.1%}"
          f"   ({a} 个 certified->cannot_certify,{len(flip)-a} 个反向;对称即噪声)")
    kc = [r for r in R if r["old"]["verdict"] == r["new"]["verdict"] == "knot_confirmed"]
    up = sum(1 for r in kc if r["new"]["k"] > r["old"]["k"])
    dn = sum(1 for r in kc if r["new"]["k"] < r["old"]["k"])
    n = up + dn
    p = sum(comb(n, i) for i in range(up, n + 1)) / 2 ** n if n else float("nan")
    print(f"\n对照:等级最大化的方向 {up} 升 / {dn} 降,符号检验单边 p={p:.3g}")
    print(f"  -> 噪声会给出大致对称的升降;这个方向不是噪声能产生的")
    json.dump({"n_replicate": len(rep), "n_flip": len(flip), "flip_rate": len(flip)/max(1, len(rep)),
               "grade_up": up, "grade_down": dn, "sign_p": p},
              open(os.path.join(HERE, "solver_noise.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
