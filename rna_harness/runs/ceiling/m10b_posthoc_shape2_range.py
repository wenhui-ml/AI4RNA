#!/usr/bin/env python3
"""POST-HOC: the shape-2 check that the M10b declaration required and the script never implemented.

Declaration `rule_64_unexpected_shapes.shape_2`:
  "共同靶标上的长度跨度比全集窄很多(< 60%)⇒ range_restriction 后缀,S 型压缩会同时缩小两条 rho。"

`m10b_pairwise_restriction.py` contains no such check -- grep for range/span/跨度 returns nothing.
This file computes it after the fact and labels it as such. It changes no frozen verdict.
Offline, $0.
"""
import os, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from m1c_skeleton_bookkeeping import build                       # noqa: E402
from m8b_target_length import CORE, BLIND, run_pop               # noqa: E402


def main():
    D, names, core_fn = {}, {}, {fn for _n, fn in CORE}
    for nm, fn in CORE + BLIND:
        keep, M = build(fn, 14)
        d = run_pop(nm, keep, M, 200, 0, 9500)
        if d is not None:
            D[fn] = d; names[fn] = nm

    def span(fn, ts):
        v = [D[fn]["len"][t] for t in ts]
        return max(v) - min(v), (st.pstdev(v) if len(v) > 1 else 0.0)

    print("对 (a × b)                       n   全集跨度a  共同跨度  比值a   全集跨度b  比值b   sd比a   sd比b")
    print("-" * 104)
    ks = list(D); bad = []
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            fa, fb = ks[i], ks[j]
            C = sorted(set(D[fa]["used"]) & set(D[fb]["used"]))
            if len(C) < 25:
                continue
            sa, da = span(fa, D[fa]["used"]); sb, db = span(fb, D[fb]["used"])
            ca, va = span(fa, C); cb, vb = span(fb, C)
            ra, rb = ca / sa, cb / sb
            sra, srb = (va / da if da else 0), (vb / db if db else 0)
            strat = "core" if (fa in core_fn and fb in core_fn) else "blind"
            flag = "  **<0.6**" if min(ra, rb) < 0.6 else ""
            if min(ra, rb) < 0.6:
                bad.append((names[fa], names[fb]))
            print(f"[{strat:<5}] {names[fa]:<11}× {names[fb]:<11} {len(C):3d}"
                  f"  {sa:7.1f}  {ca:7.1f}  {ra:.3f}  {sb:7.1f}  {rb:.3f}  {sra:.3f}  {srb:.3f}{flag}")
    print(f"\n跨度比 < 0.6 的对:**{len(bad)}** ⇒ "
          f"{'range_restriction 后缀成立' if bad else 'shape_2 不触发 —— 限制到共同靶标【没有】压掉长度跨度'}")


if __name__ == '__main__':
    main()
