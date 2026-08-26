#!/usr/bin/env python3
"""Cache the M12 sequence-intrinsic features for all six populations, once.

Everything M13 does is re-analysis of the SAME per-molecule features M12 used; nothing here
recomputes or redefines a column. Import is from m12_sequence_intrinsic unchanged.
COST: $0, offline.
"""
from __future__ import annotations
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from m12_sequence_intrinsic import build                                          # noqa: E402

POPS = (("样本2(主)", "panel_f1_fresh.json"), ("基准4560", "panel_f1_designs.json"),
        ("样本3", "panel_f1_third.json"), ("第四批", "panel_f1_fourth.json"),
        ("第五批(盲)", "panel_f1_fifth.json"), ("第六批(盲)", "panel_f1_sixth.json"))


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(HERE, "m13_features.json"))
    a = ap.parse_args()
    out = {}
    for nm, fn in POPS:
        rows = build(fn, a.workers)
        out[fn] = {"pop": nm, "rows": rows}
        print(f"{nm:12s} {fn:26s} molecules={len(rows)}", flush=True)
    json.dump(out, open(a.out, "w"))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
