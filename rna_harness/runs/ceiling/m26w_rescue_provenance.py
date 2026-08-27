#!/usr/bin/env python3
"""M26w: show that the outcome contains no model, by reproducing it from raw reactivity.

rescue_factor is the outcome of every claim in this line of work and the corpus supplies it. If it
were computed with a folding model, everything would be circular. This recomputes it from the
per-nucleotide reactivity columns alone, on the design window, and reports the agreement. Offline,
no ViennaRNA, no model of any kind.
"""
from __future__ import annotations
import csv, collections, json, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = "/data/wenhui/openknot_data/Data/OK7a_M2R_data.v4.5.1.csv"


def main():
    csv.field_size_limit(10 ** 9)
    rows = list(csv.DictReader(open(SRC)))
    rk = [c for c in rows[0] if c.startswith("reactivity_")]
    prof = lambda r: np.array([float(r[c]) if r[c] not in ("", "NA", "nan") else np.nan for c in rk])
    by = collections.defaultdict(dict)
    for r in rows:
        a = r["mutA"] if r["mutA"] not in ("", "NA", "nan") else None
        b = r["mutB"] if r["mutB"] not in ("", "NA", "nan") else None
        by[(r["puzzle"], r["method"])][(a, b)] = r
    got, mine = [], []
    for _cell, v in by.items():
        wt = v.get((None, None))
        if wt is None:
            continue
        pw = prof(wt)
        ss, se = int(float(wt["sub_start"])), int(float(wt["sub_end"]))
        for (a, b), r in v.items():
            if not (a and b) or r["rescue_factor"] in ("", "NA", "nan"):
                continue
            ra, rb = v.get((a, None)), v.get((None, b))
            if ra is None or rb is None:
                continue
            di, dj, dij = prof(ra) - pw, prof(rb) - pw, prof(r) - pw
            m = ~(np.isnan(di) | np.isnan(dj) | np.isnan(dij))
            m[:ss - 1] = False; m[se:] = False
            if m.sum() < 10:
                continue
            den = np.hypot(np.linalg.norm(di[m]), np.linalg.norm(dj[m]))
            if den <= 0:
                continue
            mine.append(1 - np.linalg.norm(dij[m]) / den)
            got.append(float(r["rescue_factor"]))
    got, mine = np.array(got), np.array(mine)
    rg, rm = got.argsort().argsort(), mine.argsort().argsort()
    out = {"n": len(got),
           "corpus_mean": round(float(got.mean()), 4), "corpus_sd": round(float(got.std()), 4),
           "reproduced_mean": round(float(mine.mean()), 4),
           "reproduced_sd": round(float(mine.std()), 4),
           "pearson": round(float(np.corrcoef(got, mine)[0, 1]), 6),
           "spearman": round(float(np.corrcoef(rg, rm)[0, 1]), 6),
           "max_abs_diff": round(float(np.abs(got - mine).max()), 6),
           "median_abs_diff": round(float(np.median(np.abs(got - mine))), 6),
           "formula": "R = 1 - |d_ij| / hypot(|d_i|,|d_j|) on the design window"}
    print(json.dumps(out, indent=1, ensure_ascii=False))
    json.dump(out, open(os.path.join(HERE, "m26w_rescue_provenance.json"), "w"),
              ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
