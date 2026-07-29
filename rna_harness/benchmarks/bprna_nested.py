"""A NESTED benchmark that is not Eterna100 -- the only way to test the phrase "common nested structures".

WHY THIS EXISTS. The product table's first row claims performance on "common nested structures" and every
number behind it comes from Eterna100-V2. Eterna100 is a curated set of hard PUZZLES with known solutions,
not a sample of the structures a user would bring, and its length distribution shows it:

    length percentile        25%    50%    75%    90%    max
    Eterna100-V2              67    104    258    389    400
    bpRNA-derived nested      74    102    127    196    500

The medians nearly coincide; Eterna's tail is far heavier, and the tail is exactly where the shipped cascade
fails (14/25 above 250 nt). So this source is a genuinely different distribution, not a re-skin.

SOURCE. RNAInvBench's inverse_rna_folding train split (bpRNA-derived, 66767 rows), filtered to
has_pk == False, has_multiplet == False, valid == True -> 60787 nested structures. The same file already
backs the crossing cross-source pool in runs/cc_ascent/bprna_pk.py, so the provenance is established.

TWO PREDICTIONS, DECLARED BEFORE THE RUN, pointing opposite ways -- which is what makes this a test:
  * EASIER  than Eterna, because the long tail is much lighter and length is where we fail.
  * HARDER  than Eterna, because these are NATURAL structures with no guarantee a sequence exists that folds
    to them under the Turner model, whereas every Eterna puzzle ships a solution.

SAMPLING IS DETERMINISTIC AND DECLARED: filter, drop duplicate dot-brackets, keep 12 <= length <= 400 (the
Eterna span, so lengths stay comparable and cost stays bounded), sort by Id, take an evenly spaced stride.
No randomness, no re-rolling, and nothing about the sample depends on how the results come out.
"""
from __future__ import annotations

import gzip
import os
import pickle
import warnings
from typing import Dict, List, Optional

SRC = os.environ.get("RNAINVBENCH_NESTED",
                     "/data/wenhui/RNAInvBench/data/inverse_rna_folding_train_dotbracket.pkl.gz")
MIN_LEN, MAX_LEN = 12, 400


def available(path: Optional[str] = None) -> bool:
    return os.path.exists(path or SRC)


def load_bprna_nested(n: int = 100, path: Optional[str] = None) -> List[Dict[str, object]]:
    """A deterministic n-target nested sample. Returns rows shaped like load_eterna100's."""
    p = path or SRC
    if not os.path.exists(p):
        raise FileNotFoundError(f"RNAInvBench nested source not found at {p}; set RNAINVBENCH_NESTED")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pickle.load(gzip.open(p, "rb"))
    d = df[(df.has_pk == False) & (df.has_multiplet == False) & (df.valid == True)]   # noqa: E712
    d = d[(d.length >= MIN_LEN) & (d.length <= MAX_LEN)]
    seen, rows = set(), []
    for _, r in d.sort_values("Id").iterrows():
        db = str(r["dotbracket"])
        if db in seen or set(db) - set("().") or len(db) != int(r["length"]):
            continue                       # nested only, and the stored length must match the string
        seen.add(db)
        rows.append({"id": str(r["Id"]), "name": str(r["Id"]), "target": db, "n": len(db)})
    if not rows:
        return []
    stride = max(1, len(rows) // n)
    out = rows[::stride][:n]
    for k, row in enumerate(out):
        row["idx"] = k                     # a stable small integer for per-cell filenames
    return out


if __name__ == "__main__":
    import statistics as st
    R = load_bprna_nested()
    L = sorted(x["n"] for x in R)
    print(f"bpRNA-derived nested sample: {len(R)} targets")
    print(f"  length  min {L[0]}  25% {L[len(L)//4]}  median {st.median(L)}  "
          f"75% {L[3*len(L)//4]}  max {L[-1]}")
    for band, lo, hi in (("<=100nt", 0, 100), ("101-250", 101, 250), (">250", 251, 10 ** 9)):
        print(f"  {band:>8}: {sum(1 for x in L if lo <= x <= hi)}")
    print(f"  first 3 ids: {[x['id'] for x in R[:3]]}")
