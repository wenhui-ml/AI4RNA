#!/usr/bin/env python3
"""Build an INDEPENDENT cross-dataset pseudoknot pool for chomsky_climb (experiment C, second source).

Source: RNAInvBench inverse_rna_folding set (bpRNA-derived; origin != Pseudobase++), the has_pk=True,
size-comparable (<=120 nt), single-pairing (no multiplet) structures. Reconstructs the CROSSING dot-bracket
from the pair list with a greedy multi-bracket assignment (the shipped pair2dotBraket flattens pseudoknots),
keeps only structures that actually cross, and saves a deterministic sample of 60 targets to
bprna_pk/targets.json. This is a genuinely different distribution from Pseudobase++ -- a true cross-dataset
generalization test, size-matched to the first held-out pool.
"""
import gzip, json, os, pickle, warnings
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = "/data/wenhui/RNAInvBench/data/inverse_rna_folding_train_dotbracket.pkl.gz"
BRACKETS = [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")]


def crosses(a, b):
    (i, j), (k, l) = a, b
    return (i < k < j < l) or (k < i < l < j)


def to_crossing_db(pairs, n):
    """Greedy multi-bracket dot-bracket from canonical pairs; returns None if >4 mutually-crossing levels."""
    # use ALL pair types: crossing (pseudoknot) pairs are encoded as type != 0; has_multiplet==False
    # guarantees each base is in <=1 pair, so the full pair list is a valid matching.
    ps = sorted([(min(i, j), max(i, j)) for i, j, t in pairs if i != j])
    if not ps:
        return None
    levels = []  # each level = list of pairs placed with one bracket type
    assign = {}
    for p in ps:
        placed = False
        for li, lv in enumerate(levels):
            if not any(crosses(p, q) for q in lv):
                lv.append(p); assign[p] = li; placed = True; break
        if not placed:
            levels.append([p]); assign[p] = len(levels) - 1
    if len(levels) > len(BRACKETS):
        return None
    db = ["."] * n
    for (i, j), li in assign.items():
        o, c = BRACKETS[li]
        db[i] = o; db[j] = c
    return "".join(db)


def has_crossing(db):
    return any(c in db for c in "[]{}<>")


def main(n_sample=60, seed=0):
    import pandas as pd
    df = pickle.load(gzip.open(SRC, "rb"))
    df = df[(df["has_pk"] == True) & (df["length"] <= 120) & (df["length"] >= 20)
            & (df["has_multiplet"] == False)]
    out = []
    for _, r in df.iterrows():
        db = to_crossing_db(r["pairs"], int(r["length"]))
        if db and has_crossing(db) and len(db) == int(r["length"]):
            out.append(db)
    # dedup + deterministic sample (no RNG: stride through the sorted-unique list)
    uniq = sorted(set(out))
    stride = max(1, len(uniq) // n_sample)
    sample = uniq[::stride][:n_sample]
    os.makedirs(os.path.join(HERE, "bprna_pk"), exist_ok=True)
    rows = [{"id": f"bprna_pk:{k}", "target": t} for k, t in enumerate(sample)]
    json.dump({"source": "RNAInvBench/inverse_rna_folding (bpRNA-derived, ArchiveII/bpRNA origin)",
               "n_candidates": len(uniq), "n_sample": len(rows), "rows": rows},
              open(os.path.join(HERE, "bprna_pk", "targets.json"), "w"), indent=1)
    from collections import Counter
    def npk(db):  # crossing-pair count = grade
        import sys; sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
        return sum(1 for c in db if c in "[]{}<>") // 2
    print(f"independent PK pool: {len(uniq)} valid crossing candidates -> sampled {len(rows)}")
    print("length range:", min(len(r["target"]) for r in rows), "-", max(len(r["target"]) for r in rows))
    print("grade (crossing-pair count) dist:", dict(sorted(Counter(npk(r["target"]) for r in rows).items())))
    print("sample:", rows[0]["target"])
    print("-> bprna_pk/targets.json")


if __name__ == "__main__":
    main()
