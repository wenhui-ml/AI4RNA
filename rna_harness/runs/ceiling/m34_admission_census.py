#!/usr/bin/env python3
"""M34 census: the admission ledger, from both implementations of the criterion.

A census, not an experiment. No effect is estimated here and none may be.

The manuscript prints how many constructs the criterion refused and under which reason. Until
this file existed those counts were TYPED INTO make_numbers.py by hand -- laundered through put()
where make_check.py, which only inspects the .tex, could not see them. Recomputing them turned up
something worth stating rather than hiding: the two implementations of the criterion keep
different books.

  designaudit.mining.mine  is the packaged criterion the paper points readers to. It walks
      constructs, and reports exclusions per construct under named reasons.
  m26_model_scored.build   is the script that produced the paper's numbers. It walks rows of the
      released table and applies the same clauses in a different order.

Both admit 4317 interventions -- that is the number every result rests on and it agrees exactly.
Their intermediate tallies do not agree, and cannot: they count different objects. This file
records both, so the manuscript can say which ledger a printed count comes from.

COST: $0, offline, no folding (pair_prob is stubbed so only the admission logic runs).
"""
from __future__ import annotations

import json, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, "/data/wenhui/openclaude_rna/rna_harness/tools")
import m26_model_scored as M26
from designaudit import corpora, mining


def main() -> int:
    out = {}

    # ---- ledger A: the packaged criterion
    cs = corpora.openknot_m2r(M26.SRC)
    admitted, rep = mining.mine(cs)
    out["designaudit"] = {
        "n_constructs": int(rep.n_input),
        "n_admitted": int(rep.n_admitted),
        "excluded": {k: int(v) for k, v in rep.excluded.items() if v},
        "excluded_total": int(sum(rep.excluded.values())),
        "by_pair_type": {k: int(v) for k, v in rep.by_pair_type.items()},
        "unaccounted": int(rep.n_input - rep.n_admitted - sum(rep.excluded.values())),
    }

    # ---- ledger B: the script that produced the paper's numbers
    M26.pair_prob = lambda seq, a, b, cache: 0.0
    recs, _dropped = M26.build()
    out["m26_build"] = {"n_rows": M26.EXCL["n_rows"], "n_admitted": len(recs),
                        "excluded": {k: v for k, v in M26.EXCL.items() if k != "n_rows" and v}}

    a, b = out["designaudit"]["n_admitted"], out["m26_build"]["n_admitted"]
    out["implementations_agree_on_admission"] = bool(a == b)
    out["n_admitted"] = a
    print(f"designaudit  constructs {out['designaudit']['n_constructs']}  "
          f"admitted {a}  excluded {out['designaudit']['excluded']}")
    print(f"m26.build    rows {out['m26_build']['n_rows']}  "
          f"admitted {b}  excluded {out['m26_build']['excluded']}")
    print(f"agree on admission: {out['implementations_agree_on_admission']}")
    if a != b:
        print("  !! the two criteria admit different sets; nothing may be reported until resolved")
        out["VOID"] = "admission_mismatch"

    json.dump(out, open(os.path.join(HERE, "m34_admission_census.json"), "w"),
              indent=2, ensure_ascii=False)
    return 0 if a == b else 1


if __name__ == "__main__":
    raise SystemExit(main())
