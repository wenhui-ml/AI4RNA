"""Eterna100 benchmark loader.

Source: https://github.com/eternagame/eterna100-benchmarking (branch `master`).
One TSV holds both V1 and V2 targets. We evaluate V2 under installed ViennaRNA 2.7.2
(turner2004 era == "vienna2" convention), per IMPLEMENTATION_SPEC.md.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional

from ..task import DesignTask, OracleConfig

_HERE = os.path.dirname(__file__)
DEFAULT_TSV = os.path.join(_HERE, "..", "data", "eterna100_puzzles.tsv")
RAW_URL = "https://raw.githubusercontent.com/eternagame/eterna100-benchmarking/master/data/eterna100_puzzles.tsv"

_STRUCT_COL = {"V1": 2, "V2": 3}
# We evaluate V2 under installed 2.7.2 (Vienna2 convention), so pick the Vienna2 sample for
# V2 (col 11 = "Sample Solution (V2/Vienna2)") and the Vienna1 sample for V1 (col 8).
_SAMPLE_COL = {"V1": 8, "V2": 11}


def _ensure_tsv(path: str) -> str:
    if os.path.exists(path):
        return path
    # Best-effort download if missing.
    import urllib.request

    os.makedirs(os.path.dirname(path), exist_ok=True)
    urllib.request.urlretrieve(RAW_URL, path)
    return path


def load_eterna100(path: Optional[str] = None, version: str = "V2") -> List[Dict[str, object]]:
    """Return a list of {id, name, target, sample} dicts, skipping placeholder cells."""
    if version not in _STRUCT_COL:
        raise ValueError("version must be 'V1' or 'V2'")
    path = _ensure_tsv(path or DEFAULT_TSV)
    col = _STRUCT_COL[version]
    scol = _SAMPLE_COL[version]
    out: List[Dict[str, object]] = []
    with open(path, newline="") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)  # header
        for row in r:
            if len(row) <= col:
                continue
            struct = row[col].strip()
            if not (set(struct) <= set("().") and struct):
                continue  # placeholder like (Unsolved)/(Undisclosed)
            sample = row[scol].strip() if len(row) > scol else ""
            if not (set(sample.upper()) <= set("ACGU") and sample):
                sample = ""
            out.append(
                {"id": int(row[0]), "name": row[1].strip(), "target": struct, "sample": sample.upper()}
            )
    return out


def eterna100_tasks(
    path: Optional[str] = None,
    version: str = "V2",
    ids: Optional[List[int]] = None,
    max_len: Optional[int] = None,
) -> List[DesignTask]:
    """Load Eterna100 as DesignTask objects (pure inverse folding, no extra constraints)."""
    puzzles = load_eterna100(path, version)
    oracle = OracleConfig(param_set="turner2004", temperature=37.0, dangles=2)
    tasks: List[DesignTask] = []
    for p in puzzles:
        if ids is not None and p["id"] not in ids:
            continue
        if max_len is not None and len(p["target"]) > max_len:
            continue
        tasks.append(
            DesignTask(
                task_id=f"eterna100:{p['id']}:{p['name']}",
                target=p["target"],
                family="eterna100",
                version=version,
                oracle=oracle,
                meta={"puzzle_id": p["id"], "name": p["name"], "sample_solution": p["sample"]},
            )
        )
    return tasks
