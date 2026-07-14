"""JSON-over-stdio bridge: the contract between the openclaude TS tools and this Python
science layer.

Usage:
    python -m rna_harness.cli <command>            # JSON request on stdin
    python -m rna_harness.cli <command> --json '<inline json>'
    echo '{"sequence":"GGGGAAAACCCC","target":"((((....))))"}' | python -m rna_harness.cli analyze

Every command returns a single JSON object on stdout with an "ok" field. Errors are
returned as {"ok": false, "error": "..."} with a non-zero exit code, so the TS side can
distinguish failures. This is deliberately stateless and deterministic.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict

from .actuation.diagnosis import diagnose
from .constraints import ConstraintSpec, check_constraints
from .determinism import four_coordinate_report
from .metrics import ensemble_metrics
from .oracles.registry import build_panel, cross_oracle_consistency
from .oracles.vienna import ViennaRNAOracle
from .task import DesignTask, OracleConfig
from .verifier import verify


def _oracle_from(d: Dict) -> ViennaRNAOracle:
    oc = OracleConfig.from_dict(d.get("oracle"))
    return ViennaRNAOracle(param_set=oc.param_set, temperature=oc.temperature, dangles=oc.dangles)


def _task_from(req: Dict) -> DesignTask:
    """Accept either a full {"task": {...}} or a lightweight {target, constraints, oracle}."""
    if "task" in req and isinstance(req["task"], dict):
        return DesignTask.from_dict(req["task"])
    return DesignTask(
        task_id=req.get("task_id", "adhoc"),
        target=req["target"],
        constraints=ConstraintSpec.from_dict(req.get("constraints")),
        family=req.get("family", "adhoc"),
        oracle=OracleConfig.from_dict(req.get("oracle")),
    )


# --------------------------------------------------------------------------- handlers
def h_fold(req: Dict) -> Dict:
    orc = _oracle_from(req)
    struct, energy = orc.fold_mfe(req["sequence"])
    return {"mfe_structure": struct, "mfe_energy": round(energy, 4)}


def h_analyze(req: Dict) -> Dict:
    orc = _oracle_from(req)
    target = req.get("target")
    em = ensemble_metrics(orc, req["sequence"], target) if target else None
    out = {"analysis": em.as_dict() if em else orc.analyze(req["sequence"]).__dict__}
    if req.get("cross_oracle") and target:
        out["cross_oracle"] = cross_oracle_consistency(
            req["sequence"], target, build_panel(req.get("panel", "default"))
        )
    return out


def h_verify(req: Dict) -> Dict:
    task = _task_from(req)
    res = verify(
        task,
        req["sequence"],
        compute_cross_oracle=bool(req.get("cross_oracle", True)),
        panel_kind=req.get("panel", "default"),
    )
    out = res.to_dict()
    # Mode-A guard: if the harness pinned the canonical target, refuse to report solved when the
    # agent verified against a DRIFTED target (its own mis-typed dot-bracket). Prevents the false
    # is_umfe=true that makes the agent stop "satisfied" against a wrong target.
    pinned = (os.environ.get("RNA_TASK_TARGET") or "").strip()
    tgt = (req.get("target") or "").strip()
    if pinned and tgt and tgt != pinned and not (len(tgt) < len(pinned) and tgt in pinned):
        out["solved"] = False
        out["is_umfe"] = False
        out["target_mismatch"] = (
            f"NOT a real solve: you verified against a {len(tgt)}-nt target, but the task target is "
            f"{len(pinned)} nt. Re-copy the EXACT task target verbatim and design against it.")
    return out


def h_diagnose(req: Dict) -> Dict:
    if "folded" in req:
        folded = req["folded"]
    else:
        orc = _oracle_from(req)
        folded, _ = orc.fold_mfe(req["sequence"])
    return diagnose(req["target"], folded).to_dict()


def h_constraints(req: Dict) -> Dict:
    spec = ConstraintSpec.from_dict(req.get("constraints"))
    return check_constraints(req["sequence"], spec).to_dict()


def h_benchmark_task(req: Dict) -> Dict:
    source = req.get("source", "eterna100")
    if source != "eterna100":
        raise ValueError(f"unknown benchmark source {source!r}")
    from .benchmarks import eterna100_tasks

    tasks = eterna100_tasks(version=req.get("version", "V2"), ids=req.get("ids"), max_len=req.get("max_len"))
    return {"tasks": [t.to_dict() for t in tasks], "count": len(tasks)}


def h_generate(req: Dict) -> Dict:
    task = _task_from(req)
    method = req.get("method", "local_search")
    budget = int(req.get("budget", 2000))
    seed = req.get("seed", 0)
    if method == "rnainverse":
        from .generators import RNAinverseGenerator

        gen = RNAinverseGenerator(gc_only=bool(req.get("gc_only", False)))
    elif method == "local_search":
        from .generators import LocalSearchGenerator

        gen = LocalSearchGenerator()
    elif method == "desirna":
        from .generators import DesiRNAGenerator

        gen = DesiRNAGenerator(replicas=int(req.get("replicas", 10)))
    else:
        raise ValueError(f"unknown generator method {method!r}")
    res = gen.design(task, budget=budget, seed=seed)
    out = res.to_dict()
    out["verification"] = verify(task, res.best_sequence, compute_cross_oracle=bool(req.get("cross_oracle", False))).to_dict()
    return out


def h_four_coords(req: Dict) -> Dict:
    task = _task_from(req)
    return four_coordinate_report(task, list(req["sequences"]))


HANDLERS = {
    "fold": h_fold,
    "analyze": h_analyze,
    "verify": h_verify,
    "diagnose": h_diagnose,
    "constraints": h_constraints,
    "benchmark_task": h_benchmark_task,
    "generate": h_generate,
    "four_coords": h_four_coords,
}


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(json.dumps({"ok": True, "commands": sorted(HANDLERS)}))
        return 0
    cmd = argv[0]
    if cmd not in HANDLERS:
        print(json.dumps({"ok": False, "error": f"unknown command {cmd!r}", "commands": sorted(HANDLERS)}))
        return 2
    # request from --json or stdin
    req: Dict = {}
    if "--json" in argv:
        req = json.loads(argv[argv.index("--json") + 1])
    else:
        raw = sys.stdin.read().strip()
        req = json.loads(raw) if raw else {}
    try:
        result = HANDLERS[cmd](req)
        print(json.dumps({"ok": True, **result}))
        return 0
    except Exception as e:  # noqa: BLE001
        import traceback

        print(json.dumps({"ok": False, "error": str(e), "trace": traceback.format_exc()[-1500:]}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
