#!/usr/bin/env python3
"""A generic strict-predicate gate: wrap any Tool so it optimises toward the criterion actually scored.

The measured regularity behind this (runs/ceiling/DIRECTION.md): an operator satisfies WHATEVER predicate
it optimises, and a predicate it never consults rejects most of its output.

    benchmark   in-loop objective     never consulted        result
    crossing    pKiss                 ProbKnot / ShapeKnots  100% / 5% / 3.4%
    crossing    pKiss + ProbKnot      ShapeKnots             100% / 100% / 9.1%
    Eterna      MFE match             unique MFE             100% / 86%
    Eterna      MFE match             ensemble robustness    100% / median P = 0.059

Every tool in this harness verifies with `verify(seq, target)` at the default strictness 'match', and both
RemcTool and DesirnaTool do exactly that, so the strict predicate is out of loop for them in the same way
ShapeKnots was out of loop for the pseudoknot operator.

This wrapper closes that gap without touching any solver: run the inner tool, apply the STRICT predicate,
and on rejection re-run with a different seed until it passes or the budget is spent. It is deliberately
the weakest possible form of the fix -- it cannot steer the inner search, only resample it -- so whatever
it buys is a lower bound on what putting the predicate in the loop would buy. The in-loop version measured
on the pseudoknot side (DecomposePKPanelTool(sk_gate=True)) gained 1.69x capability, so this is the
transferable, solver-agnostic sibling of that result.

Two things it fixes at once:
  * capability, by not stopping at a solution the scorer will reject;
  * honesty, by never returning a sequence that fails the criterion. The ungated pseudoknot arms returned
    279 sequences of which 26 were real; the gated arm returned 44, all real.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

from .tools import Tool, ToolResult, verify


class StrictGate(Tool):
    """Wrap `inner` so a result only counts when `predicate(seq, target)` holds.

    predicate: (seq, target) -> bool. Defaults to verify(..., strictness=strictness).
    max_attempts / budget_s bound the resampling; seeds are seed + k * seed_stride so the retries are
    genuinely independent draws rather than the same trajectory continued.
    """

    def __init__(self, inner: Tool, strictness: str = "umfe",
                 predicate: Optional[Callable[[Optional[str], str], bool]] = None,
                 max_attempts: int = 8, budget_s: Optional[float] = None, seed_stride: int = 7919):
        self.inner = inner
        self.strictness = strictness
        self.predicate = predicate
        self.max_attempts = int(max_attempts)
        self.seed_stride = int(seed_stride)
        self.cores = getattr(inner, "cores", 1)
        self.budget_s = float(budget_s) if budget_s is not None \
            else float(getattr(inner, "budget_s", 60.0)) * self.max_attempts
        self.name = f"gate[{strictness}]x{self.max_attempts}({inner.name})"

    def _ok(self, seq, target):
        if seq is None:
            return False
        if self.predicate is not None:
            return bool(self.predicate(seq, target))
        ok, _bp, _o = verify(seq, target, strictness=self.strictness)
        return bool(ok)

    def attempt(self, target: str, seed: int = 0) -> ToolResult:
        t0 = time.time()
        n_try = n_reject = 0
        inner_s = gate_s = 0.0
        best: Optional[ToolResult] = None
        win: Optional[ToolResult] = None
        for k in range(self.max_attempts):
            if k and (time.time() - t0) > self.budget_s:
                break
            n_try += 1
            i0 = time.time()
            r = self.inner.attempt(target, seed=seed + k * self.seed_stride)
            inner_s += time.time() - i0
            if best is None or (r.bp is not None and best.bp is not None and r.bp < best.bp):
                best = r
            if not r.solved or not r.seq:
                continue                                  # the inner tool did not even reach its own goal
            g0 = time.time()
            ok = self._ok(r.seq, target)
            gate_s += time.time() - g0
            if ok:
                win = r
                break
            n_reject += 1

        stats = dict(getattr(best, "stats", None) or {})
        stats.update({"gate_strictness": self.strictness, "gate_attempts": n_try,
                      "gate_rejects": n_reject, "gate_s": round(gate_s, 3),
                      "inner_s": round(inner_s, 3),
                      "inner_solved_any": bool(win is not None or (best is not None and best.solved))})
        ref = win or best
        if ref is None:
            return ToolResult(self.name, False, len(target), 0.0, self.cores, 0.0, "none", None,
                              stats=stats)
        return ToolResult(self.name, win is not None, ref.bp, round(time.time() - t0, 3), self.cores,
                          sum(x.cost_usd for x in [ref]) * max(n_try, 1),
                          f"{ref.oracle}+{self.strictness}", (win.seq if win else None),
                          oracle_s=round((ref.oracle_s or 0.0) * max(n_try, 1) + gate_s, 3),
                          stats=stats)
